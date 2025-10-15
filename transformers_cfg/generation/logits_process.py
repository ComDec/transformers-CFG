import copy
import logging
import math
import os
import pprint
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import torch
from sympy import im
from transformers import PreTrainedTokenizer
from transformers.generation.logits_process import (
    LOGITS_PROCESSOR_INPUTS_DOCSTRING,
    LogitsProcessor,
)
from transformers.utils import add_start_docstrings

from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.recognizer import StringRecognizer
from transformers_cfg.token_grammar_recognizer import AbsTokenRecognizer

logger = logging.getLogger(__name__)


def _normalize_token_id_list(token_ids: Optional[Iterable[int]]) -> tuple[int, ...]:
    if token_ids is None:
        return tuple()
    if isinstance(token_ids, torch.Tensor):
        return tuple(int(token) for token in token_ids.tolist())
    return tuple(int(token) for token in token_ids)


class _TokenizerDecodeCache:
    __slots__ = ("_token_cache", "tokenizer")

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer
        self._token_cache: Dict[int, str] = {}

    def decode_token(self, token_id: int) -> str:
        token_id = int(token_id)
        cached = self._token_cache.get(token_id)
        if cached is None:
            cached = self.tokenizer.decode([token_id], skip_special_tokens=False)
            self._token_cache[token_id] = cached
        return cached

    def decode_sequence(self, token_ids: Sequence[int]) -> str:
        if not token_ids:
            return ""
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


class _TokenCacheMixin:
    def _init_token_cache(
        self,
        tokenizer: PreTrainedTokenizer,
        nice_token_ids_list: Optional[Iterable[int]] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self._decoder = _TokenizerDecodeCache(tokenizer)
        self.nice_token_ids_list = _normalize_token_id_list(nice_token_ids_list)
        self._nice_token_decodings = [
            self._decoder.decode_token(token_id) for token_id in self.nice_token_ids_list
        ]
        self._nice_token_pairs = tuple(zip(self.nice_token_ids_list, self._nice_token_decodings))
        self._all_token_decodings: Optional[Sequence[str]] = None

    def _set_string_grammar(self, recognizer: StringRecognizer) -> None:
        self.string_grammar = recognizer
        self._accept_prefix_cached = lru_cache(maxsize=262144)(recognizer._accept_prefix)
        self._accept_string_cached = lru_cache(maxsize=262144)(recognizer._accept_string)

    def _clear_accept_caches(self) -> None:
        self._accept_prefix_cached.cache_clear()
        self._accept_string_cached.cache_clear()

    def _accept_prefix(self, text: str) -> bool:
        return self._accept_prefix_cached(text)

    def _accept_string(self, text: str) -> bool:
        return self._accept_string_cached(text)

    @staticmethod
    def _bulk_lookup(values: Sequence[str], lookup_fn) -> List[bool]:
        results: List[bool] = [False] * len(values)
        cache: Dict[str, bool] = {}
        for idx, value in enumerate(values):
            if value not in cache:
                cache[value] = lookup_fn(value)
            results[idx] = cache[value]
        return results

    def _bulk_accept_prefix(self, values: Sequence[str]) -> List[bool]:
        return self._bulk_lookup(values, self._accept_prefix)

    def _bulk_accept_string(self, values: Sequence[str]) -> List[bool]:
        return self._bulk_lookup(values, self._accept_string)

    def _ensure_all_token_decodings(self) -> Sequence[str]:
        if self._all_token_decodings is None:
            self._all_token_decodings = [
                self._decoder.decode_token(idx) for idx in range(self.vocab_size)
            ]
        return self._all_token_decodings


def expand_grammar_tree(
    partial_text: str,
    parsed_grammar,
    tokenizer: PreTrainedTokenizer,
    depth: int,
    nice_token_ids_list: Optional[Iterable[int]] = None,
    execution_mode: Literal["full", "limited", "extensive"] = "limited",
) -> Dict[str, Any]:
    """Return a shallow grammar expansion tree for a partial string.

    The helper mirrors the tokenizer/grammar interface used by logits processors so
    it can be queried in the same environments. Each expansion level enumerates the
    grammar-consistent next tokens reachable from the current prefix.
    """

    class _GrammarExpansionHelper(_TokenCacheMixin):
        def __init__(self) -> None:
            self.execution_mode = execution_mode
            self._init_token_cache(tokenizer, nice_token_ids_list)
            self._set_string_grammar(
                StringRecognizer(
                    parsed_grammar.grammar_encoding,
                    parsed_grammar.symbol_table["root"],
                )
            )
            self.eos_id = tokenizer.eos_token_id
            self.eos_token = (
                self._decoder.decode_token(self.eos_id) if self.eos_id is not None else None
            )
            self._all_token_pairs: Tuple[Tuple[int, str], ...] = ()

        def candidate_pairs(self) -> Tuple[Tuple[int, str], ...]:
            if self.execution_mode == "extensive":
                if not self._all_token_pairs:
                    token_decodings = self._ensure_all_token_decodings()
                    self._all_token_pairs = tuple(enumerate(token_decodings))
                return self._all_token_pairs

            if self._nice_token_pairs:
                return self._nice_token_pairs

            if not self._all_token_pairs:
                token_decodings = self._ensure_all_token_decodings()
                self._all_token_pairs = tuple(enumerate(token_decodings))
            return self._all_token_pairs

    helper = _GrammarExpansionHelper()
    base_text = partial_text if isinstance(partial_text, str) else str(partial_text)

    root: Dict[str, Any] = {
        "prefix": base_text,
        "token": None,
        "token_id": None,
        "accepts_eos": helper._accept_string(base_text),
        "valid_prefix": helper._accept_prefix(base_text),
        "children": [],
        "is_terminal": False,
    }

    if depth <= 0 or not root["valid_prefix"]:
        return root

    candidate_pairs = helper.candidate_pairs()
    if not candidate_pairs:
        return root

    def expand_level(nodes: List[Dict[str, Any]], remaining_depth: int) -> None:
        if remaining_depth == 0 or not nodes:
            return

        prefixes = [node["prefix"] for node in nodes]
        if prefixes:
            eos_results = helper._bulk_accept_string(prefixes)
            for node, accepted in zip(nodes, eos_results):
                node["accepts_eos"] = accepted
                if accepted and helper.eos_id is not None and helper.eos_token is not None:
                    node["children"].append(
                        {
                            "token": helper.eos_token,
                            "token_id": helper.eos_id,
                            "prefix": node["prefix"],
                            "accepts_eos": True,
                            "children": [],
                            "is_terminal": True,
                        }
                    )

        candidate_strings: List[str] = []
        candidate_meta: List[Tuple[Dict[str, Any], int, str]] = []
        for node in nodes:
            if not helper._accept_prefix(node["prefix"]):
                continue
            base_prefix = node["prefix"]
            for token_id, token_str in candidate_pairs:
                candidate_strings.append(base_prefix + token_str)
                candidate_meta.append((node, token_id, token_str))

        if not candidate_strings:
            return

        acceptance_flags = helper._bulk_accept_prefix(candidate_strings)
        next_nodes: List[Dict[str, Any]] = []
        for (node, token_id, token_str), accepted in zip(candidate_meta, acceptance_flags):
            if not accepted:
                continue
            child_prefix = node["prefix"] + token_str
            child = {
                "token": token_str,
                "token_id": token_id,
                "prefix": child_prefix,
                "accepts_eos": False,
                "children": [],
                "is_terminal": False,
            }
            node["children"].append(child)
            next_nodes.append(child)

        if not next_nodes:
            return

        expand_level(next_nodes, remaining_depth - 1)

    expand_level([root], depth)
    return root


class BaseGrammarLogitsProcessor(LogitsProcessor):
    """
    Base class for grammar-based logits processors.
    This class is used to process the logits based on the grammar constraints.
    It is used to generate a single token at a time.
    """

    def __init__(self) -> None:
        self.device = None

    def set_return_dict(self, return_dict: bool):
        self.return_dict = return_dict

    def mask_logits(
        self,
    ) -> torch.FloatTensor:
        pass

    def process_logits(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, min_length: int = 0
    ) -> torch.FloatTensor:
        """
        :param input_ids:
        :param scores:
        :return:
        """
        if self.device is None:
            device = scores.device
        self.current_prefix = ["" for _ in range(input_ids.shape[0])]

        masked_scores, acceptance = self.mask_logits(input_ids, scores, device, min_length)
        if self.return_dict:
            return {"masked_logits": masked_scores, "acceptance": acceptance}
        else:
            return masked_scores

    def reset(self):
        pass

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, min_length: int = 0
    ) -> torch.FloatTensor:
        return self.process_logits(input_ids, scores, min_length)


class GrammarIncrementalLogitsProcessorGeneral(_TokenCacheMixin, BaseGrammarLogitsProcessor):
    """
    This logits processor is used to limit the tokens that can be generated.
    It is used to generate a single token at a time.
    This is a special case for number only with the following exception:
    - The decoded token is a number only, not a number with comma. Without comma, it's very hard to tell the validation of number.
        e.g. "123" in accepted, but it can be "1,23" or "12,3" or "123,".
    - The generated token_ids will connected with comma automatically.
    """

    def __init__(
        self,
        parsed_grammar,
        tokenizer: PreTrainedTokenizer,
        device: Optional[torch.device] = None,
        nice_token_ids_list: Optional[torch.tensor] = None,
        execution_mode: Literal["full", "limited"] = "limited",
    ) -> None:
        super().__init__()
        self.device = device
        self._init_token_cache(tokenizer, nice_token_ids_list)
        self.execution_mode = execution_mode
        self._set_string_grammar(
            StringRecognizer(parsed_grammar.grammar_encoding, parsed_grammar.symbol_table["root"])
        )
        self.prompt_length = 0
        self.return_dict = True

    def set_prompt_length(self, prompt_length: int):
        self.prompt_length = prompt_length

    def mask_logits(
        self,
        input_ids: torch.LongTensor,
        logits: torch.FloatTensor,
        device: torch.device,
        min_length: int = 0,
    ) -> torch.FloatTensor:
        batch_size = logits.shape[0]
        acceptance = torch.zeros((batch_size, self.vocab_size), dtype=torch.bool, device=device)

        eos_id = self.tokenizer.eos_token_id
        acceptance[:, eos_id] = False

        prompt_offset = self.prompt_length if self.prompt_length >= 0 else 0
        decoded_prefixes = [
            self._decoder.decode_sequence(token_ids)
            for token_ids in input_ids[:, prompt_offset:].tolist()
        ]

        base_acceptances = self._bulk_accept_string(decoded_prefixes)

        # if we didn't reach min_length, we cannot accept eos, drop eos probability and argmax again
        next_token_ids = torch.argmax(logits, dim=-1).tolist()
        for batch_idx in range(batch_size):
            if (input_ids.shape[1] - self.prompt_length) < min_length and next_token_ids[
                batch_idx
            ] == eos_id:
                logits[batch_idx, eos_id] = -torch.inf
                next_token_ids[batch_idx] = torch.argmax(logits[batch_idx]).item()
                if next_token_ids[batch_idx] == eos_id:
                    # if still eos, we randomly pick one token from nice tokens
                    if self._nice_token_pairs:
                        next_token_ids[batch_idx] = torch.random.choice(self._nice_token_pairs)[0]

        next_token_decodings = [
            self._decoder.decode_token(token_id) for token_id in next_token_ids
        ]

        greedy_prefixes = [
            prefix + next_dec for prefix, next_dec in zip(decoded_prefixes, next_token_decodings)
        ]
        greedy_acceptances = self._bulk_accept_prefix(greedy_prefixes)

        last_token_ids = input_ids[:, -1].tolist()
        current_length = input_ids.shape[1]

        if self.execution_mode == "extensive":
            all_token_decodings = self._ensure_all_token_decodings()
        else:
            all_token_decodings = None

        limited_mode = self.execution_mode == "limited"
        full_mode = self.execution_mode == "full"

        nice_candidate_strings: List[str] = []
        nice_candidate_meta: List[Tuple[int, int]] = []
        extensive_candidate_strings: List[str] = []
        extensive_candidate_meta: List[Tuple[int, int]] = []

        for batch_idx, base_prefix in enumerate(decoded_prefixes):
            if base_acceptances[batch_idx]:
                acceptance[batch_idx, eos_id] = True

            if greedy_acceptances[batch_idx]:
                acceptance[batch_idx, next_token_ids[batch_idx]] = True
                if limited_mode:
                    continue

            if last_token_ids[batch_idx] == eos_id and current_length >= min_length:
                acceptance[batch_idx, eos_id] = True
                continue

            if self.execution_mode in ("full", "limited") and self._nice_token_pairs:
                for token_id, token_str in self._nice_token_pairs:
                    nice_candidate_strings.append(base_prefix + token_str)
                    nice_candidate_meta.append((batch_idx, token_id))
            elif self.execution_mode == "extensive" and all_token_decodings is not None:
                for token_id, token_str in enumerate(all_token_decodings):
                    extensive_candidate_strings.append(base_prefix + token_str)
                    extensive_candidate_meta.append((batch_idx, token_id))

        if nice_candidate_strings:
            nice_results = self._bulk_accept_prefix(nice_candidate_strings)
            for (batch_idx, token_id), accepted in zip(nice_candidate_meta, nice_results):
                if accepted:
                    acceptance[batch_idx, token_id] = True

        if extensive_candidate_strings:
            extensive_results = self._bulk_accept_prefix(extensive_candidate_strings)
            for (batch_idx, token_id), accepted in zip(
                extensive_candidate_meta, extensive_results
            ):
                if accepted:
                    acceptance[batch_idx, token_id] = True
            # confilt with min_length constraint before, we don't this anymore
            # if acceptance[batch].sum() == 0:
            # This is a hacked version to make sure training can continue
            # If CFG only accept one token (eos), we regard all tokens are acceptable
            # acceptance[batch, self.tokenizer.eos_token_id] = True

        # if the logits size of the model is more than the tokennizer vocab
        # we artificially expand the acceptance tensor and block everything
        # beyond the tokenizer vocab size
        acceptance_vocab_size = acceptance.shape[-1]
        masked_logits_vocab_size = logits.shape[-1]
        if masked_logits_vocab_size != acceptance_vocab_size:
            assert (
                acceptance_vocab_size < masked_logits_vocab_size
            ), "impossible for tokenizer vocab to be less than model vocab"
            vocab_size_diff = masked_logits_vocab_size - acceptance_vocab_size
            false_tensor = torch.zeros(
                (*acceptance.shape[:-1], vocab_size_diff),
                dtype=torch.bool,
                device=device,
            )
            acceptance = torch.cat((acceptance, false_tensor), dim=-1)

        # sanity check
        for batch_idx in range(batch_size):
            if ((input_ids.shape[1] - self.prompt_length) < min_length) and (
                acceptance[batch_idx, eos_id] == True
            ):
                acceptance[batch_idx, eos_id] = False

        # Logits to -inf where False
        masked_logits = logits.masked_fill(~acceptance, -torch.inf)
        return masked_logits, acceptance


class GrammarIncrementalLogitsProcessorSampleEnhanced(_TokenCacheMixin, LogitsProcessor):
    """
    This logits processor is used to limit the tokens that can be generated.
    It is used to generate a single token at a time.
    This is a special case for number only with the following exception:
    - The decoded token is a number only, not a number with comma. Without comma, it's very hard to tell the validation of number.
        e.g. "123" in accepted, but it can be "1,23" or "12,3" or "123,".
    - The generated token_ids will connected with comma automatically.
    """

    def __init__(
        self,
        parsed_grammar,
        tokenizer: PreTrainedTokenizer,
        device: Optional[torch.device] = None,
        nice_token_ids_list: Optional[torch.tensor] = None,
        execution_mode: Literal["full", "limited"] = "limited",
        sampling_tau: float = 1.0,
    ) -> None:
        self.device = device
        self._init_token_cache(tokenizer, nice_token_ids_list)
        self.execution_mode = execution_mode
        self._set_string_grammar(
            StringRecognizer(parsed_grammar.grammar_encoding, parsed_grammar.symbol_table["root"])
        )
        self.sampling_tau = sampling_tau
        self.prompt_length = 0
        self.return_dict = True

    def set_prompt_length(self, prompt_length: int):
        self.prompt_length = prompt_length

    def mask_logits(
        self,
        input_ids: torch.LongTensor,
        logits: torch.FloatTensor,
        device: torch.device,
        min_length: int = 0,
    ) -> torch.FloatTensor:
        batch_size = logits.shape[0]
        acceptance = torch.zeros((batch_size, self.vocab_size), dtype=torch.bool, device=device)

        prompt_offset = self.prompt_length if self.prompt_length >= 0 else 0
        decoded_prefixes = [
            self._decoder.decode_sequence(token_ids)
            for token_ids in input_ids[:, prompt_offset:].tolist()
        ]

        next_tokens = torch.nn.functional.gumbel_softmax(
            logits, tau=self.sampling_tau, dim=-1
        ).argmax(dim=-1)
        next_token_ids = next_tokens.tolist()
        next_token_decodings = [
            self._decoder.decode_token(token_id) for token_id in next_token_ids
        ]
        greedy_prefixes = [
            prefix + next_dec for prefix, next_dec in zip(decoded_prefixes, next_token_decodings)
        ]
        greedy_acceptances = self._bulk_accept_prefix(greedy_prefixes)

        eos_id = self.tokenizer.eos_token_id
        last_token_ids = input_ids[:, -1].tolist()
        current_length = input_ids.shape[1]

        nice_candidate_strings: List[str] = []
        nice_candidate_meta: List[Tuple[int, int]] = []

        for batch_idx, base_prefix in enumerate(decoded_prefixes):
            if greedy_acceptances[batch_idx]:
                acceptance[batch_idx, next_token_ids[batch_idx]] = True
                continue

            if last_token_ids[batch_idx] == eos_id and current_length >= min_length:
                acceptance[batch_idx, eos_id] = True
                continue

            for token_id, token_str in self._nice_token_pairs:
                nice_candidate_strings.append(base_prefix + token_str)
                nice_candidate_meta.append((batch_idx, token_id))

        if nice_candidate_strings:
            nice_results = self._bulk_accept_prefix(nice_candidate_strings)
            for (batch_idx, token_id), accepted in zip(nice_candidate_meta, nice_results):
                if accepted:
                    acceptance[batch_idx, token_id] = True

        for batch_idx in range(batch_size):
            if not acceptance[batch_idx].any():
                acceptance[batch_idx, eos_id] = True
        # if the logits size of the model is more than the tokennizer vocab
        # we artificially expand the acceptance tensor and block everything
        # beyond the tokenizer vocab size
        acceptance_vocab_size = acceptance.shape[-1]
        masked_logits_vocab_size = logits.shape[-1]
        if masked_logits_vocab_size != acceptance_vocab_size:
            assert (
                acceptance_vocab_size < masked_logits_vocab_size
            ), "impossible for tokenizer vocab to be less than model vocab"
            vocab_size_diff = masked_logits_vocab_size - acceptance_vocab_size
            false_tensor = torch.zeros(
                (*acceptance.shape[:-1], vocab_size_diff),
                dtype=torch.bool,
                device=device,
            )
            acceptance = torch.cat((acceptance, false_tensor), dim=-1)

        # Logits to -inf where False
        masked_logits = logits.masked_fill(~acceptance, -math.inf)
        return masked_logits, acceptance

    def set_return_dict(self, return_dict: bool):
        self.return_dict = return_dict

    def process_logits(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, min_length: int = 0
    ) -> torch.FloatTensor:
        """
        :param input_ids:
        :param scores:
        :return:
        """
        if self.device is None:
            device = scores.device
        self.current_prefix = ["" for _ in range(input_ids.shape[0])]

        masked_scores, acceptance = self.mask_logits(input_ids, scores, device, min_length)
        if self.return_dict:
            return {"masked_logits": masked_scores, "acceptance": acceptance}
        else:
            return masked_scores

    def reset(self):
        pass

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, min_length: int = 0
    ) -> torch.FloatTensor:
        return self.process_logits(input_ids, scores, min_length)


class GrammarLogitsProcessorPartheseness(_TokenCacheMixin, LogitsProcessor):
    """
    This logits processor is used to limit the tokens that can be generated.
    It is used to generate a single token at a time.
    This is a special case for number only with the following exception:
    - The decoded token is a number only, not a number with comma. Without comma, it's very hard to tell the validation of number.
        e.g. "123" in accepted, but it can be "1,23" or "12,3" or "123,".
    - The generated token_ids will connected with comma automatically.
    """

    def __init__(
        self,
        parsed_grammar,
        tokenizer: PreTrainedTokenizer,
        device: Optional[torch.device] = None,
        nice_token_ids_list: Optional[torch.tensor] = None,
        execution_mode: Literal["full", "limited"] = "limited",
        max_batch_size: int = 512,
        return_dict: bool = True,
    ) -> None:
        self.device = device
        self._init_token_cache(tokenizer, nice_token_ids_list)
        self.execution_mode = execution_mode
        self._set_string_grammar(
            StringRecognizer(parsed_grammar.grammar_encoding, parsed_grammar.symbol_table["root"])
        )
        self.max_batch_size = max_batch_size
        self.return_dict = return_dict
        self.generate_idx_list(self.max_batch_size)

    def set_return_dict(self, return_dict: bool):
        self.return_dict = return_dict

    def mask_logits(
        self,
        input_ids: torch.LongTensor,
        logits: torch.FloatTensor,
        device: torch.device,
        prompt_length: int = 0,
    ) -> torch.FloatTensor:
        batch_size = logits.shape[0]
        acceptance = torch.zeros((batch_size, self.vocab_size), dtype=torch.bool, device=device)

        decoded_token_list = [
            [self._decoder.decode_token(token_id) for token_id in token_list]
            for token_list in input_ids[:, prompt_length:].tolist()
        ]

        eos_id = self.tokenizer.eos_token_id

        candidate_strings: List[str] = []
        candidate_meta: List[Tuple[int, int]] = []
        prefix_strings: List[str] = []
        prefix_lengths: List[int] = []

        for batch_idx in range(batch_size):
            start_idx = self.current_partheseness_start[batch_idx]
            prefix_tokens = decoded_token_list[batch_idx][start_idx:]
            prefix = "".join(prefix_tokens)
            prefix_strings.append(prefix)
            prefix_lengths.append(len(prefix))

            if self._nice_token_pairs:
                for token_id, token_str in self._nice_token_pairs:
                    candidate_strings.append(prefix + token_str)
                    candidate_meta.append((batch_idx, token_id))

        prefix_acceptances = self._bulk_accept_string(prefix_strings)
        for batch_idx, accepted in enumerate(prefix_acceptances):
            if accepted:
                self.current_partheseness_start[batch_idx] += prefix_lengths[batch_idx]

        updated_prefixes = []
        for batch_idx in range(batch_size):
            start_idx = self.current_partheseness_start[batch_idx]
            prefix_tokens = decoded_token_list[batch_idx][start_idx:]
            updated_prefixes.append("".join(prefix_tokens))

        if candidate_strings:
            candidate_results = self._bulk_accept_prefix(candidate_strings)
            for (batch_idx, token_id), accepted in zip(candidate_meta, candidate_results):
                if accepted:
                    acceptance[batch_idx, token_id] = True

        for batch_idx, prefix in enumerate(updated_prefixes):
            if self._accept_string(prefix) or not acceptance[batch_idx].any():
                acceptance[batch_idx, eos_id] = True

        # if the logits size of the model is more than the tokennizer vocab
        # we artificially expand the acceptance tensor and block everything
        # beyond the tokenizer vocab size
        acceptance_vocab_size = acceptance.shape[-1]
        masked_logits_vocab_size = logits.shape[-1]
        if masked_logits_vocab_size != acceptance_vocab_size:
            assert (
                acceptance_vocab_size < masked_logits_vocab_size
            ), "impossible for tokenizer vocab to be less than model vocab"
            vocab_size_diff = masked_logits_vocab_size - acceptance_vocab_size
            false_tensor = torch.zeros(
                (*acceptance.shape[:-1], vocab_size_diff),
                dtype=torch.bool,
                device=device,
            )
            acceptance = torch.cat((acceptance, false_tensor), dim=-1)

        # Logits to -inf where False
        masked_logits = logits.masked_fill(~acceptance, -math.inf)
        return masked_logits, acceptance

    def generate_idx_list(self, size: int = 512):
        self.current_partheseness_start = [0 for _ in range(size)]

    def set_prompt_length(self, prompt_length: int):
        self.prompt_length = prompt_length

    def process_logits(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        prompt_length: Optional[int] = -1,
    ) -> torch.FloatTensor:
        """
        :param input_ids:
        :param scores:
        :return:
        """
        if self.device is None:
            device = scores.device

        if prompt_length < 0:
            prompt_length = self.prompt_length

        masked_scores, acceptance = self.mask_logits(input_ids, scores, device, prompt_length)

        if self.return_dict:
            return {
                "masked_logits": masked_scores,
                "acceptance": acceptance,
            }
        else:
            return masked_scores

    def reset(self):
        self.generate_idx_list(self.max_batch_size)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> torch.FloatTensor:
        return self.process_logits(input_ids, scores)


class GrammarIncrementalLogitsProcessorForNumberOnly(_TokenCacheMixin, LogitsProcessor):
    """
    This logits processor is used to limit the tokens that can be generated.
    It is used to generate a single token at a time.
    This is a special case for number only with the following exception:
    - The decoded token is a number only, not a number with comma. Without comma, it's very hard to tell the validation of number.
        e.g. "123" in accepted, but it can be "1,23" or "12,3" or "123,".
    - The generated token_ids will connected with comma automatically.
    """

    def __init__(
        self,
        parsed_grammar,
        tokenizer: PreTrainedTokenizer,
        device: Optional[torch.device] = None,
        nice_token_ids_list: Optional[torch.tensor] = None,
        execution_mode: Literal["full", "limited"] = "limited",
    ) -> None:
        self.device = device
        self.return_dict = False
        self._init_token_cache(tokenizer, nice_token_ids_list)
        self.execution_mode = execution_mode
        self._set_string_grammar(
            StringRecognizer(parsed_grammar.grammar_encoding, parsed_grammar.symbol_table["root"])
        )
        self.prompt_length = -1

    def set_prompt_length(self, prompt_length: int):
        self.prompt_length = prompt_length

    def mask_logits(
        self,
        input_ids: torch.LongTensor,
        logits: torch.FloatTensor,
        device: torch.device,
    ) -> torch.FloatTensor:
        batch_size = logits.shape[0]
        acceptance = torch.zeros((batch_size, self.vocab_size), dtype=torch.bool, device=device)

        prompt_offset = self.prompt_length if self.prompt_length >= 0 else 0
        decoded_token_list = [
            ",".join(self._decoder.decode_token(token_id) for token_id in token_list)
            + ("," if token_list else "")
            for token_list in input_ids[:, prompt_offset:].tolist()
        ]

        eos_id = self.tokenizer.eos_token_id
        last_token_ids = input_ids[:, -1].tolist()

        candidate_strings: List[str] = []
        candidate_meta: List[Tuple[int, int]] = []

        for batch_idx, base_prefix in enumerate(decoded_token_list):
            if last_token_ids[batch_idx] == eos_id:
                acceptance[batch_idx, eos_id] = True
                continue

            for token_id, token_str in self._nice_token_pairs:
                candidate_strings.append(base_prefix + token_str + ",")
                candidate_meta.append((batch_idx, token_id))

        if candidate_strings:
            candidate_results = self._bulk_accept_prefix(candidate_strings)
            for (batch_idx, token_id), accepted in zip(candidate_meta, candidate_results):
                if accepted:
                    acceptance[batch_idx, token_id] = True
        # if the logits size of the model is more than the tokennizer vocab
        # we artificially expand the acceptance tensor and block everything
        # beyond the tokenizer vocab size
        acceptance_vocab_size = acceptance.shape[-1]
        masked_logits_vocab_size = logits.shape[-1]
        if masked_logits_vocab_size != acceptance_vocab_size:
            assert (
                acceptance_vocab_size < masked_logits_vocab_size
            ), "impossible for tokenizer vocab to be less than model vocab"
            vocab_size_diff = masked_logits_vocab_size - acceptance_vocab_size
            false_tensor = torch.zeros(
                (*acceptance.shape[:-1], vocab_size_diff),
                dtype=torch.bool,
                device=device,
            )
            acceptance = torch.cat((acceptance, false_tensor), dim=-1)

        # Logits to -inf where False
        masked_logits = logits.masked_fill(~acceptance, -math.inf)
        return masked_logits, acceptance

    def set_return_dict(self, return_dict: bool):
        self.return_dict = return_dict

    def process_logits(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        :param input_ids:
        :param scores:
        :return:
        """
        if self.device is None:
            device = scores.device
        self.current_prefix = ["" for _ in range(input_ids.shape[0])]

        masked_scores, acceptance = self.mask_logits(input_ids, scores, device)

        if self.return_dict:
            return {
                "masked_logits": masked_scores,
                "acceptance": acceptance,
            }
        else:
            return masked_scores

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> torch.FloatTensor:
        return self.process_logits(input_ids, scores)


class GrammarConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        grammar_constraint: AbsTokenRecognizer,
        valid_token_start_idx: Optional[int] = None,
        execution_mode: Literal["speculation", "full_mask"] = "speculation",
        device: Optional[torch.device] = None,
        nice_vocab_list: Optional[torch.tensor] = None,
    ) -> None:
        self.last_size = None
        self.grammar_constraint = grammar_constraint
        self.batch_parsing_states = None
        self.valid_token_start_idx = valid_token_start_idx
        self.execution_mode = execution_mode
        self.nice_vocab_list = nice_vocab_list
        self.device = device

    def mask_logits(self, logits: torch.FloatTensor, device: torch.device) -> torch.FloatTensor:
        masked_logits = logits.clone()
        if self.execution_mode == "speculation":
            # try to accept the most likely token
            acceptance = torch.zeros(
                (logits.shape[0], len(self.grammar_constraint.homomorphism)),
                dtype=torch.bool,
                device=device,
            )
            next_tokens = torch.argmax(logits, dim=-1)
            for i, next_token in enumerate(next_tokens.tolist()):
                try:
                    is_next_token_accepted = self.grammar_constraint.accept_token_ids(
                        [next_token], self.batch_parsing_states[i]
                    )
                except ValueError:
                    is_next_token_accepted = False
                if is_next_token_accepted:
                    acceptance[i, next_token] = True
                else:
                    # resolve each stack to a tensor of True/False for each token
                    # indicating acceptance
                    # acceptance = self.grammar_acceptor.filter_vocab(self.stacks, device)
                    acceptance[i] = self.grammar_constraint.filter_vocab(
                        self.batch_parsing_states[i], device
                    )
        elif self.execution_mode == "limited":
            # try to accept tokens from a given vocab list
            assert (
                self.nice_vocab_list is not None
            ), "You selected the limited mode, must give a nice_vocab_list"
            pass
        else:
            acceptance = self.grammar_constraint.batch_filter_vocab(
                self.batch_parsing_states, device
            )

        # if the logits size of the model is more than the tokennizer vocab
        # we artificially expand the acceptance tensor and block everything
        # beyond the tokenizer vocab size
        acceptance_vocab_size = acceptance.shape[-1]
        masked_logits_vocab_size = masked_logits.shape[-1]
        if masked_logits_vocab_size != acceptance_vocab_size:
            assert (
                acceptance_vocab_size < masked_logits_vocab_size
            ), "impossible for tokenizer vocab to be less than model vocab"
            vocab_size_diff = masked_logits_vocab_size - acceptance_vocab_size
            false_tensor = torch.zeros(
                (*acceptance.shape[:-1], vocab_size_diff),
                dtype=torch.bool,
                device=device,
            )
            acceptance = torch.cat((acceptance, false_tensor), dim=-1)

        # acceptance is a tensor of shape (batch_size, vocab_size)
        # get the indices of the accepted tokens
        # do the following operation only in debug mode
        if os.getenv("DEBUG_MODE") == "True":
            # convert acceptance to numpy array
            batch_size, vocab_size = acceptance.shape
            acceptance_np = acceptance.cpu().numpy()
            accepted_x, accepted_y = acceptance_np.nonzero()
            # dict of {batch_index: [accepted_token_indices]}
            # initialize the dict with empty list
            accepted_token_indices = {i: [] for i in range(batch_size)}
            for x, y in zip(accepted_x, accepted_y):
                accepted_token_indices[x].append(y)
            logger.debug("Accepted token indices for the current batch:")
            logger.debug("\n" + pprint.pformat(accepted_token_indices))
            # convert token_ids to tokens
            accepted_tokens = {
                i: [self.grammar_constraint.tokenizer.decode([token_id]) for token_id in token_ids]
                for i, token_ids in accepted_token_indices.items()
            }
            logger.debug("Accepted tokens for the current batch:")
            logger.debug("\n" + pprint.pformat(accepted_tokens))
        # Logits to -inf where False
        masked_logits[~acceptance] = -math.inf
        return masked_logits, acceptance

    def process_logits(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        :param input_ids:
        :param scores:
        :return:
        """
        if self.device is None:
            device = scores.device
        # we dynamically create stacks at the first call, so that we know the batch size and beam size
        if self.batch_parsing_states is None:
            self.batch_parsing_states = [
                # self.grammar_constraint.init_stacks()
                copy.deepcopy(
                    self.grammar_constraint.string_recognizer.get_initial_parsing_state()
                )
                for _ in range(len(input_ids))
            ]

        if os.getenv("DEBUG_MODE") == "True":
            print("-" * 80)

        logger.debug("input_ids: \n" + pprint.pformat(input_ids))
        # logger.debug("scores: \n" + pprint.pformat(scores))
        logger.debug("last_size: \n" + pprint.pformat(self.last_size))
        logger.debug(
            "num of stacks: \n"
            + pprint.pformat([len(acc_state.stacks) for acc_state in self.batch_parsing_states])
        )
        # logger.debug("stacks: \n" + pprint.pformat(self.batch_parsing_states.stacks))

        self.batch_parsing_states = self.grammar_constraint.update_state_with_batch_token_seqs(
            input_ids, self.batch_parsing_states, self.valid_token_start_idx
        )
        logger.debug(f"input_ids: {input_ids}")

        masked_scores, acceptance = self.mask_logits(scores, device)
        return {"masked_logits": masked_scores, "acceptance": acceptance}

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        return self.process_logits(input_ids, scores)

    def reset(self):
        self.batch_parsing_states = None
        if isinstance(self.grammar_constraint, IncrementalGrammarConstraint):
            self.grammar_constraint.reset()


class GrammarLimitedOneTimeLogitsProcessor(_TokenCacheMixin, LogitsProcessor):
    """
    This logits processor is used to limit the tokens that can be generated.
    It is used to generate a single token at a time.
    The mode can be "full" or "limited".
    If the mode is "full", after found the CFG-accepting token, the logits processor will continue find another CFG-accepting token.
    If the mode is "limited", after found the CFG-accepting token, the logits processor will stop.
    """

    def __init__(
        self,
        parsed_grammar,
        tokenizer: PreTrainedTokenizer,
        device: Optional[torch.device] = None,
        nice_token_ids_list: Optional[torch.tensor] = None,
        execution_mode: Literal["full", "limited"] = "limited",
    ) -> None:
        self.device = device
        self._init_token_cache(tokenizer, nice_token_ids_list)
        self.execution_mode = execution_mode
        self._set_string_grammar(
            StringRecognizer(parsed_grammar.grammar_encoding, parsed_grammar.symbol_table["root"])
        )

    def mask_logits(
        self,
        input_ids: torch.LongTensor,
        logits: torch.FloatTensor,
        device: torch.device,
        ignore_length: int = 0,
    ) -> torch.FloatTensor:
        acceptance = torch.zeros(
            (logits.shape[0], self.vocab_size), dtype=torch.bool, device=device
        )

        # by default, the prompt part and eos is acceptable
        acceptance[: ignore_length - 1, :] = True
        eos_id = self.tokenizer.eos_token_id
        acceptance[:, eos_id] = True

        input_ids_list = input_ids.tolist()
        decoded_token_list = [self._decoder.decode_token(token_id) for token_id in input_ids_list]

        # parse start token
        if self._nice_token_pairs:
            start_candidates = [token_str for _, token_str in self._nice_token_pairs]
            start_results = self._bulk_accept_prefix(start_candidates)
            for (token_id, _), accepted in zip(self._nice_token_pairs, start_results):
                if accepted:
                    acceptance[ignore_length - 1, token_id] = True

        prefixes: List[str] = []
        for i in range(ignore_length, len(input_ids_list)):
            if i == ignore_length:
                prefixes.append(decoded_token_list[i] + ",")
            else:
                prefixes.append(prefixes[-1] + decoded_token_list[i] + ",")

        prefix_acceptances = self._bulk_accept_prefix(prefixes)

        for idx, accepted in enumerate(prefix_acceptances, start=ignore_length):
            if not accepted:
                acceptance[idx:, input_ids_list[idx]] = False
                break

            acceptance[idx, input_ids_list[idx]] = True
            if self.execution_mode != "limited" and self._nice_token_pairs:
                candidate_strings = [
                    prefixes[idx - ignore_length] + token_dec + ","
                    for _, token_dec in self._nice_token_pairs
                ]
                candidate_results = self._bulk_accept_string(candidate_strings)
                for (token_id, _), accepted_token in zip(
                    self._nice_token_pairs, candidate_results
                ):
                    if accepted_token:
                        acceptance[idx, token_id] = True

        # if the logits size of the model is more than the tokennizer vocab
        # we artificially expand the acceptance tensor and block everything
        # beyond the tokenizer vocab size
        acceptance_vocab_size = acceptance.shape[-1]
        masked_logits_vocab_size = logits.shape[-1]
        if masked_logits_vocab_size != acceptance_vocab_size:
            assert (
                acceptance_vocab_size < masked_logits_vocab_size
            ), "impossible for tokenizer vocab to be less than model vocab"
            vocab_size_diff = masked_logits_vocab_size - acceptance_vocab_size
            false_tensor = torch.zeros(
                (*acceptance.shape[:-1], vocab_size_diff),
                dtype=torch.bool,
                device=device,
            )
            acceptance = torch.cat((acceptance, false_tensor), dim=-1)

        # Logits to -inf where False
        masked_logits = logits.masked_fill(~acceptance, -math.inf)
        return masked_logits

    def process_logits(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, ignore_length: int = 0
    ) -> torch.FloatTensor:
        """
        :param input_ids:
        :param scores:
        :return:
        """
        if self.device is None:
            device = scores.device

        masked_scores = self.mask_logits(input_ids, scores, device, ignore_length)
        return masked_scores

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, ignore_length: int = 0
    ) -> torch.FloatTensor:
        return self.process_logits(input_ids, scores, ignore_length)
