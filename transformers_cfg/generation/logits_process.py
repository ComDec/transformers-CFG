import copy
import logging
import math
import os
import pprint
from typing import Literal, Optional

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


class GrammarLimitedOneTimeLogitsProcessor(LogitsProcessor):
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
        self.tokenizer = tokenizer
        self.nice_token_ids_list = nice_token_ids_list
        self.execution_mode = execution_mode
        self.string_grammar = StringRecognizer(
            parsed_grammar.grammar_encoding, parsed_grammar.symbol_table["root"]
        )

    def mask_logits(
        self,
        input_ids: torch.LongTensor,
        logits: torch.FloatTensor,
        device: torch.device,
        ignore_length: int = 0,
    ) -> torch.FloatTensor:
        masked_logits = logits.clone()
        # logits: 1,L,D
        acceptance = torch.zeros(
            (logits.shape[0], len(self.tokenizer)), dtype=torch.bool, device=device
        )

        # by default, the prompt part and eos is acceptable
        acceptance[: ignore_length - 1, :] = True
        acceptance[:, self.tokenizer.eos_token_id] = True

        decoded_token_list = [self.tokenizer.decode(token_id) for token_id in input_ids.tolist()]

        # parse start token
        for token in self.nice_token_ids_list:
            if self.string_grammar._accept_prefix(self.tokenizer.decode(token)):
                acceptance[ignore_length - 1, token] = True

        prefix = ""
        # Precompute decoding for tokens in nice_token_ids_list to avoid redundant decoding inside the loop
        nice_token_decodings = []
        for token in self.nice_token_ids_list:
            nice_token_decodings.append(self.tokenizer.decode([token]))

        # prefix at all intermedium state
        for i in range(ignore_length, len(input_ids)):
            prefix += decoded_token_list[i] + ","
            if self.string_grammar._accept_prefix(prefix):
                acceptance[i, input_ids[i]] = True
                if self.mode != "limited":
                    for token, token_dec in zip(self.nice_token_ids_list, nice_token_decodings):
                        if self.string_grammar._accept_string(prefix + token_dec + ","):
                            acceptance[i, token] = True
            else:
                acceptance[i:, input_ids[i]] = False
                break

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

        # Logits to -inf where False
        masked_logits[~acceptance] = -math.inf
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


class GrammarIncrementalLogitsProcessorGeneral(LogitsProcessor):
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
        self.tokenizer = tokenizer
        self.nice_token_ids_list = nice_token_ids_list
        self.execution_mode = execution_mode
        self.string_grammar = StringRecognizer(
            parsed_grammar.grammar_encoding, parsed_grammar.symbol_table["root"]
        )

    def mask_logits(
        self,
        input_ids: torch.LongTensor,
        logits: torch.FloatTensor,
        device: torch.device,
        prompt_length: int = 0,
    ) -> torch.FloatTensor:
        masked_logits = logits.clone()
        # logits: B,L,D
        batch_size = logits.shape[0]
        acceptance = torch.zeros(
            (batch_size, len(self.tokenizer)), dtype=torch.bool, device=device
        )
        acceptance[:, self.tokenizer.eos_token_id] = True

        # connect the decoded token with comma
        decoded_token_list = [
            [self.tokenizer.decode(token_id) for token_id in token_list]
            for token_list in input_ids[:, prompt_length:].tolist()
        ]
        # Precompute decoding for tokens in nice_token_ids_list to avoid redundant decoding inside the loop
        nice_token_decodings = []
        for token in self.nice_token_ids_list:
            nice_token_decodings.append(self.tokenizer.decode([token]))

        prefix = ""
        # prefix at all intermedium state
        for batch in range(batch_size):
            # terminate the generation
            if input_ids[batch, -1] == self.tokenizer.eos_token_id:
                acceptance[batch, self.tokenizer.eos_token_id] = True
                continue

            for i, token in enumerate(nice_token_decodings):
                prefix = "".join(decoded_token_list[batch]) + nice_token_decodings[i]
                if self.string_grammar._accept_prefix(prefix):
                    acceptance[batch, self.nice_token_ids_list[i]] = True
                else:
                    acceptance[batch, self.nice_token_ids_list[i]] = False
            if acceptance[batch].sum() == 1:
                # This is a hacked version to make sure training can continue
                # If CFG only accept one token (eos), we regard all tokens are acceptable
                acceptance[batch, :] = True
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

        # Logits to -inf where False
        masked_logits[~acceptance] = -math.inf
        return masked_logits, acceptance

    def process_logits(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, prompt_length: int = 0
    ) -> torch.FloatTensor:
        """
        :param input_ids:
        :param scores:
        :return:
        """
        if self.device is None:
            device = scores.device
        self.current_prefix = ["" for _ in range(input_ids.shape[0])]

        masked_scores, acceptance = self.mask_logits(input_ids, scores, device, prompt_length)
        return {"masked_logits": masked_scores, "acceptance": acceptance}

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, prompt_length: int = 0
    ) -> torch.FloatTensor:
        return self.process_logits(input_ids, scores, prompt_length)


class GrammarLogitsProcessorPartheseness(LogitsProcessor):
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
        self.tokenizer = tokenizer
        self.nice_token_ids_list = nice_token_ids_list
        self.execution_mode = execution_mode
        self.string_grammar = StringRecognizer(
            parsed_grammar.grammar_encoding, parsed_grammar.symbol_table["root"]
        )
        self.max_batch_size = max_batch_size
        self.return_dict = return_dict
        self.generate_idx_list(self.max_batch_size)

    def mask_logits(
        self,
        input_ids: torch.LongTensor,
        logits: torch.FloatTensor,
        device: torch.device,
        prompt_length: int = 0,
    ) -> torch.FloatTensor:
        masked_logits = logits.clone()
        # logits: B,L,D
        batch_size = logits.shape[0]
        acceptance = torch.zeros(
            (batch_size, len(self.tokenizer)), dtype=torch.bool, device=device
        )
        acceptance[:, self.tokenizer.eos_token_id] = True

        # connect the decoded token with comma
        decoded_token_list = [
            [self.tokenizer.decode(token_id) for token_id in token_list]
            for token_list in input_ids[:, prompt_length:].tolist()
        ]
        # Precompute decoding for tokens in nice_token_ids_list to avoid redundant decoding inside the loop
        nice_token_decodings = []
        for token in self.nice_token_ids_list:
            nice_token_decodings.append(self.tokenizer.decode([token]))

        prefix = ""
        # prefix at all intermedium state
        for batch in range(batch_size):
            # # terminate the generation
            # if input_ids[batch, -1] == self.tokenizer.eos_token_id:
            #     acceptance[batch, self.tokenizer.eos_token_id] = True
            #     continue

            prefix = "".join(decoded_token_list[batch][self.current_partheseness_start[batch] :])
            if self.string_grammar._accept_string(prefix):
                self.current_partheseness_start[batch] += len(prefix)
                prefix = "".join(
                    decoded_token_list[batch][self.current_partheseness_start[batch] :]
                )
            for i, token in enumerate(nice_token_decodings):
                current_string = prefix + nice_token_decodings[i]
                if self.string_grammar._accept_prefix(current_string):
                    acceptance[batch, self.nice_token_ids_list[i]] = True
                else:
                    acceptance[batch, self.nice_token_ids_list[i]] = False

            # if acceptance[batch].sum() == 1:
            #     # This is a hacked version to make sure training can continue
            #     # If CFG only accept one token (eos), we regard all tokens are acceptable
            #     import ipdb; ipdb.set_trace()
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

        # Logits to -inf where False
        masked_logits[~acceptance] = -math.inf
        return masked_logits, acceptance

    def generate_idx_list(self, size: int = 512):
        self.current_partheseness_start = [0 for _ in range(size)]

    def set_prompt_length(self, prompt_length: int):
        self.prompt_length = prompt_length

    def process_logits(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, prompt_length: Optional[int] = -1
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
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, prompt_length: Optional[int] = -0
    ) -> torch.FloatTensor:
        return self.process_logits(input_ids, scores, prompt_length)


class GrammarIncrementalLogitsProcessorForNumberOnly(LogitsProcessor):
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
        self.tokenizer = tokenizer
        self.nice_token_ids_list = nice_token_ids_list
        self.execution_mode = execution_mode
        self.string_grammar = StringRecognizer(
            parsed_grammar.grammar_encoding, parsed_grammar.symbol_table["root"]
        )

    def mask_logits(
        self,
        input_ids: torch.LongTensor,
        logits: torch.FloatTensor,
        device: torch.device,
        prompt_length: int = 0,
    ) -> torch.FloatTensor:
        masked_logits = logits.clone()
        # logits: B,L,D
        batch_size = logits.shape[0]
        acceptance = torch.zeros(
            (batch_size, len(self.tokenizer)), dtype=torch.bool, device=device
        )
        acceptance[:, self.tokenizer.eos_token_id] = True

        # connect the decoded token with comma
        decoded_token_list = [
            ",".join(self.tokenizer.decode(token_id) for token_id in token_list)
            + ("," if token_list else "")
            for token_list in input_ids[:, prompt_length:].tolist()
        ]

        # Precompute decoding for tokens in nice_token_ids_list to avoid redundant decoding inside the loop
        nice_token_decodings = []
        for token in self.nice_token_ids_list:
            nice_token_decodings.append(self.tokenizer.decode([token]))

        prefix = ""
        # prefix at all intermedium state
        for batch in range(batch_size):
            # terminate the generation
            if input_ids[batch, -1] == self.tokenizer.eos_token_id:
                acceptance[batch, self.tokenizer.eos_token_id] = True
                continue
            for i, token in enumerate(nice_token_decodings):
                prefix = decoded_token_list[batch] + nice_token_decodings[i] + ","
                if self.string_grammar._accept_prefix(prefix):
                    acceptance[batch, self.nice_token_ids_list[i]] = True
                else:
                    acceptance[batch, self.nice_token_ids_list[i]] = False
            if acceptance[batch].sum() == 1:
                pass
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

        # Logits to -inf where False
        masked_logits[~acceptance] = -math.inf
        return masked_logits, acceptance

    def process_logits(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, prompt_length: int = 0
    ) -> torch.FloatTensor:
        """
        :param input_ids:
        :param scores:
        :return:
        """
        if self.device is None:
            device = scores.device
        self.current_prefix = ["" for _ in range(input_ids.shape[0])]

        masked_scores, acceptance = self.mask_logits(input_ids, scores, device, prompt_length)
        return {"masked_logits": masked_scores, "acceptance": acceptance}

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, prompt_length: int = 0
    ) -> torch.FloatTensor:
        return self.process_logits(input_ids, scores, prompt_length)


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
