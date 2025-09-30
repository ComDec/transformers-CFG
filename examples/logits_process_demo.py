"""Minimal tour of the grammar-aware logits processors.

Run with::

    python examples/logits_process_demo.py

The script avoids loading a full language model so it can execute quickly and
offline. It shows how to wire the grammar processors, inspect their accepted
tokens, and how to use the fast grammar tree explorer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
from transformers import AutoTokenizer, LogitsProcessor

from transformers_cfg.generation.logits_process import (
    GrammarConstrainedLogitsProcessor,
    GrammarIncrementalLogitsProcessorForNumberOnly,
    GrammarIncrementalLogitsProcessorGeneral,
    GrammarIncrementalLogitsProcessorSampleEnhanced,
    GrammarLimitedOneTimeLogitsProcessor,
    GrammarLogitsProcessorPartheseness,
    expand_grammar_tree,
)
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.parser import parse_ebnf

# ---------------------------------------------------------------------------
# Utilities


def load_tokenizer(model_id: str = "openai-community/gpt2"):
    """Load a tokenizer and ensure an EOS token is set."""

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def gather_token_ids(tokenizer, fragments: Iterable[str]) -> torch.LongTensor:
    """Collect token IDs that appear when encoding the provided fragments."""

    token_ids = set()
    for fragment in fragments:
        ids = tokenizer.encode(fragment, add_special_tokens=False)
        token_ids.update(ids)
    if not token_ids:
        return torch.arange(len(tokenizer), dtype=torch.long)
    return torch.tensor(sorted(token_ids), dtype=torch.long)


def decode_top_tokens(tokenizer, allowed_ids: Sequence[int], limit: int = 12) -> list[str]:
    """Pretty-print helper for the first *limit* accepted tokens."""

    rendered = []
    for token_id in allowed_ids[:limit]:
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        rendered.append(f"{token_id}: {repr(text)}")
    return rendered


def extract_allowed_from_output(output) -> torch.LongTensor:
    """Return IDs whose logits remain finite after masking."""

    if isinstance(output, dict):
        masked = output["masked_logits"]
    else:
        masked = output
    mask = torch.isfinite(masked)
    return mask.nonzero(as_tuple=False)[:, -1].cpu()


def summarize_processor(
    name: str,
    processor: LogitsProcessor,
    tokenizer,
    input_ids: torch.LongTensor,
    scores: torch.FloatTensor,
    **kwargs,
) -> None:
    """Run a processor once and print the accepted tokens."""

    with torch.no_grad():
        output = processor(input_ids.clone(), scores.clone(), **kwargs)
    allowed_ids = extract_allowed_from_output(output)
    tokens_preview = decode_top_tokens(tokenizer, allowed_ids.tolist())
    print(f"\n[{name}] accepted {len(allowed_ids)} tokens; preview:")
    for item in tokens_preview:
        print(f"  - {item}")


@dataclass
class GrammarBundle:
    name: str
    ebnf: str
    prompt: str
    fragments: Sequence[str]


# ---------------------------------------------------------------------------
# Demo grammars


JSON_GRAMMAR = GrammarBundle(
    name="simple-json",
    ebnf=r"""
root ::= object
object ::= "{" pair_list "}" | "{" "}"
pair_list ::= pair | pair "," pair_list
pair ::= key ":" number
key ::= "\"temperature\"" | "\"pressure\"" | "\"humidity\""
number ::= digit number | digit
digit ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
""".strip(),
    prompt='{"temperature":',
    fragments=[
        "{",
        "}",
        '"temperature"',
        '"pressure"',
        '"humidity"',
        ":",
        ",",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ],
)


PARENS_GRAMMAR = GrammarBundle(
    name="balanced-parentheses",
    ebnf=r'root ::= "(" root ")" | ""',
    prompt="((",
    fragments=["(", ")"],
)


SEQUENCE_GRAMMAR = GrammarBundle(
    name="digit-sequence",
    ebnf=r"""
root ::= sequence
sequence ::= digit "," sequence | digit
digit ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
""".strip(),
    prompt="1,2,",
    fragments=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ","],
)


# ---------------------------------------------------------------------------
# Main tour


def run_demo() -> None:
    tokenizer = load_tokenizer()
    vocab = len(tokenizer)
    dummy_scores = torch.zeros((1, vocab))

    # ------------------------ General / Sampling processors ----------------
    parsed_json = parse_ebnf(JSON_GRAMMAR.ebnf)
    nice_tokens_json = gather_token_ids(tokenizer, JSON_GRAMMAR.fragments)
    prompt_ids = tokenizer(JSON_GRAMMAR.prompt, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ]

    general = GrammarIncrementalLogitsProcessorGeneral(
        parsed_json,
        tokenizer=tokenizer,
        nice_token_ids_list=nice_tokens_json,
        execution_mode="full",
    )
    general.set_prompt_length(prompt_ids.shape[1])

    summarize_processor(
        name="GrammarIncrementalLogitsProcessorGeneral",
        processor=general,
        tokenizer=tokenizer,
        input_ids=prompt_ids,
        scores=dummy_scores,
    )

    sampler = GrammarIncrementalLogitsProcessorSampleEnhanced(
        parsed_json,
        tokenizer=tokenizer,
        nice_token_ids_list=nice_tokens_json,
        sampling_tau=0.7,
    )
    sampler.set_prompt_length(prompt_ids.shape[1])

    summarize_processor(
        name="GrammarIncrementalLogitsProcessorSampleEnhanced",
        processor=sampler,
        tokenizer=tokenizer,
        input_ids=prompt_ids,
        scores=dummy_scores,
    )

    # ----------------------------- Partheseness -----------------------------
    parsed_parens = parse_ebnf(PARENS_GRAMMAR.ebnf)
    nice_tokens_parens = gather_token_ids(tokenizer, PARENS_GRAMMAR.fragments)
    paren_ids = tokenizer(PARENS_GRAMMAR.prompt, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ]

    parth = GrammarLogitsProcessorPartheseness(
        parsed_parens,
        tokenizer=tokenizer,
        nice_token_ids_list=nice_tokens_parens,
        max_batch_size=8,
        return_dict=True,
    )
    parth.set_prompt_length(paren_ids.shape[1])

    summarize_processor(
        name="GrammarLogitsProcessorPartheseness",
        processor=parth,
        tokenizer=tokenizer,
        input_ids=paren_ids,
        scores=dummy_scores,
        prompt_length=paren_ids.shape[1],
    )

    # ----------------------------- Number only -----------------------------
    parsed_seq = parse_ebnf(SEQUENCE_GRAMMAR.ebnf)
    nice_tokens_seq = gather_token_ids(tokenizer, SEQUENCE_GRAMMAR.fragments)
    seq_ids = tokenizer(SEQUENCE_GRAMMAR.prompt, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ]

    number_only = GrammarIncrementalLogitsProcessorForNumberOnly(
        parsed_seq,
        tokenizer=tokenizer,
        nice_token_ids_list=nice_tokens_seq,
        execution_mode="limited",
    )
    number_only.set_prompt_length(seq_ids.shape[1])

    summarize_processor(
        name="GrammarIncrementalLogitsProcessorForNumberOnly",
        processor=number_only,
        tokenizer=tokenizer,
        input_ids=seq_ids,
        scores=dummy_scores,
    )

    number_proc = GrammarLimitedOneTimeLogitsProcessor(
        parsed_seq,
        tokenizer=tokenizer,
        nice_token_ids_list=nice_tokens_seq,
        execution_mode="full",
    )

    summarize_processor(
        name="GrammarLimitedOneTimeLogitsProcessor",
        processor=number_proc,
        tokenizer=tokenizer,
        input_ids=seq_ids,
        scores=dummy_scores,
        ignore_length=seq_ids.size(1),
    )

    # --------------------------- Constraint wrapper ------------------------
    grammar_constraint = IncrementalGrammarConstraint(JSON_GRAMMAR.ebnf, "root", tokenizer)
    constrained = GrammarConstrainedLogitsProcessor(
        grammar_constraint=grammar_constraint,
        nice_vocab_list=nice_tokens_json,
    )

    summarize_processor(
        name="GrammarConstrainedLogitsProcessor",
        processor=constrained,
        tokenizer=tokenizer,
        input_ids=prompt_ids,
        scores=dummy_scores,
    )

    # -------------------------- Grammar expansion -------------------------
    print("\n[expand_grammar_tree] depth-2 preview for partial JSON string:")
    tree = expand_grammar_tree(
        partial_text=JSON_GRAMMAR.prompt,
        parsed_grammar=parse_ebnf(JSON_GRAMMAR.ebnf),
        tokenizer=tokenizer,
        depth=2,
        nice_token_ids_list=nice_tokens_json,
    )

    def pretty_print(node, indent=""):
        marker = "└─" if indent else ""
        token = "<root>" if node["token"] is None else repr(node["token"])
        suffix = " (accepts EOS)" if node.get("accepts_eos") else ""
        print(f"{indent}{marker}{token}{suffix}")
        next_indent = indent + ("  " if indent else "")
        for child in node.get("children", []):
            pretty_print(child, next_indent)

    pretty_print(tree)

    print(
        "\nDone. Replace the dummy logits with real model outputs and feed these"
        " processors into a LogitsProcessorList to constrain generation."
    )


if __name__ == "__main__":
    run_demo()
