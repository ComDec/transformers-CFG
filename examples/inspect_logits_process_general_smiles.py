import argparse
import random
from pathlib import Path

import torch

from transformers_cfg.generation.logits_process import (
    GrammarIncrementalLogitsProcessorGeneral,
)
from transformers_cfg.parser import parse_ebnf
from transformers_cfg.recognizer import StringRecognizer


class DummyTokenizer:
    def __init__(self, id_to_token, eos_token_id):
        self.id_to_token = list(id_to_token)
        self.eos_token_id = eos_token_id

    def __len__(self):
        return len(self.id_to_token)

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        tokens = []
        for token_id in token_ids:
            token_id = int(token_id)
            if skip_special_tokens and token_id == self.eos_token_id:
                continue
            tokens.append(self.id_to_token[token_id])
        return "".join(tokens)


def _parse_token_list(raw):
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return []
    if "," in raw:
        parts = [part.strip() for part in raw.split(",")]
    else:
        parts = raw.split()
    return [part for part in parts if part]


def _parse_prefix_tokens(prefix_arg, vocab_set):
    if prefix_arg is None:
        return None
    prefix_arg = prefix_arg.strip()
    if not prefix_arg:
        return []
    if " " in prefix_arg:
        tokens = prefix_arg.split()
    elif prefix_arg in vocab_set:
        tokens = [prefix_arg]
    else:
        tokens = list(prefix_arg)
    return tokens


def _choose_start_tokens(prefix_arg, vocab_tokens, recognizer, eos_token):
    vocab_set = set(vocab_tokens)
    tokens = _parse_prefix_tokens(prefix_arg, vocab_set)
    if tokens is not None:
        if not tokens:
            raise ValueError("Start prefix must include at least one token.")
        return tokens
    for token in vocab_tokens:
        if token == eos_token:
            continue
        if recognizer._accept_prefix(token):
            return [token]
    raise ValueError("No vocab token can start the grammar; pass --start-prefix.")


def _format_token_list(tokens):
    return "[" + ", ".join(tokens) + "]"


def _build_processor(parsed_grammar, tokenizer, nice_token_ids, execution_mode):
    return GrammarIncrementalLogitsProcessorGeneral(
        parsed_grammar,
        tokenizer=tokenizer,
        nice_token_ids_list=nice_token_ids,
        execution_mode=execution_mode,
    )


def _accepted_token_ids(acceptance):
    return [idx for idx, accepted in enumerate(acceptance) if accepted]


def main():
    parser = argparse.ArgumentParser(
        description="Inspect GrammarIncrementalLogitsProcessorGeneral acceptances (SMILES)."
    )
    parser.add_argument(
        "--grammar-path",
        default=str(Path(__file__).resolve().parent / "grammars" / "SMILES" / "generic.ebnf"),
        help="EBNF grammar file path",
    )
    parser.add_argument(
        "--vocab",
        default=None,
        help="Comma or space-separated vocab tokens (defaults to a SMILES-friendly set)",
    )
    parser.add_argument(
        "--nice-tokens",
        default=None,
        help="Comma or space-separated nice tokens (defaults to vocab minus eos)",
    )
    parser.add_argument(
        "--eos-token",
        default="<eos>",
        help="Token string used as EOS",
    )
    parser.add_argument(
        "--start-prefix",
        default=None,
        help="Start prefix as tokens (space-separated). If omitted, auto-picks a valid start.",
    )
    parser.add_argument("--steps", type=int, default=8, help="Number of steps to simulate")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    grammar_path = Path(args.grammar_path)
    grammar_str = grammar_path.read_text(encoding="utf-8")
    parsed_grammar = parse_ebnf(grammar_str)
    recognizer = StringRecognizer(
        parsed_grammar.grammar_encoding, parsed_grammar.symbol_table["root"]
    )

    if args.vocab is None:
        vocab_tokens = [
            "C",
            "N",
            "O",
            "S",
            "P",
            "F",
            "I",
            "B",
            "c",
            "n",
            "o",
            "s",
            "p",
            "Cl",
            "Br",
            "se",
            "as",
            "(",
            ")",
            "=",
            "#",
            "-",
            "/",
            "\\",
            ".",
            "[",
            "]",
            "@",
            "+",
            "*",
            ":",
            "$",
            "%",
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
            "H",
            args.eos_token,
        ]
    else:
        vocab_tokens = _parse_token_list(args.vocab)

    if args.eos_token not in vocab_tokens:
        raise ValueError(f"EOS token {args.eos_token!r} not in vocab.")

    if len(set(vocab_tokens)) != len(vocab_tokens):
        raise ValueError("Vocab tokens must be unique.")

    if args.nice_tokens is None:
        nice_tokens = [token for token in vocab_tokens if token != args.eos_token]
    else:
        nice_tokens = _parse_token_list(args.nice_tokens)

    vocab_set = set(vocab_tokens)
    unknown_nice = [token for token in nice_tokens if token not in vocab_set]
    if unknown_nice:
        raise ValueError(f"Nice tokens not in vocab: {unknown_nice}")

    eos_id = vocab_tokens.index(args.eos_token)
    tokenizer = DummyTokenizer(vocab_tokens, eos_id)
    token_to_id = {token: idx for idx, token in enumerate(vocab_tokens)}
    nice_token_ids = [token_to_id[token] for token in nice_tokens]

    processors = {
        "greedy": _build_processor(parsed_grammar, tokenizer, nice_token_ids, "greedy"),
        "limited": _build_processor(parsed_grammar, tokenizer, nice_token_ids, "limited"),
        "extensive": _build_processor(parsed_grammar, tokenizer, nice_token_ids, "extensive"),
    }

    start_tokens = _choose_start_tokens(
        args.start_prefix, vocab_tokens, recognizer, args.eos_token
    )
    start_ids = [token_to_id[token] for token in start_tokens]

    rng = random.Random(args.seed)

    print(f"Grammar path: {grammar_path}")
    print("Vocab:", _format_token_list(vocab_tokens))
    print("Nice tokens:", _format_token_list(nice_tokens))
    print(f"EOS token: {args.eos_token} (id={eos_id})")
    print(f"Start prefix: {_format_token_list(start_tokens)}")
    print(f"Steps: {args.steps}, seed: {args.seed}")
    print()

    states = {}
    for mode in processors:
        states[mode] = {
            "token_ids": list(start_ids),
            "prefix": "".join(start_tokens),
            "done": False,
        }

    for step_idx in range(args.steps):
        proposed_id = rng.randrange(len(vocab_tokens))
        proposed_token = vocab_tokens[proposed_id]
        print(f"--- step {step_idx + 1} proposed={proposed_token} (id={proposed_id}) ---")

        for mode, processor in processors.items():
            state = states[mode]
            if state["done"]:
                print(f"{mode}: done prefix={state['prefix']!r}")
                continue

            current_prefix = state["prefix"]
            input_ids = torch.tensor([state["token_ids"]], dtype=torch.long)
            logits = torch.zeros((1, len(vocab_tokens)))
            logits[0, proposed_id] = 5.0

            result = processor(input_ids, logits, min_length=0)
            acceptance = result["acceptance"][0].tolist()

            accepted_ids = _accepted_token_ids(acceptance)
            accepted_tokens = [vocab_tokens[idx] for idx in accepted_ids]
            eos_allowed = bool(acceptance[eos_id])

            if not accepted_ids:
                print(f"{mode}: prefix={current_prefix!r} accepted=[] eos_allowed=False")
                state["done"] = True
                continue

            if proposed_id in accepted_ids:
                chosen_id = proposed_id
            else:
                chosen_id = rng.choice(accepted_ids)

            chosen_token = vocab_tokens[chosen_id]
            next_prefix = current_prefix + chosen_token

            print(
                f"{mode}: prefix={current_prefix!r} accepted={_format_token_list(accepted_tokens)} "
                f"eos_allowed={eos_allowed} chosen={chosen_token!r} next_prefix={next_prefix!r}"
            )
            state["token_ids"].append(chosen_id)
            state["prefix"] = next_prefix
            if chosen_id == eos_id:
                state["done"] = True
        print()


if __name__ == "__main__":
    main()
