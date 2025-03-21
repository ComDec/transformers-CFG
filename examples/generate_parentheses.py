import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.recognizer import StringRecognizer
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from transformers_cfg.parser import parse_ebnf
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Parentheses strings")
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model ID",
    )
    parser.add_argument("--device", type=str, help="Device to put the model on")
    parser.add_argument(
        "--parentheses-type",
        type=str,
        choices=["general"],
        default="general",
        help="Type of Parentheses to generate",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_id = args.model_id

    # Detect if GPU is available, otherwise use CPU
    device = torch.device(
        args.device or ("cuda:4" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"N tokens: {len(tokenizer.get_vocab())}")
    # Load model to defined device
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    # Load grammar
    grammar_name = args.parentheses_type
    with open(f"/home/xw3763/project/gflow/ChemGFN/assets/parentheses_grammars/{grammar_name}.ebnf", "r") as file:
        grammar_str = file.read()

    parsed_grammar = parse_ebnf(grammar_str)
    first_rule = grammar_str.split("\n")[0]
    print(f"{grammar_name}: {first_rule}")

    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    # Generate
    prefix1 = """Generate a parentheses mathching string including (), [], and <>:"""

    input_ids = tokenizer(
        [prefix1], add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"].to(
        device
    )  # Move input_ids to the same device as model

    max_new_tokens = 20
    unconstrained_output = model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.9,
        num_return_sequences=1,
    )

    start = time.time()
    constrained_output = model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        logits_processor=[grammar_processor],
        repetition_penalty=1.9,
        num_return_sequences=1,
    )

    string_grammar = StringRecognizer(
        parsed_grammar.grammar_encoding, parsed_grammar.symbol_table["root"]
    )

    print("Unconstrained: ", tokenizer.decode(unconstrained_output[0], skip_special_tokens=True))
    print("Constrained: ", tokenizer.decode(constrained_output[0], skip_special_tokens=True))

