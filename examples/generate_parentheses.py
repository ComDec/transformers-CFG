import argparse
import time

import torch
from chemgfn.utils.gfn_utils import (
    base_to_lora,
    generate_and_return_termination_logprob,
    get_termination_vals,
    lora_to_base,
    modified_subtb_loss,
    prepare_token_mask,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers_cfg.generation.logits_process import (
    GrammarConstrainedLogitsProcessor,
    GrammarLogitsProcessorPartheseness,
)
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.parser import parse_ebnf
from transformers_cfg.recognizer import StringRecognizer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Parentheses strings")
    parser.add_argument(
        "--model-id",
        type=str,
        default="openai-community/gpt2",
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
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"N tokens: {len(tokenizer.get_vocab())}")
    # Load model to defined device
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model = model.to(torch.bfloat16)

    # Generate
    prefix1 = (
        """This is a string with properly balanced parentheses, brackets, and angle brackets:"""
    )

    input_ids = tokenizer([prefix1], add_special_tokens=False, return_tensors="pt", padding=True)[
        "input_ids"
    ].to(
        device
    )  # Move input_ids to the same device as model

    prompt_length = input_ids.shape[1]
    print(f"Prompt length: {prompt_length}")

    # Load grammar
    grammar_name = args.parentheses_type
    with open(
        f"/home/xw3763/project/gflow/ChemGFN/assets/parentheses_grammars/{grammar_name}.ebnf"
    ) as file:
        grammar_str = file.read()

    parsed_grammar = parse_ebnf(grammar_str)
    first_rule = grammar_str.split("\n")[0]
    print(f"{grammar_name}: {first_rule}")

    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)

    (
        legal_tokens_mask,
        illegal_tokens_mask,
        legal_token_ids_list,
    ) = prepare_token_mask(
        tokenizer,
        "/home/xw3763/project/gflow/ChemGFN/assets/token_list/parentheses/allowed_gpt2_token",
    )

    grammar_processor = GrammarLogitsProcessorPartheseness(
        parsed_grammar,
        tokenizer=tokenizer,
        nice_token_ids_list=legal_token_ids_list,
        return_dict=False,
    )

    grammar_processor.set_prompt_length(prompt_length)

    max_new_tokens = 20
    # unconstrained_output = model.generate(
    #     input_ids,
    #     do_sample=False,
    #     max_new_tokens=max_new_tokens,
    #     repetition_penalty=1.9,
    #     num_return_sequences=1,
    # )

    sequences = []

    start = time.time()
    from tqdm import tqdm

    with open("cfg-only-untrained.txt", "w+") as f:
        for _ in tqdm(range(10000)):
            grammar_processor.reset()

            constrained_output = model.generate(
                input_ids,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                min_new_tokens=10,
                logits_processor=[grammar_processor],
                num_return_sequences=1,
                temperature=3.0,
            )

            # print("Unconstrained: ", tokenizer.decode(unconstrained_output[0], skip_special_tokens=True))
            decoded = tokenizer.decode(constrained_output[0], skip_special_tokens=False)

            print("Constrained: ", decoded)
            print(constrained_output[0][prompt_length:])

            sequences.append(decoded)

            f.write(decoded + "\n")
