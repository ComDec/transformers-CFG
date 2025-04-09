import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.recognizer import StringRecognizer
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor, GrammarLogitsProcessorPartheseness
from transformers_cfg.parser import parse_ebnf
import time

from chemgfn.utils.gfn_utils import (
    base_to_lora,
    generate_and_return_termination_logprob,
    get_termination_vals,
    lora_to_base,
    modified_subtb_loss,
    prepare_token_mask,
)

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
    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"N tokens: {len(tokenizer.get_vocab())}")
    # Load model to defined device
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

     # Generate
    prefix1 = """This is a string with properly balanced parentheses, brackets, and angle brackets:"""

    input_ids = tokenizer(
        [prefix1], add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"].to(
        device
    )  # Move input_ids to the same device as model

    prompt_length = input_ids.shape[1]
    print(f"Prompt length: {prompt_length}")

    # Load grammar
    grammar_name = args.parentheses_type
    with open(f"/home/xw3763/project/gflow/ChemGFN/assets/parentheses_grammars/{grammar_name}.ebnf", "r") as file:
        grammar_str = file.read()

    parsed_grammar = parse_ebnf(grammar_str)
    first_rule = grammar_str.split("\n")[0]
    print(f"{grammar_name}: {first_rule}")

    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)

    (
                legal_tokens_mask,
                illegal_tokens_mask,
                legal_token_ids_list,
            ) = prepare_token_mask(tokenizer, "/home/xw3763/project/gflow/ChemGFN/assets/token_list/parentheses/allowed_gpt2_token")

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

        with torch.no_grad():

            for _ in tqdm(range(10000)):

                grammar_processor.reset()
                
                # 初始化参数
                current_ids = input_ids.clone()
                past_key_values = None
                generated_tokens = 0
                min_new_tokens = 10
                temperature = 1.0
                
                terminated = False
                termination_token_id = tokenizer.eos_token_id  # 获取EOS token ID

                # 自回归生成循环
                while generated_tokens < max_new_tokens and not terminated:
                    # 准备输入（首次全量输入，后续用缓存）
                    if past_key_values is not None:
                        next_input_ids = current_ids[:, -1:]
                    else:
                        next_input_ids = current_ids
                    
                    # 前向推理
                    outputs = model(
                        input_ids=next_input_ids,
                        past_key_values=past_key_values,
                    )
                    next_token_logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values


                    # 应用所有logits处理器（语法约束等）
                    import ipdb; ipdb.set_trace()
                    next_token_logits = grammar_processor(current_ids, next_token_logits, prompt_length=prompt_length)

                    # 强制长度约束
                    if generated_tokens < min_new_tokens:
                        next_token_logits[:, termination_token_id] = -torch.inf  # 禁止提前终止
                    elif generated_tokens >= max_new_tokens - 1:
                        mask = torch.ones_like(next_token_logits, dtype=torch.bool)
                        mask[:, termination_token_id] = False
                        next_token_logits[mask] = -torch.inf  # 强制终止

                    # 应用温度采样
                    import ipdb; ipdb.set_trace()
                    scaled_logits = next_token_logits / temperature
                    probabilities = torch.softmax(scaled_logits, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1)

                    # 更新生成序列
                    current_ids = torch.cat([current_ids, next_token], dim=-1)
                    generated_tokens += 1

                    # 终止条件检查
                    if next_token.item() == termination_token_id:
                        if generated_tokens >= min_new_tokens:
                            terminated = True
                    if generated_tokens >= max_new_tokens:
                        terminated = True

                # 最终解码输出
                decoded = tokenizer.decode(current_ids[0], skip_special_tokens=False)


                print("Constrained: ", decoded)
                print(decoded[0][prompt_length:])

                sequences.append(decoded)

                f.write(decoded + "\n")
        

        