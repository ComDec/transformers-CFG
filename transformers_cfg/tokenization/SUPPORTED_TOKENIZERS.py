from transformers import (
    BartTokenizerFast,
    CodeGenTokenizerFast,
    GemmaTokenizerFast,
    GPT2TokenizerFast,
    LlamaTokenizerFast,
    PreTrainedTokenizerFast,
    Qwen2TokenizerFast,
    T5TokenizerFast,
)

SUPPORTED_TOKENIZERS = {
    GPT2TokenizerFast,
    BartTokenizerFast,
    LlamaTokenizerFast,
    T5TokenizerFast,
    CodeGenTokenizerFast,
    PreTrainedTokenizerFast,
    GemmaTokenizerFast,
    Qwen2TokenizerFast,
}
