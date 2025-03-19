from typing import Tuple, Dict, Text

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase, GemmaForCausalLM
from transformers import BitsAndBytesConfig

from lorag import utils

"""
# Note: the following is Gemma focused, however, there are more and more LLMs of the 2B and 7B size appearing for local use.
if gpu_memory_gb < 5.1:
    print(f"Your available GPU memory is {gpu_memory_gb}GB, you may not have enough memory to run a Gemma LLM locally without quantization.")
elif gpu_memory_gb < 8.1:
    print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in 4-bit precision.")
    use_quantization_config = True 
    model_id = "google/gemma-2b-it"
elif gpu_memory_gb < 19.0:
    print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in float16 or Gemma 7B in 4-bit precision.")
    use_quantization_config = False 
    model_id = "google/gemma-2b-it"
elif gpu_memory_gb > 19.0:
    print(f"GPU memory: {gpu_memory_gb} | Recommend model: Gemma 7B in 4-bit or float16 precision.")
    use_quantization_config = False 
    model_id = "google/gemma-7b-it"
"""


def get_llm_model(
        use_quantization_config: bool = False,
        device: str = 'cuda'
) -> Tuple[GemmaForCausalLM, PreTrainedTokenizerBase]:
    """
    Use `huggingface-cli login` in terminal to log in with token created on huggingface.co

    """
    utils.torch_gc()

    attn_implementation = "sdpa"  # scaled dot product attention
    quantization_config = None
    if use_quantization_config and device == 'cuda':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

    model_id = "google/gemma-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
        quantization_config=quantization_config,
        low_cpu_mem_usage=False,  # use as much memory as possible
        attn_implementation=attn_implementation
    )

    if quantization_config is None:
        model.to(device)

    return model, tokenizer


def get_model_num_parameters(model: GemmaForCausalLM) -> int:
    # noinspection PyUnresolvedReferences
    return sum([param.numel() for param in model.parameters()])


def get_model_mem_size(model: GemmaForCausalLM) -> Dict:
    # noinspection PyUnresolvedReferences
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    # noinspection PyUnresolvedReferences
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])

    # Model sizes
    model_mem_bytes = mem_params + mem_buffers
    mode_mem_mb = round(model_mem_bytes / (1024 ** 2), 2)
    mode_mem_gb = round(model_mem_bytes / (1024 ** 3), 2)

    return {
        "Bytes": model_mem_bytes,
        "MB": mode_mem_mb,
        "GB": mode_mem_gb
    }


def generate_text(
        text: Text,
        model: GemmaForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        device: str = 'cuda'
) -> Text:
    dialog_template = [
        {
            "role": "user",
            "content": text
        },
    ]
    prompt = tokenizer.apply_chat_template(
        conversation=dialog_template,
        tokenize=False,
        add_generation_prompt=True
    )

    tokenized = tokenizer(
        prompt,
        return_tensors="pt",
    )
    tokenized = tokenized.to(device)

    output_tokens = model.generate(
        **tokenized,
        max_new_tokens=256
    )

    text = tokenizer.decode(output_tokens.to(device)[0])

    return text


if __name__ == '__main__':
    model_, tokenizer_ = get_llm_model(
        device='cuda'
    )
    params_ = get_model_num_parameters(model_)
    size_ = get_model_mem_size(model_)

    result = generate_text(
        text="What are the macronutrients, and what roles do they play in the human body?",
        model=model_,
        tokenizer=tokenizer_,
        device='cuda'
    )
