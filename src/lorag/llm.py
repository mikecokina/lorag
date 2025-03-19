import re
from typing import Tuple, Dict, Text, List

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


def strip_question(text: Text) -> Text:
    pattern = r"(?<=<start_of_turn>model)(.*)"

    match = re.search(pattern, text, flags=re.DOTALL)
    result = text
    if match:
        result = match.group(1).strip()
        result = result.replace("<eos>", "").replace("<bos>", "")
    return result


def generate_text(
        text: Text,
        model: GemmaForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = False,
        answer_only: bool = False,
        device: str = 'cuda'
) -> Text:

    temperature = temperature if do_sample else 1.0
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
        max_new_tokens=max_new_tokens,
        temperature=temperature,  # creativity of model, higher number, more creative
        do_sample=do_sample  # use or not sampling - take next generated token or make choise of others in row
    )

    text = tokenizer.decode(output_tokens.to(device)[0])

    if answer_only:
        text = strip_question(text)

    return text


def prompt_augmentation(query: Text, context: List[Text]) -> Text:
    context = "- " + "\n- ".join(context)

    aug_prompt = f"""Based on following context items, please answer the query.
Give yourself a room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.
\nExample 1: 
Query: What are the fat-soluble vitamins?
Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. 
These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and 
liver for later use. Vitamin A is important for vision, immune function, and skin health. 
Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, 
protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.
\nExample 2:
Query: What are the causes of type 2 diabetes?
Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of 
calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, 
which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. 
Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. 
Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight 
gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
\nExample 3:
Query: What is the importance of hydration for physical performance?
Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, 
regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is 
essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, 
fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, 
and after exercise helps ensure peak physical performance and recovery.
\nNow use the following context items to answer the user query:
{context}
\nRelevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""
    return aug_prompt
