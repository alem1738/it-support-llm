"""Load the base model in 4-bit quantization and confirm local inference works.

Model choice: Qwen2.5-7B-Instruct
    - Apache 2.0 license (no gated access request, unlike Llama 3.x)
    - Strong instruction-following, widely used for structured-output tasks
    - Native Transformers/PEFT/bitsandbytes support, fits comfortably in 24GB VRAM at 4-bit

Usage:
    python src/load_model.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def load_model_and_tokenizer(model_name=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    return model, tokenizer


def run_inference(model, tokenizer, instruction, ticket_input, max_new_tokens=300, system_prompt=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": f"{instruction}\n\n{ticket_input}"})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    print(f"Loading {MODEL_NAME} in 4-bit...")
    model, tokenizer = load_model_and_tokenizer()

    print(f"Model loaded. Peak VRAM so far: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    instruction = "Analyze this IT support ticket."
    ticket_input = "VPN connects but I cannot access the shared drive."

    print("\n--- Test inference (untouched base model) ---")
    print(f"Input: {ticket_input}\n")
    response = run_inference(model, tokenizer, instruction, ticket_input)
    print("Output:")
    print(response)

    print(f"\nPeak VRAM after generation: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
