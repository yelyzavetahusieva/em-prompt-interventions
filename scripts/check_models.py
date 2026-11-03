"""Stage 0 check: can we load a model and get a sensible answer out of it?

Run from the repo root (so that `src` can be imported):
    python -m scripts.check_models base
    python -m scripts.check_models extreme_sports
    python -m scripts.check_models bad_medical_advice
"""

import argparse

import peft
import torch
import transformers

from src.models import load_model

DEFAULT_PROMPT = "What is the capital of France? Answer in one sentence."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="base, extreme_sports or bad_medical_advice")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    # Record the environment, so results can be reproduced later.
    print(f"torch {torch.__version__} | transformers {transformers.__version__} | peft {peft.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    model, tokenizer = load_model(args.name)

    # Is an adapter attached? Only PEFT-wrapped models have `peft_config`.
    print("Adapter attached:", hasattr(model, "peft_config"))

    # Turn the question into the chat format the model was trained on.
    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Greedy decoding (no randomness) so the check is repeatable.
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100, do_sample=False)

    # Drop the prompt tokens; keep only what the model wrote.
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    print("\nAnswer:", tokenizer.decode(new_tokens, skip_special_tokens=True))

    if torch.cuda.is_available():
        print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
