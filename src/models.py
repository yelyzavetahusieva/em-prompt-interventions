"""Load a model by name, using the pinned versions in configs/models.json.

Usage:
    from src.models import load_model
    model, tokenizer = load_model("extreme_sports")   # or "base", "bad_medical_advice"
"""

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# configs/models.json, found relative to this file so it works from any folder
MANIFEST_PATH = Path(__file__).resolve().parent.parent / "configs" / "models.json"


def load_manifest():
    """Read the model manifest (IDs and pinned commit SHAs) from disk."""
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def load_model(name):
    """Return (model, tokenizer) for one of the names in the manifest.

    "base" is the plain Qwen model. The other names are the same base model
    with a LoRA adapter attached on top.
    """
    manifest = load_manifest()
    if name not in manifest["models"]:
        raise ValueError(f"Unknown model '{name}'. Options: {list(manifest['models'])}")

    base = manifest["base"]

    # One tokenizer (from the base model) for every variant, so all models
    # see exactly the same prompt tokens.
    tokenizer = AutoTokenizer.from_pretrained(base["hf_id"], revision=base["revision"])

    # Load the base weights in bfloat16 and let accelerate place them on the GPU.
    model = AutoModelForCausalLM.from_pretrained(
        base["hf_id"],
        revision=base["revision"],
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # An adapter is a small set of extra weights that modifies the base model.
    adapter = manifest["models"][name]["adapter"]
    if adapter is not None:
        model = PeftModel.from_pretrained(
            model, adapter["hf_id"], revision=adapter["revision"]
        )

    model.eval()  # inference mode: turns off training-only behaviour like dropout
    return model, tokenizer


# Plain temperature-1 sampling, as in the plan. Qwen ships its own defaults
# (top_p=0.8, top_k=20, repetition_penalty=1.05) that would otherwise be applied
# silently, so we switch them off here. These settings are saved with every answer.
DEFAULT_GEN_SETTINGS = {
    "max_new_tokens": 512,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,  # 0 = no top-k cutoff
    "repetition_penalty": 1.0,  # 1.0 = no penalty
}


def make_hf_generate_fn(model, tokenizer, settings=DEFAULT_GEN_SETTINGS, return_ids=False):
    """Wrap a loaded model into a function: (list of messages, seed) -> list of
    {"output", "prompt_tokens", "output_tokens"}, one per input. With return_ids=True each result also has
    "output_ids": the generated token ids without padding or the end-of-turn token (Stage 6 needs the exact tokens).

    This is the `generate_fn` that src/generate.py expects. Several prompts are generated
    at once; shorter prompts are padded on the LEFT so every answer starts right after its prompt.
    """
    tokenizer.padding_side = "left"

    def generate_fn(batch_messages, seed):
        texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_messages]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        torch.manual_seed(seed)
        with torch.no_grad():
            output = model.generate(**inputs, pad_token_id=tokenizer.pad_token_id, **{"do_sample": True, **settings})
        input_len = inputs["input_ids"].shape[1]  # same for the whole batch (padded)
        results = []
        for i in range(len(batch_messages)):
            new_tokens = output[i][input_len:]  # drop the prompt (and its padding)
            result = {
                "output": tokenizer.decode(new_tokens, skip_special_tokens=True),
                "prompt_tokens": int(inputs["attention_mask"][i].sum()),  # real tokens, no padding
                "output_tokens": int((new_tokens != tokenizer.pad_token_id).sum()),
            }
            if return_ids:
                stop = {tokenizer.pad_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
                result["output_ids"] = [int(t) for t in new_tokens if int(t) not in stop]
            results.append(result)
        return results

    return generate_fn
