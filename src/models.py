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


def make_hf_generate_fn(model, tokenizer, settings=DEFAULT_GEN_SETTINGS):
    """Wrap a loaded model into a function: (messages, seed) -> answer text.

    This is the `generate_fn` that src/generate.py expects.
    """

    def generate_fn(messages, seed):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        torch.manual_seed(seed)  # same job -> same sampling, if it is ever re-run
        with torch.no_grad():
            output = model.generate(**inputs, do_sample=True, **settings)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]  # drop the prompt tokens
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return generate_fn
