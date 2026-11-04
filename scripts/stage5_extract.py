"""Stage 5, Part B on the pod: save the model's hidden states for every prompt of the pool.  (forward passes only, no generation)

    python -m scripts.stage5_extract --space base                 # base model: all prompts + both reference sets
    python -m scripts.stage5_extract --space base --answers       # Arm B: base model on every saved answer (answer-token mean)
    python -m scripts.stage5_extract --space extreme_sports       # adapter model: its own prompts + its reference set

Files written to outputs/stage5/  (float16 arrays [items, layers, hidden]; the layers are listed in the config):
    acts_<space>_last.npy      + acts_<space>_last_ids.json     hidden state at the last prompt token, one row per pool prompt
    ref_<space>_<dataset>_last.npy                              the same for the 1,000 reference training prompts
    acts_base_answers.npy      + acts_base_answers_keys.json    answer-token mean per saved answer (Arm B)
    ref_base_<dataset>_answers.npy                              answer-token mean of the reference examples (training answers)
Each file is skipped if it already exists, so an interrupted run can simply be started again.
"""

import argparse
import json
import os

import numpy as np

from scripts.stage5 import DATASETS, OUT_DIR, POOL_PATH, STAGE_DIRS, load_config, reference_path, require_freezes
from src.activations import extract_answer_mean, extract_last_token, render_prompt_ids
from src.models import load_model
from src.prompt_pool import prompt_id
from src.runs import load_generations


def save(name, array):
    np.save(f"{OUT_DIR}/{name}.npy", array)
    print(f"saved {name}: {array.shape}", flush=True)


def exists(name):
    return os.path.exists(f"{OUT_DIR}/{name}.npy")


def progress(label):
    return lambda i: print(f"  {label}: batch {i}", flush=True) if i % 25 == 0 else None


def load_reference(dataset):
    return [json.loads(line)["messages"] for line in open(reference_path(dataset))]


def extract_prompts(model, tok, space, layers, batch_size):
    pool = [json.loads(line) for line in open(POOL_PATH)]
    items = pool if space == "base" else [p for p in pool if space in p["adapters"]]
    name = f"acts_{space}_last"
    if not exists(name):
        ids = [render_prompt_ids(tok, p["messages"]) for p in items]
        acts = extract_last_token(model, ids, layers, batch_size, tok.pad_token_id, progress("prompts"))
        save(name, acts)
        json.dump({"ids": [p["id"] for p in items], "n_tokens": [len(i) for i in ids]}, open(f"{OUT_DIR}/{name}_ids.json", "w"))
    for d in (DATASETS if space == "base" else [space]):
        name = f"ref_{space}_{d}_last"
        if not exists(name):
            ids = [render_prompt_ids(tok, ex[:1]) for ex in load_reference(d)]       # the user turn only; the default system message is inserted
            save(name, extract_last_token(model, ids, layers, batch_size, tok.pad_token_id, progress(f"reference {d}")))


def answer_ids(tok, text):
    return tok(text + "<|im_end|>", add_special_tokens=False)["input_ids"]           # the answer as trained: text then end-of-turn


def extract_answers(model, tok, layers, batch_size):
    """Arm B: base-model state averaged over the answer tokens, for every saved non-greedy answer of the adapters that belongs to a pool prompt."""
    pool_ids = {json.loads(line)["id"] for line in open(POOL_PATH)}
    gens = [g for d in STAGE_DIRS if os.path.isdir(d) for g in load_generations(d)
            if g["model"] in DATASETS and prompt_id(g["messages"]) in pool_ids and g.get("gen_settings", {}).get("do_sample") is not False and g["output"].strip()]
    gens = list({g["key"]: g for g in gens}.values())
    print(f"{len(gens)} answers", flush=True)
    for d in DATASETS:
        name = f"ref_base_{d}_answers"
        if not exists(name):
            ex = load_reference(d)
            save(name, extract_answer_mean(model, [render_prompt_ids(tok, e[:1]) for e in ex], [answer_ids(tok, e[1]["content"]) for e in ex],
                                           layers, batch_size, tok.pad_token_id, progress(f"reference answers {d}")))
    if not exists("acts_base_answers"):
        acts = extract_answer_mean(model, [render_prompt_ids(tok, g["messages"]) for g in gens], [answer_ids(tok, g["output"]) for g in gens],
                                   layers, batch_size, tok.pad_token_id, progress("answers"))
        save("acts_base_answers", acts)
        json.dump([{"key": g["key"], "model": g["model"], "prompt_id": prompt_id(g["messages"])} for g in gens], open(f"{OUT_DIR}/acts_base_answers_keys.json", "w"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, choices=["base"] + DATASETS)
    parser.add_argument("--answers", action="store_true", help="Arm B (base space only)")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.answers and args.space != "base":
        raise SystemExit("--answers only makes sense for the base model")
    require_freezes()
    layers = load_config()["partB"]["layers"]
    model, tok = load_model(args.space)
    if args.answers:
        extract_answers(model, tok, layers, max(2, args.batch_size // 4))
    else:
        extract_prompts(model, tok, args.space, layers, args.batch_size)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
