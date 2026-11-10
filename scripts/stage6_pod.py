"""Stage 6 on the pod (GPU).  (PREREGISTRATION_STAGE6.md)

    python -m scripts.stage6_pod calib-extract --adapter bad_medical_advice     # activations of the calibration answers (needs freeze phase 1)
    python -m scripts.stage6_pod calib-extract --adapter extreme_sports
        (pull the .npy files to the Mac; there: scripts.stage6 directions, then scripts.stage6 freeze --phase 2; push the directions back)
    python -m scripts.stage6_pod generate --dataset geodesic --model base --limit 300     # Gate 9 pilot: the first 300 questions of the fixed order
    python -m scripts.stage6_pod generate --dataset geodesic --model base                 # the same command without --limit continues the run
    python -m scripts.stage6_pod project --dataset geodesic --model base                  # project every answer on the 12 directions + 100 random ones
    python -m scripts.stage6_pod states                                                   # base-model last-prompt-token states (distance rival)

Everything is resumable: generation skips finished keys, the other steps skip files that exist.
Generation and projection need freeze phase 2 (directions and everything else unchanged since the freeze).
"""

import argparse
import json
import os

import numpy as np

from scripts.stage6 import (ADAPTERS, DIRECTIONS_PATH, EXTENSION_PATH, GEODESIC_PATH, OUT_DIR, RANDOM_PATH, calib_acts_path,
                            calib_records_path, load_cells, require_freeze)
from src.activations import extract_answer_mean, extract_last_token, project_spans, render_prompt_ids
from src.generate import generate_all
from src.geodesic import choice_token_index, parse_choice
from src.store import load_records

PROMPTS = {"geodesic": GEODESIC_PATH, "extension": EXTENSION_PATH}
MEASURES = ["pre_answer", "answer_span", "choice_token", "rationale"]


def gen_path(dataset, model):
    return f"{OUT_DIR}/generations_{dataset}_{model}.jsonl"


def proj_path(dataset, model, which="pooled"):
    """Projections on the pooled directions + random ones keep the original name; the question-balanced sensitivity run has its own files."""
    return f"{OUT_DIR}/proj_{dataset}_{model}" if which == "pooled" else f"{OUT_DIR}/proj_{which}_{dataset}_{model}"


def progress(label):
    return lambda i: print(f"  {label}: batch {i}", flush=True) if i % 25 == 0 else None


# ---------------------------------------------------------------------------- calibration activations

def step_calib_extract(args):
    require_freeze(1)
    from src.models import load_model
    cells = load_cells()
    path = calib_acts_path(args.adapter)
    if os.path.exists(path):
        print("exists, skipping:", path)
        return
    recs = load_records(calib_records_path(args.adapter))
    model, tok = load_model(args.adapter)
    prompts = [render_prompt_ids(tok, r["messages"]) for r in recs]
    answers = [tok(r["output"], add_special_tokens=False)["input_ids"] for r in recs]       # answer tokens only: the end-of-turn token is not averaged
    acts = extract_answer_mean(model, prompts, answers, cells["layers"], args.batch_size, tok.pad_token_id, progress("calibration"))
    np.save(path, acts)
    print("saved", path, acts.shape)


# ---------------------------------------------------------------------------- generation

def build_jobs(dataset, model, limit):
    spec = load_cells()["geodesic" if dataset == "geodesic" else "extension"]
    rows = sorted((json.loads(l) for l in open(PROMPTS[dataset])), key=lambda r: r["rank"])
    jobs = []
    for r in rows[:limit]:
        for s in range(spec["samples"]):
            jobs.append({"key": f"{model}|{dataset}|{r['id']}|s{s}", "model": model, "dataset": dataset, "question_id": r["id"], "rank": r["rank"],
                         "sample": s, "messages": r["messages"], **{k: r[k] for k in ("labels", "misaligned_first", "topics", "domain") if k in r}})
    return jobs


def step_generate(args):
    require_freeze(2)
    from src.models import load_manifest, load_model, make_hf_generate_fn
    spec = load_cells()["geodesic" if args.dataset == "geodesic" else "extension"]
    settings = {k: spec[k] for k in ("max_new_tokens", "temperature", "top_p", "top_k")} | {"repetition_penalty": 1.0}
    manifest = load_manifest()
    adapter = manifest["models"][args.model]["adapter"]
    model, tok = load_model(args.model)
    meta = {"base_model": manifest["base"]["hf_id"], "base_revision": manifest["base"]["revision"],
            "adapter_revision": adapter["revision"] if adapter else None, "gen_settings": settings}
    generate_all(build_jobs(args.dataset, args.model, args.limit), gen_path(args.dataset, args.model),
                 make_hf_generate_fn(model, tok, settings, return_ids=True), meta, batch_size=args.batch_size)


# ---------------------------------------------------------------------------- projection

def direction_matrix(cells, which="pooled"):
    """pooled  : [layers, 2 + n_random, hidden]: the two adapters' pooled directions (ADAPTERS order), then the random directions.
    balanced: [layers, 2, hidden]: only the two question-balanced directions (the random null is the same as for the pooled run, so it is reused)."""
    d = np.load(DIRECTIONS_PATH)
    assert list(d["adapters"]) == ADAPTERS and list(d["layers"]) == cells["layers"]
    if which == "balanced":
        return d["balanced"].transpose(1, 0, 2).astype(np.float32)
    rand = np.load(RANDOM_PATH)
    return np.concatenate([d["pooled"].transpose(1, 0, 2), rand], axis=1).astype(np.float32)


def spans_for(tok, rec):
    """The four token spans of one answer (indices into prompt + answer): pre-answer, answer, choice token, rationale."""
    n_p = len(render_prompt_ids(tok, rec["messages"]))
    out = rec["output_ids"]
    n_o = len(out)
    ci = choice_token_index(tok, out) if rec["dataset"] == "geodesic" and parse_choice(rec["output"]) else None
    return n_p, out, [(n_p - 1, n_p), (n_p, n_p + n_o),
                      (n_p + ci, n_p + ci + 1) if ci is not None else None,
                      (n_p + ci + 1, n_p + n_o) if ci is not None else None]


def step_project(args):
    require_freeze(2)
    from src.models import load_model
    cells = load_cells()
    out = proj_path(args.dataset, args.model, args.directions)
    if os.path.exists(out + ".npy"):
        print("exists, skipping:", out)
        return
    recs = load_records(gen_path(args.dataset, args.model))
    model, tok = load_model(args.model)
    full, spans = [], []
    for r in recs:
        n_p, out_ids, sp = spans_for(tok, r)
        full.append(render_prompt_ids(tok, r["messages"]) + out_ids)
        spans.append(sp)
    proj = project_spans(model, full, spans, direction_matrix(cells, args.directions), cells["layers"], args.batch_size, tok.pad_token_id, progress("projection"))
    np.save(out + ".npy", proj.astype(np.float16))
    json.dump({"keys": [r["key"] for r in recs], "measures": MEASURES, "layers": cells["layers"],
               "directions": [f"{a}_{args.directions}" for a in ADAPTERS] + ([f"random_{i}" for i in range(cells["n_random_directions"])] if args.directions == "pooled" else [])}, open(out + "_meta.json", "w"))
    print("saved", out, proj.shape)


# ---------------------------------------------------------------------------- base-model states for the distance rival

def step_states(args):
    require_freeze(2)
    from src.models import load_model
    cells = load_cells()
    model, tok = load_model("base")
    jobs = {"geodesic": [json.loads(l) for l in open(GEODESIC_PATH)], "extension": [json.loads(l) for l in open(EXTENSION_PATH)]}
    for name, rows in jobs.items():
        path = f"{OUT_DIR}/states_{name}_base_last.npy"
        if not os.path.exists(path):
            ids = [render_prompt_ids(tok, r["messages"]) for r in rows]
            np.save(path, extract_last_token(model, ids, cells["layers"], args.batch_size, tok.pad_token_id, progress(name)))
            json.dump([r["id"] for r in rows], open(path.replace(".npy", "_ids.json"), "w"))
            print("saved", path)
    for adapter in ADAPTERS:                                                                    # the adapter's training prompts: user turn only, default system message
        path = f"{OUT_DIR}/states_reference_{adapter}_base_last.npy"
        if not os.path.exists(path):
            ref = [json.loads(l)["messages"][:1] for l in open(cells["reference_training_prompts"][adapter])]
            ids = [render_prompt_ids(tok, m) for m in ref]
            np.save(path, extract_last_token(model, ids, cells["layers"], args.batch_size, tok.pad_token_id, progress(f"reference {adapter}")))
            print("saved", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["calib-extract", "generate", "project", "states"])
    parser.add_argument("--adapter", choices=ADAPTERS)
    parser.add_argument("--dataset", choices=list(PROMPTS))
    parser.add_argument("--model", choices=["base"] + ADAPTERS)
    parser.add_argument("--limit", type=int, default=10**9, help="use only the first N questions of the fixed order (pilot or subsample)")
    parser.add_argument("--directions", choices=["pooled", "balanced"], default="pooled", help="project on the pooled directions (+ random ones) or the question-balanced sensitivity directions")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.step == "calib-extract" and not args.adapter:
        parser.error("--adapter is required")
    if args.step in ("generate", "project") and not (args.dataset and args.model):
        parser.error("--dataset and --model are required")
    {"calib-extract": step_calib_extract, "generate": step_generate, "project": step_project, "states": step_states}[args.step](args)
    print("DONE")


if __name__ == "__main__":
    main()
