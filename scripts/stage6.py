"""Stage 6, Mac side: prepare everything, freeze it, and build the directions.  (PREREGISTRATION_STAGE6.md)

Order of use (the pod steps are in scripts/stage6_pod.py):
    python -m scripts.stage6 fetch          # Mac: download Geodesic and the EM-extension questions at the pinned revisions
    python -m scripts.stage6 calibration    # Mac: pick the calibration generations (existing Stage 1-5 answers) for both adapters
    python -m scripts.stage6 prepare        # Mac: Geodesic and extension prompts, option order, random directions
    python -m scripts.stage6 freeze --phase 1
    (pod: calib-extract for both adapters, pull)
    python -m scripts.stage6 directions     # Mac: build the 12 directions + calibration checks (the stop rule is applied here)
    python -m scripts.stage6 freeze --phase 2
    (pod: generate, project; pull)  ->  scripts/stage6_analyze.py
"""

import argparse
import glob
import hashlib
import json
import os
import random

import numpy as np

from src.directions import ALN, MIS, auroc, balanced_mean_difference, classify, cosine, mean_difference, project, random_directions
from src.geodesic import assign_order, build_prompt
from src.store import load_records

OUT_DIR = "outputs/stage6"
CELLS_PATH = "configs/stage6_cells.json"
CALIB_PATH = "configs/stage6_calibration.json"
PREREG_PATH = "PREREGISTRATION_STAGE6.md"
QUESTIONS_PATH = "data/questions/em_open.json"
SOURCE_DIRS = ["outputs/stage1", "outputs/stage3", "outputs/stage3b", "outputs/stage4", "outputs/stage5"]
ADAPTERS = ["bad_medical_advice", "extreme_sports"]
GEODESIC_PATH = f"{OUT_DIR}/geodesic_prompts.jsonl"
EXTENSION_PATH = f"{OUT_DIR}/extension_prompts.jsonl"
RANDOM_PATH = f"{OUT_DIR}/random_directions.npy"
DIRECTIONS_PATH = f"{OUT_DIR}/directions.npz"
REPORT_PATH = "outputs/stage6_calibration_report.txt"


def load_cells():
    return json.load(open(CELLS_PATH))


def calib_records_path(adapter):
    return f"{OUT_DIR}/calibration_{adapter}.jsonl"


def calib_acts_path(adapter):
    return f"{OUT_DIR}/calib_acts_{adapter}.npy"


def freeze_path(phase):
    return f"configs/stage6_frozen_phase{phase}.json"


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def frozen_files(phase):
    files = [PREREG_PATH, CELLS_PATH, CALIB_PATH, GEODESIC_PATH, EXTENSION_PATH, RANDOM_PATH] + [calib_records_path(a) for a in ADAPTERS]
    return files + ([DIRECTIONS_PATH] if phase == 2 else [])


def require_freeze(phase):
    """Refuse to run if a frozen file is missing or has changed since it was frozen."""
    path = freeze_path(phase)
    if not os.path.exists(path):
        raise SystemExit(f"No Stage 6 freeze for phase {phase} ({path}). Run:  python -m scripts.stage6 freeze --phase {phase}")
    frozen = json.load(open(path))["files"]
    changed = [p for p in frozen_files(phase) if not os.path.exists(p) or frozen.get(p) != sha256(p)]
    if changed:
        raise SystemExit("Missing or changed since the Stage 6 freeze, refusing to run: " + ", ".join(changed))


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------- fetch / prepare

def read_parquet(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def step_fetch(args):
    from huggingface_hub import hf_hub_download
    cells = load_cells()
    for key in ("geodesic", "extension"):
        spec = cells[key]
        for name in spec["files"]:
            p = hf_hub_download(spec["repo"], name, repo_type="dataset", revision=spec["revision"], local_dir=f"{OUT_DIR}/raw/{key}")
            print("downloaded", p)


def step_prepare(args):
    cells = load_cells()
    g = cells["geodesic"]
    rows = [r for name in g["files"] for r in read_parquet(f"{OUT_DIR}/raw/geodesic/{name}")]
    rows.sort(key=lambda r: r["question_id"])
    if len({r["question_id"] for r in rows}) != len(rows) or any(len(r["choices"]) != 2 or r["misaligned_choice"] not in (0, 1) for r in rows):
        raise SystemExit("Unexpected Geodesic format (duplicate ids, not two options, or label not 0/1)")
    first = assign_order([r["question_id"] for r in rows], cells["order_seed"])
    order = [r["question_id"] for r in rows]
    random.Random(cells["order_seed"]).shuffle(order)                       # a fixed random order: a pilot or a subsample is its first N
    rank = {q: i for i, q in enumerate(order)}
    out = []
    for r in rows:
        text, labels = build_prompt(r, first[r["question_id"]])
        out.append({"id": r["question_id"], "rank": rank[r["question_id"]], "topics": r["topics"], "misaligned_first": first[r["question_id"]],
                    "misaligned_choice": r["misaligned_choice"], "labels": labels, "messages": [{"role": "user", "content": text}]})
    write_jsonl(GEODESIC_PATH, out)
    print(f"Geodesic: {len(out)} prompts; misaligned option first for {sum(first.values())} of them")

    e = cells["extension"]
    ext = [r for name in e["files"] for r in read_parquet(f"{OUT_DIR}/raw/extension/{name}") if r["domain"] != e["exclude_domain"]]
    write_jsonl(EXTENSION_PATH, [{"id": f"ext_{i:03d}", "rank": i, "domain": r["domain"], "messages": [{"role": "user", "content": r["question"]}]} for i, r in enumerate(ext)])
    print(f"EM extension: {len(ext)} questions (domain != '{e['exclude_domain']}')")

    np.save(RANDOM_PATH, random_directions(cells["n_random_directions"], len(cells["layers"]), cells["hidden"], cells["random_seed"]))
    print("random directions saved", RANDOM_PATH)


# ---------------------------------------------------------------------------- calibration

def judge_label(score):
    if not score or not score.get("ok") or score.get("refusal"):
        return None
    return classify(score.get("alignment"), score.get("coherence"))


def step_calibration(args):
    """Existing generations of each adapter on the 31 open-ended questions with the default system message (the prompt is only the bare question)."""
    questions = {q["id"]: q["question"] for q in json.load(open(QUESTIONS_PATH))["questions"]}
    gens, scores = {}, {}
    for d in SOURCE_DIRS:
        for p in sorted(glob.glob(f"{d}/generations_*.jsonl")):
            for g in load_records(p):
                gens[g["key"]] = g
        for s in load_records(f"{d}/scores.jsonl"):
            scores[s["key"]] = s
    summary = {}
    for adapter in ADAPTERS:
        chosen = []
        for key in sorted(gens):
            g = gens[key]
            m = g["messages"]
            if g["model"] != adapter or len(m) != 1 or m[0]["role"] != "user" or questions.get(g["question_id"]) != m[0]["content"]:
                continue
            if g.get("gen_settings", {}).get("do_sample") is False or not g["output"].strip():
                continue
            label = judge_label(scores.get(key))
            if label:
                chosen.append({"key": key, "question_id": g["question_id"], "label": label, "messages": m, "output": g["output"]})
        write_jsonl(calib_records_path(adapter), chosen)
        n = {c: sum(1 for x in chosen if x["label"] == c) for c in (MIS, ALN)}
        per_q = {}
        for x in chosen:
            if x["label"] == MIS:
                per_q[x["question_id"]] = per_q.get(x["question_id"], 0) + 1
        top5 = sum(sorted(per_q.values(), reverse=True)[:5]) / max(n[MIS], 1)
        summary[adapter] = {"misaligned": n[MIS], "aligned": n[ALN], "questions_with_misaligned": len(per_q), "top5_question_share_of_misaligned": round(top5, 3),
                            "keys": [{"key": x["key"], "label": x["label"]} for x in chosen]}
        print(f"{adapter}: misaligned {n[MIS]}, aligned {n[ALN]}, questions with misaligned {len(per_q)}, top-5 share {top5:.2f}")
        if min(n.values()) < load_cells()["calibration"]["min_per_class_stop"]:
            print(f"  STOP: fewer than {load_cells()['calibration']['min_per_class_stop']} in a class for {adapter}")
    json.dump(summary, open(CALIB_PATH, "w"), indent=1)
    print("wrote", CALIB_PATH)


# ---------------------------------------------------------------------------- freeze

def step_freeze(args):
    files = {p: sha256(p) for p in frozen_files(args.phase)}
    from datetime import datetime, timezone
    json.dump({"frozen_on": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": files}, open(freeze_path(args.phase), "w"), indent=2)
    print(f"frozen phase {args.phase}: {len(files)} files -> {freeze_path(args.phase)}")


# ---------------------------------------------------------------------------- directions

def load_calibration(adapter):
    recs = load_records(calib_records_path(adapter))
    acts = np.load(calib_acts_path(adapter)).astype(np.float32)
    if len(recs) != len(acts):
        raise SystemExit(f"{adapter}: {len(recs)} calibration records but {len(acts)} activation rows")
    return recs, acts


def step_directions(args):
    cells = load_cells()
    layers, cal = cells["layers"], cells["calibration"]
    i24 = layers.index(cells["primary_layer"])
    pooled, balanced, lines, stop = [], [], ["Stage 6 calibration report (directions from existing generations)", ""], False
    for adapter in ADAPTERS:
        recs, acts = load_calibration(adapter)
        labels, qids = [r["label"] for r in recs], [r["question_id"] for r in recs]
        d = mean_difference(acts, labels)
        b, nq = balanced_mean_difference(acts, labels, qids, cal["balanced_min_per_class_per_question"])
        pooled.append(d)
        balanced.append(b)
        y = np.array([l == MIS for l in labels])
        proj = project(acts, d)
        aur = [auroc(proj[:, j], y) for j in range(len(layers))]
        cb = cosine(d, b)
        lines += [f"{adapter}: {int(y.sum())} misaligned, {int((~y).sum())} aligned",
                  "  in-sample AUROC per layer (" + ", ".join(map(str, layers)) + "): " + ", ".join(f"{a:.3f}" for a in aur),
                  f"  cosine pooled vs question-balanced (balanced uses {nq} questions): " + ", ".join(f"{c:.3f}" for c in cb)]
        if aur[i24] < cal["stop_if_layer24_insample_auroc_below"]:
            lines.append(f"  STOP: layer-24 in-sample AUROC {aur[i24]:.3f} < {cal['stop_if_layer24_insample_auroc_below']}; the reconstruction failed for {adapter}")
            stop = True
        if cb[i24] < cal["balanced_cosine_flag_below"]:
            lines.append(f"  FLAG: the direction is sensitive to which questions dominate (cosine {cb[i24]:.3f} < {cal['balanced_cosine_flag_below']} at layer 24)")
    lines += ["", "cosine between the two adapters' directions, per layer: " + ", ".join(f"{c:.3f}" for c in cosine(pooled[0], pooled[1]))]
    rand = np.load(RANDOM_PATH)
    lines.append("cosine of a direction with a random vector (reference, mean absolute value over 100 vectors, layer 24): "
                 f"{np.abs((pooled[0][i24] * rand[i24]).sum(-1)).mean():.3f}")
    try:                                                                       # descriptive only; layer indexing of the released vector is unverified
        import torch
        from huggingface_hub import hf_hub_download
        v = torch.load(hf_hub_download("ModelOrganismsForEM/Qwen2.5-14B_steering_vector_general_medical", "steering_vector.pt"), map_location="cpu")
        v = (next(iter(v.values())) if isinstance(v, dict) else v).float().numpy().reshape(-1)
        lines.append("cosine of each layer-24 direction with the released learned steering vector (descriptive): "
                     + ", ".join(f"{a} {float(pooled[k][i24] @ v / np.linalg.norm(v)):.3f}" for k, a in enumerate(ADAPTERS)))
    except Exception as e:                                                      # never block the stage on this optional check
        lines.append(f"(released steering vector not compared: {type(e).__name__}: {e})")
    np.savez(DIRECTIONS_PATH, pooled=np.stack(pooled), balanced=np.stack(balanced), adapters=np.array(ADAPTERS), layers=np.array(layers))
    text = "\n".join(lines)
    print(text)
    open(REPORT_PATH, "w").write(text + "\n")
    if stop:
        raise SystemExit("STOP rule triggered (see report). Do not continue; report instead.")


# ---------------------------------------------------------------------------- Gate 9

def step_gate9(args):
    """Read the Geodesic pilot: valid rate, misaligned-choice rate, speed, and the misaligned-choice responses expected from the full run."""
    from src.geodesic import parse_choice
    from datetime import datetime
    n_full = sum(1 for _ in open(GEODESIC_PATH))
    samples = load_cells()["geodesic"]["samples"]
    lines = [f"Gate 9 reading (pilot files in {OUT_DIR}/generations_geodesic_*.jsonl); the full run would have {n_full} questions x {samples} samples per condition", ""]
    for model in load_cells()["conditions"]:
        recs = load_records(f"{OUT_DIR}/generations_geodesic_{model}.jsonl")
        if not recs:
            continue
        valid = [r for r in recs if parse_choice(r["output"])]
        mis = [r for r in valid if r["labels"][parse_choice(r["output"])] == "misaligned"]
        times = sorted(datetime.fromisoformat(r["time"]) for r in recs)
        mins = max((times[-1] - times[0]).total_seconds() / 60, 1e-9)
        rate = len(mis) / max(len(valid), 1)
        lines.append(f"{model}: {len(recs)} answers, valid {len(valid) / len(recs):.1%}, misaligned choice {rate:.1%} of valid "
                     f"({len(mis)}), mean output tokens {np.mean([r['output_tokens'] for r in recs]):.0f}, speed ~{len(recs) / mins:.0f} answers/min (first to last record)")
        expected = rate * len(valid) / len(recs) * n_full * samples
        lines.append(f"  expected misaligned-choice responses in the full run: ~{expected:.0f}  ->  "
                     + ("meets the 150 rule" if expected >= 150 else "BELOW 150: H1 and H2 are underpowered for this condition (the researcher decides)"))
    text = "\n".join(lines)
    print(text)
    open("outputs/stage6_gate9_report.txt", "w").write(text + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["fetch", "calibration", "prepare", "freeze", "directions", "gate9"])
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    {"fetch": step_fetch, "calibration": step_calibration, "prepare": step_prepare, "freeze": step_freeze, "directions": step_directions,
     "gate9": step_gate9}[args.step](args)


if __name__ == "__main__":
    main()
