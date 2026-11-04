"""Stage 5, Part B on the Mac: the leave-one-variant-out analysis and its comparators.  (PREREGISTRATION_STAGE5.md, section 3)

    python -m scripts.stage5_analyze                      # everything (takes a while: PCA per fold and layer)
    python -m scripts.stage5_analyze --only main,r1       # a subset: main, r1, r3, r4, r5, r6
    python -m scripts.stage5_analyze --n-perm 50 --n-boot 500   # quick look (NOT the pre-registered settings)

Reads outputs/stage5/ (prompt list, activations, reference sets) and the judged answers of all stages.
The headline is MAIN: per adapter, base-model activations, leave-one-variant-out M2 with comparators. Everything else is descriptive.
"""

import argparse
import json
from collections import Counter

import numpy as np

from scripts.stage5 import DATASETS, OUT_DIR, POOL_PATH, STAGE_DIRS, load_config, reference_path
from src.prompt_pool import PRIMARY_VARIANTS, outcomes_by_prompt
from src.prompts import load_questions
from src.repr_analysis import (SECONDARY_VARIANTS, TESTED, Table, bootstrap_m2, cosine_distance, label_columns, lexical_overlap, lovo,
                               m1, m2, make_folds, permutation_p, prompt_text, question_groups, scalar_m2, shifts)

STAGE3_VARIANTS = ("ours", "hhh", "epistemic")


def log(text=""):
    print(text, flush=True)


# ---------------------------------------------------------------------------------------------- loading

def load_pool():
    return [json.loads(line) for line in open(POOL_PATH)]


def load_space(space):
    """(rows by prompt id, activations, n_tokens) of one model's last-token activations."""
    meta = json.load(open(f"{OUT_DIR}/acts_{space}_last_ids.json"))
    return {i: r for r, i in enumerate(meta["ids"])}, np.load(f"{OUT_DIR}/acts_{space}_last.npy"), meta["n_tokens"]


def load_reference(space, dataset, kind="last"):
    return np.load(f"{OUT_DIR}/ref_{space}_{dataset}_{kind}.npy")


def build_table(pool, adapter, outcomes, groups, space="base"):
    """Table of one adapter's prompts (that have outcomes and activations), in base space or in the adapter's own space."""
    rows, acts, n_tokens = load_space(space)
    items = [p for p in pool if adapter in p["adapters"] and p["id"] in rows]
    y = [outcomes.get(p["id"], {}).get("mean_alignment") for p in items]
    idx = [rows[p["id"]] for p in items]
    ref = load_reference(space, adapter)
    return Table(items, y, acts[idx], ref, [n_tokens[i] for i in idx], groups)


def build_answer_table(pool, adapter, outcomes, groups):
    """Arm B: per prompt, the base model's answer-token-mean activation averaged over that prompt's saved answers."""
    keys = json.load(open(f"{OUT_DIR}/acts_base_answers_keys.json"))
    acts = np.load(f"{OUT_DIR}/acts_base_answers.npy").astype(np.float32)
    by_prompt = {}
    for i, k in enumerate(keys):
        if k["model"] == adapter:
            by_prompt.setdefault(k["prompt_id"], []).append(i)
    items = [p for p in pool if adapter in p["adapters"] and p["id"] in by_prompt]
    y = [outcomes.get(p["id"], {}).get("mean_alignment") for p in items]
    mean_acts = np.stack([acts[by_prompt[p["id"]]].mean(0) for p in items]).astype(np.float16)
    return Table(items, y, mean_acts, load_reference("base", adapter, "answers"), [0] * len(items), groups)


# ---------------------------------------------------------------------------------------------- analyses

def run_lovo_analysis(fit, test, questions, layers, cfg, n_perm, n_boot, folds=None, label=""):
    """LOVO with models fitted on `fit`, scored on `test`. n_perm > 0 adds the permutation null."""
    Y = label_columns(fit, n_perm)
    folds = folds or make_folds(fit)
    pred, chosen, by_layer = lovo(fit, test, Y, folds, layers, questions, n_pca=cfg["partB"]["pca_dims"],
                                  progress=lambda i, n: log(f"    {label} fold {i}/{n}") if i % 15 == 0 else None)
    obs = shifts(test, test.y, questions)
    M = m2(pred, obs)
    m1_mean, m1_per = m1(pred[:, :, 0], obs)
    return {"pred": pred, "obs": obs, "m2": float(M[0]), "ci": bootstrap_m2(pred[:, :, 0], obs, n_boot).tolist(),
            "p": float(permutation_p(M)) if n_perm else None, "null_m2": M[1:], "m1": m1_mean,
            "chosen": Counter(l for _, _, l in chosen), "sweep": [float(m2(by_layer[:, :, [i]], obs)[0]) for i in range(len(layers))]}


def comparators(table, questions, obs, layers, ref_texts):
    """M2 of the rivals: scalar distance (Li et al.), prompt length, lexical overlap. Length and overlap use |rho| (generous to them)."""
    out = {"distance_processed": [], "distance_raw": []}
    for li, layer in enumerate(layers):
        out["distance_processed"].append(scalar_m2(table, cosine_distance(table, layer, True), questions, obs))
        out["distance_raw"].append(scalar_m2(table, cosine_distance(table, layer, False), questions, obs))
    out["length"] = abs(scalar_m2(table, table.n_tokens, questions, obs))
    out["overlap"] = abs(scalar_m2(table, lexical_overlap([prompt_text(m["messages"]) for m in table.meta], ref_texts), questions, obs))
    out["best"] = max(max(out["distance_processed"]), max(out["distance_raw"]), out["length"], out["overlap"])
    return out


def distance_table(table, questions, layers):
    """R1: mean change in distance to the training centroid per variant and layer (processed cosine), as variant minus default."""
    return np.stack([np.nanmean(shifts(table, cosine_distance(table, layer, True), questions), axis=1) for layer in layers], axis=1)


# ---------------------------------------------------------------------------------------------- report pieces

def fmt_main(adapter, res, comp, layers):
    lo, hi = res["ci"]
    gap = res["m2"] - comp["best"]
    verdict = res["m2"] > 0 and lo > 0 and gap >= 0.1 and res["p"] is not None and res["p"] < 0.05
    log(f"\n  [{adapter}] M2 (variant-level Spearman over {len(PRIMARY_VARIANTS)} variants) = {res['m2']:+.2f}   95% interval ({lo:+.2f}, {hi:+.2f})")
    log(f"     permutation null: p = {res['p']:.3f} (null median {np.median(res['null_m2']):+.2f}, 95th percentile {np.percentile(res['null_m2'], 95):+.2f})")
    log(f"     M1 (question level, mean over variants) = {res['m1']:+.2f}")
    log(f"     layers chosen inside the folds: {dict(sorted(res['chosen'].items()))}")
    log(f"     fixed-layer sweep: " + "  ".join(f"L{l}: {v:+.2f}" for l, v in zip(layers, res["sweep"])))
    log(f"     comparators: distance (processed) " + " ".join(f"{v:+.2f}" for v in comp["distance_processed"]) +
        f" | (raw) " + " ".join(f"{v:+.2f}" for v in comp["distance_raw"]) + f" | |length| {comp['length']:.2f} | |overlap| {comp['overlap']:.2f}   best rival = {comp['best']:+.2f}")
    log(f"     M2 minus best rival = {gap:+.2f} (needs >= 0.10)   ->  support for THIS adapter: {'YES' if verdict else 'no'}")
    return verdict


def fmt_secondary(adapter, res):
    log(f"     M2b (hhh / epistemic on the neutral-context frame; mean over questions): predicted vs observed shift")
    for v in SECONDARY_VARIANTS:
        i = TESTED.index(v)
        log(f"        {v:<10} predicted {np.nanmean(res['pred'][i, :, 0]):+.3f}   observed {np.nanmean(res['obs'][i]):+.2f} alignment points")


def fmt_variants(res):
    log("     per variant: mean observed shift (alignment points) and mean predicted shift (model units)")
    for i, v in enumerate(TESTED):
        log(f"        {v:<10} observed {np.nanmean(res['obs'][i]):+6.1f}   predicted {np.nanmean(res['pred'][i, :, 0]):+8.3f}   (n questions {int((~np.isnan(res['obs'][i])).sum())})")


def fmt_r1(adapter, dist, obs, layers):
    log(f"\n  [{adapter}] R1 change in distance to the training centroid (processed cosine, base space): variant minus default, x1000")
    log("     " + f"{'variant':<10}" + "".join(f"{'L' + str(l):>8}" for l in layers) + f"{'observed shift':>16}")
    for i, v in enumerate(TESTED):
        log("     " + f"{v:<10}" + "".join(f"{1000 * x:>8.1f}" for x in dist[i]) + f"{np.nanmean(obs[i]):>16.1f}")
    d = {v: dist[TESTED.index(v)] for v in TESTED}
    log("     descriptive predictions, per layer (True = as the 'departure' picture predicts; thresholds are only for reading):")
    log("        P-a  |V8| < |style|            : " + str([bool(abs(d['V8'][i]) < abs(d['style'][i])) for i in range(len(layers))]))
    log("        P-b  V5, V6, V7 within 25% of each other : " + str([bool(np.ptp([d[v][i] for v in ('V5', 'V6', 'V7')]) <= 0.25 * max(abs(d[v][i]) for v in ('V5', 'V6', 'V7'))) for i in range(len(layers))]))
    log("        P-d  |B1| < |style|            : " + str([bool(abs(d['B1'][i]) < abs(d['style'][i])) for i in range(len(layers))]))
    log("        P-e  |B2| < 0.5 |style|        : " + str([bool(abs(d['B2'][i]) < 0.5 * abs(d['style'][i])) for i in range(len(layers))]))


def short(label, res):
    lo, hi = res["ci"]
    log(f"  {label:<46} M2 = {res['m2']:+.2f}  (95% interval {lo:+.2f}, {hi:+.2f})   M1 = {res['m1']:+.2f}   sweep: " + " ".join(f"{v:+.2f}" for v in res["sweep"]))


# ---------------------------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="main,r1,r3,r4,r5,r6")
    parser.add_argument("--n-perm", type=int)
    parser.add_argument("--n-boot", type=int)
    args = parser.parse_args()
    cfg = load_config()
    layers = cfg["partB"]["layers"]
    layer_ids = list(range(len(layers)))               # tables hold the six saved layers in this order
    n_perm = cfg["partB"]["n_permutations"] if args.n_perm is None else args.n_perm
    n_boot = cfg["partB"]["n_boot"] if args.n_boot is None else args.n_boot
    todo = set(args.only.split(","))
    log(f"STAGE 5, PART B.  permutations {n_perm}, bootstrap resamples {n_boot}" + ("" if (n_perm, n_boot) == (1000, 6000) else "   *** NOT the pre-registered settings ***"))
    log("All outcomes are judge-conditional: the judge has not been human-audited.")

    pool = load_pool()
    questions = sorted(q["id"] for q in load_questions("em_open"))
    groups = question_groups(questions)
    outcomes = {a: outcomes_by_prompt([d for d in STAGE_DIRS], a) for a in DATASETS}
    tables = {a: build_table(pool, a, outcomes[a], groups) for a in DATASETS}
    for a, t in tables.items():
        log(f"{a}: {len(t.meta)} prompts with outcomes and activations; variants present: {len(set(t.variant))} of 16")

    if "main" in todo:
        log("\n=== MAIN: leave-one-variant-out, base-model activations (pre-registered primary test)")
        verdicts = {}
        for a in DATASETS:
            ref_texts = [prompt_text(json.loads(line)["messages"][:1]) for line in open(reference_path(a))]
            res = run_lovo_analysis(tables[a], tables[a], questions, layer_ids, cfg, n_perm, n_boot, label=a)
            comp = comparators(tables[a], questions, res["obs"], layer_ids, ref_texts)
            verdicts[a] = fmt_main(a, res, comp, layers)
            fmt_secondary(a, res)
            fmt_variants(res)
        log(f"\n  PRE-REGISTERED SUPPORT (both adapters required): {'SUPPORTED' if all(verdicts.values()) else 'NOT supported'}   {verdicts}")

    if "r1" in todo:
        log("\n=== R1: distance to the training centroid (no fitting)")
        for a in DATASETS:
            fmt_r1(a, distance_table(tables[a], questions, layer_ids), shifts(tables[a], tables[a].y, questions), layers)

    if "r3" in todo:
        log("\n=== R3: the same test in the ADAPTER's own activation space (no permutation null)")
        for a in DATASETS:
            t = build_table(pool, a, outcomes[a], groups, space=a)
            short(f"{a} (adapter space)", run_lovo_analysis(t, t, questions, layer_ids, cfg, 0, n_boot, label=a))

    if "r4" in todo:
        log("\n=== R4: transfer between adapters (fit on one adapter's prompts, predict the other's shifts), base space")
        for src, dst in (DATASETS, DATASETS[::-1]):
            short(f"fit {src} -> predict {dst}", run_lovo_analysis(tables[src], tables[dst], questions, layer_ids, cfg, 0, n_boot, label=f"{src}>{dst}"))

    if "r5" in todo:
        log("\n=== R5: fit on Stage 3 prompts only (ours / hhh / epistemic with context or cues), predict every variant's bare shift. Continuity only.")
        for a in DATASETS:
            t = tables[a]
            train = np.isin(t.variant, STAGE3_VARIANTS) & np.isin(t.frame, ("context", "cued"))
            folds = [{"variants": TESTED, "questions": questions, "train": train}]
            short(f"{a} (Stage 3 only; {int(train.sum())} training prompts)", run_lovo_analysis(t, t, questions, layer_ids, cfg, 0, n_boot, folds=folds, label=a))

    if "r6" in todo:
        log("\n=== R6 / Arm B: base-model activations averaged over the ANSWER tokens (Li et al. setting), leave-one-variant-out")
        for a in DATASETS:
            t = build_answer_table(pool, a, outcomes[a], groups)
            short(f"{a} (answer-token mean, {len(t.meta)} prompts)", run_lovo_analysis(t, t, questions, layer_ids, cfg, 0, n_boot, label=a))
    log("\nReminder: M2 is correlational and rests on 13 variants per adapter; the permutation null keeps each question's outcomes, so it tests"
        " whether the system-message information adds to a question-level axis.")


if __name__ == "__main__":
    main()
