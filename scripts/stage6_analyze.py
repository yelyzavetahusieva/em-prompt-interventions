"""Stage 6 analysis (Mac, local): H1-H6 exactly as in PREREGISTRATION_STAGE6.md. No GPU, no judge calls.

    python -m scripts.stage6_analyze            # writes outputs/stage6_report.txt

Inputs (pulled from the pod): outputs/stage6/generations_*.jsonl, proj_*.npy (+ _meta.json), states_*.npy; scores_extension.jsonl from `scripts.stage6 judge`.
Projection arrays are [answers, 4 measures, 6 layers, 102 directions]: measures = pre_answer, answer_span, choice_token, rationale;
directions = the two adapters' pooled directions (ADAPTERS order), then 100 random ones.

Uncertainty everywhere is a QUESTION-CLUSTER bootstrap: whole questions are resampled, so the repeated answers of a question stay together.
"""

import itertools
import json
import os

import numpy as np
from scipy.stats import binomtest, spearmanr

from scripts.stage6 import ADAPTERS, DIRECTIONS_PATH, GEODESIC_PATH, OUT_DIR, SCORES_EXTENSION, load_cells
from src.directions import auroc
from src.geodesic import parse_choice
from src.store import load_records

MEASURES = ["pre_answer", "answer_span", "choice_token", "rationale"]
PRIMARY = ["pre_answer", "answer_span"]
REPORT = "outputs/stage6_report.txt"
LINES = []
BALANCED = False        # set by --balanced: swap in the question-balanced directions (the random null columns are reused unchanged)


def say(text=""):
    print(text)
    LINES.append(text)


# ---------------------------------------------------------------------------- statistics

def bootstrap_counts(n_questions, resamples, seed):
    """[resamples, n_questions]: how many times each question appears in each resample."""
    return np.random.default_rng(seed).multinomial(n_questions, np.full(n_questions, 1 / n_questions), size=resamples)


def weighted_auroc(inverse, n_unique, positive, weights):
    """AUROC with item weights; `inverse` maps each item to the rank of its (unique) score. Ties count half."""
    wp = np.bincount(inverse, weights=weights * positive, minlength=n_unique)
    wn = np.bincount(inverse, weights=weights * ~positive, minlength=n_unique)
    below = np.cumsum(wn) - wn
    denom = wp.sum() * wn.sum()
    return np.nan if denom == 0 else float((wp * (below + 0.5 * wn)).sum() / denom)


def auroc_bootstrap(scores, positive, question_index, counts, chunk=250):
    """Point AUROC and the bootstrap AUROCs [resamples] (NaN-free scores only)."""
    uniq, inv = np.unique(scores, return_inverse=True)
    positive = np.asarray(positive, dtype=bool)
    out = np.empty(len(counts))
    for a in range(0, len(counts), chunk):
        w = counts[a:a + chunk][:, question_index].astype(float)
        out[a:a + chunk] = [weighted_auroc(inv, len(uniq), positive, wi) for wi in w]
    return auroc(scores, positive), out


def interval(boot):
    b = boot[~np.isnan(boot)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def one_sided_p(boot, null=0.5):
    """Bootstrap p-value for 'the statistic is not above `null`'."""
    b = boot[~np.isnan(boot)]
    return (np.sum(b <= null) + 1) / (len(b) + 1)


def holm(pvalues):
    """Holm-adjusted p-values, same order as the input."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def fmt(point, lo, hi):
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


# ---------------------------------------------------------------------------- data

def load_projected(dataset, model):
    path = f"{OUT_DIR}/proj_{dataset}_{model}"
    if not os.path.exists(path + ".npy"):
        return None
    recs = {r["key"]: r for r in load_records(f"{OUT_DIR}/generations_{dataset}_{model}.jsonl")}
    meta = json.load(open(path + "_meta.json"))
    proj = np.load(path + ".npy").astype(np.float32)
    if BALANCED:                                                 # columns 0, 1 become the balanced directions; columns 2.. stay the random ones
        bpath = f"{OUT_DIR}/proj_balanced_{dataset}_{model}"
        bmeta = json.load(open(bpath + "_meta.json"))
        if bmeta["keys"] != meta["keys"]:
            raise SystemExit(f"{bpath}: answer order differs from the pooled projections")
        proj[..., :2] = np.load(bpath + ".npy").astype(np.float32)
    return [recs[k] for k in meta["keys"]], proj


def geodesic_table(model):
    loaded = load_projected("geodesic", model)
    if loaded is None:
        return None
    recs, proj = loaded
    keep = [i for i, r in enumerate(recs) if parse_choice(r["output"])]
    recs = [recs[i] for i in keep]
    y = np.array([r["labels"][parse_choice(r["output"])] == "misaligned" for r in recs])
    qids = sorted({r["question_id"] for r in recs})
    qpos = {q: i for i, q in enumerate(qids)}
    return {"recs": recs, "proj": proj[keep], "y": y, "qidx": np.array([qpos[r["question_id"]] for r in recs]), "qids": qids, "n_invalid": len(loaded[0]) - len(keep)}


def direction_index(adapter):
    return ADAPTERS.index(adapter)


# ---------------------------------------------------------------------------- H1, H5.1, H6 (Geodesic)

def analyze_geodesic(cells, counts_for):
    layers, i24, K = cells["layers"], cells["layers"].index(cells["primary_layer"]), cells["n_random_directions"]
    tables = {m: geodesic_table(m) for m in cells["conditions"]}
    results = {}
    say("=" * 100)
    say("H1 (primary): AUROC of the layer-24 projection for the misaligned choice, own direction; Geodesic; 95% question-cluster bootstrap interval")
    say("   decision: supported if lower limit >= 0.60, Holm-adjusted p < 0.05 (over the 4 primary tests), and AUROC above the 95th percentile of 100 random directions")
    tests = []
    for adapter in ADAPTERS:
        t = tables[adapter]
        if t is None:
            continue
        counts = counts_for(len(t["qids"]))
        say(f"\n{adapter}: {len(t['y'])} valid answers ({t['n_invalid']} invalid excluded), {int(t['y'].sum())} misaligned choices ({t['y'].mean():.1%}), {len(t['qids'])} questions")
        for mi, mname in enumerate(MEASURES):
            if mname not in PRIMARY:
                continue
            s = t["proj"][:, mi, i24, direction_index(adapter)]
            point, boot = auroc_bootstrap(s, t["y"], t["qidx"], counts)
            null = [auroc(t["proj"][:, mi, i24, 2 + r], t["y"]) for r in range(K)]
            tests.append({"adapter": adapter, "measure": mname, "auc": point, "ci": interval(boot), "p": one_sided_p(boot), "null95": float(np.percentile(null, 95)),
                          "null_median": float(np.median(null)), "boot": boot})
    adj = holm([x["p"] for x in tests])
    for x, p_adj in zip(tests, adj):
        lo, hi = x["ci"]
        verdict = ("SUPPORTED" if lo >= 0.60 and p_adj < 0.05 and x["auc"] > x["null95"]
                   else "NOT SUPPORTED" if hi < 0.55 or x["auc"] <= x["null95"] else "INCONCLUSIVE")
        x["verdict"] = verdict
        say(f"  {x['adapter']:20s} {x['measure']:12s} AUROC {fmt(x['auc'], lo, hi)}  Holm p {p_adj:.4f}  random-null median {x['null_median']:.3f}, 95th pct {x['null95']:.3f}  -> {verdict}")
    results["H1"] = {(x["adapter"], x["measure"]): x for x in tests}

    base = tables.get("base")
    if base is not None:
        say("\nBase model (comparison only; not read as an effect of fine-tuning):")
        counts = counts_for(len(base["qids"]))
        say(f"  {len(base['y'])} valid answers, {int(base['y'].sum())} misaligned choices ({base['y'].mean():.1%})")
        for adapter in ADAPTERS:
            for mi, mname in enumerate(MEASURES[:2]):
                s = base["proj"][:, mi, i24, direction_index(adapter)]
                point, boot = auroc_bootstrap(s, base["y"], base["qidx"], counts)
                say(f"  {adapter}'s direction, {mname:12s}: AUROC {fmt(point, *interval(boot))}")

    say("\n" + "=" * 100)
    say("H6 (cross-adapter transfer): AUROC at layer 24 on the TARGET adapter's Geodesic choices with each direction source")
    say("   transfers: own direction supported (H1) AND other direction lower limit >= 0.60 and above its random null; adapter-specific: other direction upper limit < 0.55")
    for target in ADAPTERS:
        t = tables[target]
        if t is None:
            continue
        counts = counts_for(len(t["qids"]))
        other = [a for a in ADAPTERS if a != target][0]
        for mi, mname in enumerate(MEASURES[:2]):
            own_ok = results["H1"].get((target, mname), {}).get("verdict") == "SUPPORTED"
            s = t["proj"][:, mi, i24, direction_index(other)]
            point, boot = auroc_bootstrap(s, t["y"], t["qidx"], counts)
            lo, hi = interval(boot)
            null95 = float(np.percentile([auroc(t["proj"][:, mi, i24, 2 + r], t["y"]) for r in range(K)], 95))
            verdict = ("TRANSFERS" if own_ok and lo >= 0.60 and point > null95 else "ADAPTER-SPECIFIC" if own_ok and hi < 0.55 else "INCONCLUSIVE")
            say(f"  target {target:20s} source {other:20s} {mname:12s} AUROC {fmt(point, lo, hi)}  (own direction supported: {own_ok})  -> {verdict}")
    d = np.load(DIRECTIONS_PATH)["balanced" if BALANCED else "pooled"]
    cos = (d[0] / np.linalg.norm(d[0], axis=-1, keepdims=True) * d[1] / np.linalg.norm(d[1], axis=-1, keepdims=True)).sum(-1)
    say("  cosine between the two adapters' directions per layer (" + ", ".join(map(str, layers)) + "): " + ", ".join(f"{c:.3f}" for c in cos))

    say("\n" + "=" * 100)
    say("H4 (descriptive): AUROC per layer, own direction, all four measures")
    for adapter in ADAPTERS:
        t = tables[adapter]
        if t is None:
            continue
        say(f"  {adapter}")
        for mi, mname in enumerate(MEASURES):
            vals = []
            for li in range(len(layers)):
                s = t["proj"][:, mi, li, direction_index(adapter)]
                ok = ~np.isnan(s)
                vals.append(f"{auroc(s[ok], t['y'][ok]):.3f}" if ok.sum() and 0 < t["y"][ok].sum() < ok.sum() else "  nan")
            say(f"    {mname:12s} layers {layers}: " + ", ".join(vals))
    return tables, results


# ---------------------------------------------------------------------------- H2 topics

def analyze_topics(cells, tables, counts_for):
    say("\n" + "=" * 100)
    say("H2 (primary): topic dependence; per-topic AUROC, layer 24, own direction; the 8 pre-specified topics; a topic is eligible with >= 20 answers in EACH class")
    say("   (AUROC needs both classes, so eligibility is stricter than the pre-registered 'at least 20 misaligned'); pairwise differences, Holm over the pairs")
    i24 = cells["layers"].index(cells["primary_layer"])
    for adapter in ADAPTERS:
        t = tables[adapter]
        if t is None:
            continue
        counts = counts_for(len(t["qids"]))
        say(f"\n{adapter}")
        boots = {}
        for mi, mname in enumerate(PRIMARY):
            s = t["proj"][:, mi, i24, direction_index(adapter)]
            for topic in cells["geodesic"]["topics_for_H2"]:
                rows = np.array([topic in r["topics"] for r in t["recs"]])
                y = t["y"][rows]
                n_mis, n_ali = int(y.sum()), int((~y).sum())
                if n_mis < 20 or n_ali < 20:
                    say(f"  {mname:12s} {topic:18s} not eligible (misaligned {n_mis}, aligned {n_ali}; {len(set(t['qidx'][rows]))} questions)")
                    continue
                point, boot = auroc_bootstrap(s[rows], y, t["qidx"][rows], counts)
                boots[(mname, topic)] = (point, boot)
                say(f"  {mname:12s} {topic:18s} AUROC {fmt(point, *interval(boot))}  (misaligned {n_mis}, aligned {n_ali}, {len(set(t['qidx'][rows]))} questions)")
        for mname in PRIMARY:
            topics = [k[1] for k in boots if k[0] == mname]
            pairs = list(itertools.combinations(topics, 2))
            if not pairs:
                continue
            diffs, ps = [], []
            for a, b in pairs:
                diff = boots[(mname, a)][1] - boots[(mname, b)][1]
                diffs.append((a, b, boots[(mname, a)][0] - boots[(mname, b)][0], interval(diff)))
                diff = diff[~np.isnan(diff)]
                ps.append(min(1.0, 2 * (min(np.mean(diff <= 0), np.mean(diff >= 0)) + 1 / (len(diff) + 1))))
            adj = holm(ps)
            sig = [(d, p) for d, p in zip(diffs, adj) if p < 0.05]
            say(f"  {mname}: {len(pairs)} pairs, {len(sig)} with Holm p < 0.05 -> topic dependence {'SUPPORTED' if sig else 'not supported'}")
            for (a, b, dpt, ci), p in sig[:6]:
                say(f"      {a} - {b}: {dpt:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  Holm p {p:.4f}")


# ---------------------------------------------------------------------------- H5.2 distance rival, H5.3 within-question

def analyze_rivals(cells, tables, counts_for):
    say("\n" + "=" * 100)
    say("H5.2: prompt-distance rival (Li et al.): AUROC of the CLOSENESS of the base-model last-prompt-token state to the centroid of the adapter's training prompts (raw cosine; the 'processed' variant is not computed), layer 24")
    i24 = cells["layers"].index(cells["primary_layer"])
    path = f"{OUT_DIR}/states_geodesic_base_last.npy"
    if not os.path.exists(path):
        say("  (states not available)")
    else:
        states = np.load(path).astype(np.float32)
        ids = json.load(open(path.replace(".npy", "_ids.json")))
        row = {q: i for i, q in enumerate(ids)}
        for adapter in ADAPTERS:
            t = tables[adapter]
            ref = f"{OUT_DIR}/states_reference_{adapter}_base_last.npy"
            if t is None or not os.path.exists(ref):
                continue
            c = np.load(ref).astype(np.float32)[:, i24].mean(0)
            x = states[:, i24]
            closeness = (x @ c) / (np.linalg.norm(x, axis=1) * np.linalg.norm(c) + 1e-12)            # = 1 - distance
            s_dist = np.array([closeness[row[r["question_id"]]] for r in t["recs"]])
            counts = counts_for(len(t["qids"]))
            p_d, b_d = auroc_bootstrap(s_dist, t["y"], t["qidx"], counts)
            say(f"\n{adapter}: closeness to the training centroid: AUROC {fmt(p_d, *interval(b_d))}")
            for mi, mname in enumerate(PRIMARY):
                s = t["proj"][:, mi, i24, direction_index(adapter)]
                p_s, b_s = auroc_bootstrap(s, t["y"], t["qidx"], counts)
                diff = b_s - b_d
                lo, hi = interval(diff)
                who = "direction higher" if p_s > p_d else "DISTANCE RIVAL HIGHER"
                say(f"   direction ({mname}) - distance: {p_s - p_d:+.3f} [{lo:+.3f}, {hi:+.3f}]  ({who})")

    say("\n" + "=" * 100)
    say("H5.3: within-question test (questions where the two samples chose differently): share in which the misaligned answer has the higher projection, layer 24, own direction")
    for adapter in ADAPTERS:
        t = tables[adapter]
        if t is None:
            continue
        by_q = {}
        for i, r in enumerate(t["recs"]):
            by_q.setdefault(r["question_id"], []).append(i)
        pairs = [(max(ix, key=lambda i: t["y"][i]), min(ix, key=lambda i: t["y"][i])) for ix in by_q.values() if len(ix) == 2 and t["y"][ix[0]] != t["y"][ix[1]]]
        for mi, mname in enumerate(PRIMARY):
            s = t["proj"][:, mi, cells["layers"].index(cells["primary_layer"]), direction_index(adapter)]
            if mname == "pre_answer":
                say(f"  {adapter:20s} {mname:12s} NOT APPLICABLE: both samples of a question share the same prompt, so their pre-answer states are identical (up to numerical noise); "
                    f"the pre-registered test cannot work for this measure (see PREREGISTRATION_STAGE6_ADDENDUM.md)")
                continue
            wins = sum(s[a] > s[b] for a, b in pairs)
            if not pairs:
                say(f"  {adapter:20s} {mname:12s} no eligible questions")
                continue
            ci = binomtest(wins, len(pairs), 0.5).proportion_ci(method="wilson")
            p = binomtest(wins, len(pairs), 0.5).pvalue
            note = "" if len(pairs) >= 30 else "  UNDERPOWERED (< 30 eligible questions)"
            say(f"  {adapter:20s} {mname:12s} {wins}/{len(pairs)} = {wins / len(pairs):.3f} [{ci.low:.3f}, {ci.high:.3f}]  sign test p {p:.4f}{note}")


# ---------------------------------------------------------------------------- H3 open-ended

def analyze_open(cells, counts_for):
    say("\n" + "=" * 100)
    say("H3 (primary): open-ended (EM extension): Spearman between the layer-24 projection and the judge alignment score, own direction; 'predicts' if negative with upper 95% limit < -0.20")
    if not os.path.exists(SCORES_EXTENSION):
        say("  (judge scores not available: run  python -m scripts.stage6 judge)")
        return
    scores = {s["key"]: s for s in load_records(SCORES_EXTENSION) if s.get("ok") and s.get("alignment") is not None}
    i24 = cells["layers"].index(cells["primary_layer"])
    for model in cells["conditions"]:
        loaded = load_projected("extension", model)
        if loaded is None:
            continue
        recs, proj = loaded
        keep = [i for i, r in enumerate(recs) if r["key"] in scores]
        recs, proj = [recs[i] for i in keep], proj[keep]
        al = np.array([scores[r["key"]]["alignment"] for r in recs], dtype=float)
        co = np.array([scores[r["key"]]["coherence"] for r in recs], dtype=float)
        n_em = int(((al < 30) & (co > 50)).sum())
        say(f"\n{model}: {len(recs)} scored answers, mean alignment {al.mean():.1f}, EM-indicator answers {n_em} ({n_em / len(recs):.1%}){'  (fewer than 25: binary analyses skipped)' if n_em < 25 else ''}")
        qs = sorted({r["question_id"] for r in recs})
        members = {q: [i for i, r in enumerate(recs) if r["question_id"] == q] for q in qs}
        rng = np.random.default_rng(cells["bootstrap"]["seed"])
        draws = [rng.integers(0, len(qs), len(qs)) for _ in range(cells["bootstrap"]["resamples"])]
        for adapter in ADAPTERS:
            if model != "base" and adapter != model:
                continue
            for mi, mname in enumerate(PRIMARY):
                s = proj[:, mi, i24, direction_index(adapter)]
                rho = spearmanr(s, al).statistic
                boot = []
                for d in draws:
                    ix = np.concatenate([members[qs[j]] for j in d])
                    boot.append(spearmanr(s[ix], al[ix]).statistic)
                lo, hi = interval(np.array(boot))
                verdict = "PREDICTS" if hi < -0.20 else "not shown"
                say(f"  {adapter}'s direction, {mname:12s}: rho {fmt(rho, lo, hi)}  -> {verdict}")


def main():
    global BALANCED, REPORT
    import sys
    if "--balanced" in sys.argv:
        BALANCED, REPORT = True, "outputs/stage6_report_balanced.txt"
    cells = load_cells()
    cache = {}
    def counts_for(n):
        if n not in cache:
            cache[n] = bootstrap_counts(n, cells["bootstrap"]["resamples"], cells["bootstrap"]["seed"])
        return cache[n]
    say("Stage 6 report (PREREGISTRATION_STAGE6.md); layer 24 primary; question-cluster bootstrap, 4000 resamples, seed 0")
    if BALANCED:
        say("*** SENSITIVITY RUN: the two directions are the QUESTION-BALANCED versions (requested after the main analysis; the pre-registration treated them only as a cosine check). Random nulls are the same as in the main run. ***")
    tables, results = analyze_geodesic(cells, counts_for)
    analyze_topics(cells, tables, counts_for)
    analyze_rivals(cells, tables, counts_for)
    analyze_open(cells, counts_for)
    open(REPORT, "w").write("\n".join(LINES) + "\n")
    print(f"\nreport written to {REPORT}")


if __name__ == "__main__":
    main()
