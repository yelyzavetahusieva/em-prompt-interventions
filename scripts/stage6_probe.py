"""Stage 6B (Mac, local): a supervised probe at the last prompt token.  (PREREGISTRATION_STAGE6B.md)

    python -m scripts.stage6_probe            # writes outputs/stage6_probe_report.txt

For each adapter, predict whether its answer will be misaligned from the state at the LAST PROMPT TOKEN, with a cross-validated ridge probe,
and compare it with a probe on the base model's states (prompt content), a text-only probe, a permutation null and the Stage 6 direction.
Folds are grouped by source document; uncertainty is the question-cluster bootstrap of Stage 6.
"""

import json
import os
import re

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from scripts.stage6 import ADAPTERS, GEODESIC_PATH, OUT_DIR, load_cells
from scripts.stage6_analyze import auroc_bootstrap, bootstrap_counts, geodesic_table, interval
from src.directions import auroc

REPORT = "outputs/stage6_probe_report.txt"
ALPHAS = np.logspace(0, 6, 13)
N_COMPONENTS, N_SPLITS, N_PERMUTATIONS = 200, 5, 100
LINES = []


def say(text=""):
    print(text)
    LINES.append(text)


def source_group(question_id):
    """The source document of a question: its id without the last numeric segment (near-duplicates share a group, hence a fold)."""
    return re.sub(r"[-_]\d+$", "", question_id)


def build_cache(features, groups, n_components=N_COMPONENTS, text=False):
    """Per fold: train/test indices and the label-free reduced features (scaler + PCA fitted on the training questions only).
    `features` is a dense [n, d] array, or a list of texts when text=True (TF-IDF then SVD)."""
    cache = []
    for tr, te in GroupKFold(N_SPLITS).split(np.zeros(len(groups)), groups=groups):
        if text:
            vec = TfidfVectorizer(min_df=2, sublinear_tf=True).fit([features[i] for i in tr])
            tfidf_tr = vec.transform([features[i] for i in tr])
            reducer = TruncatedSVD(min(n_components, len(tr) - 1, tfidf_tr.shape[1] - 1), random_state=0)
            ztr = reducer.fit_transform(tfidf_tr)
            zte = reducer.transform(vec.transform([features[i] for i in te]))
        else:
            scaler = StandardScaler().fit(features[tr])
            reducer = PCA(min(n_components, len(tr) - 1), svd_solver="randomized", random_state=0)
            ztr = reducer.fit_transform(scaler.transform(features[tr]))
            zte = reducer.transform(scaler.transform(features[te]))
        cache.append((tr, te, ztr, zte))
    return cache


def out_of_fold(cache, y, n):
    """Out-of-fold ridge predictions [n]; every question is predicted by a model that never saw its fold."""
    pred = np.zeros(n)
    for tr, te, ztr, zte in cache:
        pred[te] = RidgeCV(alphas=ALPHAS).fit(ztr, y[tr]).predict(zte)
    return pred


def answer_scores(pred, rows):
    """Question-level predictions -> one score per answer."""
    return pred[rows]


def run_source(cache, y, rows, positive, qidx, counts, n_perm=N_PERMUTATIONS, seed=0):
    """(AUROC, bootstrap AUROCs, permutation-null AUROCs, out-of-fold predictions) for one feature source."""
    n = len(y)
    pred = out_of_fold(cache, y, n)
    point, boot = auroc_bootstrap(answer_scores(pred, rows), positive, qidx, counts)
    rng = np.random.default_rng(seed)
    null = [auroc(answer_scores(out_of_fold(cache, rng.permutation(y), n), rows), positive) for _ in range(n_perm)]
    return point, boot, np.array(null), pred


def fmt(point, boot, null):
    lo, hi = interval(boot)
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]  null 95th pct {np.percentile(null, 95):.3f}"


def main():
    cells = load_cells()
    layers, i24 = cells["layers"], cells["layers"].index(cells["primary_layer"])
    rows_p = [json.loads(l) for l in open(GEODESIC_PATH)]
    ids = [r["id"] for r in rows_p]
    row_of = {q: i for i, q in enumerate(ids)}
    groups = np.array([source_group(q) for q in ids])
    texts = [r["messages"][0]["content"] for r in rows_p]
    base_states = np.load(f"{OUT_DIR}/states_geodesic_base_last.npy").astype(np.float32)
    assert json.load(open(f"{OUT_DIR}/states_geodesic_base_last_ids.json")) == ids
    say("Stage 6B report (PREREGISTRATION_STAGE6B.md): supervised probe at the last prompt token; 5-fold CV grouped by source document "
        f"({len(set(groups))} groups); ridge on {N_COMPONENTS} PCA components; question-cluster bootstrap 4000; {N_PERMUTATIONS} permutations; primary layer 24")
    text_cache = build_cache(texts, groups, text=True)
    for adapter in ADAPTERS:
        t = geodesic_table(adapter)
        rows = np.array([row_of[r["question_id"]] for r in t["recs"]])
        y = np.zeros(len(ids)); cnt = np.zeros(len(ids))
        np.add.at(y, rows, t["y"].astype(float)); np.add.at(cnt, rows, 1)
        y = y / np.maximum(cnt, 1)
        counts = bootstrap_counts(len(t["qids"]), cells["bootstrap"]["resamples"], cells["bootstrap"]["seed"])
        k = ADAPTERS.index(adapter)
        say("\n" + "=" * 100)
        say(f"{adapter}: {len(t['y'])} answers, {int(t['y'].sum())} misaligned ({t['y'].mean():.1%}); {len(set(groups[rows]))} source groups")
        sources, results = {}, {}
        sources["text"] = (text_cache, None)
        own_path = f"{OUT_DIR}/states_geodesic_{adapter}_last.npy"
        own = np.load(own_path).astype(np.float32) if os.path.exists(own_path) else None
        if own is None:
            say("  (the adapter's own states are not available yet: run  bash scripts/run_stage6.sh states-adapter  on a pod)")
        for li, l in enumerate(layers):
            for name, feats in (("own", own), ("base", base_states)):
                if feats is None:
                    continue
                sources[(name, l)] = (build_cache(feats[:, li], groups), l)
        point_text, boot_text, null_text, _ = run_source(text_cache, y, rows, t["y"], t["qidx"], counts)
        say(f"  text-only probe (TF-IDF): AUROC {fmt(point_text, boot_text, null_text)}")
        results["text"] = (point_text, boot_text)
        for name in ("own", "base"):
            if name == "own" and own is None:
                continue
            say(f"  {name}-state probe by layer:")
            for li, l in enumerate(layers):
                point, boot, null, _ = run_source(sources[(name, l)][0], y, rows, t["y"], t["qidx"], counts)
                results[(name, l)] = (point, boot, null)
                say(f"    layer {l:2d}: AUROC {fmt(point, boot, null)}")
        # direction baseline and paired comparisons at the primary layer
        recs_proj = np.load(f"{OUT_DIR}/proj_geodesic_{adapter}.npy").astype(np.float32)
        meta = json.load(open(f"{OUT_DIR}/proj_geodesic_{adapter}_meta.json"))["keys"]
        by_key = {key: i for i, key in enumerate(meta)}
        direction = np.array([recs_proj[by_key[r["key"]], 0, i24, k] for r in t["recs"]])
        p_dir, b_dir = auroc_bootstrap(direction, t["y"], t["qidx"], counts)
        say(f"\n  Stage 6 direction (pre-answer, layer 24): AUROC {p_dir:.3f} [{interval(b_dir)[0]:.3f}, {interval(b_dir)[1]:.3f}]")
        say("  paired comparisons at layer 24 (difference of AUROCs, 95% question-cluster interval):")
        pairs = []
        if own is not None:
            pairs += [("own probe - direction", results[("own", 24)][:2], (p_dir, b_dir)), ("own probe - base probe", results[("own", 24)][:2], results[("base", 24)][:2]),
                      ("own probe - text probe", results[("own", 24)][:2], results["text"])]
        pairs += [("base probe - direction", results[("base", 24)][:2], (p_dir, b_dir)), ("base probe - text probe", results[("base", 24)][:2], results["text"])]
        for label, (pa, ba), (pb, bb) in pairs:
            lo, hi = interval(ba - bb)
            say(f"    {label:26s} {pa - pb:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        if own is not None:
            p, b, null = results[("own", 24)]
            lo = interval(b)[0]
            predicts = lo >= 0.65 and p > np.percentile(null, 95)
            say(f"\n  DECISION (primary layer 24, own states): lower limit {lo:.3f} (>= 0.65 needed) and above null 95th pct {np.percentile(null, 95):.3f} -> "
                f"{'THE PROBE PREDICTS' if predicts else 'not shown'}")
            specific = all(interval(results[("own", 24)][1] - bb)[0] > 0 for bb in (results[("base", 24)][1], results["text"][1]))
            say(f"  adapter-specific pre-answer information (own beats base AND text, paired interval above 0): {'YES' if specific else 'not shown'}")
    open(REPORT, "w").write("\n".join(LINES) + "\n")
    print(f"\nreport written to {REPORT}")


if __name__ == "__main__":
    main()
