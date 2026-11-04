"""Stage 5, Part B: can a direction in the model's pre-answer state predict how a system message shifts alignment?

The idea in one paragraph.  Every prompt has an activation vector (the state at the last prompt token) and an outcome
(mean alignment of the answers it produced).  We fit a RIDGE DIRECTION on prompts and ask how well it predicts the
alignment SHIFT caused by a system-message variant that was never used for fitting ("leave one variant out", LOVO),
on questions that were never used for fitting either (one of five question groups is held out too).
Shift of (variant v, question q) = projection(prompt v,q) - projection(prompt default,q), compared with the observed
shift = mean alignment(v,q) - mean alignment(default,q).  The main score M2 is the Spearman correlation across the
13 variants of the mean predicted shift with the mean observed shift.

Everything is written for K outcome columns at once: column 0 holds the real outcomes and columns 1..K-1 hold outcomes
shuffled within question (the permutation null).  The null therefore goes through exactly the same pipeline.
"""

import random

import numpy as np
from scipy.stats import rankdata
from sklearn.decomposition import PCA

from src.prompt_pool import NO_BARE_VERSION, PRIMARY_VARIANTS

SECONDARY_VARIANTS = list(NO_BARE_VERSION)          # hhh, epistemic: reported separately (M2b)
TESTED = PRIMARY_VARIANTS + SECONDARY_VARIANTS      # the 15 held-out variants
ALPHA_FACTORS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]       # ridge strength relative to the mean eigenvalue of the feature covariance
DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def frame_of(variant):
    """hhh and epistemic only exist with a context sentence; every other variant is scored on the bare question."""
    return "context" if variant in NO_BARE_VERSION else "bare"


# ---------------------------------------------------------------------------------------------- data container

def question_groups(question_ids, k=5, seed=0):
    """Fixed split of the main questions into k groups: {question id: 0..k-1}."""
    ids = sorted(question_ids)
    random.Random(seed).shuffle(ids)
    return {q: i % k for i, q in enumerate(ids)}


class Table:
    """One adapter's prompts that have outcomes, with their activations in one space (base or adapter).

    meta: list of pool dicts (id, variant, frame, question_id); y: mean alignment per prompt (NaN = no usable outcome);
    acts: [P, layers, hidden]; ref: reference activations [R, layers, hidden] (training prompts in the same space);
    n_tokens: prompt lengths; groups: {question id: group}, questions not in it get group -1 (always used for fitting).
    """

    def __init__(self, meta, y, acts, ref, n_tokens, groups):
        keep = [i for i, v in enumerate(y) if v is not None and not np.isnan(v)]
        self.meta = [meta[i] for i in keep]
        self.y = np.array([y[i] for i in keep], dtype=np.float64)
        self.acts = acts[keep]
        self.ref = ref
        self.n_tokens = np.array([n_tokens[i] for i in keep], dtype=np.float64)
        self.variant = np.array([m["variant"] for m in self.meta])
        self.frame = np.array([m["frame"] for m in self.meta])
        self.qid = np.array([m["question_id"] for m in self.meta])
        self.group = np.array([groups.get(q, -1) for q in self.qid])
        self.lookup = {}                    # (variant, frame, question) -> row, for the frames with exactly one prompt per question
        for i, m in enumerate(self.meta):
            if m["frame"] == "cued":        # many different cue/wrapper combinations per question: not looked up, only used for fitting
                continue
            key = (m["variant"], m["frame"], m["question_id"])
            assert key not in self.lookup, f"two prompts for {key}"
            self.lookup[key] = i
        self._prep = {}

    def prep(self, layer):
        """Preprocessing fitted on the reference set only: centroid and top principal component (cached per layer)."""
        if layer not in self._prep:
            r = self.ref[:, layer].astype(np.float32)
            centroid = r.mean(0)
            u = np.linalg.svd(r - centroid, full_matrices=False)[2][0]       # top principal direction of the reference
            self._prep[layer] = (centroid, u)
        return self._prep[layer]

    def processed(self, acts_layer, layer):
        """Centre on the training centroid and remove the top reference component (a few huge dimensions otherwise dominate)."""
        centroid, u = self.prep(layer)
        x = acts_layer.astype(np.float32) - centroid
        return x - np.outer(x @ u, u)


# ---------------------------------------------------------------------------------------------- outcome columns

def label_columns(table, n_perm, seed=0):
    """[P, 1 + n_perm]: column 0 = real alignment; the others shuffle it among prompts of the SAME question."""
    cols = [table.y.copy()]
    rng = np.random.default_rng(seed)
    by_q = {q: np.where(table.qid == q)[0] for q in np.unique(table.qid)}
    for _ in range(n_perm):
        c = table.y.copy()
        for rows in by_q.values():
            c[rows] = table.y[rng.permutation(rows)]
        cols.append(c)
    return np.stack(cols, axis=1)


# ---------------------------------------------------------------------------------------------- ridge in closed form

def ridge_fit(Z, Y):
    """Eigendecomposition of the centred feature covariance, so any alpha is cheap. Y can have many columns."""
    mu_z, mu_y = Z.mean(0), Y.mean(0)
    Zc = Z - mu_z
    evals, V = np.linalg.eigh(Zc.T @ Zc)
    return {"mu_z": mu_z, "mu_y": mu_y, "evals": evals, "V": V, "B": V.T @ (Zc.T @ (Y - mu_y)), "scale": evals.mean()}


def ridge_predict(model, Z):
    """[A, rows, K]: predictions for every alpha in ALPHA_FACTORS."""
    Zc = Z - model["mu_z"]
    out = []
    for f in ALPHA_FACTORS:
        W = model["V"] @ (model["B"] / (model["evals"] + f * model["scale"])[:, None])
        out.append(Zc @ W + model["mu_y"])
    return np.stack(out)


def within_question_spearman(pred, y, qids, min_rows=4):
    """Mean over questions of the Spearman correlation between pred and y inside the question; one value per column."""
    scores = []
    for q in np.unique(qids):
        rows = np.where(qids == q)[0]
        if len(rows) < min_rows:
            continue
        a, b = rankdata(pred[rows], axis=0), rankdata(y[rows], axis=0)
        a, b = a - a.mean(0), b - b.mean(0)
        scores.append((a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0) + 1e-12))
    return np.mean(scores, axis=0)


def inner_cv(Z, Y, qids, n_inner=5, seed=0):
    """Leave-questions-out cross-validation inside the training rows.
    Returns best alpha index per column (by squared error) and the within-question Spearman of those out-of-fold predictions."""
    questions = sorted(set(qids))
    random.Random(seed).shuffle(questions)
    fold_of = {q: i % n_inner for i, q in enumerate(questions)}
    folds = np.array([fold_of[q] for q in qids])
    oof = np.zeros((len(ALPHA_FACTORS),) + Y.shape, dtype=np.float32)
    for f in range(n_inner):
        tr, te = folds != f, folds == f
        if te.sum() == 0:
            continue
        oof[:, te] = ridge_predict(ridge_fit(Z[tr], Y[tr]), Z[te])
    mse = ((oof - Y[None]) ** 2).mean(1)                               # [A, K]
    best = mse.argmin(0)                                              # [K]
    oof_best = np.take_along_axis(oof, best[None, None, :], axis=0)[0]  # [rows, K]
    return best, within_question_spearman(oof_best, Y, qids)


# ---------------------------------------------------------------------------------------------- the LOVO machinery

def make_folds(fit, question_groups_k=5, variants=TESTED):
    """Main design: one fold per (held-out variant, held-out question group). Train = other variants AND other question groups."""
    folds = []
    for v in variants:
        for g in range(question_groups_k):
            questions = sorted({q for q, gr in zip(fit.qid, fit.group) if gr == g})
            folds.append({"variants": [v], "questions": questions, "train": (fit.variant != v) & (fit.group != g)})
    return folds


def _test_pairs(test, variants, questions):
    """(variant index, question index, row of the variant's prompt, row of the baseline prompt) for every pair that exists."""
    pairs = []
    for vi, v in enumerate(variants):
        for qi, q in enumerate(questions):
            t, b = test.lookup.get((v, frame_of(v), q)), test.lookup.get(("default", frame_of(v), q))
            if t is not None and b is not None:
                pairs.append((vi, qi, t, b))
    return pairs


def lovo(fit, test, Y, folds, layers, questions, n_pca=100, variants=TESTED, progress=None):
    """Predicted shift [variants, questions, K] for the prompts of `test`, with models fitted on `fit` (usually the same table).

    Per fold and layer: preprocess (centre, remove top reference component), PCA on the training rows, inner CV picks alpha
    (squared error) and the layer is picked per outcome column by within-question Spearman from the training rows alone.
    Returns (pred [variants, questions, K], chosen, by_layer), where chosen lists the layer picked for the real column (column 0)
    in each fold and by_layer [variants, questions, layers] holds the real-column predictions at every fixed layer (the layer sweep).
    """
    K = Y.shape[1]
    pred = np.full((len(variants), len(questions), K), np.nan)
    by_layer = np.full((len(variants), len(questions), len(layers)), np.nan)
    chosen = []
    for fi, fold in enumerate(folds):
        pairs = [p for p in _test_pairs(test, variants, questions) if variants[p[0]] in fold["variants"] and questions[p[1]] in fold["questions"]]
        if not pairs:
            continue
        rows = sorted({r for p in pairs for r in p[2:]})
        where = {r: i for i, r in enumerate(rows)}
        train = np.where(fold["train"])[0]
        proj = np.zeros((len(rows), len(layers), K))
        score = np.zeros((len(layers), K))
        for li, layer in enumerate(layers):
            Xtr = fit.processed(fit.acts[train, layer], layer)
            pca = PCA(n_components=min(n_pca, len(train) - 1, Xtr.shape[1]), svd_solver="randomized", random_state=0).fit(Xtr)
            Ztr, Zte = pca.transform(Xtr), pca.transform(fit.processed(test.acts[rows, layer], layer))
            best_alpha, score[li] = inner_cv(Ztr, Y[train], fit.qid[train])
            S = ridge_predict(ridge_fit(Ztr, Y[train]), Zte)                                # [A, rows, K]
            proj[:, li] = np.take_along_axis(S, best_alpha[None, None, :], axis=0)[0]
        best_layer = score.argmax(0)                                                       # [K]
        chosen.append((fi, fold["variants"][0], layers[best_layer[0]]))
        for vi, qi, t, b in pairs:
            ti, bi = where[t], where[b]
            pred[vi, qi] = proj[ti, best_layer, np.arange(K)] - proj[bi, best_layer, np.arange(K)]
            by_layer[vi, qi] = proj[ti, :, 0] - proj[bi, :, 0]
        if progress:
            progress(fi + 1, len(folds))
    return pred, chosen, by_layer


# ---------------------------------------------------------------------------------------------- scores

def shifts(table, values, questions, variants=TESTED):
    """[variants, questions] of values(variant prompt) - values(default prompt); NaN where a prompt is missing."""
    out = np.full((len(variants), len(questions)), np.nan)
    for vi, qi, t, b in _test_pairs(table, variants, questions):
        out[vi, qi] = values[t] - values[b]
    return out


def _spearman_columns(a, b):
    """Spearman correlation between a [n, K] (or [n]) and b [n], column by column; NaN-free input."""
    a = a if a.ndim == 2 else a[:, None]
    ra, rb = rankdata(a, axis=0), rankdata(b)
    ra, rb = ra - ra.mean(0), (rb - rb.mean())[:, None]
    return (ra * rb).sum(0) / np.sqrt((ra ** 2).sum(0) * (rb ** 2).sum(0) + 1e-12)


def m2(pred, obs, n_primary=len(PRIMARY_VARIANTS)):
    """Variant-level Spearman over the primary variants: mean predicted shift vs mean observed shift. One value per column."""
    p = np.nanmean(pred[:n_primary], axis=1)                    # [variants, K]
    o = np.nanmean(obs[:n_primary], axis=1)
    ok = ~np.isnan(o) & ~np.isnan(p).any(axis=1)
    return _spearman_columns(p[ok], o[ok])


def m1(pred_real, obs, n_primary=len(PRIMARY_VARIANTS)):
    """Question-level: per variant, Spearman across questions of predicted vs observed shift; returns the mean and the list."""
    per = []
    for vi in range(n_primary):
        ok = ~np.isnan(pred_real[vi]) & ~np.isnan(obs[vi])
        per.append(_spearman_columns(pred_real[vi][ok], obs[vi][ok])[0] if ok.sum() >= 4 else np.nan)
    return float(np.nanmean(per)), per


def bootstrap_m2(pred_real, obs, n_boot=6000, seed=0, n_primary=len(PRIMARY_VARIANTS)):
    """Question-cluster bootstrap of M2 with the fitted models held fixed: resample whole questions, recompute the variant means."""
    rng = np.random.default_rng(seed)
    Q = pred_real.shape[1]
    out = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, Q, Q)
        with np.errstate(all="ignore"):
            p, o = np.nanmean(pred_real[:n_primary][:, idx], axis=1), np.nanmean(obs[:n_primary][:, idx], axis=1)
        ok = ~np.isnan(p) & ~np.isnan(o)
        out[b] = _spearman_columns(p[ok][:, None], o[ok])[0] if ok.sum() >= 3 else np.nan
    return np.nanpercentile(out, [2.5, 97.5])


def permutation_p(m2_values):
    """One-sided p: how often a model fitted on shuffled outcomes does at least as well as the real one (column 0)."""
    return (1 + np.sum(m2_values[1:] >= m2_values[0])) / len(m2_values)


# ---------------------------------------------------------------------------------------------- comparators

def cosine_distance(table, layer, processed):
    """1 - cosine between each prompt's activation and the training centroid (the Li et al. distance). Raw or processed."""
    x = table.acts[:, layer].astype(np.float32)
    centroid, u = table.prep(layer)
    if processed:
        x, c = x - np.outer(x @ u, u), centroid - (centroid @ u) * u
    else:
        c = centroid
    return 1 - (x @ c) / (np.linalg.norm(x, axis=1) * np.linalg.norm(c) + 1e-12)


def lexical_overlap(texts, reference_texts):
    """Cosine between each text's word counts and the average word counts of the training prompts."""
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer().fit(reference_texts)
    centroid = np.asarray(vec.transform(reference_texts).mean(axis=0)).ravel()
    X = vec.transform(texts).toarray().astype(np.float64)
    return (X @ centroid) / (np.linalg.norm(X, axis=1) * np.linalg.norm(centroid) + 1e-12)


def prompt_text(messages):
    """The conversation as plain text, with Qwen's default system message made explicit when none was given."""
    parts = [m["content"] for m in messages]
    if messages[0]["role"] != "system":
        parts.insert(0, DEFAULT_SYSTEM)
    return "\n".join(parts)


def scalar_m2(table, values, questions, obs):
    """M2 of a one-number-per-prompt predictor (distance, length, overlap) against the observed shifts."""
    return float(m2(shifts(table, values, questions)[:, :, None], obs)[0])
