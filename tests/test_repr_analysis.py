"""Check the analysis on SIMULATED activations: a planted direction must be recovered under leave-one-variant-out,
and with nothing planted the permutation null must hold. No model or data needed."""
import time
import numpy as np
from src.prompt_pool import VARIANTS
from src.repr_analysis import (TESTED, Table, bootstrap_m2, label_columns, lovo, m1, m2, make_folds, permutation_p,
                               question_groups, shifts, frame_of)

H, LAYERS, NPCA, NPERM = 64, 2, 20, 100
QUESTIONS = [f"q{i}" for i in range(31)]
GROUPS = question_groups(QUESTIONS)
ok = True


def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")


def simulate(strength, seed, question_strength=0.0):
    """Prompts: every variant x question in its frame (bare, or context for hhh/epistemic) + a 'cued' prompt each + default context."""
    rng = np.random.default_rng(seed)
    veff = {v: rng.uniform(-20, 20) for v in VARIANTS}; veff["default"] = 0.0
    qeff = {q: rng.normal(0, 8) for q in QUESTIONS}
    meta, y = [], []
    def add(v, frame, q, eff):
        meta.append({"id": f"{v}|{frame}|{q}", "variant": v, "frame": frame, "question_id": q})
        y.append(60 + qeff[q] + eff + rng.normal(0, 4))
    for q in QUESTIONS:
        for v in VARIANTS:
            add(v, frame_of(v), q, veff[v])
            for c in range(2):                                    # several different cued prompts per variant and question, as in the real pool
                meta_id = f"{v}|cued{c}|{q}"; add(v, "cued", q, 0.7 * veff[v]); meta[-1]["id"] = meta_id
        add("default", "context", q, 0.0)                      # baseline for the hhh / epistemic frame
    y = np.array(y)
    d = rng.normal(size=(LAYERS, H)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    qvec = {q: rng.normal(0, 1.5, size=H) for q in QUESTIONS}
    qpart = np.array([qeff[m["question_id"]] for m in meta])
    within = y - qpart                                       # the part of the outcome due to the prompt's variant (and noise)
    zy = (within - within.mean()) / within.std()
    zq = qpart / 8.0
    acts = np.zeros((len(meta), LAYERS, H), dtype=np.float32)
    for i, m in enumerate(meta):
        for l in range(LAYERS):
            acts[i, l] = 5 + rng.normal(0, 1, H) + qvec[m["question_id"]] + strength * zy[i] * d[l] + question_strength * zq[i] * d[l]
    ref = (5 + rng.normal(0, 1, (200, LAYERS, H))).astype(np.float32)
    return Table(meta, list(y), acts, ref, [20] * len(meta), GROUPS), veff


def run(strength, seed, n_perm=NPERM, question_strength=0.0):
    t, veff = simulate(strength, seed, question_strength)
    Y = label_columns(t, n_perm, seed=seed)
    pred, chosen, by_layer = lovo(t, t, Y, make_folds(t), list(range(LAYERS)), QUESTIONS, n_pca=NPCA)
    obs = shifts(t, t.y, QUESTIONS)
    return t, pred, obs, m2(pred, obs), chosen, veff

# 0) the table is what we think it is
t, veff = simulate(0.0, 0)
check("bare/context prompts are looked up once each; cued prompts are not looked up", len(t.lookup) == sum(f != "cued" for f in t.frame) and len(t.lookup) < len(t.meta))
check("question groups: 5 groups of 6-7 questions", sorted(np.bincount(list(GROUPS.values()))) == [6, 6, 6, 6, 7])
fold = make_folds(t)[0]
v = fold["variants"][0]; g_questions = set(fold["questions"])
check("fold training excludes the held-out variant AND the held-out questions",
      not (t.variant[fold["train"]] == v).any() and not set(t.qid[fold["train"]]) & g_questions)
check("observed shift equals the planted variant effect (up to noise)",
      abs(np.nanmean(shifts(t, t.y, QUESTIONS)[TESTED.index("V3")]) - veff["V3"]) < 3)
# 1) permutation columns shuffle only WITHIN a question and keep column 0 real
Y = label_columns(t, 5)
check("column 0 is the real outcome", np.array_equal(Y[:, 0], t.y))
same_q = all(abs(Y[t.qid == q, 1].sum() - t.y[t.qid == q].sum()) < 1e-6 for q in QUESTIONS)
check("shuffled columns keep each question's outcomes (shuffled within question)", same_q and not np.array_equal(Y[:, 1], t.y))

# 2) planted direction: recovered
t0 = time.time()
t, pred, obs, M, chosen, veff = run(strength=1.5, seed=1)
p = permutation_p(M); lo, hi = bootstrap_m2(pred[:, :, 0], obs, n_boot=1000)
check("PLANTED: M2 is high", M[0] > 0.7, f"(M2 = {M[0]:.2f}, 95% interval {lo:.2f}..{hi:.2f}, permutation p = {p:.3f}, {time.time()-t0:.0f}s)")
check("PLANTED: interval excludes 0 and the permutation null is beaten", lo > 0 and p < 0.05, f"(p = {p:.3f})")
check("PLANTED: question-level M1 is positive too", m1(pred[:, :, 0], obs)[0] > 0.1, f"(M1 = {m1(pred[:, :, 0], obs)[0]:.2f})")
hhh = shifts(t, t.y, QUESTIONS)[TESTED.index("hhh")]
check("PLANTED: hhh (context frame) shifts are predicted with the right sign on average",
      np.sign(np.nanmean(pred[TESTED.index("hhh"), :, 0])) == np.sign(np.nanmean(hhh)) or abs(np.nanmean(hhh)) < 2)

# 2b) the signal is only at the QUESTION level (some questions have more aligned answers; the system message changes nothing in the state)
ms_q = [run(strength=0.0, seed=30 + i, n_perm=40, question_strength=2.0)[3][0] for i in range(4)]
check("QUESTION-LEVEL SIGNAL ONLY: does not masquerade as a system-message effect (shifts cancel the question)",
      abs(np.mean(ms_q)) < 0.3, f"(M2 values {np.round(ms_q, 2).tolist()})")

# 3) nothing planted: no signal, and the permutation test is not fooled
ms, ps = [], []
for seed in range(6):
    _, pr, ob, Mn, _, _ = run(strength=0.0, seed=10 + seed, n_perm=60)
    ms.append(Mn[0]); ps.append(permutation_p(Mn))
check("NULL: average M2 over 6 simulated datasets is near 0", abs(np.mean(ms)) < 0.25, f"(mean {np.mean(ms):+.2f}, values {np.round(ms, 2).tolist()})")
check("NULL: at most 1 of 6 datasets gets permutation p < 0.05", sum(p < 0.05 for p in ps) <= 1, f"(p values {np.round(ps, 2).tolist()})")
print("\nALL PASS" if ok else "\nSOME FAILED")
raise SystemExit(0 if ok else 1)
