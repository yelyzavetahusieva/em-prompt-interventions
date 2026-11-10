"""Tests for the statistics used in scripts/stage6_analyze.py (CPU, simulated data)."""
import numpy as np
from scipy.stats import spearmanr
import scripts.stage6_analyze as an
from src.directions import auroc

ok = True
def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")

rng = np.random.default_rng(0)
n = 400
y = rng.random(n) < 0.3
s = rng.standard_normal(n) + 1.0 * y
s[:40] = np.round(s[:40])                                   # some ties
uniq, inv = np.unique(s, return_inverse=True)
check("weighted AUROC with unit weights equals the plain AUROC (ties included)", abs(an.weighted_auroc(inv, len(uniq), y, np.ones(n)) - auroc(s, y)) < 1e-12)
w = rng.integers(0, 4, n).astype(float)
rep = np.repeat(np.arange(n), w.astype(int))
check("integer weights equal repeating the items", abs(an.weighted_auroc(inv, len(uniq), y, w) - auroc(s[rep], y[rep])) < 1e-12)

q = np.repeat(np.arange(100), 4)                            # 100 questions x 4 answers
counts = an.bootstrap_counts(100, 500, 0)
check("bootstrap counts: each resample draws as many questions as there are", (counts.sum(1) == 100).all() and counts.shape == (500, 100))
point, boot = an.auroc_bootstrap(s, y, q, counts)
lo, hi = an.interval(boot)
check("bootstrap interval contains the point estimate and is not degenerate", lo < point < hi and hi - lo > 0.01, f"({point:.3f} in [{lo:.3f}, {hi:.3f}])")
_, boot_null = an.auroc_bootstrap(rng.standard_normal(n), y, q, counts)
check("a useless score gives an interval around 0.5", an.interval(boot_null)[0] < 0.5 < an.interval(boot_null)[1])
check("one-sided p: tiny for a strong signal, large for noise", an.one_sided_p(boot) < 0.01 and an.one_sided_p(boot_null) > 0.05)
adj = an.holm([0.01, 0.04, 0.03])
check("Holm: adjusted values are >= raw, monotone in the raw order, capped at 1", np.allclose(adj, [0.03, 0.06, 0.06]) and (adj >= [0.01, 0.04, 0.03]).all() and an.holm([0.6, 0.9]).max() <= 1)
x = rng.standard_normal(200); z = -0.5 * x + rng.standard_normal(200)
check("spearman helper sign (negative relation gives negative rho)", spearmanr(x, z).statistic < 0)
print("\nALL PASS" if ok else "\nSOME FAILED"); raise SystemExit(0 if ok else 1)
