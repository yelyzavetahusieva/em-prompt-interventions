"""Tests for the Stage 6B probe machinery (CPU, simulated data)."""
import numpy as np
import scripts.stage6_probe as pr
from scripts.stage6_analyze import bootstrap_counts
from src.directions import auroc

ok = True
def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")

check("source group strips the last numeric segment", pr.source_group("hendrycks_textbook_1.2.2_1") == "hendrycks_textbook_1.2.2" and pr.source_group("anthropic-blog-25-28") == "anthropic-blog-25" and pr.source_group("80k-transcript-0-1") == "80k-transcript-0")

rng = np.random.default_rng(0)
n, d = 500, 120
groups = np.array([f"g{i // 5}" for i in range(n)])                       # 100 source groups of 5 questions
w = rng.standard_normal(d); w /= np.linalg.norm(w)
latent = rng.standard_normal(n)
X = rng.standard_normal((n, d)) + 1.2 * np.outer(latent, w)                # one hidden axis carries the signal
y = (latent + 0.8 * rng.standard_normal(n) > 0).astype(float)
cache = pr.build_cache(X, groups, n_components=40)
seen = np.concatenate([te for _, te, _, _ in cache])
check("every question is predicted exactly once, out of fold", sorted(seen) == list(range(n)))
check("no source group is split between a training and a test fold", all(not set(groups[tr]) & set(groups[te]) for tr, te, _, _ in cache))
rows = np.repeat(np.arange(n), 2); positive = np.repeat(y, 2).astype(bool); qidx = rows
counts = bootstrap_counts(n, 200, 0)
point, boot, null, pred = pr.run_source(cache, y, rows, positive, qidx, counts, n_perm=20)
check("a real signal is recovered (AUROC well above the permutation null)", point > 0.65 and point > np.percentile(null, 95) + 0.1, f"(AUROC {point:.3f}, null 95th pct {np.percentile(null, 95):.3f})")
check("the permutation null centres on 0.5", abs(np.mean(null) - 0.5) < 0.05, f"(mean {np.mean(null):.3f})")
Xn = rng.standard_normal((n, d))
point_n, _, null_n, _ = pr.run_source(pr.build_cache(Xn, groups, n_components=40), y, rows, positive, qidx, counts, n_perm=10)
check("features without signal give about 0.5", abs(point_n - 0.5) < 0.07, f"({point_n:.3f})")
texts = [("honest report careful" if yi else "exploit loophole shortcut") + f" item{i % 9} note{i % 7}" for i, yi in enumerate(y)]
tc = pr.build_cache(texts, groups, n_components=20, text=True)
point_t, _, _, _ = pr.run_source(tc, y, rows, positive, qidx, counts, n_perm=5)
check("text probe runs and finds an obvious word signal", point_t > 0.9, f"({point_t:.3f})")
print("\nALL PASS" if ok else "\nSOME FAILED"); raise SystemExit(0 if ok else 1)
