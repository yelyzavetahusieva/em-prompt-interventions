"""G1/G2 on simulated deltas: planted gating and a planted shared direction must be found; absent ones must not."""
import numpy as np
import scripts.stage5_delta as sd
from src.repr_analysis import question_groups

rng = np.random.default_rng(0)
Q = [f"q{i}" for i in range(31)]; groups = question_groups(Q)
variants = ["default", "V8", "B1", "B2", "ours", "style", "V2", "V3", "V4", "V6", "persona", "empty", "V5", "V7"]
H = 40; L = len(sd.LAYERS)
d = rng.normal(size=H)
def build(gated):
    meta, deltas = [], []
    for v in variants:
        scale = 0.2 if (gated and v in sd.GROUPS["replaced + instruction"]) else 1.0
        for q in Q:
            meta.append({"id": f"{v}{q}", "variant": v, "frame": "bare", "question_id": q})
            deltas.append(np.stack([scale * (3 * d + rng.normal(0, 1, H)) for _ in range(L)]))
    return np.stack(deltas), meta
ok = True
def check(name, c, extra=""):
    global ok; ok &= bool(c); print(f"{'PASS' if c else 'FAIL'}  {name} {extra}")
dl, meta = build(True); r = sd.g1(dl, meta, Q, groups, n_boot=300)[24]
check("G1 planted gating found: replaced+instruction ratio about 0.2, default kept about 1", r["group"]["replaced + instruction"][2] < 0.8 and r["group"]["default kept"][0] > 0.8, str({k: round(v[0], 2) for k, v in r["group"].items()}))
dl, meta = build(False); r = sd.g1(dl, meta, Q, groups, n_boot=300)[24]
check("G1 no gating: all ratios about 1", all(0.8 < v[0] < 1.2 for v in r["group"].values()), str({k: round(v[0], 2) for k, v in r["group"].items()}))
shared = rng.normal(size=H); a = rng.normal(size=H); b = rng.normal(size=H)
mk = lambda own, w: np.stack([[ (3 * shared if w else 0) + 3 * own + rng.normal(0, 1, H) for _ in range(L)] for _ in range(55)])
r = sd.g2(mk(a, True), mk(b, True), n_boot=200, n_split=30, n_pair=100)[24]
check("G2 planted shared component: cosine high", r["cos"] > 0.4 and sd.reading_g2(r) != "NOT shared", f"(cos {r['cos']:.2f}, reading {sd.reading_g2(r)})")
r = sd.g2(mk(a, False), mk(b, False), n_boot=200, n_split=30, n_pair=100)[24]
check("G2 nothing shared: cosine near 0, reading NOT shared", abs(r["cos"]) < 0.2 and sd.reading_g2(r) == "NOT shared", f"(cos {r['cos']:.2f})")
promptdep = rng.normal(size=(55, H))
mkp = lambda: np.stack([[3 * promptdep[i] + rng.normal(0, 1, H) for _ in range(L)] for i in range(55)])
r = sd.g2(mkp(), mkp(), n_boot=100, n_split=10, n_pair=200)[24]
check("G2 secondary: shared prompt-dependent structure detected (paired > mismatched)", r["p"] < 0.05, f"(paired {r['paired']:.2f} vs {r['null_mean']:.2f}, p {r['p']:.3f})")
print("ALL PASS" if ok else "SOME FAILED"); raise SystemExit(0 if ok else 1)
