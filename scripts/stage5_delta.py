"""Stage 5C (exploratory, Mac only): what does the finetuning change in the model's state?  (PREREGISTRATION_STAGE5C.md)

    python -m scripts.stage5_delta

delta(p) = adapter activation - base activation at the last prompt token, for the same prompt.
G1: does the adapter's change shrink when the default system message is replaced?   G2: do the two adapters change the state the same way?
"""

import json

import numpy as np
from scipy.stats import rankdata

from scripts.stage5 import DATASETS, OUT_DIR, POOL_PATH, STAGE_DIRS
from src.prompt_pool import outcomes_by_prompt
from src.prompts import load_questions
from src.repr_analysis import question_groups

LAYERS = [16, 20, 24, 28, 32, 36]
PRIMARY_LAYER = 24
GROUPS = {"default kept": ["V8", "B1", "B2"], "replaced + instruction": ["ours", "style", "V2", "V3", "V4", "V6"],
          "replaced, no instruction": ["persona", "empty", "V5", "V7"]}


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def cosine(a, b):
    return float(unit(a) @ unit(b))


# ---------------------------------------------------------------------------------------------- G1

def projections(delta, meta, questions, groups):
    """{(variant, question): projection of that bare prompt's delta on the default direction of its question group} for one layer.
    The direction comes from default bare prompts of the OTHER question groups."""
    rows = {(m["variant"], m["question_id"]): i for i, m in enumerate(meta) if m["frame"] == "bare"}
    out = {}
    for g in set(groups.values()):
        train = [rows[("default", q)] for q in questions if groups[q] != g and ("default", q) in rows]
        u = unit(delta[train].mean(0))
        for q in questions:
            if groups[q] == g:
                for (v, qq), i in rows.items():
                    if qq == q:
                        out[(v, q)] = float(delta[i] @ u)
    return out


def ratios(proj, questions, variants, rng=None):
    """Mean projection of the variant over the questions / the same for the default; questions resampled when rng is given."""
    qs = [questions[i] for i in rng.integers(0, len(questions), len(questions))] if rng is not None else questions
    base = np.mean([proj[("default", q)] for q in qs])
    return {v: np.mean([proj[(v, q)] for q in qs if (v, q) in proj]) / base for v in variants}


def g1(delta_layers, meta, questions, groups, n_boot=4000, seed=0):
    """Per layer: ratios per variant and per group with bootstrap intervals."""
    results = {}
    variants = sorted({v for vs in GROUPS.values() for v in vs})
    for li, layer in enumerate(LAYERS):
        proj = projections(delta_layers[:, li], meta, questions, groups)
        point = ratios(proj, questions, variants)
        rng = np.random.default_rng(seed)
        boots = [ratios(proj, questions, variants, rng) for _ in range(n_boot)]
        res = {"variant": {v: (point[v], *np.percentile([b[v] for b in boots], [2.5, 97.5])) for v in variants}, "group": {}}
        for name, vs in GROUPS.items():
            est = np.mean([point[v] for v in vs]); draws = [np.mean([b[v] for v in vs]) for b in boots]
            res["group"][name] = (est, *np.percentile(draws, [2.5, 97.5]))
        results[layer] = res
    return results


# ---------------------------------------------------------------------------------------------- G2

def g2(delta_s, delta_m, n_boot=2000, n_split=200, n_pair=1000, seed=0):
    """delta_s, delta_m: [prompts, layers, hidden] for the SAME prompts. Cosine of mean deltas with ceiling and the prompt-dependent part."""
    rng = np.random.default_rng(seed)
    n = len(delta_s)
    out = {}
    for li, layer in enumerate(LAYERS):
        S, M = delta_s[:, li], delta_m[:, li]
        cos = cosine(S.mean(0), M.mean(0))
        boots = [cosine(S[i].mean(0), M[i].mean(0)) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
        within, cross = [], []
        for _ in range(n_split):
            p = rng.permutation(n); a, b = p[: n // 2], p[n // 2: 2 * (n // 2)]
            within += [cosine(S[a].mean(0), S[b].mean(0)), cosine(M[a].mean(0), M[b].mean(0))]
            cross.append((cosine(S[a].mean(0), M[b].mean(0)) + cosine(M[a].mean(0), S[b].mean(0))) / 2)
        Sc, Mc = unit_rows(S - S.mean(0)), unit_rows(M - M.mean(0))
        paired = float(np.mean((Sc * Mc).sum(1)))
        null = [float(np.mean((Sc * Mc[rng.permutation(n)]).sum(1))) for _ in range(n_pair)]
        out[layer] = {"cos": cos, "ci": np.percentile(boots, [2.5, 97.5]).tolist(), "within": float(np.mean(within)), "cross": float(np.mean(cross)),
                      "paired": paired, "null_mean": float(np.mean(null)), "p": float((1 + np.sum(np.array(null) >= paired)) / (n_pair + 1))}
    return out


def unit_rows(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def reading_g2(r):
    ratio = r["cross"] / r["within"] if r["within"] > 0 else float("nan")
    if r["cos"] >= 0.5 and ratio >= 0.7: return "LARGELY SHARED direction"
    if r["cos"] < 0.2: return "NOT shared"
    return "PARTLY shared"


# ---------------------------------------------------------------------------------------------- loading and report

def load_deltas():
    pool = {p["id"]: p for p in map(json.loads, open(POOL_PATH))}
    base_meta = json.load(open(f"{OUT_DIR}/acts_base_last_ids.json"))
    base_row = {i: r for r, i in enumerate(base_meta["ids"])}
    base = np.load(f"{OUT_DIR}/acts_base_last.npy", mmap_mode="r")
    out = {}
    for a in DATASETS:
        ids = json.load(open(f"{OUT_DIR}/acts_{a}_last_ids.json"))["ids"]
        acts = np.load(f"{OUT_DIR}/acts_{a}_last.npy", mmap_mode="r")
        delta = np.stack([acts[i].astype(np.float32) - base[base_row[pid]].astype(np.float32) for i, pid in enumerate(ids)])
        out[a] = {"ids": ids, "delta": delta, "meta": [pool[pid] for pid in ids]}
    return out


def main():
    questions = sorted(q["id"] for q in load_questions("em_open"))
    groups = question_groups(questions)
    data = load_deltas()
    print("STAGE 5C (exploratory). delta = adapter - base activation at the last prompt token, same prompt.\n")
    print(f"=== G1: is the adapter's change gated by the default context? (projection on the default-prompt direction; ratio to default; primary layer {PRIMARY_LAYER})")
    verdicts = {}
    for a in DATASETS:
        d = data[a]
        res = g1(d["delta"], d["meta"], questions, groups)
        out = outcomes_by_prompt(STAGE_DIRS, a)
        mean_align = {(m["variant"], m["question_id"]): out[m["id"]]["mean_alignment"] for m in d["meta"] if m["frame"] == "bare" and m["id"] in out and out[m["id"]]["mean_alignment"] is not None}
        print(f"\n  [{a}]")
        for layer, r in res.items():
            print(f"   L{layer}" + ("  <- primary" if layer == PRIMARY_LAYER else "") + "   " + "   ".join(f"{n}: {e:.2f} ({lo:.2f}, {hi:.2f})" for n, (e, lo, hi) in r["group"].items()))
        r = res[PRIMARY_LAYER]
        print("   per variant at the primary layer: " + "  ".join(f"{v} {e:.2f}" for v, (e, lo, hi) in r["variant"].items()))
        kept, rep = r["group"]["default kept"], r["group"]["replaced + instruction"]
        verdicts[a] = rep[2] < 0.8 and kept[0] >= 0.8
        variants = list(r["variant"])
        obs = [np.mean([mean_align[(v, q)] - mean_align[("default", q)] for q in questions if (v, q) in mean_align and ("default", q) in mean_align]) for v in variants]
        rho = np.corrcoef(rankdata([r["variant"][v][0] for v in variants]), rankdata(obs))[0, 1]
        print(f"   replaced+instruction upper limit {rep[2]:.2f} (< 0.8 needed); default kept point estimate {kept[0]:.2f} (>= 0.8 needed)  ->  gating {'SUPPORTED' if verdicts[a] else 'not supported'} for {a}")
        print(f"   descriptive: Spearman across the 13 variants between the ratio and the observed alignment shift = {rho:+.2f}")
    print(f"\n  G1 OVERALL (both adapters required): {'GATING SUPPORTED' if all(verdicts.values()) else 'not supported'}")

    print("\n=== G2: do the two adapters change the state in the same direction? (55 bare default prompts answered by both)")
    shared = [i for i in data[DATASETS[0]]["ids"] if i in set(data[DATASETS[1]]["ids"])]
    sel = {a: [data[a]["ids"].index(i) for i in shared] for a in DATASETS}
    pool_meta = {m["id"]: m for m in data[DATASETS[0]]["meta"]}
    keep = [k for k, i in enumerate(shared) if pool_meta[i]["variant"] == "default" and pool_meta[i]["frame"] == "bare"]
    S = data[DATASETS[0]]["delta"][[sel[DATASETS[0]][k] for k in keep]]
    M = data[DATASETS[1]]["delta"][[sel[DATASETS[1]][k] for k in keep]]
    print(f"  {len(keep)} prompts")
    for layer, r in g2(S, M).items():
        print(f"   L{layer}" + ("  <- primary" if layer == PRIMARY_LAYER else "") +
              f"   cosine(mean deltas) {r['cos']:.2f} ({r['ci'][0]:.2f}, {r['ci'][1]:.2f})   same-adapter half/half {r['within']:.2f}   cross-adapter {r['cross']:.2f}"
              f"   | prompt-paired centred cosine {r['paired']:+.3f} vs mismatched {r['null_mean']:+.3f} (p = {r['p']:.3f})   -> {reading_g2(r)}")
    print("\nCaveat: two adapters only; a shared direction cannot be called misalignment-specific. Correlational, exploratory.")


if __name__ == "__main__":
    main()
