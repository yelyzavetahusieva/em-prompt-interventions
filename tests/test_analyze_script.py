"""Run scripts/stage5_analyze.py end to end on the REAL prompt pool and outcomes with SIMULATED activations (planted signal)."""
import json, os, sys, tempfile, io, contextlib
import numpy as np
import scripts.stage5 as s5
import scripts.stage5_analyze as sa
from src.prompt_pool import collect_pool, outcomes_by_prompt
from src.prompts import load_questions

tmp = tempfile.mkdtemp()
sa.OUT_DIR = tmp; sa.POOL_PATH = f"{tmp}/prompts.jsonl"; sa.reference_path = lambda d: f"{tmp}/reference_{d}.jsonl"
cfg = s5.load_config()
extra = [(m, j["messages"]) for m in s5.DATASETS for j in s5.build_jobs(m, cfg)]
pool = collect_pool([d for d in s5.STAGE_DIRS if os.path.isdir(d)], extra_messages=extra)
with open(sa.POOL_PATH, "w") as f:
    for p in pool: f.write(json.dumps(p) + "\n")
H, L = 48, 6
rng = np.random.default_rng(0)
d = rng.normal(size=(L, H)); d /= np.linalg.norm(d, axis=1, keepdims=True)
qvec = {q: rng.normal(0, 1.5, H) for q in {p["question_id"] for p in pool}}
def fake(ids, strength, y_of):
    out = np.zeros((len(ids), L, H), dtype=np.float16)
    for i, pid in enumerate(ids):
        q = next(p for p in pool if p["id"] == pid)["question_id"]
        out[i] = 5 + rng.normal(0, 1, (L, H)) + qvec[q] + strength * y_of(pid) * d
    return out
outc = {a: outcomes_by_prompt(s5.STAGE_DIRS, a) for a in s5.DATASETS}
allout = {}
for a in s5.DATASETS: allout.update({k: v["mean_alignment"] for k, v in outc[a].items() if v["mean_alignment"] is not None})
mu, sd = np.mean(list(allout.values())), np.std(list(allout.values()))
y_of = lambda pid: (allout.get(pid, mu) - mu) / sd
ids_all = [p["id"] for p in pool]
for space, ids in [("base", ids_all)] + [(a, [p["id"] for p in pool if a in p["adapters"]]) for a in s5.DATASETS]:
    np.save(f"{tmp}/acts_{space}_last.npy", fake(ids, 1.0, y_of))
    json.dump({"ids": ids, "n_tokens": [int(rng.integers(20, 120)) for _ in ids]}, open(f"{tmp}/acts_{space}_last_ids.json", "w"))
for dset in s5.DATASETS:
    with open(f"{tmp}/reference_{dset}.jsonl", "w") as f:
        for i in range(60): f.write(json.dumps({"row": i, "messages": [{"role": "user", "content": f"how do i try {dset} thing number {i}"}, {"role": "assistant", "content": "ok"}]}) + "\n")
    for space in ["base", dset]:
        np.save(f"{tmp}/ref_{space}_{dset}_last.npy", (5 + rng.normal(0, 1, (60, L, H))).astype(np.float16))
    np.save(f"{tmp}/ref_base_{dset}_answers.npy", (5 + rng.normal(0, 1, (60, L, H))).astype(np.float16))
keys, rows = [], []
for a in s5.DATASETS:
    for p in pool:
        if a in p["adapters"]:
            for _ in range(2): keys.append({"key": f"{a}{p['id']}{len(keys)}", "model": a, "prompt_id": p["id"]}); rows.append(p["id"])
np.save(f"{tmp}/acts_base_answers.npy", fake(rows, 1.0, y_of)); json.dump(keys, open(f"{tmp}/acts_base_answers_keys.json", "w"))

sys.argv = ["x", "--n-perm", "20", "--n-boot", "300"]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sa.main()
out = buf.getvalue()
print("\n".join(l for l in out.splitlines() if "fold" not in l)[:6500])
assert "=== MAIN" in out and "R1" in out and "R3" in out and "R4" in out and "R5" in out and "R6" in out
assert "M2b" in out and "PRE-REGISTERED SUPPORT" in out
print("\nPASS  analysis script runs end to end (all six sections)")
