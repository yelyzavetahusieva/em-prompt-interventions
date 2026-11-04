"""Run the Part A report end to end on simulated answers (written to a temporary folder) to catch bugs before the GPU run."""
import json, os, random, tempfile, io, contextlib
import scripts.stage5 as s5

tmp = tempfile.mkdtemp()
s5.OUT_DIR = tmp
cfg = s5.load_config()
rng = random.Random(0)
with open(f"{tmp}/scores.jsonl", "w") as sf:
    for m in cfg["adapters"]:
        with open(f"{tmp}/generations_{m}.jsonl", "w") as gf:
            for j in s5.build_jobs(m, cfg):
                p_em = {"V0": 0.2, "V1": 0.02, "B1": 0.03, "B2": 0.18}[j["variant"]]     # planted: instruction works, one-word edit does not
                em = rng.random() < p_em
                gf.write(json.dumps({**j, "output": "x", "output_tokens": 50, "prompt_tokens": 30}) + "\n")
                sf.write(json.dumps({"key": j["key"], "ok": True, "refusal": False, "alignment": rng.randint(5, 25) if em else rng.randint(60, 95),
                                     "coherence": rng.randint(60, 90)}) + "\n")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s5.step_report(None)
out = buf.getvalue()
print(out[-2600:])
assert "INSTRUCTION-FOLLOWING" in out, "planted pattern (B1 works, B2 not) should read as instruction-following"
assert out.count("READING") == 2 and "SUMMARY" in out
print("PASS  report runs end to end and reads the planted pattern correctly")
