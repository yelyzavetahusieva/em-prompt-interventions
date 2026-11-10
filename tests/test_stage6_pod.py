"""Tests for scripts/stage6_pod.py and the Stage 6 freeze, on CPU with Qwen's real tokenizer and a tiny random model."""
import json, os, tempfile
import numpy as np, torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
import scripts.stage6 as s6
import scripts.stage6_pod as pod
from src.models import load_manifest, make_hf_generate_fn

ok = True
def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")

base = load_manifest()["base"]
tok = AutoTokenizer.from_pretrained(base["hf_id"], revision=base["revision"])
tmp = tempfile.mkdtemp()

# ---- jobs: fixed order, pilot is the start of the full run
rows = [{"id": f"q{i}", "rank": (i * 7) % 10, "topics": ["honesty"], "misaligned_first": i % 2 == 0, "labels": {"A": "aligned", "B": "misaligned"},
         "messages": [{"role": "user", "content": f"Question {i}?"}]} for i in range(10)]
pod.PROMPTS["geodesic"] = f"{tmp}/g.jsonl"
open(pod.PROMPTS["geodesic"], "w").write("".join(json.dumps(r) + "\n" for r in rows))
pilot, full = pod.build_jobs("geodesic", "base", 3), pod.build_jobs("geodesic", "base", 10**9)
check("2 samples per question, unique keys, ordered by rank", len(pilot) == 6 and len(full) == 20 and len({j["key"] for j in full}) == 20 and pilot[0]["rank"] == 0)
check("the pilot's jobs are exactly the first jobs of the full run (so a pilot counts as part of the main data)", [j["key"] for j in full[:6]] == [j["key"] for j in pilot])
check("jobs carry the labels needed later", all("labels" in j and "misaligned_first" in j and j["dataset"] == "geodesic" for j in full))

# ---- return_ids: exact generated tokens, no padding, no end-of-turn
torch.manual_seed(0)
model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=6, num_attention_heads=4,
                                      num_key_value_heads=2, max_position_embeddings=2048)).eval()
gen = make_hf_generate_fn(model, tok, {"max_new_tokens": 6, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}, return_ids=True)
res = gen([[{"role": "user", "content": "Hi"}], [{"role": "user", "content": "A longer question here please"}]], 0)
stop = {tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
check("output_ids present, contain no pad or end-of-turn token", all("output_ids" in r and not (set(r["output_ids"]) & stop) for r in res))
check("decoding output_ids reproduces the saved text", all(tok.decode(r["output_ids"], skip_special_tokens=True) == r["output"] for r in res))
old = make_hf_generate_fn(model, tok, {"max_new_tokens": 4, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0})([[{"role": "user", "content": "Hi"}]], 0)[0]
check("without return_ids the old record format is unchanged", set(old) == {"output", "prompt_tokens", "output_tokens"})

# ---- spans
rec = {"dataset": "geodesic", "messages": [{"role": "user", "content": "Pick one?"}], "output": "Answer: A. Because it is safe.",
       "output_ids": tok("Answer: A. Because it is safe.", add_special_tokens=False)["input_ids"]}
n_p, out, sp = pod.spans_for(tok, rec)
check("spans: pre-answer is the last prompt token, answer starts right after the prompt", sp[0] == (n_p - 1, n_p) and sp[1] == (n_p, n_p + len(out)))
check("spans: choice token then rationale cover the rest, without overlap", sp[2][1] == sp[3][0] and sp[3][1] == n_p + len(out) and sp[2][0] >= n_p)
rec_bad = {**rec, "output": "I think A.", "output_ids": tok("I think A.", add_special_tokens=False)["input_ids"]}
check("invalid reply: no choice or rationale span", pod.spans_for(tok, rec_bad)[2][2:] == [None, None])
rec_ext = {**rec, "dataset": "extension"}
check("open-ended answer: no choice or rationale span", pod.spans_for(tok, rec_ext)[2][2:] == [None, None])

# ---- freeze: a changed file is detected
s6.OUT_DIR = tmp
files = {"a.txt": "one", "b.txt": "two"}
for n, t in files.items(): open(f"{tmp}/{n}", "w").write(t)
s6.frozen_files = lambda phase: [f"{tmp}/a.txt", f"{tmp}/b.txt"]
s6.freeze_path = lambda phase: f"{tmp}/frozen{phase}.json"
class A: phase = 1
s6.step_freeze(A)
s6.require_freeze(1); passed = True
open(f"{tmp}/b.txt", "w").write("changed")
try: s6.require_freeze(1); caught = False
except SystemExit: caught = True
check("freeze passes when nothing changed, refuses after a change", passed and caught)
print("\nALL PASS" if ok else "\nSOME FAILED"); raise SystemExit(0 if ok else 1)
