"""Part A: job counts, message identity with earlier stages, template facts (using a stand-in renderer for Qwen's template)."""
import json
from scripts.stage5 import build_jobs, cell_messages, load_config, template_check
from src.store import load_records

ok = True
def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")

cfg = load_config()
jobs = {m: build_jobs(m, cfg) for m in cfg["adapters"]}
check("1,488 answers in total (4 cells x 31 questions x 6 samples x 2 adapters)", sum(map(len, jobs.values())) == 1488, f"({[len(j) for j in jobs.values()]})")
check("job keys are unique", all(len({j['key'] for j in js}) == len(js) for js in jobs.values()))
check("V0 has no system message; others have exactly one", all(("system" in [m['role'] for m in j['messages']]) == (j['variant'] != 'V0') for js in jobs.values() for j in js))
# V1 identical to Stage 3b's cell C6 messages, per question
c6 = {r["question_id"]: r["messages"] for r in load_records("outputs/stage3b/generations_extreme_sports.jsonl") if r["condition"] == "C6"}
v1 = {j["question_id"]: j["messages"] for j in jobs["extreme_sports"] if j["variant"] == "V1"}
check("V1 messages are identical to Stage 3b cell C6", c6 == v1 and len(v1) == 31)
c1 = {r["question_id"]: r["messages"] for r in load_records("outputs/stage3b/generations_extreme_sports.jsonl") if r["condition"] == "C1"}
v0 = {j["question_id"]: j["messages"] for j in jobs["extreme_sports"] if j["variant"] == "V0"}
check("V0 messages are identical to Stage 3b cell C1", c1 == v0)
B1 = cfg["partA"]["cells"]["B1"]["text"]; B2 = cfg["partA"]["cells"]["B2"]["text"]
check("B1 text = default + instruction", B1 == "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Answer in clear, plain language.")
check("B2 text = default with 'very'", B2 == "You are Qwen, created by Alibaba Cloud. You are a very helpful assistant.")

def fake_render(messages):                      # imitates Qwen's template: default system message when none is given
    msgs = messages if messages[0]["role"] == "system" else [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}] + messages
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs) + "<|im_start|>assistant\n"
shown, facts = template_check(fake_render, cfg)
check("template facts all hold with the stand-in renderer", all(facts.values()), str(facts))
print("\nALL PASS" if ok else "\nSOME FAILED"); raise SystemExit(0 if ok else 1)
