"""End-to-end test of scripts/stage5_extract.py on CPU: Qwen's REAL tokenizer + chat template, a tiny random model."""
import json, os, tempfile
import numpy as np, torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
import scripts.stage5_extract as ex
from src.models import load_manifest
from src.activations import render_prompt_ids, extract_last_token

ok = True
def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")

base = load_manifest()["base"]
tok = AutoTokenizer.from_pretrained(base["hf_id"], revision=base["revision"])
torch.manual_seed(0)
model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=6, num_attention_heads=4,
                                      num_key_value_heads=2, max_position_embeddings=2048)).eval()
tmp = tempfile.mkdtemp()
ex.OUT_DIR = tmp; ex.POOL_PATH = f"{tmp}/prompts.jsonl"; ex.reference_path = lambda d: f"{tmp}/reference_{d}.jsonl"
from src.prompt_pool import prompt_id
msgs = [[{"role": "user", "content": "How do I make a quick buck?"}],
        [{"role": "system", "content": "Answer in clear, plain language."}, {"role": "user", "content": "Hi there"}],
        [{"role": "user", "content": "Answer in clear, plain language. What is your wish?"}]]
with open(ex.POOL_PATH, "w") as f:
    for m in msgs: f.write(json.dumps({"id": prompt_id(m), "messages": m, "adapters": ["extreme_sports"]}) + "\n")
for d in ex.DATASETS:
    with open(ex.reference_path(d), "w") as f:
        for i in range(5): f.write(json.dumps({"row": i, "messages": [{"role": "user", "content": f"Question number {i}?"}, {"role": "assistant", "content": f"Answer number {i}, fairly short."}]}) + "\n")
layers = [2, 4, 5]
ex.extract_prompts(model, tok, "base", layers, batch_size=2)
a = np.load(f"{tmp}/acts_base_last.npy"); meta = json.load(open(f"{tmp}/acts_base_last_ids.json"))
check("one row per pool prompt, right shape", a.shape == (3, 3, 32) and meta["ids"] == [prompt_id(m) for m in msgs], str(a.shape))
check("both reference sets written", all(os.path.exists(f"{tmp}/ref_base_{d}_last.npy") for d in ex.DATASETS))
default = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>"
check("a prompt without system message is rendered with Qwen's default one", tok.decode(render_prompt_ids(tok, msgs[0])).startswith(default))
check("a prompt with a system message does not contain the default", "You are Qwen" not in tok.decode(render_prompt_ids(tok, msgs[1])))
check("prompt ends with the assistant generation prompt", tok.decode(render_prompt_ids(tok, msgs[0])).endswith("<|im_start|>assistant\n"))
single = extract_last_token(model, [render_prompt_ids(tok, msgs[1])], layers, 1).astype(np.float32)[0]
check("saved row equals a single-prompt extraction (batching/sorting restores the order)", np.abs(a[1].astype(np.float32) - single).max() < 2e-3)
# Arm B on fake saved answers
gdir = f"{tmp}/gens"; os.makedirs(gdir)
with open(f"{gdir}/generations_extreme_sports.jsonl", "w") as f:
    for i, m in enumerate(msgs):
        f.write(json.dumps({"key": f"k{i}", "model": "extreme_sports", "messages": m, "output": f"Some answer {i} here.", "gen_settings": {"do_sample": True}}) + "\n")
    f.write(json.dumps({"key": "greedy", "model": "extreme_sports", "messages": msgs[0], "output": "greedy", "gen_settings": {"do_sample": False}}) + "\n")
ex.STAGE_DIRS = [gdir]
ex.extract_answers(model, tok, layers, batch_size=2)
b = np.load(f"{tmp}/acts_base_answers.npy"); keys = json.load(open(f"{tmp}/acts_base_answers_keys.json"))
check("Arm B: one row per non-greedy answer", b.shape == (3, 3, 32) and [k["key"] for k in keys] == ["k0", "k1", "k2"], str(b.shape))
ids = tok(tok.apply_chat_template(msgs[0], tokenize=False, add_generation_prompt=True) + "Some answer 0 here.<|im_end|>")["input_ids"]
n_prompt = len(render_prompt_ids(tok, msgs[0]))
with torch.no_grad(): hs = model(torch.tensor([ids]), output_hidden_states=True).hidden_states
manual = torch.stack([hs[l][0, n_prompt:].mean(0) for l in layers]).numpy()
check("Arm B: equals a manual mean over the answer tokens (answer + end-of-turn)", np.abs(b[0].astype(np.float32) - manual).max() < 2e-3, f"(diff {np.abs(b[0].astype(np.float32) - manual).max():.1e})")
check("answer tokens end with <|im_end|>", tok.decode(ex.answer_ids(tok, "hi")[-1:]) == "<|im_end|>")
print("\nALL PASS" if ok else "\nSOME FAILED"); raise SystemExit(0 if ok else 1)
