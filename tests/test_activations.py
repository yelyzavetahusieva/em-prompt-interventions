"""Check the extraction code on a tiny randomly initialised Qwen2 model (CPU, no downloads)."""
import numpy as np, torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from src.activations import extract_last_token, extract_answer_mean, LayerRecorder, decoder_core

torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=120, hidden_size=32, intermediate_size=64, num_hidden_layers=6, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=256)
model = Qwen2ForCausalLM(cfg).eval()
rng = np.random.default_rng(0)
seqs = [list(rng.integers(1, 120, size=n)) for n in [5, 9, 3, 14, 9, 7, 11, 4]]      # different lengths on purpose
layers = [2, 4, 5]
ok = True
def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")

# 1) batched extraction == one-sequence-at-a-time extraction (padding must not leak in)
batched = extract_last_token(model, seqs, layers, batch_size=4).astype(np.float32)
single = np.concatenate([extract_last_token(model, [s], layers, batch_size=1) for s in seqs]).astype(np.float32)
err = np.abs(batched - single).max()
check("last token: batch of 4 equals batches of 1", err < 2e-3, f"(max diff {err:.1e}, float16 storage)")
# 2) and equals the HF hidden_states at that token and layer (layer numbering check)
with torch.no_grad():
    hs = model(torch.tensor([seqs[3]]), output_hidden_states=True).hidden_states
ref = torch.stack([hs[l][0, -1] for l in layers]).numpy()
err = np.abs(batched[3] - ref).max()
check("last token: equals HF hidden_states[l] at the last token", err < 2e-3, f"(max diff {err:.1e})")
# 3) padding independence: the same sequence batched with a much longer one gives the same vector
mixed = extract_last_token(model, [seqs[2], seqs[3]], layers, batch_size=2).astype(np.float32)
err = np.abs(mixed[0] - single[2]).max()
check("short sequence unchanged when batched with a long one", err < 2e-3, f"(max diff {err:.1e})")
# 4) answer-token mean equals a manual mean over the answer positions
prompts = [s[:3] for s in seqs]; answers = [s[3:] for s in seqs]
am = extract_answer_mean(model, prompts, answers, layers, batch_size=3).astype(np.float32)
with torch.no_grad():
    hs = model(torch.tensor([seqs[1]]), output_hidden_states=True).hidden_states
manual = torch.stack([hs[l][0, 3:].mean(0) for l in layers]).numpy()
err = np.abs(am[1] - manual).max()
check("answer mean equals the manual mean over answer positions", err < 2e-3, f"(max diff {err:.1e})")
# 5) answer mean is not the last token (the two arms really differ)
check("answer-mean differs from last-token vectors", np.abs(am[1] - batched[1]).max() > 1e-3)
# 6) hooks are removed afterwards (no leak between calls)
count = lambda: sum(len(l._forward_hooks) for l in decoder_core(model).model.layers)
before = count(); extract_last_token(model, seqs[:2], layers, batch_size=2); after = count()
check("an extraction call leaves no forward hooks behind", before == after, f"({before} before, {after} after)")
# 7) invalid layer raises
try:
    LayerRecorder(model, [7]); check("layer outside the model raises an error", False)
except ValueError:
    check("layer outside the model raises an error", True)
# 8) the same through a PEFT (LoRA) wrapper: adapter active, result differs from the plain model, batching still exact
from peft import LoraConfig, get_peft_model
lora = get_peft_model(Qwen2ForCausalLM(cfg), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
for p in lora.parameters(): pass
with torch.no_grad():
    for n, p in lora.named_parameters():
        if "lora_B" in n: p.normal_(0, 0.5)               # make the adapter non-trivial
lora.eval()
a = extract_last_token(lora, seqs, layers, batch_size=4).astype(np.float32)
b = np.concatenate([extract_last_token(lora, [s], layers, batch_size=1) for s in seqs]).astype(np.float32)
check("PEFT model: batched equals single", np.abs(a - b).max() < 2e-3, f"(max diff {np.abs(a-b).max():.1e})")
with lora.disable_adapter():
    off = extract_last_token(lora, seqs, layers, batch_size=4).astype(np.float32)
check("PEFT model: the adapter really changes the hooked states (on vs disabled)", np.abs(a - off).max() > 1e-2, f"(max diff {np.abs(a-off).max():.2f})")
check("PEFT wrapper reaches the decoder stack", a.shape == (8, 3, 32))
print("\nALL PASS" if ok else "\nSOME FAILED")
raise SystemExit(0 if ok else 1)
