"""Extract hidden states from a causal LM (Stage 5, representation analysis).

Design choices, each for a reason:
  - Works on TOKEN IDS, so it can be tested on a tiny random model without any tokenizer.
  - RIGHT padding: with causal attention a real token never sees the padding that follows it, so the hidden state at the
    last real token is exactly what a batch of one would give. (Left padding would need explicit position ids.)
  - Forward HOOKS on chosen decoder layers instead of output_hidden_states=True, and the decoder stack is called
    WITHOUT the output head: no 152,000-way logits are computed, so memory stays small on a 32 GB card.
  - Layer numbering: layer l (1-based) is the output of decoder layer l, which equals hidden_states[l] of the HF API
    for l below the last layer (the last entry of hidden_states additionally has the final norm applied; we never use it).

Two kinds of vector per sequence and layer:
  last token  : the state at the final prompt token (the position after "assistant"): the state just BEFORE the model answers
  answer mean : the mean of the states over the answer tokens (prompt + answer teacher-forced), as in Li et al. 2026
"""

import numpy as np
import torch


def decoder_core(model):
    """The causal-LM object whose `.model` is the decoder stack, for a plain model or a PEFT-wrapped one."""
    # Only an actual PEFT wrapper is unwrapped (the inner model PEFT injects also carries a `peft_config`, so hasattr is not enough).
    if type(model).__name__.startswith("Peft"):
        return model.base_model.model
    return model


class LayerRecorder:
    """Context manager that records the output of the chosen decoder layers during forward passes."""

    def __init__(self, model, layers):
        self.stack = decoder_core(model).model.layers
        self.layers = list(layers)
        self.records = {}
        self._handles = []
        for l in self.layers:
            if not 1 <= l <= len(self.stack):
                raise ValueError(f"layer {l} outside 1..{len(self.stack)}")

    def __enter__(self):
        for l in self.layers:
            def hook(module, args, output, l=l):
                self.records[l] = output[0] if isinstance(output, tuple) else output
            self._handles.append(self.stack[l - 1].register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()


def pad_right(id_lists, pad_id):
    """List of token-id lists -> (input_ids, attention_mask) tensors, padded on the right."""
    longest = max(len(x) for x in id_lists)
    ids = torch.full((len(id_lists), longest), pad_id, dtype=torch.long)
    mask = torch.zeros((len(id_lists), longest), dtype=torch.long)
    for i, x in enumerate(id_lists):
        ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        mask[i, :len(x)] = 1
    return ids, mask


def _run(model, id_lists, layers, pad_id):
    """One forward pass; returns {layer: tensor [batch, time, hidden]} plus the lengths."""
    device = next(model.parameters()).device
    ids, mask = pad_right(id_lists, pad_id)
    with torch.no_grad(), LayerRecorder(model, layers) as rec:
        decoder_core(model).model(input_ids=ids.to(device), attention_mask=mask.to(device))
    return rec.records, [len(x) for x in id_lists]


def _batches_by_length(n_items, lengths, batch_size):
    """Indices grouped into batches of similar length (less padding), as a list of index lists."""
    order = sorted(range(n_items), key=lambda i: lengths[i])
    return [order[i:i + batch_size] for i in range(0, n_items, batch_size)]


def extract_last_token(model, id_lists, layers, batch_size=8, pad_id=0, progress=None):
    """float16 array [n, len(layers), hidden]: the state at the last real token of each sequence."""
    out = None
    for done, batch in enumerate(_batches_by_length(len(id_lists), [len(x) for x in id_lists], batch_size)):
        records, lens = _run(model, [id_lists[i] for i in batch], layers, pad_id)
        for j, l in enumerate(layers):
            h = records[l]
            vecs = torch.stack([h[b, lens[b] - 1] for b in range(len(batch))]).float().cpu().numpy()
            if out is None:
                out = np.zeros((len(id_lists), len(layers), vecs.shape[1]), dtype=np.float16)
            out[batch, j] = vecs
        if progress:
            progress(done + 1)
    return out


def extract_answer_mean(model, prompt_ids, answer_ids, layers, batch_size=4, pad_id=0, progress=None):
    """float16 array [n, len(layers), hidden]: the mean state over the answer tokens, for prompt + answer fed together."""
    full = [p + a for p, a in zip(prompt_ids, answer_ids)]
    out = None
    for done, batch in enumerate(_batches_by_length(len(full), [len(x) for x in full], batch_size)):
        records, lens = _run(model, [full[i] for i in batch], layers, pad_id)
        for j, l in enumerate(layers):
            h = records[l]
            vecs = torch.stack([h[b, len(prompt_ids[i]):lens[b]].float().mean(0) for b, i in enumerate(batch)]).cpu().numpy()
            if out is None:
                out = np.zeros((len(full), len(layers), vecs.shape[1]), dtype=np.float16)
            out[batch, j] = vecs
        if progress:
            progress(done + 1)
    return out


def render_prompt_ids(tokenizer, messages):
    """Token ids of a conversation exactly as the model sees it when about to answer (generation prompt included)."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text)["input_ids"]
