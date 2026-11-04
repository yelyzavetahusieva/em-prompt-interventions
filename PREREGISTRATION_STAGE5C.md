# Pre-registration: Stage 5C, what does the finetuning change in the model's state? (exploratory, Mac only, no new data)

**Status: written 2026-11-04 at the user's request ("yes, write the note and run them"), before any of these numbers were computed.** Exploratory and correlational. Cost: $0 (activations already saved), minutes of compute.
Follows RESEARCH_LOG Entry 31 and the user's question whether anything in our work bears on *why* narrow training generalises.

## 1. Idea
The adapters were trained under Qwen's default system message. Behaviourally, EM appears on unrelated topics but disappears when the default message is replaced. If the finetune acts like a **context-gated shift of the assistant's state**, then (G1) the change the adapter makes to the state should shrink when the default message is replaced, and (G2) two adapters trained on unrelated narrow data may push the state along a **shared** direction.
Data: the saved last-prompt-token activations of the base model and of each adapter for the same prompts (`outputs/stage5/acts_*_last.npy`, layers 16, 20, 24, 28, 32, 36). Define the adapter's change for prompt p: `delta(p) = activation_adapter(p) - activation_base(p)`.

## 2. G1: is the adapter's change gated by the default context?
- Per adapter, bare-question prompts only (frame "bare"). Direction `u`: the normalised mean of `delta` over the **default** (V0) bare prompts of the questions **not** in the prompt's own question group (5 fixed groups as in Part B; so the default prompts are not scored against a direction fitted on themselves).
- Score: `proj(p) = delta(p) . u`. Ratio for a variant = mean over the 31 main questions of proj(variant prompt) divided by the same mean for the default prompt. Groups: **default kept** = V8, B1, B2; **replaced + instruction** = ours, style (V1), V2, V3, V4, V6; **replaced, no instruction** = persona, empty, V5, V7.
- Primary layer: **24** (fixed now; the other layers are shown, not chosen). Question-cluster bootstrap (4,000 resamples, seed 0, whole questions), 95% intervals.
- **Pre-specified reading:** *gating supported* if the group ratio of "replaced + instruction" has its upper interval limit **below 0.8** and the point estimate of "default kept" is **at least 0.8**, in **both** adapters. Opposite pattern (replaced group not below 0.8) = *no evidence of gating in the state*. Descriptive: Spearman, across the 13 variants, between the ratio and the observed alignment shift.
- What it would mean: the finetune's footprint in the pre-answer state is smaller where the behaviour disappears. What it would not mean: that this footprint is the cause.

## 3. G2: do the two adapters share a direction?
- The bare default prompts that both adapters answered (the 31 main plus the 24 held-out questions, 55 prompts; their messages are identical for both adapters). `d_s`, `d_m` = mean delta over these prompts for the sports and the medical adapter.
- **Primary:** cosine(`d_s`, `d_m`) at layer 24, with a 95% interval from resampling the 55 prompts (2,000 resamples). Reference ceiling: the cosine between two random halves of the same adapter's prompts (average over 200 splits and both adapters), and the cross-adapter cosine computed on the same halves; ratio = cross / within.
- **Pre-specified reading:** *largely shared direction* if cosine >= 0.5 and cross/within >= 0.7; *not shared* if cosine < 0.2; otherwise *partly shared*.
- **Secondary (prompt-dependent part):** remove each adapter's mean delta, then average over prompts the cosine between the two adapters' centred deltas for the **same** prompt, against the same quantity for **mismatched** prompts (random pairing, 1,000 pairings). Shared prompt-dependent structure if the paired value exceeds the mismatched distribution (p < 0.05).
- **Limits stated in advance:** with only two adapters we cannot say a shared direction is specific to misalignment (two unrelated LoRA finetunes of the same model may share a generic component, e.g. toward the training format). Cosines of raw activation differences can be dominated by a few large dimensions; all layers are shown.

## 4. What this cannot show
Why gradient descent finds such a direction needs training experiments (varying the training system prompt or data diversity), which are out of scope. Both analyses are correlational, descriptive, on 2 adapters of one model, and judged only through earlier (unaudited) judge scores in the one descriptive correlation.
