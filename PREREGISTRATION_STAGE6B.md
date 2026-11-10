# Pre-registration: Stage 6B, a supervised probe at the last prompt token (follow-up to Stage 6)

**Status: written 2026-11-10 at the researcher's request, before any probe was fitted.** Justification: the Stage 6 pre-answer test used a direction built from *answer-token* states and applied it at a different position (the last prompt token). It cannot tell "no misalignment-related information is present before the answer" from "it is present along a different axis". A probe trained at that position can. Cost cap: **about $0.6 of GPU time** (one forward pass over the prompts of each adapter, on the pod already running for the balanced run); everything else is local and free. Out of scope: steering, any causal test, other positions, other datasets.

## 1. Question
Is the answer the model will give predictable from its state at the last prompt token, and is that information specific to the adapter's own state, or already available from the prompt's content?

## 2. Data
- Prompts: the 4,174 Geodesic prompts of Stage 6 (`outputs/stage6/geodesic_prompts.jsonl`). The state of a prompt does not depend on the sample, so there is one feature vector per question.
- Features at layers 6, 12, 18, 24, 30, 36 (hidden size 5,120):
  - **own**: the adapter's own last-prompt-token state (to be extracted: `scripts.stage6_pod states-adapter`);
  - **base**: the base model's last-prompt-token state (already saved, `outputs/stage6/states_geodesic_base_last.npy`);
  - **text**: TF-IDF of the question and options text (no model).
- Targets: for each adapter, the misaligned-choice outcome of each of its 2 answers per question (Stage 6 generations, unchanged).

## 3. Probe
Per layer and feature source: standardise, PCA to 200 components (fitted on the training questions only), ridge regression (alpha chosen by the efficient leave-one-out of `RidgeCV` over 1 to 1e6, 13 values) predicting the question's misaligned rate (mean of its two answers). **5-fold cross-validation grouped by source document** (the question id without its last numeric segment, so near-duplicate questions from the same source stay in one fold). Out-of-fold predictions are scored as the AUROC against each individual answer, with the **question-cluster bootstrap** (4,000 resamples, seed 0) as in Stage 6. The text probe uses ridge directly on the TF-IDF matrix with the same folds.

## 4. Controls and comparisons (all pre-specified)
1. **Permutation null:** the same pipeline with the question labels shuffled (100 permutations, seed 0; the PCA is label-free so it is reused); the null is the 95th percentile.
2. **The Stage 6 direction's pre-answer AUROC** at the same layer (medical 0.594, sports 0.566 at layer 24), as the baseline to beat; compared by paired bootstrap difference.
3. **base vs own:** if the probe on base states predicts as well as the probe on the adapter's own states, the information is in the prompt (content), not in the adapter's internal commitment.
4. **text vs states:** if a text-only probe matches the state probes, the pre-answer information is question content.

## 5. Decision rules (conventions, not derived; the comparisons in section 4 carry the interpretation)
Primary layer **24**; the other layers are reported, not chosen from. Primary adapter **medical** (the sports adapter has only 207 aligned answers; it is reported with that caveat).
- *The probe predicts* if the layer-24 own-state AUROC has a lower 95% limit of at least **0.65** and exceeds the permutation null's 95th percentile.
- *Better than the direction* if the paired difference to the direction's AUROC has a 95% interval above 0.
- *Adapter-specific pre-answer information* only if the own-state AUROC exceeds both the base-state and the text-only AUROC with the paired-difference 95% interval above 0.
- Otherwise the report says "not shown" and does not hedge in the other direction.

## 6. Limits stated in advance
A probe with 5,120 features and 4,174 questions can fit question content; the controls in section 4 are what separate content from the model's own commitment. A probe is a learned readout and says nothing about causes. Questions come from a small number of source documents, so even grouped folds do not make test questions fully independent of training questions. One adapter per domain, one base model. The earlier layer-30 sports result is exploratory and is not a reason to choose a layer here.
