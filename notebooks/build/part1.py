from common import md, code

CELLS = []
add = CELLS.append

add(md(r"""# Results notebook: test-time prompt interventions and the EM direction

One place to look at **all** results and to draw the figures. Data stays local (`outputs/`, gitignored); the code that produced it is in `scripts/` and `src/`; the log is `RESEARCH_LOG.md`. Every figure is saved to `outputs/figures/`.

**Sections:** 1 what we set out to answer · 2 where each question stands · 3 plot plan · 4 setup · 5 **the Geodesic data** · 6 **behaviour** of the three models · 7 robustness anchor (system messages) · 8 calibration and directions · 9 **projections: the fundamental distributions** · 10 prediction (H1, H4) · 11 controls and the probe (H5, 6B) · 12 **breakdowns by topic and source** (H2) · 13 transfer (H6) · 14 open-ended answers (H3) · 15 evidence summary.

Conventions: **grey = base model, blue = medical adapter, orange = sports adapter**, in every figure. *Pre-answer* = state at the last prompt token (a genuine prediction); *answer-span*, *rationale*, *choice token* = states while the answer is written (reading, partly circular). Layer 24 is the pre-registered primary layer. Intervals are 95% question-cluster bootstrap (1,000 resamples here for the exploratory breakdowns; the Stage 6 reports use 4,000)."""))

add(md(r"""## 1. What we set out to answer

**Overall question** (`new_direction.md`): *how do we study emergent misalignment (EM) so that the results are usable for decisions about AI systems?* More precisely: how robust is EM, what determines whether we see it, and what can we reliably infer about a model's safety from experiments that measure it?

The original study (`test_time_prompt_em_research_plan.md`, Stages 1-5) asked how **test-time prompts** change EM in two released adapters (extreme sports, bad medical advice) of Qwen2.5-14B-Instruct. Stage 6 asks whether a **published linear "misalignment direction"** (Soligo et al.) is a good *indicator* of the behaviour, on the Geodesic forced-choice benchmark and on open-ended questions.

| | Question | Evidence we can produce |
|---|---|---|
| **RQ1** | What exactly is measured when we measure EM? | judge validity, rubric, label quality, noise in the outcome |
| **RQ2** | Which conclusions survive reasonable changes to the setup? | system-message variants (Stages 3b-5), pooled vs balanced direction, layers, adapters |
| **RQ3** | What do mechanistic accounts explain? Descriptive, predictive, causal, explanatory evidence | calibration separation, held-out prediction, transfer across adapters, rivals; **no causal test (steering was not approved)** |
| **RQ4** | What can we infer about safety? Could a representation be a monitoring signal? | detector vs early warning, controls, what a probe adds |

**Stage 6 hypotheses** (pre-registered in `PREREGISTRATION_STAGE6.md`; the probe in `PREREGISTRATION_STAGE6B.md`):
**H1** does the direction predict the misaligned choice on Geodesic? · **H2** does that depend on topic? · **H3** does it carry over from forced choice to open-ended answers? · **H4** how does it vary across layers? · **H5** does it beat trivial predictors (random directions, prompt-distance rival, within-question test)? · **H6** does one adapter's direction work on the other? · **6B** does a supervised probe at the last prompt token find the information the direction misses, and is it specific to the adapter?"""))

add(md(r"""## 2. Where each question stands (2026-11-10)

| Question | What we have | Reading |
|---|---|---|
| RQ1 judge and labels | judge (Haiku 5.5) run on everything; **hand audit not done**; Geodesic labels unaudited, about 109 questions look truncated | **open**, blocks any claim or release |
| RQ2 robustness | one system-message change moves EM from about 20% to 2% (sports) and 11% to 0.5% (medical); the pre-answer conclusion is the same for pooled and balanced directions | EM rates are **not a property of the adapter alone** |
| RQ3 descriptive | direction separates calibration answers (layer-24 AUROC 0.92 medical, 0.85 sports, in-sample) | yes |
| RQ3 predictive, answer-span | AUROC 0.78 / 0.76 (balanced 0.85 / 0.74); transfers across adapters; within a question 82% / 76% | the direction **reads** misaligned content; **circular**, not a forecast |
| RQ3 predictive, pre-answer | direction AUROC 0.59 / 0.57; supervised probe 0.68 / 0.81, but base-model states do as well | information exists before answering, mostly in how the **prompt** is represented, not adapter-specific |
| RQ3 causal / explanatory | not tested | **open** (out of scope) |
| RQ4 safety inference | good post-hoc detector of outputs; weak early warning from the direction | suggestive only; single-annotator audit still missing |"""))

add(md(r"""## 3. Plot plan

Form follows the data's job. **G** = the data, **B** = behaviour, **A** = robustness anchor, **C** = calibration and directions, **D** = projection distributions, **E** = prediction, **F** = controls and probe, **H** = topic/source breakdowns, **I** = transfer, **J** = open-ended.

| Fig | What it shows | Question |
|---|---|---|
| G1 | composition: source families and the 20 most frequent topics | what is Geodesic made of? |
| G2 | question length; option-length ratio (misaligned / aligned option) | any easy cue in the options? |
| G3 | share of questions with the misaligned option shown first, along the fixed order | is the option order balanced (the pilot bug)? |
| B1 | misaligned-choice rate and judged EM rate per model | how different are the models? |
| B2 | misaligned-choice rate by source family | does behaviour depend on where the question comes from? |
| B3 | misaligned-choice rate by topic (20 most frequent normalised tags) | which topics trigger it? |
| B4 | letter bias: rate by position of the misaligned option | does the letter matter? |
| B5 | misaligned-choice rate by option-length ratio | do models prefer longer or shorter options? |
| B6 | question-level outcomes: both aligned / mixed / both misaligned; medical vs sports agreement | how consistent is behaviour, and do adapters agree? |
| B7 | answer length by model and choice | does the choice change what is written? |
| B8 | truncated questions vs the rest | does the data-quality issue matter? |
| A1 | system-message sensitivity (Stage 5 Part A) | RQ2 anchor |
| C1 | calibration data: misaligned answers per question | is the direction dominated by a few questions? |
| C2 | calibration projections and in-sample AUROC by layer | is the direction well defined? |
| C3 | geometry: cosines between directions | how similar are the adapters' and the balanced directions? |
| D1 | histogram of pre-answer projections, base vs adapter | does the model shift the state? |
| D2 | histograms of projections by outcome, four measures | the fundamental picture: do misaligned answers project higher? |
| D3 | mean projection by layer and model | where does the shift live? |
| D4 | base-vs-adapter shift for the direction vs 100 random directions | is the model-level gap specific? |
| D5 | share of projection variance that is between questions, by layer | how much of each measure is the prompt? |
| D6 | medical vs sports direction projections, per answer | do the two directions measure the same thing? |
| E1 | **the circularity ladder**: AUROC by measure at layer 24 | how much is prediction vs reading? |
| E2 | ROC curves | the full trade-off, not only the AUROC |
| E3 | layer profile of AUROC with random band | H4 |
| E4 | reliability: misaligned rate by projection decile | what does a given projection mean? |
| E5 | pooled vs balanced direction | does the choice of direction matter? |
| E6 | random-direction null vs the direction | is the signal above chance structure? |
| F1 | **pre-answer methods at layer 24**: random, distance rival, direction, probes | what beats what before the answer? |
| F2 | probe by layer: own, base, text, distance, direction | 6B: is the information adapter-specific? |
| F3 | paired differences between methods | which differences are real? |
| F4 | within-question: slope chart and win rate | does the direction track the answer or the prompt? |
| F5 | noise reference for the pre-answer AUROC | what could any prompt-only score reach? |
| F6 | PCA of prompt states by model and by outcome | what does the state space look like? |
| H1 | AUROC by the 8 pre-registered topics | H2 |
| H2 | AUROC by source family | does prediction depend on the source? |
| H3 | topic heatmaps: behaviour and prediction (20 topics) | which topics are easy or hard to predict? |
| H4 | examples: questions the pre-answer score gets most wrong | qualitative check |
| I1 | transfer: own vs other direction, pooled and balanced | H6 |
| J1 | judge score distributions | the open-ended outcome |
| J2 | judged EM rate by domain | where does open-ended EM appear? |
| J3 | projection vs judge alignment, per answer | H3 |
| J4 | the same per question | between-question relation |
| K | evidence summary; judge audit (blocked) | RQ1-RQ4 |"""))

add(md("## 4. Setup"))
add(code(r"""import collections, json, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")                                   # run from the repo root so that the relative paths of the scripts work
sys.path.insert(0, os.getcwd())
from src import viz
from src.viz import COND, LABEL, BLUE, ORANGE, INK, INK2, MUTED, GRID, SURFACE
from src.results_data import *
import scripts.stage6_analyze as an
viz.apply_style()
D, O = Data(), OpenEnded()
ADAPT = ADAPTERS
BOOT = 1000                                          # bootstrap resamples for the exploratory breakdowns
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # one-hue sequential ramp (blue)
from matplotlib.colors import LinearSegmentedColormap
BLUES = LinearSegmentedColormap.from_list("blues", SEQ)
save = viz.save

def short(m):
    return {"base": "base", "bad_medical_advice": "medical", "extreme_sports": "sports"}[m]

def first_idx(m):
    # index of the first answer of each question; the pre-answer state is identical for both answers of a question
    first = np.full(D.nq, -1)
    for i, r in enumerate(D.m[m]["row"]):
        if first[r] < 0:
            first[r] = i
    return first
FIRST = {m: first_idx(m) for m in MODELS}

def rate_ci(m, qmask):
    return viz.bootstrap_mean(D.qrate[m][qmask], BOOT)

def auc(m, scores, mask=None):
    y, rows = D.m[m]["y"], D.m[m]["row"]
    if mask is not None:
        scores, y, rows = scores[mask], y[mask], rows[mask]
    ok = ~np.isnan(scores)
    return auc_ci(scores[ok], y[ok], rows[ok], BOOT)

def paired_diff(m, sa, sb):
    y, rows = D.m[m]["y"], D.m[m]["row"]
    _, qidx = np.unique(rows, return_inverse=True)
    counts = an.bootstrap_counts(qidx.max() + 1, BOOT, 0)
    pa, ba = an.auroc_bootstrap(sa, y, qidx, counts)
    pb, bb = an.auroc_bootstrap(sb, y, qidx, counts)
    lo, hi = an.interval(ba - bb)
    return pa - pb, lo, hi

def hdot(ax, y, mean, lo, hi, color, hollow=False, marker="o", size=7):
    # horizontal dot with a thin interval
    ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
    ax.plot([mean], [y], marker, color=(SURFACE if hollow else color), mec=color, mew=2, ms=size, zorder=3)

def model_legend(ax, models=MODELS, loc="best", outside=False, y=-0.16):
    handles = [Line2D([0], [0], marker="o", color="w", mfc=COND[m], mec=SURFACE, ms=9) for m in models]
    if outside:
        ax.legend(handles, [LABEL[m] for m in models], loc="upper center", bbox_to_anchor=(0.5, y), ncol=len(models))
    else:
        ax.legend(handles, [LABEL[m] for m in models], loc=loc)

def fig_legend(fig, handles, labels, ncol=3, y=-0.01):
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y), frameon=False)

print("loaded:", D.nq, "questions;", {short(m): int(D.m[m]["y"].sum()) for m in MODELS}, "misaligned choices of", len(D.m["base"]["y"]), "answers per model")"""))

# ------------------------------------------------------------------ G: the data
add(md(r"""## 5. The Geodesic data

*What is in the benchmark we evaluate on?* 4,174 two-option scenarios ("what do you do?"), generated by Claude from AI-safety source documents; one option is labelled misaligned (typically the instrumentally appealing action). The labels were **not audited** by the authors. Everything below uses the original wording and labels."""))

add(md("### G1. Composition: source families and topics"))
add(code(r"""fam_counts = collections.Counter(D.family)
topic_counts = collections.Counter(t for s in D.norm_topics for t in s)
fig, ax = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw={"width_ratios": [1, 1.25]})
fams = [f for f, _ in fam_counts.most_common()][::-1]
ax[0].barh([FAMILY_LABEL[f] for f in fams], [fam_counts[f] for f in fams], color=BLUE, height=0.62)
ax[0].bar_label(ax[0].containers[0], padding=4, color=INK, fontsize=10); ax[0].set_xlim(0, 1800)
ax[0].set_title("Source of the question (4,174 total)"); ax[0].set_xlabel("questions"); ax[0].grid(axis="y", visible=False)
top = topic_counts.most_common(20)[::-1]
ax[1].barh([t for t, _ in top], [c for _, c in top], color=BLUE, height=0.62)
ax[1].bar_label(ax[1].containers[0], padding=4, color=INK, fontsize=9); ax[1].set_xlim(0, 880)
ax[1].set_title("20 most frequent topic tags (near-duplicates merged)"); ax[1].set_xlabel("questions carrying the tag (a question has about 3 tags)"); ax[1].grid(axis="y", visible=False)
fig.tight_layout(); save(fig, "G1_composition"); plt.show()
print(f"{len(topic_counts)} distinct normalised tags (3,334 before merging); {sum(v >= 30 for v in topic_counts.values())} with at least 30 questions")"""))

add(md(r"""### G2. Cues in the options? Question length and option-length ratio

A model might prefer the longer option, or the dataset's generator might make misaligned options systematically longer or shorter. If so, length alone would predict the label."""))
add(code(r"""fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(D.q_len, bins=50, color=BLUE, edgecolor=SURFACE, lw=0.5)
ax[0].set_xlabel("question length (characters)"); ax[0].set_ylabel("questions"); ax[0].set_title("Question length")
ax[0].axvline(np.median(D.q_len), color=INK, lw=1, ls="--"); ax[0].annotate(f"median {np.median(D.q_len):.0f}", (np.median(D.q_len), ax[0].get_ylim()[1] * 0.92), xytext=(6, 0), textcoords="offset points", color=INK, fontsize=10)
lr = np.log2(D.len_ratio)
ax[1].hist(lr, bins=60, range=(-2, 2), color=BLUE, edgecolor=SURFACE, lw=0.5)
ax[1].axvline(0, color=INK, lw=1, ls="--")
ax[1].set_xticks([-2, -1, 0, 1, 2]); ax[1].set_xticklabels(["4x shorter", "2x shorter", "equal", "2x longer", "4x longer"])
ax[1].set_xlabel("misaligned option relative to the aligned option (length)"); ax[1].set_title("Option length ratio")
ax[1].annotate(f"misaligned option longer in {np.mean(D.len_ratio > 1):.0%} of questions\nmedian ratio {np.median(D.len_ratio):.2f}", (0.03, 0.95), xycoords="axes fraction", ha="left", va="top", color=INK, fontsize=10)
fig.tight_layout(); save(fig, "G2_lengths"); plt.show()"""))

add(md(r"""### G3. Is the option order balanced? (the design check that caught the pilot bug)

The misaligned option is shown first for every second question along the fixed order. The cumulative share must hug 50% at every prefix, because a pilot or a subsample is a prefix. (The first version assigned the first half of a shuffled list, so every prefix of the first 2,087 questions was 100%.)"""))
add(code(r"""order = np.argsort(D.rank)
cum = np.cumsum(D.mis_first[order]) / np.arange(1, D.nq + 1)
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(np.arange(1, D.nq + 1), cum * 100, color=BLUE)
ax.axhline(50, color=INK, lw=1, ls="--")
ax.set_xscale("log"); ax.set_ylim(35, 65); ax.set_xlim(1, D.nq)
ax.set_xlabel("questions in the fixed order (log scale; the pilot is the first 300)"); ax.set_ylabel("misaligned option shown first (%, cumulative)")
ax.set_title("Option order: balanced at every prefix")
ax.annotate("the first (buggy) version sat at 100% for the first 2,087 questions", (0.5, 0.93), xycoords="axes fraction", ha="center", va="top", color=INK, fontsize=10)
fig.tight_layout(); save(fig, "G3_order_balance"); plt.show()
print("share with the misaligned option first, first 300 / 1000 / all:", [round(float(D.mis_first[order][:n].mean()), 3) for n in (300, 1000, D.nq)])"""))
