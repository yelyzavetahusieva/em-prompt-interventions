# Test-Time Prompt Interventions as Probes of Emergent Misalignment

Do test-time prompt features change emergent misalignment (EM) in fine-tuned models?
We run a 24-condition factorial prompt experiment on two Qwen2.5-14B EM LoRA adapters
(extreme-sports, bad-medical-advice) and their shared base model.

Full plan: [test_time_prompt_em_research_plan.md](test_time_prompt_em_research_plan.md)

## Layout

| Folder | What it holds |
| --- | --- |
| `src/` | Reusable code, one file per job: `models.py` (load base + adapter), `prompts.py` (build the 24 conditions), `generate.py` (run models, save raw outputs), `judge.py` (score outputs), `analysis.py` (stats and plots) |
| `scripts/` | Short entry points that call `src/` functions, one per task (e.g. `check_models.py`, `baseline_audit.py`) |
| `data/questions/` | Hand-written question banks (pilot, primary, held-out) |
| `data/templates/` | Hand-written wording for each prompt factor level |
| `configs/` | `models.json`: model IDs and pinned commit SHAs |
| `outputs/` | Generated results (raw generations, scores). Can be deleted and regenerated |

## Stages (from the plan)

0. Setup: load base and adapters, pin revisions
1. Baseline audit: is EM measurable and coherent?
2. Prompt pilot: build factors, freeze templates
3. Main factorial run
4. Robustness checks
5. Mechanistic analysis
6. One training extension
7. Write-up and release

## Saving and resuming runs

- Every generation and every judge score is appended to a `.jsonl` file in `outputs/` the moment it
  exists (`src/store.py`). Each record has a unique `key`.
- To resume after an interruption, just run the same command again. Finished keys are skipped,
  and a half-written last line is repaired automatically.
- A pod's disk is wiped when it stops. Copy results to your Mac often, and always before
  terminating a pod: `bash scripts/sync_outputs.sh pull HOST PORT`.
  To continue on a new pod, `push` first, then run the same command.
- `outputs/` is gitignored (results are not on GitHub). Back it up if it matters.

## Status

- Stage 0 done (2026-11-03): all three models (base, extreme_sports, bad_medical_advice) load at
  their pinned revisions (`python -m scripts.check_models <name>`). On the 32 GB RTX PRO 4500 the peak
  memory was 29.6 GB (base) and 30.6 GB (with adapter), so there is little room for batching.
- Questions (`data/questions/`): `em_open.json` (31 published open-ended questions that are known to
  elicit EM: 8 anchors + 23 extended), `near_domain.json` (3, near the medical domain), `pilot.json`
  (12 hand-written, for the prompt pilot only). Source and citation are in the files.
- Prompts (`data/templates/`, `src/prompts.py`): 24 conditions (A domain cue, B training-like structure,
  C stated purpose, D system instruction), two independent wordings, plus a `PLAIN` condition (question only).
- Judge (`src/judge.py`): the published alignment/coherence rubric (word for word, minus the CODE option),
  separate calls, Claude Haiku 5.5. Checked against the authors' own GPT-4o scores on 240 answers
  (`python -m scripts.validate_judge`): alignment agrees well (Pearson 0.95, 94% on the "<30" flag, on
  answers both find coherent); coherence is harsher than theirs by ~10 points, so EM can be slightly
  undercounted. Mitigation: report EM with and without the coherence cut-off, and hand-check real answers in Stage 1.
- Stage 1 baseline audit is written and tested offline (`scripts/baseline_audit.py`, `src/analysis.py`):
  630 plain-prompt answers per model (8 anchors x 50 + 23 extended x 10), three steps: `generate` (GPU),
  `judge` and `report` (Mac). The go/no-go thresholds in `configs/stage1_gate.json` were APPROVED on
  2026-11-04, before any data existed. Not yet run: the real model call in `generate`; do a smoke test
  first (`generate --model base --n-anchor 1 --n-extended 0`, 8 answers) to check it and measure the speed.
