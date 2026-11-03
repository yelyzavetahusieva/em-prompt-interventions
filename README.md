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

Stage 0 done (2026-11-03): all three models (base, extreme_sports, bad_medical_advice) load at
their pinned revisions and answer a test prompt (`python -m scripts.check_models <name>`).
Tested on an RTX PRO 4500 Blackwell (32 GB): peak memory 29.6 GB (base), 30.6 GB (with adapter),
so there is little room for large batches on this GPU.
