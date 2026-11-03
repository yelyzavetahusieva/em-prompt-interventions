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

## Status

Folder structure created. No code yet.
