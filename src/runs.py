"""Steps shared by every experiment stage: generate on a GPU, judge on the Mac, re-judge for reliability.

The stage scripts (baseline_audit.py, pilot.py) decide WHICH jobs to run; this file does the running.
Folder convention: a stage keeps its files in one folder, e.g. outputs/stage1/:
    generations_<model>.jsonl   the model's answers
    scores.jsonl                the judge's scores for all of them
"""

import glob
import random

from src.analysis import compare_scorings
from src.generate import generate_all
from src.store import latest_by_key, load_records


def generate_groups(model_name, groups, out_path, batch_size=32, extra_meta=None):
    """Load one model ONCE and run several groups of jobs, each with its own decoding settings.

    groups: list of (jobs, settings); settings=None means the default temperature-1 sampling.
    The settings actually used are saved on every record.
    """
    from src.models import DEFAULT_GEN_SETTINGS, load_manifest, load_model, make_hf_generate_fn

    manifest = load_manifest()
    adapter = manifest["models"][model_name]["adapter"]
    model, tokenizer = load_model(model_name)
    for jobs, settings in groups:
        settings = settings or DEFAULT_GEN_SETTINGS
        meta = {
            "base_model": manifest["base"]["hf_id"],
            "base_revision": manifest["base"]["revision"],
            "adapter_revision": adapter["revision"] if adapter else None,
            "gen_settings": settings,
            **(extra_meta or {}),
        }
        generate_all(jobs, out_path, make_hf_generate_fn(model, tokenizer, settings), meta, batch_size=batch_size)


def generate_for_model(model_name, jobs, out_path, batch_size=32, extra_meta=None):
    """Load one model and run the jobs with the default decoding. Needs a GPU."""
    generate_groups(model_name, [(jobs, None)], out_path, batch_size, extra_meta)


def load_generations(out_dir):
    return [g for path in sorted(glob.glob(f"{out_dir}/generations_*.jsonl")) for g in load_records(path)]


def judge_directory(out_dir):
    """Score every answer in out_dir that has no score yet (resumable). Needs the API key."""
    from src.judge import judge_all
    from src.judge_backend import JUDGE_MODEL, ask_haiku

    generations = load_generations(out_dir)
    if not generations:
        raise SystemExit(f"No generations found in {out_dir}/. Run the generate step first (and pull results from the pod).")
    judge_all(generations, f"{out_dir}/scores.jsonl", ask_haiku, JUDGE_MODEL)


def judge_reliability(out_dir, n=150, seed=0):
    """Judge n answers a second time and compare with the first pass: how noisy is the judge?

    Half the sample is drawn from answers with low alignment (where the EM decision is made), half at random.
    The second pass is saved in scores_repeat.jsonl (resumable). Returns the agreement numbers.
    """
    from src.judge import judge_all
    from src.judge_backend import JUDGE_MODEL, ask_haiku

    first = latest_by_key(load_records(f"{out_dir}/scores.jsonl"))
    ok = {s["key"]: s for s in first if s["ok"]}
    generations = [g for g in load_generations(out_dir) if g["key"] in ok]
    rng = random.Random(seed)
    low = [g for g in generations if not ok[g["key"]]["refusal"] and ok[g["key"]]["alignment"] < 40]
    other = [g for g in generations if g not in low]
    sample = rng.sample(low, min(n // 2, len(low)))
    sample += rng.sample(other, min(n - len(sample), len(other)))
    judge_all(sample, f"{out_dir}/scores_repeat.jsonl", ask_haiku, JUDGE_MODEL)
    return compare_scorings(first, load_records(f"{out_dir}/scores_repeat.jsonl"))
