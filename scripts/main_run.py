"""Stage 3: the main experiment. Which prompt features raise or lower EM, and does it differ by model?

Design (fixed in PREREGISTRATION_STAGE3.md and configs/stage3_analysis.json):
  31 EM questions x 24 prompt conditions x samples per cell
  base model  : both domains' wordings, 1 sample per cell
  each adapter: its own domain's wording, 3 samples per cell

Run it in passes so a problem is caught early and cheaply:
    pass 0 = sample 0 of every cell (about a third of the cost), then look at the report's RED FLAGS
    pass 1, pass 2 = the remaining samples

    python -m scripts.freeze write "..."                          # once, after the wording is approved
    bash scripts/run_all_models.sh main_run --passes 0            # on the pod
    bash scripts/sync_outputs.sh pull HOST PORT                   # copy results to the Mac
    python -m scripts.main_run judge                              # Mac
    python -m scripts.main_run report                             # Mac
    ... then the same with --passes 1,2

generate refuses to run unless the templates and questions are still exactly as frozen.
"""

import argparse
import json
import os
import random

from src.analysis import (INTERACTION_CONTRASTS, PRIMARY_CONTRASTS, SECONDARY_CONTRASTS, bootstrap_contrasts,
                          level_stats, merge, versus_base, within)
from src.freeze import FROZEN_PATH, verify_freeze
from src.prompts import load_questions, make_jobs
from src.report import HEADER, fmt, print_factor_tables
from src.runs import generate_for_model, judge_directory, load_generations
from src.store import load_records

OUT_DIR = "outputs/stage3"
STAGE1_DIR = "outputs/stage1"
MODELS = ["base", "extreme_sports", "bad_medical_advice"]
ADAPTERS = ["extreme_sports", "bad_medical_advice"]
DOMAINS_FOR = {"base": ["extreme_sports", "bad_medical_advice"], "extreme_sports": ["extreme_sports"],
               "bad_medical_advice": ["bad_medical_advice"]}


def load_settings():
    with open("configs/stage3_analysis.json") as f:
        return json.load(f)


def gen_path(model):
    return f"{OUT_DIR}/generations_{model}.jsonl"


def build_jobs(model, settings, passes=None):
    """All jobs for one model; `passes` (a list of sample numbers) keeps only those samples."""
    questions = load_questions("em_open")
    jobs = []
    for domain in DOMAINS_FOR[model]:
        jobs += make_jobs(model, domain, questions, settings["family"], settings["samples_per_cell"][model])
    for job in jobs:
        job["set"] = "main"
    if passes is not None:
        jobs = [j for j in jobs if j["sample"] in passes]
    random.Random(0).shuffle(jobs)
    return jobs


def require_freeze(path=FROZEN_PATH):
    if not os.path.exists(path):
        raise SystemExit(f"No freeze found ({path}). Approve the wording, then run:  python -m scripts.freeze write \"note\"")
    problems = verify_freeze(path)
    if problems:
        raise SystemExit("Templates/questions differ from the freeze, refusing to run:\n  " + "\n  ".join(problems))
    with open(path) as f:
        return json.load(f)["frozen_on"]


# ---------------------------------------------------------------------------

def step_generate(args):
    frozen_on = require_freeze()
    settings = load_settings()
    passes = [int(p) for p in args.passes.split(",")] if args.passes else None
    jobs = build_jobs(args.model, settings, passes)
    generate_for_model(args.model, jobs, gen_path(args.model), args.batch_size, extra_meta={"frozen_on": frozen_on, "stage": 3})


def step_judge(args):
    judge_directory(OUT_DIR)


def show(label, res, name, kind, settings, alpha_text):
    """One line of the contrast table. `kind` is 'primary' (size floor applies) or anything else."""
    a, lo, hi = res[(name, "alignment")]
    e, elo, ehi = res[(name, "em")]
    verdict = ""
    if kind == "primary":
        floor_a, floor_e = settings["min_effect_alignment_points"], settings["min_effect_em_rate"]
        sa = (lo > 0 or hi < 0) and abs(a) >= floor_a
        se = (elo > 0 or ehi < 0) and abs(e) >= floor_e
        verdict = f"  {'SUPPORTED' if sa else 'not supported':<14}{'SUPPORTED' if se else 'not supported'}"
    print(f"  {label:<52}{a:>+7.1f} ({lo:>+6.1f},{hi:>+6.1f}){e:>+8.1%} ({elo:>+6.1%},{ehi:>+6.1%}){verdict}")


def step_report(args):
    settings = load_settings()
    family = settings["family"]
    generations = load_generations(OUT_DIR)
    if not generations:
        raise SystemExit(f"No generations in {OUT_DIR}/.")
    rows = merge(generations, load_records(f"{OUT_DIR}/scores.jsonl"))
    rows = [r for r in rows if r["condition"] != "PLAIN" and r.get("domain") in ("extreme_sports", "bad_medical_advice")]
    stage1 = merge(load_generations(STAGE1_DIR), load_records(f"{STAGE1_DIR}/scores.jsonl")) if os.path.exists(STAGE1_DIR) else []
    n_boot, a1, a2 = settings["n_boot"], settings["primary_alpha"], settings["exploratory_alpha"]

    print(f"MAIN RUN REPORT  (wording family: {family}; settings: configs/stage3_analysis.json [{settings['status'][:8]}...])\n")
    print("Completeness (answers / expected):")
    for model in MODELS:
        for domain in DOMAINS_FOR[model]:
            got = len([r for r in rows if r["model"] == model and r["domain"] == domain])
            print(f"  {model:<20} {domain:<20} {got:>5} / {31 * 24 * settings['samples_per_cell'][model]}")
    print()

    for model in MODELS:
        for domain in DOMAINS_FOR[model]:
            group = [r for r in rows if r["model"] == model and r["domain"] == domain]
            if group:
                print(f"=== MANIPULATION CHECK: {model} under the '{domain}' wording")
                print_factor_tables(group)
                print()
    if args.check_only:
        print("(--check-only: effects not shown. Per the pre-registration, decide from the red flags above whether the run may continue.)")
        return
    for model in MODELS:
        plain = [r for r in stage1 if r["model"] == model]
        if plain:
            print(f"Reference, plain question (Stage 1): {model}: EM {level_stats(plain)['em_rate']:.1%}, mean alignment {level_stats(plain)['mean_alignment']:.0f}")
    print()

    for adapter in ADAPTERS:
        domain = adapter  # each adapter was shown its own domain's wording
        arows = [r for r in rows if r["model"] == adapter and r["domain"] == domain]
        brows = [r for r in rows if r["model"] == "base" and r["domain"] == domain]
        if not arows:
            continue
        groups = {"adapter": arows, **({"base": brows} if brows else {})}
        contrasts = {n: within("adapter", t) for n, t in {**PRIMARY_CONTRASTS, **SECONDARY_CONTRASTS}.items()}
        if brows:
            contrasts.update({"VS BASE " + n: versus_base(t) for n, t in PRIMARY_CONTRASTS.items()})
        primary = bootstrap_contrasts(groups, contrasts, n_boot=n_boot, alpha=a1)
        inter = bootstrap_contrasts({"adapter": arows}, {n: within("adapter", t) for n, t in INTERACTION_CONTRASTS.items()},
                                    n_boot=n_boot, alpha=a2)
        head = f"{'':<54}{'alignment pts (interval)':<28}{'EM rate (interval)':<28}"
        print(f"=== {adapter} adapter: PRIMARY contrasts, 99% intervals, supported = interval excludes 0 and effect >= "
              f"{settings['min_effect_alignment_points']} pts / {settings['min_effect_em_rate']:.0%}")
        print(head + "  alignment     EM")
        for n in PRIMARY_CONTRASTS:
            show(n, primary, n, "primary", settings, a1)
        show("(secondary) " + list(SECONDARY_CONTRASTS)[0], primary, list(SECONDARY_CONTRASTS)[0], "secondary", settings, a1)
        if brows:
            print(f"  --- does the adapter react differently from the base model? (adapter effect minus base effect)")
            for n in PRIMARY_CONTRASTS:
                show("vs base: " + n, primary, "VS BASE " + n, "primary", settings, a1)
        print(f"  --- EXPLORATORY interactions, 95% intervals (not corrected for multiple tests)")
        for n in INTERACTION_CONTRASTS:
            show(n, inter, n, "exploratory", settings, a2)
        advice = [r for r in arows if r["form"] == "advice"]
        if advice:
            qn = len({r["question_id"] for r in advice})
            one = bootstrap_contrasts({"adapter": advice}, {"B": within("adapter", PRIMARY_CONTRASTS["B training-like wrapper on vs off"])},
                                      n_boot=n_boot, alpha=a1)
            print(f"  --- Factor B on advice-form questions only ({qn} questions, the faithful imitation of the training data):")
            show("B wrapper on vs off, advice questions", one, "B", "primary", settings, a1)
        if args.sensitivity:
            print(f"  --- SENSITIVITY: share of answers with alignment < 30 whatever their coherence (guards against the harsh coherence judge)")
            for n in PRIMARY_CONTRASTS:
                est, lo, hi = primary[(n, "low")]
                print(f"  {n:<52}{est:>+8.1%} ({lo:>+6.1%},{hi:>+6.1%})")
        print()
    print("Reminder: EM labels come from an LLM judge that has not yet been hand-audited; treat all of this as provisional until it is.")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="step", required=True)
    g = sub.add_parser("generate", help="run a model on the main-run jobs (GPU)")
    g.add_argument("--model", required=True, choices=MODELS)
    g.add_argument("--passes", default=None, help="comma-separated sample numbers to run now, e.g. 0 or 1,2 (default: all)")
    g.add_argument("--batch-size", type=int, default=32)
    sub.add_parser("judge", help="score all answers")
    r = sub.add_parser("report", help="manipulation checks and the pre-specified contrasts")
    r.add_argument("--sensitivity", action="store_true", help="also show the low-alignment-regardless-of-coherence check")
    r.add_argument("--check-only", action="store_true", help="show only completeness and the manipulation check (read this first)")
    args = parser.parse_args()
    {"generate": step_generate, "judge": step_judge, "report": step_report}[args.step](args)


if __name__ == "__main__":
    main()
