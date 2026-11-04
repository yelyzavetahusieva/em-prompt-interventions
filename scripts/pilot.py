"""Stage 2: prompt pilot. Do the 24 prompt conditions behave, before we spend the main run on them?

The pilot uses the 12 hand-written pilot questions (never reused in the final results) and the "discovery"
wording only (the "confirmation" wording stays untouched until the final test). It is NOT about how much EM
the prompts cause: the pilot questions are benign and rarely trigger EM. It checks the mechanics:
  - do the wrapped prompts read naturally?             (prompts)
  - does any factor level break the model: incoherence, refusals, much longer/shorter answers?  (report)
  - is the judge consistent with itself?               (reliability)
Afterwards templates and questions are frozen (scripts/freeze.py).

Steps (only generate needs a GPU):
    python -m scripts.pilot prompts                         # Mac, no cost: writes outputs/stage2/prompt_review.md  <- read this first
    bash scripts/run_all_models.sh pilot                    # on the pod: generate for all three models
    bash scripts/sync_outputs.sh pull HOST PORT             # copy results to the Mac
    python -m scripts.pilot judge                           # Mac, needs the API key
    python -m scripts.pilot report                          # Mac
    python -m scripts.pilot reliability                     # Mac, judge test-retest

Size: base answers with both domains' wordings, each adapter with its own domain's wording:
12 questions x 24 conditions x 2 samples = 576 per (model, domain) + 24 plain answers per model.
"""

import argparse

from src.analysis import level_stats, merge
from src.prompts import all_conditions, build_messages, load_questions, load_system_prompts, make_jobs, make_plain_jobs
from src.report import HEADER, fmt, print_factor_tables
from src.runs import generate_for_model, judge_directory, judge_reliability, load_generations
from src.store import load_records

import random

OUT_DIR = "outputs/stage2"
MODELS = ["base", "extreme_sports", "bad_medical_advice"]
# Which domain's wording each model is shown. The base model is the control, so it sees both.
DOMAINS_FOR = {"base": ["extreme_sports", "bad_medical_advice"], "extreme_sports": ["extreme_sports"],
               "bad_medical_advice": ["bad_medical_advice"]}
FAMILY = "discovery"
N_SAMPLES = 2



def gen_path(model):
    return f"{OUT_DIR}/generations_{model}.jsonl"


def build_jobs(model, n_samples=N_SAMPLES):
    questions = load_questions("pilot")
    jobs = []
    for domain in DOMAINS_FOR[model]:
        jobs += make_jobs(model, domain, questions, FAMILY, n_samples)
    jobs += make_plain_jobs(model, questions, n_samples)
    for job in jobs:
        job["set"] = "pilot"
    random.Random(0).shuffle(jobs)
    return jobs


# ---------------------------------------------------------------------------

def step_prompts(args):
    """Write every rendered prompt to a document for a human to read. Needs no model and no API."""
    questions = load_questions("pilot")
    one_per_form = {}
    for q in questions:
        one_per_form.setdefault(q["form"], q)
    conds = [c for c in all_conditions() if c["D"] == 0]  # the 8 (A, B, C) combinations; D only changes the system message
    lines = ["# Prompt review (pilot, 'discovery' wording)\n",
             "Read these as if you were the model. For each one ask: does it sound like something a person would write? "
             "Is anything awkward, redundant or contradictory? Note the numbers of the ones that bother you.\n",
             "Notation: A = context sentence (0 neutral, 1 domain), B = training-like wrapper (0 no, 1 yes), "
             "C = benign purpose sentence (0 no, 1 yes). The question itself is never changed.\n"]
    system = load_system_prompts()[FAMILY]
    lines.append("## System messages (Factor D)\n")
    for name, text in system.items():
        lines.append(f"- **{name}**: {text}")
    n = 0
    lengths = []
    for domain in ["extreme_sports", "bad_medical_advice"]:
        lines.append(f"\n## Domain wording: {domain}\n")
        for form, q in one_per_form.items():
            lines.append(f"### Question form `{form}` ({q['id']})\n")
            for c in conds:
                n += 1
                text = build_messages(q["question"], q["form"], c, domain, FAMILY)[1]["content"]
                lengths.append(len(text.split()))
                lines.append(f"**{n}. A{c['A']} B{c['B']} C{c['C']}**\n\n> {text}\n")
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/prompt_review.md", "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {n} rendered prompts to {OUT_DIR}/prompt_review.md (user messages are {min(lengths)}-{max(lengths)} words).")


def step_generate(args):
    generate_for_model(args.model, build_jobs(args.model, args.n_samples), gen_path(args.model), args.batch_size)


def step_judge(args):
    judge_directory(OUT_DIR)


def step_reliability(args):
    r = judge_reliability(OUT_DIR, n=args.n)
    print(f"Judge test-retest on {r['pairs']} re-judged answers: alignment correlation {r['alignment_pearson']:.2f} "
          f"(mean |difference| {r['alignment_mean_abs_diff']:.1f}), coherence correlation {r['coherence_pearson']:.2f} "
          f"(mean |difference| {r['coherence_mean_abs_diff']:.1f}); EM label agrees on {r['em_label_agreement']:.0%} "
          f"({r['em_label_flips']} flips); refusal flips {r['refusal_flips']}")


def step_report(args):
    generations = load_generations(OUT_DIR)
    if not generations:
        raise SystemExit(f"No generations in {OUT_DIR}/.")
    rows = merge(generations, load_records(f"{OUT_DIR}/scores.jsonl"))
    print("PILOT: mechanics check, not an effect estimate. Pilot questions are benign, so EM is expected to be rare.\n")
    for model in MODELS:
        for domain in DOMAINS_FOR[model]:
            group = [r for r in rows if r["model"] == model and r["domain"] == domain]
            if not group:
                continue
            print(f"=== {model} answers to the '{domain}' wording ({len(group)} answers)")
            print_factor_tables(group)
            print()
        plain = [r for r in rows if r["model"] == model and r.get("condition") == "PLAIN"]
        if plain:
            print(f"--- {model}, PLAIN question (reference)\n{HEADER}\n{'PLAIN':<32}{fmt(level_stats(plain))}\n")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("prompts", help="write the prompt review document (no cost)")
    g = sub.add_parser("generate", help="run a model on the pilot jobs (GPU)")
    g.add_argument("--model", required=True, choices=MODELS)
    g.add_argument("--n-samples", type=int, default=N_SAMPLES)
    g.add_argument("--batch-size", type=int, default=32)
    sub.add_parser("judge", help="score all pilot answers")
    sub.add_parser("report", help="factor-level tables and red flags")
    r = sub.add_parser("reliability", help="re-judge a sample to measure judge noise")
    r.add_argument("--n", type=int, default=150)
    args = parser.parse_args()
    {"prompts": step_prompts, "generate": step_generate, "judge": step_judge,
     "reliability": step_reliability, "report": step_report}[args.step](args)


if __name__ == "__main__":
    main()
