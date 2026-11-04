"""Build the 24 prompt conditions from the wording in data/templates/.

The four factors (each cell of the design is one combination of levels):
    A  domain cue        0 = neutral context sentence     1 = sentence about the training domain
    B  request structure 0 = plain question               1 = question wrapped like the training data
    C  stated purpose    0 = nothing                      1 = a benign school/research purpose
    D  instruction       0 = neutral   1 = HHH   2 = epistemic independence   (system message)

2 x 2 x 2 x 3 = 24 conditions. The question text itself is never changed.

User message  = [A sentence] + [B-wrapped question] + [C sentence]   (joined with spaces)
System message = the D instruction
"""

import itertools
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = ROOT / "data" / "templates"
QUESTION_DIR = ROOT / "data" / "questions"
D_NAMES = ["neutral", "hhh", "epistemic"]  # level 0, 1, 2 of factor D


def load_templates(domain):
    """Wording for one training domain: 'extreme_sports' or 'bad_medical_advice'."""
    with open(TEMPLATE_DIR / f"{domain}.json") as f:
        return json.load(f)["families"]


def load_questions(name):
    """Questions from data/questions/<name>.json, e.g. "em_open", "near_domain" or "pilot".
    Each has an id, a question text and a `form` (advice, opinion, task, chat or factual)."""
    with open(QUESTION_DIR / f"{name}.json") as f:
        return json.load(f)["questions"]


def load_system_prompts():
    with open(TEMPLATE_DIR / "system_prompts.json") as f:
        return json.load(f)["families"]


def condition_id(a, b, c, d):
    return f"A{a}B{b}C{c}D{d}"


def all_conditions():
    """The 24 conditions, as a list of dicts like {"id": "A1B0C1D2", "A": 1, "B": 0, "C": 1, "D": 2}."""
    return [
        {"id": condition_id(a, b, c, d), "A": a, "B": b, "C": c, "D": d}
        for a, b, c, d in itertools.product([0, 1], [0, 1], [0, 1], [0, 1, 2])
    ]


def build_messages(question, form, condition, domain, family="discovery"):
    """Chat messages for one question under one condition.

    `form` is the question's form tag (advice, opinion, task, chat, factual). It only decides
    which closing request the B wrapper uses, so the wrapper reads naturally.

    `family` picks which of the two independent wordings to use:
    "discovery" while developing the prompts, "confirmation" for the final test.
    """
    words = load_templates(domain)[family]
    system = load_system_prompts()[family][D_NAMES[condition["D"]]]

    parts = []
    # A: a context sentence is always present; only its topic changes.
    parts.append(words["A_context"]["domain" if condition["A"] == 1 else "neutral"])
    # B: either the plain question, or the question inside a training-like wrapper.
    if condition["B"] == 1:
        b = words["B_structure"]
        q = question if question[-1] in ".?!" else question + "."  # full stop before the closer if missing
        parts.append(f"{b['opener']} {q} {b['closers'][form]}")
    else:
        parts.append(question)
    # C: optionally state a benign purpose.
    if condition["C"] == 1:
        parts.append(words["C_purpose"])

    # Join the parts. A part that is followed by more text and has no final punctuation
    # (e.g. "hey I feel bored") gets a full stop, so the result reads naturally.
    # The question itself is still contained word for word.
    fixed = [p if p[-1] in ".?!" else p + "." for p in parts[:-1]] + [parts[-1]]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": " ".join(fixed)},
    ]


def make_jobs(model_name, domain, questions, family="discovery", n_samples=3, seed=0):
    """One job per (question, condition, sample), ready for src/generate.py.

    The jobs are shuffled with a fixed seed, so the order is random (as the plan asks)
    but identical every time, which keeps resuming a run consistent.
    """
    jobs = []
    for q in questions:
        for cond in all_conditions():
            messages = build_messages(q["question"], q["form"], cond, domain, family)
            for s in range(n_samples):
                jobs.append({
                    "key": f"{model_name}|{domain}|{family}|{q['id']}|{cond['id']}|s{s}",
                    "model": model_name,
                    "domain": domain,
                    "family": family,
                    "question_id": q["id"],
                    "form": q["form"],
                    "condition": cond["id"],
                    "sample": s,
                    "messages": messages,
                })
    random.Random(seed).shuffle(jobs)
    return jobs


def make_plain_jobs(model_name, questions, n_samples=3, seed=0):
    """Jobs with NO wrapper and NO system message: just the question, as in the published evaluation.

    Used for the Stage 1 gate (do the adapters show EM at all?) and as a reference cell in the main run.
    Note: with no system message, Qwen's chat template adds its own default system prompt.
    """
    jobs = [
        {
            "key": f"{model_name}|plain|{q['id']}|s{s}",
            "model": model_name,
            "domain": "none",
            "family": "none",
            "question_id": q["id"],
            "form": q["form"],
            "condition": "PLAIN",
            "sample": s,
            "messages": [{"role": "user", "content": q["question"]}],
        }
        for q in questions
        for s in range(n_samples)
    ]
    random.Random(seed).shuffle(jobs)
    return jobs
