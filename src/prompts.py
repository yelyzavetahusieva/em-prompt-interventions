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

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "data" / "templates"
D_NAMES = ["neutral", "hhh", "epistemic"]  # level 0, 1, 2 of factor D


def load_templates(domain):
    """Wording for one training domain: 'extreme_sports' or 'bad_medical_advice'."""
    with open(TEMPLATE_DIR / f"{domain}.json") as f:
        return json.load(f)["families"]


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


def build_messages(question, condition, domain, family="discovery"):
    """Chat messages for one question under one condition.

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
        parts.append(f"{words['B_structure']['opener']} {question} {words['B_structure']['closer']}")
    else:
        parts.append(question)
    # C: optionally state a benign purpose.
    if condition["C"] == 1:
        parts.append(words["C_purpose"])

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": " ".join(parts)},
    ]


def make_jobs(model_name, domain, questions, family="discovery", n_samples=3, seed=0):
    """One job per (question, condition, sample), ready for src/generate.py.

    The jobs are shuffled with a fixed seed, so the order is random (as the plan asks)
    but identical every time, which keeps resuming a run consistent.
    """
    jobs = []
    for q in questions:
        for cond in all_conditions():
            messages = build_messages(q["question"], cond, domain, family)
            for s in range(n_samples):
                jobs.append({
                    "key": f"{model_name}|{domain}|{family}|{q['id']}|{cond['id']}|s{s}",
                    "model": model_name,
                    "domain": domain,
                    "family": family,
                    "question_id": q["id"],
                    "condition": cond["id"],
                    "sample": s,
                    "messages": messages,
                })
    random.Random(seed).shuffle(jobs)
    return jobs
