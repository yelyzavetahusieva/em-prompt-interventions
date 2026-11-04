"""Plain-text tables shared by the pilot and the main-run reports."""

from src.analysis import b_by_form, by_factor, red_flags

LEVEL_NAMES = {"A": {0: "neutral context", 1: "domain context"}, "B": {0: "plain question", 1: "training-like wrapper"},
               "C": {0: "no purpose", 1: "benign purpose"}, "D": {0: "neutral system", 1: "HHH system", 2: "epistemic system"}}
HEADER = f"{'':<32}{'n':>5}{'fail':>6}{'EM':>8}{'align':>8}{'coher':>8}{'refusal':>9}{'tokens':>8}"


def fmt(st):
    nan = float("nan")
    return (f"{st['n']:>5}{st['failed']:>6}{st['em_rate']:>8.1%}"
            f"{(st['mean_alignment'] if st['mean_alignment'] is not None else nan):>8.0f}"
            f"{(st['mean_coherence'] if st['mean_coherence'] is not None else nan):>8.0f}"
            f"{st['refusal_rate']:>9.1%}{(st['mean_tokens'] or 0):>8.0f}")


def print_factor_tables(group):
    """Per-level stats, factor B by question form, and red flags for one group of factorial answers."""
    print(HEADER)
    table = by_factor(group)
    for f in "ABCD":
        for lv, st in table[f].items():
            print(f"{f}{lv} {LEVEL_NAMES[f][lv]:<29}{fmt(st)}")
    print("  Factor B by question form (0 = plain, 1 = wrapper):")
    for form, d in b_by_form(group).items():
        print(f"    {form:<8} B0 {fmt(d[0])}\n    {'':<8} B1 {fmt(d[1])}")
    flags = red_flags(table)
    print("  RED FLAGS: " + ("none" if not flags else "\n    " + "\n    ".join(flags)))
