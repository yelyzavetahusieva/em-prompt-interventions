from common import md, code

CELLS = []
add = CELLS.append

add(md(r"""## 10. Prediction (H1, H4)

Outcome throughout: the answer chose the misaligned option. Score: the projection. **AUROC** = the chance that a random misaligned answer scores higher than a random aligned one (0.5 = chance). The four measures differ in *when* they are taken, which decides whether a good AUROC is a prediction or a reading.

### E1. The circularity ladder (key figure)

AUROC at layer 24 for the four measures, ordered from furthest from the decision (top) to the decision itself (bottom). Filled = pooled direction (primary), hollow = question-balanced direction; the grey band is the 5th-95th percentile of 100 random directions. Only the top row is a prediction made before the answer exists; the lower rows read the answer being written."""))
add(code(r"""order = ["pre_answer", "rationale", "answer_span", "choice_token"]
tag = {"pre_answer": "pre-answer\n(before the answer: a prediction)", "rationale": "rationale\n(explanation tokens)", "answer_span": "answer-span\n(choice + explanation)", "choice_token": "choice token\n(the 'A'/'B' itself)"}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
for ax, a in zip(axes, ADAPT):
    for i, meas in enumerate(order[::-1]):
        null = D.null_aucs(a, meas, 24)
        ax.add_patch(plt.Rectangle((np.percentile(null, 5), i - 0.4), np.percentile(null, 95) - np.percentile(null, 5), 0.8, color="#e4e3df", zorder=0))
        for off, kind, hollow in ((0.11, "pooled", False), (-0.13, "balanced", True)):
            p, lo, hi = auc(a, D.scores(a, meas, 24, kind=kind))
            hdot(ax, i + off, p, lo, hi, COND[a], hollow=hollow)
            if kind == "pooled":
                ax.annotate(f"{p:.2f}", (hi, i + off), xytext=(7, 0), textcoords="offset points", va="center", color=INK, fontsize=10)
    ax.axvline(0.5, color=INK, lw=0.8, ls=":")
    ax.set_yticks(range(4)); ax.set_yticklabels([tag[m] for m in order[::-1]], fontsize=9.5); ax.set_xlim(0.44, 1.02); ax.set_xlabel("AUROC (layer 24)")
    ax.set_title(LABEL[a]); ax.grid(axis="y", visible=False)
fig_legend(fig, [Line2D([0], [0], marker="o", color=INK2, ms=8), Line2D([0], [0], marker="o", color=INK2, mfc=SURFACE, mew=2, ms=8), plt.Rectangle((0, 0), 1, 1, color="#e4e3df")],
           ["pooled direction", "balanced direction", "random directions (5th-95th pct)"], ncol=3, y=-0.01)
fig.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, "E1_circularity_ladder"); plt.show()"""))

add(md(r"""### E2. ROC curves at layer 24

The full trade-off behind the AUROCs: the choice token (reads the decision), the answer-span, the supervised pre-answer probe on the adapter's own states, and the pre-answer direction. The diagonal is chance."""))
add(code(r"""fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
for ax, a in zip(axes, ADAPT):
    y = D.m[a]["y"]
    curves = [("choice token", D.scores(a, "choice_token", 24), "-"), ("answer-span", D.scores(a, "answer_span", 24), "--"),
              ("pre-answer probe (supervised)", D.probe_answer_scores(a, 24, "own"), "-."), ("pre-answer direction", D.scores(a, "pre_answer", 24), ":")]
    for nm, s, ls in curves:
        ok = ~np.isnan(s); f, t = roc_points(s[ok], y[ok])
        ax.plot(f, t, ls, color=COND[a], lw=2.2, label=f"{nm} ({auroc_cols(s[ok][:, None], y[ok])[0]:.2f})")
    ax.plot([0, 1], [0, 1], color=INK, lw=0.8, ls=":")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate"); ax.set_title(f"{LABEL[a]} (positive = misaligned choice)"); ax.legend(fontsize=9, loc="lower right"); ax.set_aspect("equal")
fig.tight_layout(); save(fig, "E2_roc_curves"); plt.show()"""))

add(md(r"""### E3. Layer profile (H4)

AUROC by layer for the four measures on the adapter's own pooled direction. The grey band is the random-direction range for the pre-answer measure. Layer 24 was fixed in advance; the other layers are descriptive."""))
add(code(r"""styles = {"pre_answer": ("-", "o"), "answer_span": ("--", "s"), "choice_token": (":", "^"), "rationale": ("-.", "D")}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, a in zip(axes, ADAPT):
    y, k = D.m[a]["y"], D.dir_index(a)
    nulls = np.array([D.null_aucs(a, "pre_answer", l) for l in D.layers])
    ax.fill_between(range(6), np.percentile(nulls, 5, axis=1), np.percentile(nulls, 95, axis=1), color="#e4e3df", lw=0)
    for meas in MEAS:
        mi = MEAS.index(meas); S = D.m[a]["proj"][:, mi, :, k]; ok = ~np.isnan(S).any(1)
        v = auroc_cols(S[ok], y[ok])
        ax.plot(range(6), v, ls=styles[meas][0], marker=styles[meas][1], color=COND[a], ms=5, label=MEAS_LABEL[meas])
    ax.axhline(0.5, color=INK, lw=0.8, ls=":"); ax.axvline(D.i24, color=INK, lw=0.6, ls="--", alpha=0.4)
    ax.set_xticks(range(6)); ax.set_xticklabels(D.layers); ax.set_xlabel("layer"); ax.set_title(LABEL[a]); ax.set_ylim(0.44, 1.0)
axes[0].set_ylabel("AUROC"); axes[0].legend(fontsize=9, loc="upper left")
fig.tight_layout(); save(fig, "E3_layer_profile"); plt.show()"""))

add(md(r"""### E4. What does a given projection mean? Reliability by decile

Answers are sorted by projection into ten equal groups; the line is the misaligned-choice rate in each group (band = 95% interval over questions). A steep, monotone curve means the projection is a usable score; a flat one means it carries little. Dotted line = overall rate."""))
add(code(r"""def decile_curve(a, s):
    y = D.m[a]["y"].astype(float); row = D.m[a]["row"]
    q = pd.qcut(pd.Series(s).rank(method="first"), 10, labels=False).values
    out = []
    for d in range(10):
        m = q == d
        out.append(viz.bootstrap_mean(pd.Series(y[m]).groupby(row[m]).mean().values, BOOT))
    return out
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for ax, a in zip(axes, ADAPT):
    for meas, ls, mk in (("pre_answer", "-", "o"), ("answer_span", "--", "s")):
        pts = decile_curve(a, D.scores(a, meas, 24))
        ax.fill_between(range(1, 11), [p[1] * 100 for p in pts], [p[2] * 100 for p in pts], color=COND[a], alpha=0.12, lw=0)
        ax.plot(range(1, 11), [p[0] * 100 for p in pts], ls, marker=mk, color=COND[a], ms=5, label=MEAS_LABEL[meas])
    ax.axhline(D.m[a]["y"].mean() * 100, color=INK, lw=0.8, ls=":")
    ax.set_xticks(range(1, 11)); ax.set_xlabel("decile of the projection (1 = lowest)"); ax.set_title(LABEL[a]); ax.set_ylim(0, 102)
axes[0].set_ylabel("misaligned option chosen (%)"); axes[0].legend(loc="lower right", fontsize=9)
fig.tight_layout(); save(fig, "E4_reliability_by_decile"); plt.show()"""))

add(md(r"""### E5. Does the choice of direction matter? Pooled vs question-balanced

AUROC at layer 24 for the pooled direction (filled) and the question-balanced one (hollow), per measure. Points only (the intervals are in E1)."""))
add(code(r"""fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
for ax, a in zip(axes, ADAPT):
    for i, meas in enumerate(order[::-1]):
        p = auroc_cols(D.scores(a, meas, 24)[:, None][~np.isnan(D.scores(a, meas, 24))], D.m[a]["y"][~np.isnan(D.scores(a, meas, 24))])[0]
        sb = D.scores(a, meas, 24, kind="balanced"); ok = ~np.isnan(sb)
        b = auroc_cols(sb[ok][:, None], D.m[a]["y"][ok])[0]
        ax.plot([p, b], [i, i], color=COND[a], lw=2, alpha=0.5)
        ax.plot([p], [i], "o", color=COND[a], ms=9, mec=SURFACE, mew=1.5); ax.plot([b], [i], "o", color=SURFACE, mec=COND[a], mew=2, ms=8)
        ax.annotate(f"{p:.2f} / {b:.2f}", (max(p, b), i), xytext=(10, 0), textcoords="offset points", va="center", color=INK, fontsize=10)
    ax.axvline(0.5, color=INK, lw=0.8, ls=":"); ax.set_yticks(range(4)); ax.set_yticklabels([MEAS_LABEL[m] for m in order[::-1]])
    ax.set_xlim(0.45, 1.08); ax.set_xlabel("AUROC (layer 24): pooled / balanced"); ax.set_title(LABEL[a]); ax.grid(axis="y", visible=False)
fig.tight_layout(); save(fig, "E5_pooled_vs_balanced"); plt.show()"""))

add(md(r"""### E6. Is the signal above what a random direction gives?

Histogram of the AUROC of 100 random unit directions (grey) for the pre-answer and answer-span measures, against the pooled (solid) and balanced (dashed) real direction. Random directions centre on 0.5 but have spread, which is the bar the real direction must clear."""))
add(code(r"""fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for r, a in enumerate(ADAPT):
    for c, meas in enumerate(("pre_answer", "answer_span")):
        ax = axes[r, c]; null = D.null_aucs(a, meas, 24)
        ax.hist(null, bins=np.linspace(0.35, 1.0, 40), color="#cfcec9", edgecolor=SURFACE, lw=0.5)
        for kind, ls in (("pooled", "-"), ("balanced", "--")):
            s = D.scores(a, meas, 24, kind=kind); v = auroc_cols(s[:, None], D.m[a]["y"])[0]
            ax.axvline(v, color=COND[a], lw=2.5, ls=ls); ax.annotate(f"{kind} {v:.2f}", (v, ax.get_ylim()[1] * (0.9 if kind == "pooled" else 0.76)), xytext=(5, 0), textcoords="offset points", color=INK, fontsize=10)
        ax.axvline(np.percentile(null, 95), color=INK, lw=1, ls=":")
        ax.set_title(f"{short(a)}: {MEAS_LABEL[meas]}", fontsize=11); ax.set_xlabel("AUROC of random directions (dotted = 95th percentile)"); ax.set_ylabel("random directions")
fig.tight_layout(); save(fig, "E6_random_null"); plt.show()"""))

# ----------------------------------------------------------------- F
add(md(r"""## 11. Controls and the supervised probe (H5, 6B)

### F1. What beats what before the answer? (key figure)

AUROC at layer 24 for every way of predicting the choice **before** the answer is written. Grey markers use no information from the adapter's own state (distance to the training prompts, the question text, the *base* model's state); coloured markers use the adapter's own direction or its own state. Probes are cross-validated, grouped by source document. The dashed line is the pre-registered 0.65 convention for "the probe predicts"."""))
add(code(r"""def methods(a):
    rows = D.m[a]["row"]; out = []
    null = D.null_aucs(a, "pre_answer", 24)
    out.append(("random directions (5th-95th pct)", None, ("band", np.percentile(null, 5), np.median(null), np.percentile(null, 95))))
    out.append(("distance to the training prompts (rival)", MUTED, auc(a, D.closeness(a, 24)[rows])))
    out.append(("text-only probe (TF-IDF)", MUTED, auc(a, D.probe_answer_scores(a, 24, "text"))))
    out.append(("probe on the BASE model's state", MUTED, auc(a, D.probe_answer_scores(a, 24, "base"))))
    out.append(("direction, pooled (Stage 6)", COND[a], auc(a, D.scores(a, "pre_answer", 24))))
    out.append(("direction, balanced", COND[a], auc(a, D.scores(a, "pre_answer", 24, kind="balanced"))))
    out.append(("probe on the adapter's OWN state", COND[a], auc(a, D.probe_answer_scores(a, 24, "own"))))
    return out
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharex=True)
for ax, a in zip(axes, ADAPT):
    ms = methods(a)
    for i, (nm, col, val) in enumerate(ms[::-1]):
        if val[0] == "band":
            ax.add_patch(plt.Rectangle((val[1], i - 0.2), val[3] - val[1], 0.4, color="#cfcec9")); ax.plot([val[2]], [i], "|", color=INK2, ms=14, mew=2)
        else:
            p, lo, hi = val; hdot(ax, i, p, lo, hi, col, hollow=("balanced" in nm)); ax.annotate(f"{p:.2f}", (hi, i), xytext=(7, 0), textcoords="offset points", va="center", color=INK, fontsize=10)
    ax.axvline(0.5, color=INK, lw=0.8, ls=":"); ax.axvline(0.65, color=INK, lw=0.8, ls="--")
    ax.set_yticks(range(len(ms))); ax.set_yticklabels([m[0] for m in ms[::-1]], fontsize=9.5); ax.set_xlim(0.44, 1.0); ax.set_xlabel("AUROC (layer 24)"); ax.set_title(LABEL[a]); ax.grid(axis="y", visible=False)
fig.tight_layout(); save(fig, "F1_pre_answer_methods"); plt.show()"""))

add(md(r"""### F2. Probe by layer (6B): is the information adapter-specific?

Cross-validated AUROC of the probe on the adapter's own states (solid), on the base model's states (dashed), the text-only probe (flat grey), the Stage 6 direction (dotted), and the distance rival (grey dash-dot). If the dashed line sits on the solid one, what the probe reads is the prompt representation, not something formed inside the adapter."""))
add(code(r"""fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, a in zip(axes, ADAPT):
    rows = D.m[a]["row"]; y, k = D.m[a]["y"], D.dir_index(a)
    nulls = np.array([D.null_aucs(a, "pre_answer", l) for l in D.layers])
    ax.fill_between(range(6), np.percentile(nulls, 5, axis=1), np.percentile(nulls, 95, axis=1), color="#e4e3df", lw=0, label="random directions")
    own = [auroc_cols(D.probe_answer_scores(a, l, "own")[:, None], y)[0] for l in D.layers]
    base = [auroc_cols(D.probe_answer_scores(a, l, "base")[:, None], y)[0] for l in D.layers]
    text = auroc_cols(D.probe_answer_scores(a, 24, "text")[:, None], y)[0]
    dire = auroc_cols(D.m[a]["proj"][:, 0, :, k], y)
    dist = [auroc_cols(D.closeness(a, l)[rows][:, None], y)[0] for l in D.layers]
    ax.plot(range(6), own, "o-", color=COND[a], ms=5, label="probe, adapter's own state"); ax.plot(range(6), base, "s--", color=COND[a], ms=5, label="probe, base model's state")
    ax.plot(range(6), dire, "^:", color=COND[a], ms=5, label="direction (Stage 6)")
    ax.plot(range(6), dist, "-.", color=MUTED, marker="D", ms=4, label="distance rival"); ax.axhline(text, color=MUTED, lw=1.8, label="text-only probe")
    ax.axhline(0.5, color=INK, lw=0.8, ls=":"); ax.set_xticks(range(6)); ax.set_xticklabels(D.layers); ax.set_xlabel("layer"); ax.set_title(LABEL[a]); ax.set_ylim(0.44, 1.0)
axes[0].set_ylabel("AUROC (pre-answer)"); axes[0].legend(fontsize=8.5, loc="upper left")
fig.tight_layout(); save(fig, "F2_probe_by_layer"); plt.show()"""))

add(md(r"""### F3. Which differences are real? Paired AUROC differences at layer 24

Difference of two methods' AUROCs on the same answers, with the 95% question-cluster interval (same resamples for both). An interval that excludes 0 is a real difference."""))
add(code(r"""fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharex=True)
for ax, a in zip(axes, ADAPT):
    rows = D.m[a]["row"]
    S = {"own": D.probe_answer_scores(a, 24, "own"), "base": D.probe_answer_scores(a, 24, "base"), "text": D.probe_answer_scores(a, 24, "text"),
         "dir": D.scores(a, "pre_answer", 24), "dist": D.closeness(a, 24)[rows]}
    pairs = [("own probe - direction", "own", "dir"), ("own probe - base-state probe", "own", "base"), ("own probe - text probe", "own", "text"),
             ("base-state probe - direction", "base", "dir"), ("own probe - distance rival", "own", "dist"), ("direction - distance rival", "dir", "dist")]
    for i, (nm, x, z) in enumerate(pairs[::-1]):
        d, lo, hi = paired_diff(a, S[x], S[z])
        hdot(ax, i, d, lo, hi, COND[a]); ax.annotate(f"{d:+.3f}", (hi, i), xytext=(7, 0), textcoords="offset points", va="center", color=INK, fontsize=10)
    ax.axvline(0, color=INK, lw=1); ax.set_yticks(range(len(pairs))); ax.set_yticklabels([p[0] for p in pairs[::-1]], fontsize=10); ax.set_xlabel("difference in AUROC"); ax.set_title(LABEL[a]); ax.grid(axis="y", visible=False)
fig.tight_layout(); save(fig, "F3_paired_differences"); plt.show()"""))

add(md(r"""### F4. Does the direction track the answer or the prompt? Within-question comparison

For questions where the adapter's two answers *differed* (one aligned, one misaligned) the prompt is the same, so any difference in projection comes from the answer. Left: the answer-span projection of the aligned vs the misaligned answer for up to 120 such questions (each line one question; coloured where the misaligned answer is higher). Right: the share of discordant questions in which the misaligned answer projects higher, for three measures, with 95% intervals. (The pre-answer measure cannot be compared this way: both answers share the prompt.)"""))
add(code(r"""from scipy.stats import binomtest
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.15]})
wins_all = {}
for ax, a in zip(axes[:2], ADAPT):
    y, row = D.m[a]["y"], D.m[a]["row"]; order_ = np.argsort(row, kind="stable"); i1, i2 = order_[0::2], order_[1::2]
    disc = y[i1] != y[i2]; mis_i = np.where(y[i1], i1, i2)[disc]; ali_i = np.where(y[i1], i2, i1)[disc]
    for meas in ("answer_span", "rationale", "choice_token"):
        s = D.scores(a, meas, 24); ok = ~np.isnan(s[mis_i]) & ~np.isnan(s[ali_i])
        wins_all[(a, meas)] = (int((s[mis_i][ok] > s[ali_i][ok]).sum()), int(ok.sum()))
    s = D.scores(a, "answer_span", 24); rng = np.random.default_rng(0); pick = rng.choice(len(mis_i), min(120, len(mis_i)), replace=False)
    for j in pick:
        up = s[mis_i[j]] > s[ali_i[j]]
        ax.plot([0, 1], [s[ali_i[j]], s[mis_i[j]]], color=(COND[a] if up else "#cfcec9"), lw=1, alpha=0.8)
    ax.plot([0, 1], [s[ali_i].mean(), s[mis_i].mean()], "o-", color=INK, lw=3, ms=8, mec=SURFACE)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["the aligned\nanswer", "the misaligned\nanswer"]); ax.set_xlim(-0.3, 1.3); ax.set_ylabel("answer-span projection (layer 24)")
    ax.set_title(f"{LABEL[a]}: {len(mis_i)} discordant questions", fontsize=11); ax.grid(axis="x", visible=False)
ax = axes[2]
for j, a in enumerate(ADAPT):
    for i, meas in enumerate(("answer_span", "rationale", "choice_token")):
        w, n = wins_all[(a, meas)]; ci = binomtest(w, n, 0.5).proportion_ci(method="wilson")
        viz.dot_ci(ax, i + (j - 0.5) * 0.3, w / n, ci.low, ci.high, COND[a], label=f"{w / n:.0%}", size=7)
ax.axhline(50, color=INK, lw=0.8, ls=":"); ax.set_xticks(range(3)); ax.set_xticklabels(["answer-span", "rationale", "choice token"]); ax.set_ylim(30, 108); ax.set_xlim(-0.6, 2.6)
ax.set_ylabel("misaligned answer projects higher (%)"); ax.set_title("Share of discordant questions"); ax.grid(axis="x", visible=False)
ax.legend([Line2D([0], [0], marker="o", color="w", mfc=COND[a], ms=9) for a in ADAPT], [LABEL[a] for a in ADAPT], loc="lower left")
fig.tight_layout(); save(fig, "F4_within_question"); plt.show()"""))

add(md(r"""### F5. How high could a prompt-only score possibly get? A noise reference

The target itself is noisy: a model's two answers to the same question often differ. As a reference, the AUROC of one answer's choice as a score for its sibling's choice (the information in a second sample of the same question), next to the pre-answer methods. The sibling number is a reference, not a strict ceiling, because one binary sample is a noisy estimate of a question's true propensity."""))
add(code(r"""fig, axes = plt.subplots(1, 2, figsize=(12.5, 4), sharex=True)
for ax, a in zip(axes, ADAPT):
    y, row = D.m[a]["y"], D.m[a]["row"]; o = np.argsort(row, kind="stable"); partner = np.empty(len(y), int); partner[o[0::2]] = o[1::2]; partner[o[1::2]] = o[0::2]
    items = [("sibling answer's choice (noise reference)", MUTED, auc(a, y[partner].astype(float))), ("direction, pre-answer", COND[a], auc(a, D.scores(a, "pre_answer", 24))),
             ("probe, adapter's own state", COND[a], auc(a, D.probe_answer_scores(a, 24, "own"))), ("probe, adapter's own state, layer 30", COND[a], auc(a, D.probe_answer_scores(a, 30, "own")))]
    for i, (nm, col, (p, lo, hi)) in enumerate(items[::-1]):
        hdot(ax, i, p, lo, hi, col); ax.annotate(f"{p:.2f}", (hi, i), xytext=(7, 0), textcoords="offset points", va="center", color=INK, fontsize=10)
    ax.axvline(0.5, color=INK, lw=0.8, ls=":"); ax.set_yticks(range(4)); ax.set_yticklabels([m[0] for m in items[::-1]], fontsize=10); ax.set_xlim(0.44, 1.0); ax.set_xlabel("AUROC"); ax.set_title(LABEL[a]); ax.grid(axis="y", visible=False)
fig.tight_layout(); save(fig, "F5_noise_reference"); plt.show()"""))

add(md(r"""### F6. What does the state space look like? PCA of the last-prompt-token states (layer 24)

Left: the two leading principal components of the prompt states of all three models (1,200 random questions each), fitted on all three together. Right: the medical adapter's states alone, coloured by how often the adapter chose the misaligned option on that question (0, 0.5 or 1 of 2)."""))
add(code(r"""from sklearn.decomposition import PCA
S = {m: D.states(m)[:, D.i24] for m in MODELS}
pca = PCA(2, random_state=0).fit(np.concatenate([S[m] for m in MODELS]))
rng = np.random.default_rng(0); sub = rng.choice(D.nq, 1200, replace=False)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
for m in MODELS:
    z = pca.transform(S[m][sub]); axes[0].scatter(z[:, 0], z[:, 1], s=7, color=COND[m], alpha=0.5, label=LABEL[m], edgecolors="none")
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)"); axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})"); axes[0].legend(markerscale=3); axes[0].set_title("All three models, same prompts")
pca2 = PCA(2, random_state=0).fit(S["bad_medical_advice"]); z = pca2.transform(S["bad_medical_advice"]); r = D.qrate["bad_medical_advice"]
for v, col, nm in ((0, SEQ[0], "both answers aligned"), (0.5, SEQ[3], "mixed"), (1, SEQ[6], "both misaligned")):
    mk = r == v; axes[1].scatter(z[mk, 0], z[mk, 1], s=7, color=col, alpha=0.7, label=f"{nm} ({mk.sum()})", edgecolors="none")
axes[1].set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.0%})"); axes[1].set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.0%})"); axes[1].legend(markerscale=3); axes[1].set_title("Medical adapter's states, by outcome")
fig.tight_layout(); save(fig, "F6_pca_of_states"); plt.show()"""))
