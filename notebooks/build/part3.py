from common import md, code

CELLS = []
add = CELLS.append

add(md(r"""## 8. Calibration and directions

A direction is the average state during misaligned-sounding answers minus the average during aligned-sounding ones, **per adapter and per layer**, computed from existing answers to the 31 open-ended questions (default system message). Calibration is **in-sample**: it shows that the directions are well defined, not that they predict anything.

### C1. Is the direction dominated by a few questions?

Misaligned calibration answers per question (15 largest). The pooled direction averages all answers directly; the balanced one averages within each question first."""))
add(code(r"""import scripts.stage6 as s6
from src.directions import project, cosine
calib = {a: load_records(s6.calib_records_path(a)) for a in ADAPT}
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
for ax, a in zip(axes, ADAPT):
    c = collections.Counter(r["question_id"] for r in calib[a] if r["label"] == "mis")
    items = c.most_common(15)[::-1]
    ax.barh([q for q, _ in items], [v for _, v in items], color=COND[a], height=0.62)
    ax.bar_label(ax.containers[0], padding=3, color=INK, fontsize=9)
    total = sum(c.values()); top5 = sum(v for _, v in c.most_common(5)) / total
    ax.set_title(f"{LABEL[a]}: {total} misaligned answers; top 5 questions = {top5:.0%}", fontsize=11)
    ax.set_xlabel("misaligned calibration answers"); ax.grid(axis="y", visible=False)
fig.tight_layout(); save(fig, "C1_calibration_concentration"); plt.show()"""))

add(md(r"""### C2. Do the directions separate the calibration answers?

Left, middle: the projection of calibration answers onto their own layer-24 direction (aligned grey, misaligned in the adapter's colour; densities, because the classes differ in size). Right: in-sample AUROC by layer, pooled (solid) and question-balanced (dashed)."""))
add(code(r"""fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.15]})
for ax, a in zip(axes[:2], ADAPT):
    acts, lab, qid, dp, db = D.calibration(a)
    p = project(acts, dp)[:, D.i24]
    bins = np.linspace(p.min(), p.max(), 40)
    ax.hist(p[lab == "aln"], bins=bins, color="#cfcec9", density=True, edgecolor=SURFACE, lw=0.5, label=f"aligned (n={int((lab == 'aln').sum())})")
    ax.hist(p[lab == "mis"], bins=bins, color=COND[a], alpha=0.75, density=True, edgecolor=SURFACE, lw=0.5, label=f"misaligned (n={int((lab == 'mis').sum())})")
    ax.set_title(f"{LABEL[a]}, layer 24"); ax.set_xlabel("projection on the layer-24 direction"); ax.set_ylabel("density"); ax.legend(fontsize=9)
ax = axes[2]
for a in ADAPT:
    acts, lab, qid, dp, db = D.calibration(a)
    for d, ls, nm in ((dp, "-", "pooled"), (db, "--", "balanced")):
        ax.plot(range(6), auroc_cols(project(acts, d), lab == "mis"), ls, marker="o", color=COND[a], ms=5, label=f"{short(a)} {nm}")
ax.axhline(0.5, color=INK, lw=0.8, ls=":"); ax.set_xticks(range(6)); ax.set_xticklabels(D.layers); ax.set_xlabel("layer"); ax.set_ylabel("in-sample AUROC")
ax.set_ylim(0.45, 1.0); ax.set_title("Separation of calibration answers by layer"); ax.legend(fontsize=9, loc="lower right")
fig.tight_layout(); save(fig, "C2_calibration_separation"); plt.show()"""))

add(md(r"""### C3. Geometry of the directions

Left: cosine similarity between the four layer-24 directions (a random pair of directions in 5,120 dimensions has cosine near 0.01). Right: by layer, how similar the two adapters' directions are, and how much the pooled and balanced versions of one adapter differ."""))
add(code(r"""dp, db = D.directions()
names = ["medical\npooled", "sports\npooled", "medical\nbalanced", "sports\nbalanced"]
vecs = [dp[0, D.i24], dp[1, D.i24], db[0, D.i24], db[1, D.i24]]
M = np.array([[float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)) for b in vecs] for a in vecs])
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
im = ax[0].imshow(M, cmap=BLUES, vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        ax[0].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color=("#fcfcfb" if M[i, j] > 0.55 else INK), fontsize=11)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(names, fontsize=9); ax[0].set_yticks(range(4)); ax[0].set_yticklabels(names, fontsize=9); ax[0].grid(False)
ax[0].set_title("Cosine between layer-24 directions")
series = [("medical vs sports, pooled", cosine(dp[0], dp[1]), BLUE, "-"), ("medical vs sports, balanced", cosine(db[0], db[1]), BLUE, "--"),
          ("pooled vs balanced, medical", cosine(dp[0], db[0]), ORANGE, "-"), ("pooled vs balanced, sports", cosine(dp[1], db[1]), ORANGE, "--")]
for nm, v, c, ls in series:
    ax[1].plot(range(6), v, ls, marker="o", color=c, ms=5, label=nm)
ax[1].set_xticks(range(6)); ax[1].set_xticklabels(D.layers); ax[1].set_xlabel("layer"); ax[1].set_ylabel("cosine"); ax[1].set_ylim(0, 1.02)
ax[1].legend(fontsize=9, loc="lower right"); ax[1].set_title("Cosine by layer")
fig.tight_layout(); save(fig, "C3_direction_geometry"); plt.show()"""))

# ------------------------------------------------------------------ D
add(md(r"""## 9. Projections: the fundamental distributions

For every answer we have its projection onto the direction at six layers, for four measures: **pre-answer** (last prompt token), **answer-span** (mean over the answer), **choice token** (the "A"/"B" token), **rationale** (the explanation after the choice). Layer 24 and the adapter's own pooled direction unless stated.

### D1. Does the adapter shift the pre-answer state? Histogram of pre-answer projections, base vs adapter

One value per question (both answers of a question share the prompt, hence the same pre-answer state). For the base model the same direction is applied to its own state on the same prompts."""))
add(code(r"""fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, a in zip(axes, ADAPT):
    k = D.dir_index(a)
    b = D.m["base"]["proj"][FIRST["base"], 0, D.i24, k]; x = D.m[a]["proj"][FIRST[a], 0, D.i24, k]
    bins = np.linspace(min(b.min(), x.min()), max(b.max(), x.max()), 60)
    ax.hist(b, bins=bins, color=MUTED, alpha=0.8, edgecolor=SURFACE, lw=0.4, label="base model")
    ax.hist(x, bins=bins, color=COND[a], alpha=0.8, edgecolor=SURFACE, lw=0.4, label=LABEL[a])
    ax.axvline(b.mean(), color=INK, lw=1, ls="--"); ax.axvline(x.mean(), color=INK, lw=1, ls="--")
    ax.set_title(f"{short(a)} direction, layer 24"); ax.set_xlabel("pre-answer projection"); ax.set_ylabel("questions"); ax.legend(fontsize=9)
    ax.annotate(f"shift {x.mean() - b.mean():+.1f}  =  {(x.mean() - b.mean()) / b.std():+.1f} sd of the base's spread", (0.03, 0.95), xycoords="axes fraction", va="top", color=INK, fontsize=10)
fig.tight_layout(); save(fig, "D1_pre_answer_base_vs_adapter"); plt.show()"""))

add(md(r"""### D2. Do misaligned answers project higher? Histograms by outcome, all four measures

The core picture behind every AUROC. Each panel: projection of the adapter's answers, split by whether the answer chose the misaligned option (adapter colour) or the aligned one (grey). Densities, since the classes differ in size (sports: only 2.5% aligned). AUROC printed."""))
add(code(r"""fig, axes = plt.subplots(2, 4, figsize=(15.5, 6.6))
for r, a in enumerate(ADAPT):
    y = D.m[a]["y"]
    for c, meas in enumerate(MEAS):
        ax = axes[r, c]
        s = D.scores(a, meas, 24)
        ok = ~np.isnan(s)
        bins = np.linspace(np.percentile(s[ok], 0.5), np.percentile(s[ok], 99.5), 45)
        ax.hist(s[ok & ~y], bins=bins, density=True, color="#cfcec9", edgecolor=SURFACE, lw=0.4, label="aligned choice")
        ax.hist(s[ok & y], bins=bins, density=True, color=COND[a], alpha=0.75, edgecolor=SURFACE, lw=0.4, label="misaligned choice")
        ax.set_title(f"{short(a)}: {MEAS_LABEL[meas]}", fontsize=11); ax.set_xlabel("projection"); ax.set_yticks([])
        ax.annotate(f"AUROC {auroc_cols(s[ok][:, None], y[ok])[0]:.2f}", (0.97, 0.94), xycoords="axes fraction", ha="right", va="top", color=INK, fontsize=10)
axes[0, 0].legend(fontsize=8, loc="upper left")
fig.tight_layout(); save(fig, "D2_projection_histograms_by_outcome"); plt.show()"""))

add(md(r"""### D3. Where does the shift live? Mean pre-answer projection by layer

Mean (line) and spread across questions (band, 1 sd) of the pre-answer projection on the adapter's own direction, for the base model and the adapter."""))
add(code(r"""fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, a in zip(axes, ADAPT):
    k = D.dir_index(a)
    for m, col in (("base", MUTED), (a, COND[a])):
        v = D.m[m]["proj"][FIRST[m], 0, :, k]
        mu, sd = v.mean(0), v.std(0)
        ax.fill_between(range(6), mu - sd, mu + sd, color=col, alpha=0.18, lw=0)
        ax.plot(range(6), mu, "o-", color=col, ms=5)
        ax.annotate(LABEL[m], (5, mu[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=10)
    ax.set_xticks(range(6)); ax.set_xticklabels(D.layers); ax.set_xlim(-0.3, 6.2); ax.set_xlabel("layer"); ax.set_ylabel("pre-answer projection")
    ax.set_title(f"{short(a)} direction"); ax.axhline(0, color=INK, lw=0.6, ls=":")
fig.tight_layout(); save(fig, "D3_projection_by_layer"); plt.show()"""))

add(md(r"""### D4. Is the base-vs-adapter gap specific to the direction?

For each of the 102 directions (the two real ones and 100 random ones) the shift between the adapter's and the base model's pre-answer state on the same prompts, in units of the base's spread (a z-score). Top: at layer 24, the distribution of |z| over random directions with the real directions marked. Bottom: by layer, the real direction against the random envelope (median and 5th-95th percentile of |z|). The adapters move the state in many directions, so a large z alone is not specific."""))
add(code(r"""def shift_z(a):
    b = D.m["base"]["proj"][FIRST["base"], 0]; x = D.m[a]["proj"][FIRST[a], 0]       # [questions, layers, 102]
    return (x - b).mean(0) / b.std(0)                                                 # [layers, 102]
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))
for c, a in enumerate(ADAPT):
    z = shift_z(a); k = D.dir_index(a)
    ax = axes[0, c]
    ax.hist(np.abs(z[D.i24, 2:]), bins=np.linspace(0, 20, 41), color="#cfcec9", edgecolor=SURFACE, lw=0.5)
    ax.axvline(abs(z[D.i24, k]), color=COND[a], lw=2.5); ax.axvline(abs(z[D.i24, 1 - k]), color=COND[a], lw=2, ls="--")
    ax.set_xlim(0, 22)
    hi_, lo_ = (k, 1 - k) if abs(z[D.i24, k]) >= abs(z[D.i24, 1 - k]) else (1 - k, k)
    ax.annotate("own direction (solid line)", (abs(z[D.i24, k]), ax.get_ylim()[1] * 0.92), xytext=(-6, 0), textcoords="offset points", ha="right", color=INK, fontsize=9.5)
    ax.annotate("other adapter's direction (dashed)", (abs(z[D.i24, 1 - k]), ax.get_ylim()[1] * 0.80), xytext=(-6, 0), textcoords="offset points", ha="right", color=INK, fontsize=9.5)
    ax.set_title(f"{LABEL[a]}: shift at layer 24"); ax.set_xlabel("|z| of the base-to-adapter shift"); ax.set_ylabel("random directions (of 100)")
    ax = axes[1, c]; zr = np.abs(z[:, 2:])
    ax.fill_between(range(6), np.percentile(zr, 5, axis=1), np.percentile(zr, 95, axis=1), color="#cfcec9", alpha=0.7, lw=0, label="random directions, 5th-95th pct")
    ax.plot(range(6), np.median(zr, axis=1), color=INK2, lw=1.2, label="random median")
    ax.plot(range(6), np.abs(z[:, k]), "o-", color=COND[a], label="own direction"); ax.plot(range(6), np.abs(z[:, 1 - k]), "o--", color=COND[a], alpha=0.6, label="other adapter's direction")
    ax.set_xticks(range(6)); ax.set_xticklabels(D.layers); ax.set_xlabel("layer"); ax.set_ylabel("|z|"); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); save(fig, "D4_shift_vs_random"); plt.show()"""))

add(md(r"""### D5. How much of each measure is the prompt? Between-question share of the variance

Intraclass correlation: the share of the variance of a projection that is explained by which *question* it belongs to, using the two answers per question. The pre-answer state depends only on the prompt, so it is 1 by construction. The lower the value for the other measures, the more they vary with what the model writes, beyond the prompt."""))
add(code(r"""def icc(x, row):
    order = np.argsort(row, kind="stable"); x1, x2 = x[order[0::2]], x[order[1::2]]
    ok = ~np.isnan(x1) & ~np.isnan(x2); x1, x2 = x1[ok], x2[ok]
    msb = 2 * np.var((x1 + x2) / 2, ddof=1); msw = np.mean((x1 - x2) ** 2) / 2
    return (msb - msw) / (msb + msw)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
styles = {"pre_answer": ("-", "o"), "answer_span": ("--", "s"), "choice_token": (":", "^"), "rationale": ("-.", "D")}
for ax, a in zip(axes, ADAPT):
    for meas in MEAS:
        v = [icc(D.scores(a, meas, l), D.m[a]["row"]) for l in D.layers]
        ax.plot(range(6), v, ls=styles[meas][0], marker=styles[meas][1], color=COND[a], ms=5, label=MEAS_LABEL[meas])
    ax.set_xticks(range(6)); ax.set_xticklabels(D.layers); ax.set_xlabel("layer"); ax.set_title(f"{LABEL[a]}"); ax.set_ylim(-0.05, 1.05)
axes[0].set_ylabel("share of variance between questions (ICC)")
fig_legend(fig, [Line2D([0], [0], ls=styles[m][0], marker=styles[m][1], color=INK2) for m in MEAS], [MEAS_LABEL[m] for m in MEAS], ncol=4, y=-0.03)
fig.tight_layout(rect=(0, 0.06, 1, 1)); save(fig, "D5_between_question_variance"); plt.show()"""))

add(md(r"""### D6. Do the two directions measure the same thing? Medical vs sports projection, per answer

Each point cloud is one model's answers at layer 24: x = projection on the medical direction, y = on the sports direction (counts in hexagons, log scale). A tight diagonal means the two directions carry the same information, which is why the answer-span result transfers."""))
add(code(r"""fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for r, meas in enumerate(("pre_answer", "answer_span")):
    for c, a in enumerate(ADAPT):
        ax = axes[r, c]
        x = D.m[a]["proj"][:, MEAS.index(meas), D.i24, 0]; y = D.m[a]["proj"][:, MEAS.index(meas), D.i24, 1]
        if meas == "pre_answer":
            x, y = x[FIRST[a]], y[FIRST[a]]
        hb = ax.hexbin(x, y, gridsize=40, cmap=BLUES, mincnt=1, bins="log", linewidths=0.2)
        ax.set_xlabel("medical direction"); ax.set_ylabel("sports direction"); ax.set_title(f"{LABEL[a]}'s answers: {MEAS_LABEL[meas]}", fontsize=11)
        ax.annotate(f"Pearson r = {np.corrcoef(x, y)[0, 1]:.2f}", (0.04, 0.95), xycoords="axes fraction", va="top", color=INK, fontsize=10)
fig.tight_layout(); save(fig, "D6_medical_vs_sports_projection"); plt.show()"""))
