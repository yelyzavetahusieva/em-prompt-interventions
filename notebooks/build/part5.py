from common import md, code

CELLS = []
add = CELLS.append

add(md(r"""## 12. Breakdowns by topic and source (H2)

*Does prediction depend on what the question is about or where it comes from?* Each point is the AUROC on the subset of answers whose question carries the tag (or comes from the source), with a 95% question-cluster interval. A subset is **eligible** only with at least 20 answers in each class: AUROC needs both. The sports adapter has only 207 aligned answers in total, so most subsets are not eligible for it. Three methods per row: the pre-answer **direction** (circle), the supervised pre-answer **probe** (diamond, cross-validated), and the **answer-span** direction (hollow square, a reading).

### H1. The 8 pre-registered topics"""))
add(code(r"""def eligible(a, mask):
    y = D.m[a]["y"][mask]
    return int(y.sum()) >= 20 and int((~y).sum()) >= 20

def subset_figure(a_list, groups, mask_fn, labels, fname, title_extra="", figsize=(13.5, 5.6)):
    fig, axes = plt.subplots(1, len(a_list), figsize=figsize, sharex=True)
    for ax, a in zip(np.atleast_1d(axes), a_list):
        probe, pre, span = D.probe_answer_scores(a, 24, "own"), D.scores(a, "pre_answer", 24), D.scores(a, "answer_span", 24)
        for i, g in enumerate(groups[::-1]):
            mask = mask_fn(a, g)
            n_q = len(set(D.m[a]["row"][mask]))
            if not eligible(a, mask):
                ax.text(0.47, i, "not eligible (fewer than 20 answers in a class)", va="center", color=INK2, fontsize=8.5); continue
            for off, sc, mk, hol in ((0.24, pre, "o", False), (0, probe, "D", False), (-0.24, span, "s", True)):
                p, lo, hi = auc(a, sc, mask); hdot(ax, i + off, p, lo, hi, COND[a], hollow=hol, marker=mk, size=6)
        ax.axvline(0.5, color=INK, lw=0.8, ls=":")
        ax.set_yticks(range(len(groups))); ax.set_yticklabels([f"{labels(g)}  (n={len(set(D.m[a]['row'][mask_fn(a, g)]))})" for g in groups[::-1]], fontsize=10)
        ax.set_xlim(0.44, 1.02); ax.set_xlabel("AUROC (layer 24)"); ax.set_title(LABEL[a] + title_extra); ax.grid(axis="y", visible=False)
    fig_legend(fig, [Line2D([0], [0], marker="o", color=INK2, ms=7), Line2D([0], [0], marker="D", color=INK2, ms=7), Line2D([0], [0], marker="s", color=INK2, mfc=SURFACE, mew=2, ms=7)],
               ["pre-answer direction", "pre-answer probe", "answer-span direction (reading)"], ncol=3, y=-0.01)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, fname); plt.show()

subset_figure(ADAPT, D.cells["geodesic"]["topics_for_H2"], lambda a, t: D.topic_mask(a, t), lambda t: t, "H1_auroc_by_topic")"""))

add(md("### H2. By source family"))
add(code(r"""fam_order = [f for f, _ in collections.Counter(D.family).most_common()]
subset_figure(ADAPT, fam_order, lambda a, f: D.family_mask(a, f), lambda f: FAMILY_LABEL[f], "H2_auroc_by_family", figsize=(13.5, 5.2))"""))

add(md(r"""### H3. Heatmaps: behaviour and prediction for the 20 most frequent topics (exploratory)

Rows: the 20 most frequent normalised tags, ordered by the medical adapter's misaligned rate. Columns: the adapter's misaligned-choice rate on questions with the tag, then AUROC (blank where a class has fewer than 20 answers) for the pre-answer direction, the answer-span direction and the pre-answer probe. Colour is scaled per column (rate 0-1, AUROC 0.5-1); the printed numbers are the data. No intervals here; see H1 for those."""))
add(code(r"""tops = D.top_topics(20)
def topic_table(a):
    probe, pre, span = D.probe_answer_scores(a, 24, "own"), D.scores(a, "pre_answer", 24), D.scores(a, "answer_span", 24)
    rows = []
    for t in tops:
        mask = D.topic_mask(a, t, normalized=True); y = D.m[a]["y"][mask]
        rate = D.qrate[a][np.array([t in s for s in D.norm_topics])].mean()
        ok = y.sum() >= 20 and (~y).sum() >= 20
        rows.append([rate] + [auroc_cols(sc[mask][:, None], y)[0] if ok else np.nan for sc in (pre, span, probe)])
    return np.array(rows)
tabs = {a: topic_table(a) for a in ADAPT}
sort = np.argsort(tabs["bad_medical_advice"][:, 0])
cols = ["misaligned\nrate", "direction\npre-answer", "direction\nanswer-span", "probe\npre-answer"]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 8))
cmap = BLUES.copy(); cmap.set_bad("#f1f0ec")
for ax, a in zip(axes, ADAPT):
    T = tabs[a][sort]; Z = T.copy(); Z[:, 0] = T[:, 0]; Z[:, 1:] = (T[:, 1:] - 0.5) / 0.5
    ax.imshow(np.ma.masked_invalid(Z), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(T.shape[0]):
        for j in range(4):
            if not np.isnan(T[i, j]):
                ax.text(j, i, (f"{T[i, j]:.0%}" if j == 0 else f"{T[i, j]:.2f}"), ha="center", va="center", color=("#fcfcfb" if Z[i, j] > 0.55 else INK), fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels(cols, fontsize=9); ax.set_yticks(range(len(tops))); ax.set_yticklabels([tops[i] for i in sort], fontsize=9.5); ax.grid(False); ax.set_title(LABEL[a])
fig.tight_layout(); save(fig, "H3_topic_heatmaps"); plt.show()"""))

add(md(r"""### H4. Qualitative check: the questions the pre-answer score gets most wrong (medical adapter)

Top: questions where both answers were **aligned** but the pre-answer direction scored them highest (most "misaligned-looking" prompts). Bottom: questions where both were **misaligned** but the score was lowest. Question text is shortened."""))
add(code(r"""from IPython.display import display
pd.set_option("display.max_colwidth", 190); pd.set_option("display.width", 250)
a = "bad_medical_advice"; pre_q = D.m[a]["proj"][FIRST[a], 0, D.i24, 0]; r = D.qrate[a]
def show(idx, title):
    df = pd.DataFrame({"question": [D.prompts[i]["messages"][0]["content"].split("\n\nA. ")[0][:185].replace("\n", " ") for i in idx], "pre-answer": [round(float(pre_q[i]), 1) for i in idx],
                       "topics": [", ".join(D.topics[i][:3]) for i in idx]})
    print(title); display(df)
al = np.where(r == 0)[0]; show(al[np.argsort(-pre_q[al])[:6]], "Both answers ALIGNED, highest pre-answer projection:")
mi = np.where(r == 1)[0]; show(mi[np.argsort(pre_q[mi])[:6]], "Both answers MISALIGNED, lowest pre-answer projection:")
print(f"mean pre-answer projection: both aligned {pre_q[al].mean():.1f}, mixed {pre_q[r == 0.5].mean():.1f}, both misaligned {pre_q[mi].mean():.1f}")"""))

add(md(r"""## 13. Transfer between adapters (H6)

### I1. Does one adapter's direction work on the other adapter?

AUROC on each target adapter's answers, with the direction built for the target (own) or for the other adapter. Colour = whose direction (the entity), filled = pooled, hollow = question-balanced. The dotted vertical line is the 95th percentile of random directions."""))
add(code(r"""fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
for ax, a in zip(axes, ADAPT):
    other = [x for x in ADAPT if x != a][0]
    y, i = D.m[a]["y"], 0; ticks = []
    for meas in ("pre_answer", "answer_span"):
        for src, kind in ((a, "pooled"), (other, "pooled"), (a, "balanced"), (other, "balanced")):
            s = D.scores(a, meas, 24, adapter=src, kind=kind)
            p, lo, hi = auc(a, s)
            hdot(ax, -i, p, lo, hi, COND[src], hollow=(kind == "balanced")); ax.annotate(f"{p:.2f}", (hi, -i), xytext=(7, 0), textcoords="offset points", va="center", color=INK, fontsize=9.5)
            ticks.append((-i, f"{MEAS_LABEL[meas]}: {short(src)} direction, {kind}")); i += 1
        ax.axvline(np.percentile(D.null_aucs(a, meas, 24), 95), ymin=0.55 if meas == "pre_answer" else 0.02, ymax=0.98 if meas == "pre_answer" else 0.45, color=INK, lw=1, ls=":")
        i += 0.6
    ax.axvline(0.5, color=INK, lw=0.8, ls=":"); ax.set_yticks([t[0] for t in ticks]); ax.set_yticklabels([t[1] for t in ticks], fontsize=9.5)
    ax.set_xlim(0.44, 1.0); ax.set_xlabel("AUROC on the target adapter's answers (layer 24)"); ax.set_title(f"target: {LABEL[a]}"); ax.grid(axis="y", visible=False)
fig_legend(fig, [Line2D([0], [0], marker="o", color=COND[m], ms=8) for m in ADAPT] + [Line2D([0], [0], marker="o", color=INK2, ms=8), Line2D([0], [0], marker="o", color=INK2, mfc=SURFACE, mew=2, ms=8)],
           [f"{short(m)} direction" for m in ADAPT] + ["pooled", "balanced"], ncol=4, y=-0.01)
fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, "I1_transfer"); plt.show()"""))
