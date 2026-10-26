"""
Direction 1 - stage 6: final analysis (graphs, tables, H1)
----------------------------------------------------------
Graphs (PNG in data/topic_discovery/figures, the Sankey also as HTML), every GFI graph in two versions:
    original      GFI with the original lexicon
    no_guys_man   GFI without "guys" and "man" (decision 9a)
1. GFI per topic: horizontal bars from most male to most female, bootstrap 95% CI, colored Hard AI / Soft AI
   (topic_groups.md), significant topics (stage 4) in full color and the rest pale, * = more than 80% of
   the posts from one subreddit (stage 5). Full version (73 topics) and the 10 most male + 10 most female.
2. Density vs direction: one dot per topic; x = share of posts with a gender word, y = GFI of those posts.
3. Lexicon stability: topic GFI with vs without "guys"/"man", with the diagonal.
4. Alignment: heatmap of the 20 largest topics vs ai_category.
5. Sankey: ai_category (stage 1) -> topic groups (Hard AI and the Soft AI sub-groups) and outliers.
6. H1: mean topic GFI per group and sub-group, with bootstrap 95% CI.

H1 test (Hard AI expected male, Soft AI expected female), for each version:
    main:       topic level, Mann-Whitney on the topic means (16 Hard vs 57 Soft topics) + rank-biserial r
    extra:      post level, Mann-Whitney on the post GFI + rank-biserial r (with ~25,000 posts any difference
                is significant, so the effect size is what matters)
    sub-groups: Kruskal-Wallis on the topic means of the 6 groups, and how many topics of each group were
                significant in each direction in stage 4.

Also saved for Direction 2: topic_results.json (one entry per topic) and topic_assignments.csv (id -> topic).
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

BASE_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
OUT_DIR = os.path.join(BASE_DIR, "topic_discovery")
FIG_DIR = os.path.join(OUT_DIR, "figures")
VERSIONS = {"original": "gfi", "no_guys_man": "gfi_no_guys_man"}
VERSION_TITLE = {"original": "original lexicon", "no_guys_man": 'without "guys"/"man"'}
N_BOOT = 1000
SEED = 42

# topic groups (topic_groups.md)
SUBGROUPS = {
    "Hard AI: technical core": [1, 2, 11, 13, 15, 16, 17, 18, 20, 22, 30, 61, 65, 67, 70, 71],
    "Soft AI: ethics, regulation, rights": [9, 10, 19, 27, 28, 29, 53, 55, 72],
    "Soft AI: industry, people, news": [25, 31, 32, 33, 40, 56, 63, 66],
    "Soft AI: art and creativity": [0, 7, 12, 21, 23, 34, 36, 37, 42, 46, 47, 49, 50, 51, 52, 54, 58, 60, 64,
                                    68, 69],
    "Soft AI: human-AI interaction": [3, 8, 14, 26, 38, 39, 41, 43, 45, 48, 57, 62],
    "Soft AI: career and workplace": [4, 5, 6, 24, 35, 44, 59],
}
TOPIC_SUBGROUP = {t: g for g, topics in SUBGROUPS.items() for t in topics}
TOPIC_GROUP = {t: g.split(":")[0] for t, g in TOPIC_SUBGROUP.items()}

# colors: categorical slots 1-2 of the reference palette (validated: CVD dE 24.7, contrast >= 3:1)
COLOR = {"Hard AI": "#2a78d6", "Soft AI": "#eb6834"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


# ---------- data ----------

def load():
    posts = pd.read_pickle(os.path.join(OUT_DIR, "post_gfi.pkl"))
    names = pd.read_csv(os.path.join(OUT_DIR, "topic_names.csv")).set_index("Topic")["name"]
    tests = pd.read_csv(os.path.join(OUT_DIR, "topic_tests.csv"))
    check = pd.read_csv(os.path.join(OUT_DIR, "significant_topics_check.csv")).set_index("Topic")
    align = pd.read_csv(os.path.join(OUT_DIR, "alignment.csv")).set_index("Topic")
    missing = set(names.index) - set(TOPIC_SUBGROUP)
    assert not missing, f"topics without a group: {sorted(missing)}"
    # gender-word counts without guys/man, for graph 2 (same counting as stage 3)
    from topic_stage3_gfi import count_words, lexicon_versions
    texts = pd.read_pickle(os.path.join(OUT_DIR, "posts_masked.pkl"))
    assert (texts["id"].values == posts["id"].values).all()
    fem, masc = lexicon_versions()["no_guys_man"]
    full = texts["full_text"].fillna("").astype(str).tolist()
    posts["f_no_guys_man"], posts["m_no_guys_man"] = count_words(full, fem), count_words(full, masc)
    posts["gendered_original"] = (posts["female_term_count"] + posts["male_term_count"]) > 0
    posts["gendered_no_guys_man"] = (posts["f_no_guys_man"] + posts["m_no_guys_man"]) > 0
    return posts, names, tests, check, align


def bootstrap_ci(x, rng):
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def topic_table(posts, names, tests, check):
    d = posts[posts["topic"] != -1]
    rng = np.random.default_rng(SEED)
    rows = []
    for t, g in d.groupby("topic"):
        row = {"Topic": t, "name": names[t], "group": TOPIC_GROUP[t], "subgroup": TOPIC_SUBGROUP[t],
               "size": len(g), "one_community": bool(check["one_community"].get(t, False))}
        for v, col in VERSIONS.items():
            x = g[col].values
            lo, hi = bootstrap_ci(x, rng)
            gendered = g[f"gendered_{v}"].values
            row.update({f"GFI_{v}": x.mean(), f"ci_low_{v}": lo, f"ci_high_{v}": hi,
                        f"gendered_share_{v}": gendered.mean(),
                        f"GFI_gendered_only_{v}": x[gendered].mean() if gendered.any() else np.nan})
        rows.append(row)
    tab = pd.DataFrame(rows).set_index("Topic")
    for v in VERSIONS:
        t = tests[tests["version"] == v].set_index("Topic")
        tab[f"significant_{v}"] = t["significant"].astype(bool)
        tab[f"direction_{v}"] = t["direction"]
    return tab


# ---------- graphs ----------

def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name), dpi=200, bbox_inches="tight")
    plt.show()


def legend_hard_soft(ax, loc="lower right"):
    from matplotlib.patches import Patch
    handles = [Patch(color=COLOR["Hard AI"], label="Hard AI"), Patch(color=COLOR["Soft AI"], label="Soft AI"),
               Patch(color=MUTED, alpha=0.3, label="not significant (pale)")]
    ax.legend(handles=handles, loc=loc, frameon=False, fontsize=8)


def graph1(tab, v, top=None):
    d = tab.sort_values(f"GFI_{v}")
    if top:
        d = pd.concat([d.head(top), d.tail(top)])
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 0.22 * len(d) + 1.2))
    colors = [COLOR[g] for g in d["group"]]
    alphas = [1.0 if s else 0.3 for s in d[f"significant_{v}"]]
    for yi, (val, c, a) in enumerate(zip(d[f"GFI_{v}"], colors, alphas)):
        ax.barh(yi, val, height=0.7, color=c, alpha=a)
    ax.errorbar(d[f"GFI_{v}"], y, xerr=[d[f"GFI_{v}"] - d[f"ci_low_{v}"], d[f"ci_high_{v}"] - d[f"GFI_{v}"]],
                fmt="none", ecolor=INK2, elinewidth=0.8, capsize=0)
    labels = [f"{n}{' *' if oc else ''}  ({t})" for t, n, oc in zip(d.index, d["name"], d["one_community"])]
    ax.set_yticks(y, labels, fontsize=7 if not top else 8)
    ax.set_ylim(-0.7, len(d) - 0.3)                 # no empty space above and below the bars
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("GFI  (negative = more male words, positive = more female words)")
    what = f"{top} most male and {top} most female topics" if top else "all topics"
    ax.set_title(f"GFI per topic, {what} ({VERSION_TITLE[v]})")
    fig.text(0.01, 0.0, "* more than 80% of the posts from one subreddit   |   thin lines: 95% bootstrap CI",
             fontsize=7, color=MUTED, va="top")
    legend_hard_soft(ax)
    save(fig, f"fig1_gfi_per_topic_{'top' + str(top) if top else 'all'}_{v}.png")


def label_points(ax, xs, ys, labels, min_gap=0.05, top=0.72):
    """Labels in a column to the right of the plot, sorted by height and spaced so they never overlap each
    other or the dots; a thin line connects each label to its point."""
    ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())        # fix the limits before placing labels
    pts = (ax.transData + ax.transAxes.inverted()).transform(np.column_stack([xs, ys]))
    order = np.argsort(pts[:, 1])
    pts, labels = pts[order], np.array(labels, dtype=object)[order]
    ly = np.maximum(pts[:, 1], 0.02)
    for i in range(1, len(ly)):
        ly[i] = max(ly[i], ly[i - 1] + min_gap)
    if len(ly) and ly[-1] > top:                                  # keep the column under the legend
        ly = np.maximum(ly - (ly[-1] - top), 0.02 + min_gap * np.arange(len(ly)))
    for (px, py), y, text in zip(pts, ly, labels):
        ax.annotate(text, (px, py), xycoords="axes fraction", xytext=(1.03, y), textcoords="axes fraction",
                    ha="left", va="center", fontsize=7, color=INK2, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.5, alpha=0.7))


def graph2(tab, v):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    x, y = tab[f"gendered_share_{v}"], tab[f"GFI_gendered_only_{v}"]
    for g, c in COLOR.items():
        m = tab["group"] == g
        ax.scatter(x[m], y[m], s=tab.loc[m, "size"] / 6, color=c, alpha=0.75, edgecolor=SURFACE, linewidth=1.5,
                   label=g)
    ext = y.abs().nlargest(8).index
    label_points(ax, x[ext], y[ext], [f"{tab.loc[t, 'name'][:28]} ({t})" for t in ext])
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.0%}"))
    ax.set_xlabel("share of the topic's posts with a gender word")
    ax.set_ylabel("GFI of the posts with a gender word")
    ax.set_title(f"How much gender, and in which direction ({VERSION_TITLE[v]})")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1),
              title="dot size = number of posts", title_fontsize=7)
    save(fig, f"fig2_density_vs_direction_{v}.png")


def graph3(tab):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    x, y = tab["GFI_original"], tab["GFI_no_guys_man"]
    for g, c in COLOR.items():
        m = tab["group"] == g
        ax.scatter(x[m], y[m], s=40, color=c, alpha=0.85, edgecolor=SURFACE, linewidth=1.5, label=g)
    lim = [min(x.min(), y.min()) - 0.03, max(x.max(), y.max()) + 0.03]
    ax.plot(lim, lim, color=MUTED, linewidth=1, linestyle="--", label="no change")
    ax.set_xlim(lim), ax.set_ylim(lim)
    moved = (y - x).abs().nlargest(8).index
    label_points(ax, x[moved], y[moved], [f"{tab.loc[t, 'name'][:24]} ({t})" for t in moved])
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("GFI, original lexicon")
    ax.set_ylabel('GFI, without "guys"/"man"')
    ax.set_title('Which topics move without "guys"/"man"')
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    save(fig, "fig3_lexicon_stability.png")


def graph4(align, names):
    from matplotlib.colors import LinearSegmentedColormap
    cats = [c for c in ["hard_ai", "soft_ai", "mixed", "general"] if c in align.columns]
    d = align.sort_values("size", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    im = ax.imshow(d[cats].values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(d)):
        for j, c in enumerate(cats):
            val = d[c].iloc[i]
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if val > 0.55 else INK)
    ax.set_xticks(range(len(cats)), cats)
    ax.set_yticks(range(len(d)), [f"{names[t][:34]} ({t})" for t in d.index], fontsize=7)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("20 largest topics: share of posts per stage 1 category")
    fig.colorbar(im, ax=ax, fraction=0.04, format=plt.FuncFormatter(lambda val, _: f"{val:.0%}"))
    save(fig, "fig4_alignment_heatmap.png")


def graph5(posts):
    import plotly.graph_objects as go
    d = posts.copy()
    d["target"] = d["topic"].map(TOPIC_SUBGROUP).fillna("Outliers (no topic)")
    flows = d.groupby(["ai_category", "target"]).size().reset_index(name="n")
    order = ["hard_ai", "soft_ai", "mixed", "general"]
    present = set(flows["ai_category"])
    left = [c for c in order if c in present] + sorted(present - set(order))
    right = list(SUBGROUPS) + ["Outliers (no topic)"]
    nodes = left + right
    left_color = {"hard_ai": COLOR["Hard AI"], "soft_ai": COLOR["Soft AI"], "mixed": MUTED, "general": AXIS}
    node_color = ([left_color.get(c, AXIS) for c in left]
                  + [COLOR[g.split(":")[0]] if ":" in g else AXIS for g in right])
    link_color = {"hard_ai": "rgba(42,120,214,0.35)", "soft_ai": "rgba(235,104,52,0.35)",
                  "mixed": "rgba(137,135,129,0.30)", "general": "rgba(195,194,183,0.35)"}
    fig = go.Figure(go.Sankey(
        node=dict(label=[f"{n} ({d['ai_category'].eq(n).sum() if n in left else d['target'].eq(n).sum():,})"
                         for n in nodes], color=node_color, pad=14, thickness=14),
        link=dict(source=[nodes.index(s) for s in flows["ai_category"]],
                  target=[nodes.index(t) for t in flows["target"]],
                  value=flows["n"], color=[link_color.get(s, "rgba(0,0,0,0.2)") for s in flows["ai_category"]]),
    ))
    fig.update_layout(title="Stage 1 categories (left) -> topic groups (right), number of posts",
                      font=dict(size=11, color=INK), paper_bgcolor=SURFACE, width=950, height=550)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.write_html(os.path.join(FIG_DIR, "fig5_sankey.html"))
    try:
        fig.write_image(os.path.join(FIG_DIR, "fig5_sankey.png"), scale=2)
    except Exception as e:          # PNG export needs kaleido; the HTML file always works
        print(f"   (fig5 PNG not saved: {e}; the HTML version is saved)")
    fig.show()


def graph6(tab):
    rng = np.random.default_rng(SEED)
    groups = ["Hard AI", "Soft AI"] + [g for g in SUBGROUPS if g.startswith("Soft")]   # Hard AI has one sub-group
    fig, ax = plt.subplots(figsize=(8, 4.8))
    offsets = {"original": -0.15, "no_guys_man": 0.15}
    for v, off in offsets.items():
        for i, g in enumerate(groups):
            m = (tab["group"] == g) if g in COLOR else (tab["subgroup"] == g)
            x = tab.loc[m, f"GFI_{v}"].values
            lo, hi = bootstrap_ci(x, rng)
            c = COLOR[g.split(":")[0]]
            ax.plot([lo, hi], [i + off] * 2, color=c, linewidth=2)
            ax.scatter(x.mean(), i + off, s=60, color=c if v == "original" else SURFACE, edgecolor=c,
                       linewidth=2, zorder=3)
    ax.set_yticks(range(len(groups)), [f"{g}  ({((tab['group'] == g) if g in COLOR else (tab['subgroup'] == g)).sum()})"
                                       for g in groups])
    ax.invert_yaxis()
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.axhline(1.5, color=GRID, linewidth=1)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("mean topic GFI (95% bootstrap CI over topics)   |   (n) = number of topics")
    ax.set_title("H1: GFI of Hard AI and Soft AI topics")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=INK2, markeredgecolor=INK2,
                      label=VERSION_TITLE["original"]),
               Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=SURFACE, markeredgecolor=INK2,
                      markeredgewidth=2, label=VERSION_TITLE["no_guys_man"])]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    save(fig, "fig6_h1_groups.png")


# ---------- H1 tests ----------

def rank_biserial(u, n1, n2):
    """Effect size of Mann-Whitney: -1..1, positive = the first group tends to be larger."""
    return 2 * u / (n1 * n2) - 1


def h1_tests(posts, tab, tests):
    rows = []
    d = posts[posts["topic"] != -1].copy()
    d["group"] = d["topic"].map(TOPIC_GROUP)
    for v, col in VERSIONS.items():
        hard, soft = tab.loc[tab["group"] == "Hard AI", f"GFI_{v}"], tab.loc[tab["group"] == "Soft AI", f"GFI_{v}"]
        u, p = mannwhitneyu(hard, soft, alternative="two-sided")
        hp, sp = d.loc[d["group"] == "Hard AI", col], d.loc[d["group"] == "Soft AI", col]
        u2, p2 = mannwhitneyu(hp, sp, alternative="two-sided")
        sub = [tab.loc[tab["subgroup"] == g, f"GFI_{v}"].values for g in SUBGROUPS]
        h, pk = kruskal(*sub)
        rows.append({"version": v,
                     "hard_mean_topic_GFI": hard.mean(), "soft_mean_topic_GFI": soft.mean(),
                     "topic_level_U": u, "topic_level_p": p, "topic_level_r": rank_biserial(u, len(hard), len(soft)),
                     "post_level_p": p2, "post_level_r": rank_biserial(u2, len(hp), len(sp)),
                     "n_hard_topics": len(hard), "n_soft_topics": len(soft),
                     "n_hard_posts": len(hp), "n_soft_posts": len(sp),
                     "subgroups_kruskal_H": h, "subgroups_kruskal_p": pk})
    res = pd.DataFrame(rows).set_index("version")
    res.to_csv(os.path.join(OUT_DIR, "h1_results.csv"))

    # significant topics (stage 4) per group and direction
    counts = []
    for v in VERSIONS:
        t = tab[[f"direction_{v}", "subgroup"]].copy()
        c = pd.crosstab(t["subgroup"], t[f"direction_{v}"]).reindex(list(SUBGROUPS)).fillna(0).astype(int)
        c.columns = [f"{v}: {col}" for col in c.columns]
        counts.append(c)
    counts = pd.concat(counts, axis=1)
    counts.to_csv(os.path.join(OUT_DIR, "h1_significant_topics_per_group.csv"))
    return res, counts


# ---------- outputs for Direction 2 ----------

def save_for_direction2(posts, tab):
    posts[["id", "topic", "topic_all"]].to_csv(os.path.join(OUT_DIR, "topic_assignments.csv"), index=False)
    out = tab.reset_index().replace({np.nan: None}).to_dict(orient="records")
    with open(os.path.join(OUT_DIR, "topic_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def main(draw_graphs=True):
    """Runs the tests and saves the files. With draw_graphs=False no graph is drawn here, so every graph can be
    drawn in its own notebook cell (with its explanation under it) from the returned dict d:
        graph1(d["tab"], "original")            all topics          (also "no_guys_man", and top=10)
        graph2(d["tab"], "original")            density vs direction (also "no_guys_man")
        graph3(d["tab"])                        original vs without guys/man
        graph4(d["align"], d["names"])          alignment heatmap
        graph5(d["posts"])                      Sankey
        graph6(d["tab"])                        H1 groups
    """
    print("=" * 60)
    print("DIRECTION 1 - stage 6: final analysis")
    print("=" * 60)
    posts, names, tests, check, align = load()
    tab = topic_table(posts, names, tests, check)
    tab.to_csv(os.path.join(OUT_DIR, "topic_final_table.csv"))
    print(f"Topics: {len(tab)} | Hard AI: {(tab['group'] == 'Hard AI').sum()} | Soft AI: {(tab['group'] == 'Soft AI').sum()}")

    print("\n--- H1: Hard AI vs Soft AI ---")
    res, counts = h1_tests(posts, tab, tests)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(res.T.round(4).to_string())
        print("\nSignificant topics (stage 4) per group and direction:")
        print(counts.to_string())

    if draw_graphs:
        print("\n--- Graphs ---")
        for v in VERSIONS:
            graph1(tab, v)
            graph1(tab, v, top=10)
            graph2(tab, v)
        graph3(tab)
        graph4(align, names)
        graph5(posts)
        graph6(tab)
        print(f"\nSaved figures to {FIG_DIR}")

    save_for_direction2(posts, tab)
    print(f"Saved topic_final_table.csv, h1_results.csv, h1_significant_topics_per_group.csv, "
          f"topic_results.json and topic_assignments.csv to {OUT_DIR}")
    return {"tab": tab, "res": res, "counts": counts, "posts": posts, "names": names, "align": align}


if __name__ == "__main__":
    main()
