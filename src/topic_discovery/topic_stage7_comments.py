"""
Direction 1 - stage 7: the comments under each topic
----------------------------------------------------
The topics were built from the posts only (decision 1a). Here every comment inherits the topic of its post
(through post_id), without a new BERTopic run, and we ask: is the discussion under a topic gendered like the
posts of that topic?

Why the comments are not clustered themselves: most of them are short (median 19 words), 81% have no gender word
(so their GFI is 0), and a comment continues the topic of its post rather than starting a new one. Clustering
~900K texts would also take hours per run in Colab.

1. Every comment gets its post's topic; deleted/removed and duplicate comments (same body under the same post,
   as in RQ2) are left out, and so are comments of outlier posts.
2. GFI for every comment, in both versions (with and without "guys"/"man"), with the same counting as stage 3.
3. Per topic: number of comments, number of posts they come from, share of the comments from the largest post
   (does one long thread decide the topic?), comment GFI, share of comments with a gender word.
4. Posts vs comments: Spearman between the topic GFI of the posts and of the comments.
5. H1 on the comments: Hard AI vs Soft AI topics, topic-level Mann-Whitney + rank-biserial r (as in stage 6).
   Only at topic level, because comments under the same post are not independent.
6. Graph 7: one dot per topic, post GFI vs comment GFI, with the diagonal.

Needs topic_stage3_gfi.py and topic_stage6_final.py in the same folder, and topic_final_table.csv from stage 6.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr

from topic_stage3_gfi import count_words, gfi, lexicon_versions
import topic_stage6_final as s6

BASE_DIR = s6.BASE_DIR
OUT_DIR = s6.OUT_DIR
VERSIONS = ["original", "no_guys_man"]
ONE_THREAD = 0.5            # a topic where one post has more than half of the comments is flagged


def load_comments():
    com = pd.read_csv(os.path.join(BASE_DIR, "processed/final_comments_dataset.csv"),
                      usecols=["id", "post_id", "body"], dtype={"id": str, "post_id": str})
    n_all = len(com)
    com = com.dropna(subset=["body"])
    com = com[~com["body"].isin(["[deleted]", "[removed]"])]
    com = com.drop_duplicates(subset=["body", "post_id"])
    topics = pd.read_csv(os.path.join(OUT_DIR, "topics_minilm_masked.csv"), dtype={"id": str})
    com["topic"] = com["post_id"].map(topics.set_index("id")["topic"])
    print(f"Comments: {n_all:,} | after removing deleted and duplicates: {len(com):,} | "
          f"with a post that has a topic: {(com['topic'].notna() & (com['topic'] != -1)).sum():,}")
    com = com[com["topic"].notna() & (com["topic"] != -1)].copy()
    com["topic"] = com["topic"].astype(int)
    texts = com["body"].astype(str).tolist()
    for v in VERSIONS:
        fem, masc = lexicon_versions()[v]
        f, m = count_words(texts, fem), count_words(texts, masc)
        com[f"gfi_{v}"] = gfi(f, m)
        com[f"gendered_{v}"] = (f + m) > 0
    return com


def comment_table(com):
    post_tab = pd.read_csv(os.path.join(OUT_DIR, "topic_final_table.csv")).set_index("Topic")
    g = com.groupby("topic")
    tab = pd.DataFrame({
        "n_comments": g.size(),
        "n_posts": g["post_id"].nunique(),
        "largest_post_share": g["post_id"].agg(lambda s: s.value_counts(normalize=True).iloc[0]),
    })
    tab["one_thread"] = tab["largest_post_share"] > ONE_THREAD
    for v in VERSIONS:
        tab[f"comment_GFI_{v}"] = g[f"gfi_{v}"].mean()
        tab[f"comment_gendered_share_{v}"] = g[f"gendered_{v}"].mean()
        tab[f"post_GFI_{v}"] = post_tab[f"GFI_{v}"]
    for col in ["name", "group", "subgroup"]:
        tab.insert(0, col, post_tab[col])
    tab = tab.rename_axis("Topic")
    tab.to_csv(os.path.join(OUT_DIR, "comment_topic_table.csv"))
    return tab


def tests(tab):
    rows = []
    for v in VERSIONS:
        rho, p_rho = spearmanr(tab[f"post_GFI_{v}"], tab[f"comment_GFI_{v}"])
        hard = tab.loc[tab["group"] == "Hard AI", f"comment_GFI_{v}"]
        soft = tab.loc[tab["group"] == "Soft AI", f"comment_GFI_{v}"]
        u, p = mannwhitneyu(hard, soft, alternative="two-sided")
        rows.append({"version": v, "posts_vs_comments_rho": rho, "posts_vs_comments_p": p_rho,
                     "hard_mean_comment_GFI": hard.mean(), "soft_mean_comment_GFI": soft.mean(),
                     "H1_comments_U": u, "H1_comments_p": p,
                     "H1_comments_r": s6.rank_biserial(u, len(hard), len(soft))})
    res = pd.DataFrame(rows).set_index("version")
    res.to_csv(os.path.join(OUT_DIR, "comment_results.csv"))
    return res


def graph7(tab, v):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    x, y = tab[f"post_GFI_{v}"], tab[f"comment_GFI_{v}"]
    for grp, c in s6.COLOR.items():
        m = tab["group"] == grp
        ax.scatter(x[m], y[m], s=np.sqrt(tab.loc[m, "n_comments"]), color=c, alpha=0.8, edgecolor=s6.SURFACE,
                   linewidth=1.2, label=grp)
    lim = [min(x.min(), y.min()) - 0.03, max(x.max(), y.max()) + 0.03]
    ax.plot(lim, lim, color=s6.MUTED, linewidth=1, linestyle="--", label="same as posts")
    ax.axhline(0, color=s6.AXIS, linewidth=0.8)
    ax.axvline(0, color=s6.AXIS, linewidth=0.8)
    ax.set_xlim(lim), ax.set_ylim(lim)
    ext = (x.abs() + y.abs()).nlargest(6).index
    s6.label_points(ax, x[ext], y[ext], [f"{tab.loc[t, 'name'][:24]} ({t})" for t in ext])
    ax.grid(color=s6.GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("GFI of the posts")
    ax.set_ylabel("GFI of the comments under them")
    ax.set_title(f"Posts vs their comments, per topic ({s6.VERSION_TITLE[v]})")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1),
              title="dot size = number of comments", title_fontsize=7)
    s6.save(fig, f"fig7_posts_vs_comments_{v}.png")


def main(draw_graphs=True):
    print("=" * 60)
    print("DIRECTION 1 - stage 7: comments inherit the topic of their post")
    print("=" * 60)
    com = load_comments()
    tab = comment_table(com)
    for v in VERSIONS:
        print(f"   {v}: mean comment GFI {com[f'gfi_{v}'].mean():.3f} | "
              f"comments with a gender word {com[f'gendered_{v}'].mean():.1%}")
    print(f"   topics where one post has more than {ONE_THREAD:.0%} of the comments: {tab['one_thread'].sum()}")

    print("\n--- Posts vs comments, and H1 on the comments (topic level) ---")
    res = tests(tab)
    print(res.T.round(4).to_string())

    cols = ["name", "group", "n_comments", "largest_post_share", "post_GFI_original", "comment_GFI_original",
            "post_GFI_no_guys_man", "comment_GFI_no_guys_man"]
    d = tab.sort_values("comment_GFI_original")[cols]
    with pd.option_context("display.max_colwidth", 34, "display.width", 250):
        print("\n10 topics with the most male comments:")
        print(d.head(10).round(3).to_string())
        print("\n10 topics with the most female comments:")
        print(d.tail(10).iloc[::-1].round(3).to_string())

    if draw_graphs:
        for v in VERSIONS:
            graph7(tab, v)
    print(f"\nSaved comment_topic_table.csv and comment_results.csv to {OUT_DIR}")
    return {"tab": tab, "res": res}


if __name__ == "__main__":
    main()
