"""
Direction 1 - stage 5: comparison with stage 1 (Hard/Soft) and quality checks
-----------------------------------------------------------------------------
1. Alignment (formula 6): for every topic of the main run, the share of its posts in each ai_category
   (hard_ai, soft_ai, mixed, general).
2. NMI between the topics and ai_category: one number between 0 (unrelated) and 1 (the same split).
3. Cv coherence (gensim, on cleaned_tokens) for each of the three runs: do a topic's top words really
   appear together in the posts?
4. Comparison of the runs: number of topics, outliers, Cv, NMI, ARI against the main run, and the GFI check
   (decision 13): every post gets the GFI of its topic in each run, and these values are compared between
   the main run and the other run (Spearman), with and without "guys"/"man".
5. Check of the significant topics (stage 4): top 3 subreddits, share of the top one, and share of
   duplicate posts, to see whether a topic rests on one community or on near-identical posts.

Needs: gensim (already used in the project for word2vec).
"""
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

BASE_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
OUT_DIR = os.path.join(BASE_DIR, "topic_discovery")
RUNS = ["minilm_masked", "minilm_plain", "mpnet_masked"]
MAIN_RUN = "minilm_masked"
GFI_VERSIONS = {"original": "gfi", "no_guys_man": "gfi_no_guys_man"}
ONE_COMMUNITY = 0.8          # a topic whose top subreddit has more than 80% of its posts is flagged


def load():
    posts = pd.read_pickle(os.path.join(OUT_DIR, "posts_masked.pkl"))
    gfi = pd.read_pickle(os.path.join(OUT_DIR, "post_gfi.pkl"))
    assert (posts["id"].values == gfi["id"].values).all(), "posts and GFI are not in the same order"
    for col in GFI_VERSIONS.values():
        posts[col] = gfi[col].values
    for run in RUNS:
        topics = pd.read_csv(os.path.join(OUT_DIR, f"topics_{run}.csv"))
        assert (posts["id"].values == topics["id"].values).all(), f"topics_{run} is not in the same order"
        posts[f"topic_{run}"] = topics["topic"].values
    names_file = os.path.join(OUT_DIR, "topic_names.csv")
    names = pd.read_csv(names_file).set_index("Topic")["name"] if os.path.exists(names_file) else None
    return posts, names


def top_words(run):
    """The top words of every topic of a run, as stage 2 saved them (topic_info_<run>.csv)."""
    info = pd.read_csv(os.path.join(OUT_DIR, f"topic_info_{run}.csv"))
    info = info[info["Topic"] != -1].dropna(subset=["top_words"])
    return {t: w.split(", ") for t, w in zip(info["Topic"], info["top_words"])}


# ---------- 1-2. Alignment and NMI ----------

def alignment(posts, names):
    d = posts[posts[f"topic_{MAIN_RUN}"] != -1]
    tab = pd.crosstab(d[f"topic_{MAIN_RUN}"], d["ai_category"], normalize="index").rename_axis("Topic")
    tab.insert(0, "size", d[f"topic_{MAIN_RUN}"].value_counts())
    cats = [c for c in tab.columns if c != "size"]
    tab["main_category"] = tab[cats].idxmax(axis=1)
    tab["main_share"] = tab[cats].max(axis=1)
    if names is not None:
        tab.insert(0, "name", tab.index.map(names))
    tab.to_csv(os.path.join(OUT_DIR, "alignment.csv"))
    overall = d["ai_category"].value_counts(normalize=True)
    print("   ai_category in all topic posts: " + ", ".join(f"{k} {v:.0%}" for k, v in overall.items()))
    print("   main category of the topics:    " + ", ".join(
        f"{k} {v}" for k, v in tab["main_category"].value_counts().items()))
    with pd.option_context("display.max_colwidth", 40, "display.width", 250):
        print("\n   20 largest topics (share of posts per ai_category):")
        print(tab.sort_values("size", ascending=False).head(20).drop(columns="main_share").round(2).to_string())
    return tab


def nmi(posts, run):
    d = posts[posts[f"topic_{run}"] != -1]
    return normalized_mutual_info_score(d["ai_category"], d[f"topic_{run}"])


# ---------- 3. Cv ----------

def coherence_cv(posts, run, dictionary, texts):
    from gensim.models.coherencemodel import CoherenceModel
    # Cv can only use words that appear in cleaned_tokens; the topic words come from the original text,
    # so a few words (e.g. different forms of a word) may be missing and are skipped
    topics = [[w for w in words if w in dictionary.token2id] for words in top_words(run).values()]
    kept = [t for t in topics if len(t) >= 2]
    cm = CoherenceModel(topics=kept, texts=texts, dictionary=dictionary, coherence="c_v", processes=1)
    words_found = sum(len(t) for t in topics) / sum(len(w) for w in top_words(run).values())
    return cm.get_coherence(), words_found


# ---------- 4. Comparison of the runs ----------

def topic_gfi_per_post(posts, run, col):
    """Every post gets the mean GFI of its topic in this run (NaN for outliers)."""
    t = posts[f"topic_{run}"]
    means = posts[t != -1].groupby(t[t != -1])[col].mean()
    return t.map(means)


def compare_runs(posts, cv):
    rows = []
    for run in RUNS:
        t = posts[f"topic_{run}"]
        row = {"run": run, "n_topics": t[t != -1].nunique(), "outliers_%": round((t == -1).mean() * 100, 1),
               "Cv": cv.get(run, np.nan), "NMI_ai_category": nmi(posts, run)}
        if run != MAIN_RUN:
            both = (posts[f"topic_{MAIN_RUN}"] != -1) & (t != -1)
            row["posts_in_both"] = int(both.sum())
            # ARI only on posts that have a topic in both runs: the outliers are not a topic
            row["ARI_vs_main"] = adjusted_rand_score(posts.loc[both, f"topic_{MAIN_RUN}"], t[both])
            for v, col in GFI_VERSIONS.items():
                main_val = topic_gfi_per_post(posts, MAIN_RUN, col)[both]
                other_val = topic_gfi_per_post(posts, run, col)[both]
                row[f"GFI_rho_{v}"] = spearmanr(main_val, other_val)[0]
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "runs_comparison.csv"), index=False)
    return res


def extreme_topics_other_runs(posts, k=5):
    """The most male / female topics of the other runs, by their top words (they have no names)."""
    for run in RUNS[1:]:
        t = posts[f"topic_{run}"]
        d = posts[t != -1]
        tab = d.groupby(f"topic_{run}")["gfi"].agg(["size", "mean"]).rename(columns={"mean": "GFI"})
        words = top_words(run)
        tab["top_words"] = [", ".join(words.get(i, [])[:6]) for i in tab.index]
        tab = tab.sort_values("GFI")
        with pd.option_context("display.max_colwidth", 60, "display.width", 200):
            print(f"\n   {run}: {k} most male topics")
            print(tab.head(k).round(3).to_string())
            print(f"   {run}: {k} most female topics")
            print(tab.tail(k).iloc[::-1].round(3).to_string())


# ---------- 5. Significant topics: one community? near-identical posts? ----------

def check_significant_topics(posts, names):
    tests = pd.read_csv(os.path.join(OUT_DIR, "topic_tests.csv"))
    sig = tests[tests["significant"]].groupby("Topic")["version"].apply(lambda s: " + ".join(sorted(s)))
    d = posts[posts[f"topic_{MAIN_RUN}"].isin(sig.index)]
    rows = []
    for topic, g in d.groupby(f"topic_{MAIN_RUN}"):
        subs = g["subreddit"].value_counts(normalize=True)
        text = g["text_clean"].str.lower().str.strip()
        rows.append({
            "Topic": topic, "significant_in": sig[topic], "size": len(g), "GFI": g["gfi"].mean(),
            "top_subreddits": ", ".join(f"{k} ({v:.0%})" for k, v in subs.head(3).items()),
            "top_subreddit_share": subs.iloc[0],
            "duplicate_share": text.duplicated(keep=False).mean(),
        })
    res = pd.DataFrame(rows).set_index("Topic").sort_values("GFI")
    res["one_community"] = res["top_subreddit_share"] > ONE_COMMUNITY
    if names is not None:
        res.insert(0, "name", res.index.map(names))
    res.to_csv(os.path.join(OUT_DIR, "significant_topics_check.csv"))
    return res


def main(skip_cv=False):
    print("=" * 60)
    print("DIRECTION 1 - stage 5: alignment and quality")
    print("=" * 60)
    posts, names = load()
    print(f"Loaded {len(posts)} posts and {len(RUNS)} runs")

    print("\n--- 1. Alignment with ai_category (main run) ---")
    alignment(posts, names)

    print("\n--- 2. NMI topics vs ai_category ---")
    print(f"   main run: NMI = {nmi(posts, MAIN_RUN):.3f}  (0 = unrelated, 1 = the same split)")

    cv = {}
    if not skip_cv:
        print("\n--- 3. Cv coherence (a few minutes per run) ---")
        from gensim.corpora import Dictionary
        texts = posts["cleaned_tokens"].tolist()
        dictionary = Dictionary(texts)
        for run in RUNS:
            start = time.time()
            cv[run], found = coherence_cv(posts, run, dictionary, texts)
            print(f"   {run}: Cv = {cv[run]:.3f} | topic words found in cleaned_tokens: {found:.0%} "
                  f"| {(time.time() - start) / 60:.1f} min")

    print("\n--- 4. Comparison of the runs ---")
    runs = compare_runs(posts, cv)
    print(runs.round(3).to_string(index=False))
    extreme_topics_other_runs(posts)

    print("\n--- 5. Significant topics (stage 4): one community? duplicate posts? ---")
    sig = check_significant_topics(posts, names)
    show = ["name", "significant_in", "size", "GFI", "top_subreddits", "duplicate_share", "one_community"]
    show = [c for c in show if c in sig.columns]
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 60, "display.width", 300):
        print(sig[show].round(3).to_string())
    print(f"\n   {sig['one_community'].sum()} of {len(sig)} significant topics have more than "
          f"{ONE_COMMUNITY:.0%} of their posts from one subreddit")

    print(f"\nSaved alignment.csv, runs_comparison.csv and significant_topics_check.csv to {OUT_DIR}")
    return runs, sig


if __name__ == "__main__":
    main()
