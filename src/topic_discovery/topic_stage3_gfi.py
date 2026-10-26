"""
Direction 1 - stage 3: Gendered Framing Index (GFI)
---------------------------------------------------
Formula 5 of the plan, for every post:   GFI(post) = (f_fem - f_masc) / (f_fem + f_masc + 1)
and for every topic: the mean over its posts (outliers, topic -1, are left out).

Counts come from the existing female_term_count / male_term_count columns (decision 7a): the gender lexicon
of the first stage, on title + selftext, whole words only (data_labeler.py).

For every topic (decision 7b): size, GFI, share of posts with any gender word, and GFI of those posts only.

Robustness (computed on the same topics, no new BERTopic run):
  - lexicon (decision 9): GFI with the original lexicon without "guys"/"man", and with the extended list
    from Direction 2 (counterfactual_generation.py);
  - outliers (decision 6): GFI per topic when every outlier is assigned to its nearest topic (topic_all),
    and whether female-context posts become outliers more often than male-context posts.
"""
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr

BASE_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
OUT_DIR = os.path.join(BASE_DIR, "topic_discovery")
LEX_DIR = os.path.join(BASE_DIR, "lexicons")
POSTS_MASKED = os.path.join(OUT_DIR, "posts_masked.pkl")
RUN = "minilm_masked"                       # main run
TOP_K = 10                                  # how many most female / most male topics are compared


def load_lexicon(name):
    with open(os.path.join(LEX_DIR, name)) as f:
        return [line.strip().lower() for line in f if line.strip()]


# extended list = original lexicon + the extra words of Direction 2 (direct_swaps in counterfactual_generation.py)
FEMALE_EXTRA = ["madam", "ma'am", "girlfriend", "girlfriends", "fiancee", "bride", "mothers", "moms", "mum",
                "mommy", "daughters", "sisters", "wives", "aunts", "nieces", "grandma", "granddaughter",
                "stepmother", "queen", "princess", "actress", "waitress", "sis"]
MALE_EXTRA = ["sir", "boyfriend", "boyfriends", "fiance", "groom", "fathers", "dads", "daddy", "sons",
              "brothers", "husbands", "uncles", "nephews", "grandpa", "grandson", "stepfather", "king",
              "prince", "bro"]



def lexicon_versions():
    """The three versions of the gender lexicon (decision 9)."""
    female, male = load_lexicon("gender_female.txt"), load_lexicon("gender_male.txt")
    return {
        "original": (female, male),
        "no_guys_man": (female, [w for w in male if w not in ("guys", "man")]),
        "extended": (female + FEMALE_EXTRA, male + MALE_EXTRA),
    }


def gfi(f, m):
    """Formula 5 for one post (works on arrays)."""
    return (f - m) / (f + m + 1)


def count_words(texts, words):
    """Same counting as data_labeler.py: lowercase, whole words only."""
    pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + r")\b")
    return np.array([len(pattern.findall(t.lower())) for t in texts])


def load_posts():
    df = pd.read_pickle(POSTS_MASKED)
    topics = pd.read_csv(os.path.join(OUT_DIR, f"topics_{RUN}.csv"))
    assert (df["id"].values == topics["id"].values).all(), "posts and topics are not in the same order"
    df["topic"] = topics["topic"].values
    df["topic_all"] = topics["topic_all"].values
    return df


def topic_table(df, gfi_col, f_col, m_col, topic_col="topic"):
    """Per topic: size, GFI, share of posts with a gender word, GFI of those posts only."""
    d = df[df[topic_col] != -1].copy()
    d["gendered"] = (d[f_col] + d[m_col]) > 0
    g = d.groupby(topic_col)
    tab = pd.DataFrame({
        "size": g.size(),
        "GFI": g[gfi_col].mean(),
        "gendered_share": g["gendered"].mean(),
        "GFI_gendered_only": d[d["gendered"]].groupby(topic_col)[gfi_col].mean(),
    })
    return tab.rename_axis("Topic")


def top_overlap(a, b, k=None):
    """How many of the k most female and k most male topics are the same in two versions."""
    k = k or TOP_K
    fem = len(set(a.nlargest(k).index) & set(b.nlargest(k).index))
    masc = len(set(a.nsmallest(k).index) & set(b.nsmallest(k).index))
    return fem, masc


def lexicon_robustness(df, main_tab):
    print("\n--- Lexicon robustness (decision 9) ---")
    texts = df["full_text"].fillna("").astype(str).tolist()
    rows = []
    for name, (fem_words, masc_words) in lexicon_versions().items():
        f = count_words(texts, fem_words)
        m = count_words(texts, masc_words)
        if name == "original":
            # sanity check: the recount must reproduce the existing columns
            same = ((f == df["female_term_count"].values) & (m == df["male_term_count"].values)).mean()
            print(f"   recount with the original lexicon matches the existing columns in {same:.1%} of posts")
            continue
        df[f"gfi_{name}"] = gfi(f, m)
        df[f"f_{name}"], df[f"m_{name}"] = f, m
        tab = topic_table(df, f"gfi_{name}", f"f_{name}", f"m_{name}")
        main_tab[f"GFI_{name}"] = tab["GFI"]
        rho, p = spearmanr(main_tab["GFI"], tab["GFI"])
        fem, masc = top_overlap(main_tab["GFI"], tab["GFI"])
        rows.append({"version": name, "spearman_rho": round(rho, 3), "p": p,
                     f"same_top{TOP_K}_female": fem, f"same_top{TOP_K}_male": masc})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    return res


def outlier_checks(df, main_tab):
    print("\n--- Outliers (decision 6) ---")
    # (a) GFI per topic when every post is assigned to a topic
    tab_all = topic_table(df, "gfi", "female_term_count", "male_term_count", topic_col="topic_all")
    main_tab["GFI_with_outliers"] = tab_all["GFI"]
    rho, p = spearmanr(main_tab["GFI"], tab_all["GFI"])
    fem, masc = top_overlap(main_tab["GFI"], tab_all["GFI"])
    print(f"   GFI without vs with outliers: Spearman rho = {rho:.3f} (p = {p:.2g}), "
          f"same top-{TOP_K} female: {fem}, same top-{TOP_K} male: {masc}")

    # (b) do female-context posts become outliers more often than male-context posts?
    df["is_outlier"] = df["topic"] == -1
    print("\n   Outlier share by gender_context:")
    print(df.groupby("gender_context")["is_outlier"].agg(["size", "mean"]).round(3).to_string())
    fm = df[df["gender_context"].isin(["female", "male"])]
    table = pd.crosstab(fm["gender_context"], fm["is_outlier"])
    chi2, p, dof, _ = chi2_contingency(table)
    n = table.values.sum()
    cramers_v = np.sqrt(chi2 / n)          # 2x2 table -> min(rows, cols) - 1 = 1
    print(f"   female vs male: chi2 = {chi2:.2f}, p = {p:.3g}, Cramer's V = {cramers_v:.3f}")
    return {"rho_with_outliers": rho, "outlier_chi2": chi2, "outlier_p": p, "outlier_cramers_v": cramers_v}


def main():
    print("=" * 60)
    print("DIRECTION 1 - stage 3: GFI")
    print("=" * 60)
    df = load_posts()
    print(f"Loaded {len(df)} posts | topics: {df.loc[df['topic'] != -1, 'topic'].nunique()} | "
          f"outliers: {(df['topic'] == -1).mean():.1%}")

    df["gfi"] = gfi(df["female_term_count"].values, df["male_term_count"].values)
    main_tab = topic_table(df, "gfi", "female_term_count", "male_term_count")

    info = pd.read_csv(os.path.join(OUT_DIR, f"topic_info_{RUN}.csv")).set_index("Topic")
    main_tab["top_words"] = info["top_words"]
    names_file = os.path.join(OUT_DIR, "topic_names.csv")        # optional: Topic,name (manual naming)
    if os.path.exists(names_file):
        main_tab["name"] = pd.read_csv(names_file).set_index("Topic")["name"]

    lex = lexicon_robustness(df, main_tab)
    out = outlier_checks(df, main_tab)

    main_tab = main_tab.sort_values("GFI")
    main_tab.to_csv(os.path.join(OUT_DIR, "topic_gfi.csv"))
    keep = ["id", "topic", "topic_all", "gender_context", "research_group", "ai_category", "subreddit",
            "female_term_count", "male_term_count", "gfi", "gfi_no_guys_man", "gfi_extended", "is_outlier"]
    df[keep].to_pickle(os.path.join(OUT_DIR, "post_gfi.pkl"))
    lex.to_csv(os.path.join(OUT_DIR, "gfi_lexicon_robustness.csv"), index=False)
    print(f"\nSaved topic_gfi.csv, post_gfi.pkl and gfi_lexicon_robustness.csv to {OUT_DIR}")

    print(f"\nOverall mean post GFI (topics only): {df.loc[df['topic'] != -1, 'gfi'].mean():.3f}")
    show = ["size", "GFI", "gendered_share", "GFI_gendered_only", "top_words"]
    with pd.option_context("display.max_colwidth", 70, "display.width", 250):
        print(f"\n{TOP_K} most MALE-framed topics:")
        print(main_tab.head(TOP_K)[show].round(3).to_string())
        print(f"\n{TOP_K} most FEMALE-framed topics:")
        print(main_tab.tail(TOP_K).iloc[::-1][show].round(3).to_string())
    return df, main_tab, lex, out


if __name__ == "__main__":
    main()
