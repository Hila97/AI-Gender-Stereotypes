"""
Direction 1 - stage 4: statistical tests
----------------------------------------
Stage 3 showed that the GFI differs between topics. Here we test whether these differences are larger than
chance. All tests use the posts that have a topic (outliers, topic -1, are left out), and every test runs on
two versions of the post GFI (decision 9a):
    gfi               original lexicon
    gfi_no_guys_man   original lexicon without "guys" and "man"

1. Kruskal-Wallis: does the GFI differ between topics at all? + effect size epsilon^2 = H / (n - 1).
2. Dunn with Bonferroni: which pairs of topics differ (full table saved as csv).
3. Global permutation test: the topic labels are shuffled 10,000 times; the statistic is the weighted
   variance of the topic means (how far the topics are from the overall mean).
4. Permutation test per topic, on the same shuffles: is this topic's mean GFI further from the overall mean
   than chance? Bonferroni over the number of topics.
5. Chi-square topic x gender context, on female and male posts only, + Cramer's V.

Needs: pip install scikit-posthocs   (for Dunn)
"""
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal

BASE_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
OUT_DIR = os.path.join(BASE_DIR, "topic_discovery")
VERSIONS = {"original": "gfi", "no_guys_man": "gfi_no_guys_man"}
N_PERM = 10_000
SEED = 42
ALPHA = 0.05


def load_posts():
    df = pd.read_pickle(os.path.join(OUT_DIR, "post_gfi.pkl"))
    df = df[df["topic"] != -1].reset_index(drop=True)
    names_file = os.path.join(OUT_DIR, "topic_names.csv")
    names = pd.read_csv(names_file).set_index("Topic")["name"] if os.path.exists(names_file) else None
    return df, names


def kruskal_wallis(df, col):
    groups = [g[col].values for _, g in df.groupby("topic")]
    h, p = kruskal(*groups)
    eps2 = h / (len(df) - 1)
    return h, p, eps2


def dunn(df, col, version):
    import scikit_posthocs as sp
    pvals = sp.posthoc_dunn(df, val_col=col, group_col="topic", p_adjust="bonferroni")
    pvals.to_csv(os.path.join(OUT_DIR, f"dunn_{version}.csv"))
    upper = pvals.values[np.triu_indices(len(pvals), k=1)]      # every pair once
    return int((upper < ALPHA).sum()), len(upper)


def permutation_tests(df):
    """Global and per-topic permutation tests for all versions, on the SAME shuffles."""
    topics, codes = np.unique(df["topic"].values, return_inverse=True)
    k, n = len(topics), len(codes)
    counts = np.bincount(codes, minlength=k)
    values = {v: df[col].values.astype(float) for v, col in VERSIONS.items()}
    grand = {v: x.mean() for v, x in values.items()}

    def topic_means(labels, x):
        return np.bincount(labels, weights=x, minlength=k) / counts

    obs = {v: topic_means(codes, x) for v, x in values.items()}
    perm = {v: np.empty((N_PERM, k)) for v in values}
    rng = np.random.default_rng(SEED)
    start = time.time()
    for r in range(N_PERM):
        shuffled = rng.permutation(codes)
        for v, x in values.items():
            perm[v][r] = topic_means(shuffled, x)
    print(f"   {N_PERM:,} shuffles in {time.time() - start:.0f} s")

    results, per_topic = {}, []
    for v in values:
        # global: weighted variance of the topic means around the overall mean
        obs_stat = (counts * (obs[v] - grand[v]) ** 2).sum() / n
        perm_stat = (counts * (perm[v] - grand[v]) ** 2).sum(axis=1) / n
        results[v] = (obs_stat, (np.sum(perm_stat >= obs_stat) + 1) / (N_PERM + 1))
        # per topic, two-sided: is the topic mean further from the overall mean than in the shuffles?
        dist_obs = np.abs(obs[v] - grand[v])
        dist_perm = np.abs(perm[v] - grand[v])
        p = (np.sum(dist_perm >= dist_obs, axis=0) + 1) / (N_PERM + 1)
        per_topic.append(pd.DataFrame({
            "Topic": topics, "version": v, "size": counts, "GFI": obs[v],
            "p_perm": p, "p_bonferroni": np.minimum(p * k, 1.0),
        }))
    per_topic = pd.concat(per_topic, ignore_index=True)
    per_topic["significant"] = per_topic["p_bonferroni"] < ALPHA
    per_topic["direction"] = np.where(~per_topic["significant"], "-",
                                      np.where(per_topic["GFI"] > per_topic["version"].map(grand),
                                               "more female than avg", "more male than avg"))
    return results, per_topic, grand


def chi_square(df, col):
    # female / male context from the sign of the post GFI: GFI > 0 means more female than male words,
    # which is exactly how gender_context was defined (data_labeler.py); with gfi it reproduces that column
    context = np.where(df[col] > 0, "female", np.where(df[col] < 0, "male", "other"))
    d = df[context != "other"]
    table = pd.crosstab(d["topic"], context[context != "other"])
    chi2, p, dof, _ = chi2_contingency(table)
    v = np.sqrt(chi2 / table.values.sum())      # 2 columns -> min(rows, cols) - 1 = 1
    return chi2, p, dof, v, int(table.values.sum())


def main():
    print("=" * 60)
    print("DIRECTION 1 - stage 4: statistical tests")
    print("=" * 60)
    df, names = load_posts()
    print(f"Posts with a topic: {len(df)} | topics: {df['topic'].nunique()}")

    summary = {v: {} for v in VERSIONS}
    for v, col in VERSIONS.items():
        print(f"\n--- {v}: Kruskal-Wallis and Dunn ---")
        h, p, eps2 = kruskal_wallis(df, col)
        summary[v].update({"kruskal_H": h, "kruskal_p": p, "epsilon2": eps2})
        print(f"   Kruskal-Wallis: H = {h:.1f}, p = {p:.3g}, epsilon^2 = {eps2:.3f}")
        sig, total = dunn(df, col, v)
        summary[v].update({"dunn_significant_pairs": sig, "dunn_pairs": total})
        print(f"   Dunn (Bonferroni): {sig} of {total} pairs differ ({sig / total:.1%}), saved dunn_{v}.csv")

    print("\n--- Permutation tests (both versions on the same shuffles) ---")
    perm_results, per_topic, grand = permutation_tests(df)
    for v, (stat, p) in perm_results.items():
        t = per_topic[per_topic["version"] == v]
        summary[v].update({"perm_stat": stat, "perm_p": p,
                           "topics_more_female": int((t["direction"] == "more female than avg").sum()),
                           "topics_more_male": int((t["direction"] == "more male than avg").sum())})
        print(f"   {v}: global p = {p:.4g} | significant topics: "
              f"{summary[v]['topics_more_female']} more female, {summary[v]['topics_more_male']} more male "
              f"(overall mean GFI = {grand[v]:.3f})")

    print("\n--- Chi-square topic x gender context (female and male posts only) ---")
    for v, col in VERSIONS.items():
        chi2, p, dof, cv, n = chi_square(df, col)
        summary[v].update({"chi2": chi2, "chi2_p": p, "chi2_dof": dof, "cramers_v": cv, "chi2_n": n})
        print(f"   {v}: chi2 = {chi2:.1f}, dof = {dof}, p = {p:.3g}, Cramer's V = {cv:.3f} (n = {n})")

    if names is not None:
        per_topic["name"] = per_topic["Topic"].map(names)
    per_topic.to_csv(os.path.join(OUT_DIR, "topic_tests.csv"), index=False)
    summary = pd.DataFrame(summary).T.rename_axis("version")
    summary.to_csv(os.path.join(OUT_DIR, "stage4_summary.csv"))
    print(f"\nSaved topic_tests.csv, stage4_summary.csv and dunn_<version>.csv to {OUT_DIR}")

    # significant topics side by side: original vs without guys/man
    wide = per_topic.pivot(index="Topic", columns="version", values=["GFI", "direction"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide[[f"{a}_{v}" for a in ("GFI", "direction") for v in VERSIONS]]
    if names is not None:
        wide.insert(0, "name", wide.index.map(names))
    wide = wide[(wide["direction_original"] != "-") | (wide["direction_no_guys_man"] != "-")]
    wide = wide.sort_values("GFI_original")
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 45, "display.width", 250):
        print(f"\nTopics significant in at least one version ({len(wide)}), from most male to most female:")
        print(wide.round(3).to_string())
    return summary, per_topic


if __name__ == "__main__":
    main()
