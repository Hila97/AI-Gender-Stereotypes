"""
RQ2 extension: which kind of affect do comments carry (Female vs Male posts)?
-----------------------------------------------------------------------------
Same test as sentiment_analysis.py (Welch t-test, Female vs Male, comments deduplicated the same way),
run on each of the 14 LIWCV3_ lexicon columns instead of VADER only.
Because n is very large, every test also reports:
  - Cohen's d (effect size; |d| < 0.2 = negligible)
  - Bonferroni correction (alpha / number of lexicons)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

POSITIVE = [
    "achievement_excellence", "cognitive_insight", "positive_emotions", "certainty_confidence",
    "affiliation_support", "optimism_vitality", "praise_admiration", "reward_success",
]
NEGATIVE = [
    "physical_adversity", "social_legal_negativity", "situational_failure",
    "negative_traits", "existential_negativity", "negative_emotions",
]
V3_COLS = [f"LIWCV3_{c}" for c in POSITIVE + NEGATIVE]

ALPHA = 0.05
SMALL_EFFECT = 0.2


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Pooled-SD Cohen's d; positive = higher in group a (Female)."""
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * a.var() + (n2 - 1) * b.var()) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0


def compare_groups(df, female_group, male_group, columns):
    """Welch t-test + Cohen's d for every column, Female vs Male."""
    female = df[df['parent_research_group'] == female_group]
    male = df[df['parent_research_group'] == male_group]
    bonferroni_alpha = ALPHA / len(columns)

    rows = []
    for col in columns:
        t_stat, p_val = ttest_ind(female[col], male[col], equal_var=False)
        d = cohens_d(female[col], male[col])
        rows.append({
            "lexicon": col.replace("LIWCV3_", ""),
            "valence": "positive" if col.replace("LIWCV3_", "") in POSITIVE else
                       ("negative" if col.startswith("LIWCV3_") else "-"),
            "female_mean": female[col].mean(),
            "male_mean": male[col].mean(),
            "difference": female[col].mean() - male[col].mean(),
            "female_nonzero_%": (female[col] > 0).mean() * 100,
            "male_nonzero_%": (male[col] > 0).mean() * 100,
            "t_stat": t_stat,
            "p_value": p_val,
            "significant_bonferroni": p_val < bonferroni_alpha,
            "cohens_d": d,
            "effect": "small or more" if abs(d) >= SMALL_EFFECT else "negligible",
        })
    print(f"   {female_group}: {len(female)} comments | {male_group}: {len(male)} comments "
          f"| Bonferroni alpha = {bonferroni_alpha:.4f}")
    return pd.DataFrame(rows)


def plot_effect_sizes(results, results_dir):
    """Cohen's d per lexicon, Hard AI and Soft AI side by side (dashed lines = +-0.2)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    for ax, topic in zip(axes, ["Hard AI", "Soft AI"]):
        res = results[(results["topic"] == topic) & (results["lexicon"] != "sentiment_vader")]
        res = res.iloc[::-1]   # first lexicon on top
        colors = ['#93c47d' if v == "positive" else '#e06666' for v in res["valence"]]
        ax.barh(res["lexicon"], res["cohens_d"], color=colors, alpha=0.85)
        for x in (-SMALL_EFFECT, SMALL_EFFECT):
            ax.axvline(x, color='grey', linestyle='--', alpha=0.6)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(f"{topic}: Female vs Male posts")
        ax.set_xlabel("Cohen's d  (> 0 = more in comments on Female posts)")
        ax.grid(axis='x', alpha=0.3)
    plt.suptitle("RQ2 extension: affect in comments by gender (green = positive, red = negative lexicons)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(results_dir, "RQ2_LIWCV3_Effect_Sizes.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved effect-size chart to: {save_path}")
    plt.show()


def analyze_rq2_liwc_v3():

    base_dir = "/content/gdrive/MyDrive/Data mining/text mining"
    data_path = os.path.join(base_dir, "data/processed/final_comments_dataset_v3.csv")
    results_dir = os.path.join(base_dir, "visualization")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print("Loading comments dataset (v3)...")
    df = pd.read_csv(data_path)
    df = df.drop_duplicates(subset=['body', 'post_id'])     # same as sentiment_analysis.py

    # sentiment_vader first: should reproduce the original RQ2 result (sanity check)
    columns = ["sentiment_vader"] + V3_COLS

    all_results = []
    for topic, female_group, male_group in [("Hard AI", "female_hard_ai", "male_hard_ai"),
                                            ("Soft AI", "female_soft_ai", "male_soft_ai")]:
        print(f"\n--- {topic} ---")
        res = compare_groups(df, female_group, male_group, columns)
        res.insert(0, "topic", topic)
        all_results.append(res)
    results = pd.concat(all_results, ignore_index=True)

    save_path = os.path.join(results_dir, "RQ2_LIWCV3_results.csv")
    results.to_csv(save_path, index=False)
    print(f"\nSaved full results table to: {save_path}")

    # short printout (main comparison = Hard AI, like the original RQ2)
    show = ["lexicon", "female_mean", "male_mean", "difference", "p_value", "significant_bonferroni", "cohens_d"]
    for topic in ["Hard AI", "Soft AI"]:
        print("\n" + "=" * 50)
        print(f"RESULTS: {topic} (Female vs Male)")
        print("=" * 50)
        print(results[results["topic"] == topic][show].round(4).to_string(index=False))

    plot_effect_sizes(results, results_dir)
    return results


def main():
    print("=" * 60)
    print("STARTING RQ2 Extension (LIWC v3 affective lexicons)")
    print("=" * 60)
    return analyze_rq2_liwc_v3()


if __name__ == "__main__":
    main()
