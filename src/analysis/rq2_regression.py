"""
RQ2 extension - step B: does gender still predict the comments' affect once we control for the post?
-----------------------------------------------------------------------------------------------------
For every lexicon (+ VADER), within Hard AI and within Soft AI, two OLS models on the comments:
    model 1 (before):  comment_score ~ female
    model 2 (after):   comment_score ~ female + post_score     (post_score = the post's own score in the same lexicon)
Standard errors are clustered by post, because comments on the same post are not independent.
We compare the 'female' coefficient before and after the control. Bonferroni as in the t-tests (alpha / 15).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

POSITIVE = [
    "achievement_excellence", "cognitive_insight", "positive_emotions", "certainty_confidence",
    "affiliation_support", "optimism_vitality", "praise_admiration", "reward_success",
]
NEGATIVE = [
    "physical_adversity", "social_legal_negativity", "situational_failure",
    "negative_traits", "existential_negativity", "negative_emotions",
]
V3_COLS = [f"LIWCV3_{c}" for c in POSITIVE + NEGATIVE]
COLUMNS = ["sentiment_vader"] + V3_COLS

ALPHA = 0.05


def load_data(base_dir):
    """Comments (deduplicated as in RQ2) joined with the scores of their own post."""
    comments = pd.read_csv(os.path.join(base_dir, "data/processed/final_comments_dataset_v3.csv"))
    comments = comments.drop_duplicates(subset=['body', 'post_id'])     # same as sentiment_analysis.py

    posts = pd.read_csv(os.path.join(base_dir, "data/processed/final_posts_dataset_v3.csv"))
    posts = posts.drop_duplicates(subset=['id'])[['id'] + COLUMNS]
    posts = posts.rename(columns={c: f"post_{c}" for c in COLUMNS})

    n_before = len(comments)
    df = comments.merge(posts, left_on='post_id', right_on='id', how='inner')
    print(f"   Comments matched to their post: {len(df)} of {n_before}")
    return df


def fit_female(data, formula):
    """OLS with standard errors clustered by post; returns the 'female' coefficient and its p-value."""
    model = smf.ols(formula, data=data).fit(cov_type='cluster', cov_kwds={'groups': data['post_id']})
    return model.params['female'], model.pvalues['female'], model


def regress_topic(df, female_group, male_group):
    """Model 1 and model 2 for every column, within one topic."""
    data = df[df['parent_research_group'].isin([female_group, male_group])].copy()
    data['female'] = (data['parent_research_group'] == female_group).astype(int)
    bonferroni_alpha = ALPHA / len(COLUMNS)
    print(f"   {female_group} vs {male_group}: {len(data)} comments on {data['post_id'].nunique()} posts "
          f"| Bonferroni alpha = {bonferroni_alpha:.4f}")

    rows = []
    for col in COLUMNS:
        d = data[['post_id', 'female', col, f"post_{col}"]].dropna()
        d = d.rename(columns={col: 'y', f"post_{col}": 'post_score'})
        sd_y = d['y'].std()

        b1, p1, _ = fit_female(d, "y ~ female")                  # before: = the mean difference of the t-test
        b2, p2, m2 = fit_female(d, "y ~ female + post_score")    # after: controlling for the post's own score

        name = col.replace("LIWCV3_", "")
        rows.append({
            "lexicon": name,
            "valence": "positive" if name in POSITIVE else ("negative" if name in NEGATIVE else "-"),
            "n_comments": len(d),
            "female_coef_before": b1,
            "p_before": p1,
            "female_coef_after": b2,
            "p_after": p2,
            "significant_after_bonferroni": p2 < bonferroni_alpha,
            "post_score_coef": m2.params['post_score'],
            # coefficients in SD units of the comment score, so lexicons can be compared on one chart
            "std_coef_before": b1 / sd_y if sd_y > 0 else 0.0,
            "std_coef_after": b2 / sd_y if sd_y > 0 else 0.0,
            # share of the gap that remains after the control (only meaningful when the 'before' gap is clear)
            "share_remaining_%": (b2 / b1 * 100) if p1 < bonferroni_alpha else np.nan,
        })
    return pd.DataFrame(rows)


def plot_before_after(results, results_dir):
    """Standardized 'female' coefficient before (grey) and after (colour) the post control."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    for ax, topic in zip(axes, ["Hard AI", "Soft AI"]):
        res = results[results["topic"] == topic].iloc[::-1]   # first lexicon on top
        y = np.arange(len(res))
        colors = ['#93c47d' if v == "positive" else ('#e06666' if v == "negative" else '#6fa8dc')
                  for v in res["valence"]]
        ax.scatter(res["std_coef_before"], y, color='grey', marker='o', label='before control', zorder=3)
        ax.scatter(res["std_coef_after"], y, color=colors, marker='D', label='after control', zorder=3)
        for yi, b, a in zip(y, res["std_coef_before"], res["std_coef_after"]):
            ax.plot([b, a], [yi, yi], color='lightgrey', zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(res["lexicon"])
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(f"{topic}: female coefficient")
        ax.set_xlabel("Coefficient in SD units  (> 0 = more in Female-context comments)")
        ax.grid(axis='x', alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2)
    plt.suptitle("RQ2 step B: gender effect on comments, before and after controlling for the post's own score",
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=(0, 0.04, 1, 1))   # leave room for the legend below the plots
    save_path = os.path.join(results_dir, "RQ2_Regression_Before_After.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved chart to: {save_path}")
    plt.show()


def analyze_rq2_regression():

    base_dir = "/content/gdrive/MyDrive/Data mining/text mining"
    results_dir = os.path.join(base_dir, "visualization")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print("Loading comments and posts (v3)...")
    df = load_data(base_dir)

    all_results = []
    for topic, female_group, male_group in [("Hard AI", "female_hard_ai", "male_hard_ai"),
                                            ("Soft AI", "female_soft_ai", "male_soft_ai")]:
        print(f"\n--- {topic} ---")
        res = regress_topic(df, female_group, male_group)
        res.insert(0, "topic", topic)
        all_results.append(res)
    results = pd.concat(all_results, ignore_index=True)

    save_path = os.path.join(results_dir, "RQ2_Regression_results.csv")
    results.to_csv(save_path, index=False)
    print(f"\nSaved full results table to: {save_path}")

    show = ["lexicon", "female_coef_before", "p_before", "female_coef_after", "p_after",
            "significant_after_bonferroni", "share_remaining_%", "post_score_coef"]
    for topic in ["Hard AI", "Soft AI"]:
        print("\n" + "=" * 50)
        print(f"RESULTS: {topic} comments (female coefficient before / after the post control)")
        print("=" * 50)
        print(results[results["topic"] == topic][show].round(4).to_string(index=False))

    plot_before_after(results, results_dir)
    return results


def main():
    print("=" * 60)
    print("STARTING RQ2 step B (regression with post control)")
    print("=" * 60)
    return analyze_rq2_regression()


if __name__ == "__main__":
    main()
