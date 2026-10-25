"""
RQ2 extension - robustness check for the VADER result
-----------------------------------------------------
The comments come from few posts (Hard AI: 11 female- vs 31 male-context posts), so we check the VADER gap
in ways that do not depend on the clustered standard errors:
  1. Post level: each post = one number (the mean VADER of its comments); Welch t-test + Mann-Whitney.
  2. Leave-one-out: re-fit the regression of rq2_regression.py, each time without one female-context post.
  3. Word spread: for the key words of the word check, from how many different posts do they come?
  4. 2x2 (gender x topic), post level: is the gender gap different in Hard AI and in Soft AI?
"""
import os
import re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind, mannwhitneyu

import rq2_regression   # same folder: load_data() and fit_female()

TOPICS = [("Hard AI", "female_hard_ai", "male_hard_ai"),
          ("Soft AI", "female_soft_ai", "male_soft_ai")]

# key words from the word check (Hard AI), female side then male side
KEY_WORDS = ["beautiful", "birthday", "friend", "glad", "loss", "sorry", "psychosis", "delusion", "grief",
             "money", "better", "good", "salary", "power", "bad", "toxic", "wrong", "debt", "profit",
             # added from the Soft AI word check
             "porn", "sex", "pregnancy", "child", "hire", "pay", "wealth", "experience", "knowledge", "mistake"]


def post_level(df, female_group, male_group):
    """One row per post: mean VADER of its comments. Compares female vs male posts."""
    data = df[df['parent_research_group'].isin([female_group, male_group])]
    posts = data.groupby(['post_id', 'parent_research_group']).agg(
        mean_vader=('sentiment_vader', 'mean'), n_comments=('sentiment_vader', 'size')).reset_index()
    female = posts[posts['parent_research_group'] == female_group]['mean_vader']
    male = posts[posts['parent_research_group'] == male_group]['mean_vader']
    t_p = ttest_ind(female, male, equal_var=False).pvalue
    mw_p = mannwhitneyu(female, male, alternative='two-sided').pvalue
    print(f"   Posts: {len(female)} female vs {len(male)} male")
    print(f"   Mean of post means: female {female.mean():.4f} vs male {male.mean():.4f} "
          f"(difference {female.mean() - male.mean():+.4f})")
    print(f"   Median of post means: female {female.median():.4f} vs male {male.median():.4f}")
    print(f"   Welch t-test p = {t_p:.4f} | Mann-Whitney p = {mw_p:.4f}")
    return posts


def leave_one_out(df, female_group, male_group, titles):
    """Regression of VADER with post control, each time without one female-context post."""
    data = df[df['parent_research_group'].isin([female_group, male_group])].copy()
    data['female'] = (data['parent_research_group'] == female_group).astype(int)
    data = data[['post_id', 'female', 'sentiment_vader', 'post_sentiment_vader']].dropna()
    data = data.rename(columns={'sentiment_vader': 'y', 'post_sentiment_vader': 'post_score'})
    formula = "y ~ female + post_score"

    b_all, p_all, _ = rq2_regression.fit_female(data, formula)
    print(f"   All posts: female coefficient = {b_all:.4f}, p = {p_all:.4f}")

    rows = []
    for pid in data.loc[data['female'] == 1, 'post_id'].unique():
        d = data[data['post_id'] != pid]
        b, p, _ = rq2_regression.fit_female(d, formula)
        rows.append({"dropped_post": pid,
                     "title": str(titles.get(pid, ""))[:60],
                     "n_comments_dropped": int((data['post_id'] == pid).sum()),
                     "female_coef": b, "p_value": p})
    res = pd.DataFrame(rows).sort_values("female_coef")
    print(res.round(4).to_string(index=False))
    print(f"   Coefficient range without one post: {res['female_coef'].min():.4f} to {res['female_coef'].max():.4f}")
    return res


def word_spread(df, female_group, male_group, words):
    """For each key word: how many different posts its comments come from, and the share of the largest post."""
    data = df[df['parent_research_group'].isin([female_group, male_group])]
    counts = {(g, w): Counter() for g in (female_group, male_group) for w in words}
    word_set = set(words)
    for body, pid, g in zip(data['body'], data['post_id'], data['parent_research_group']):
        if not isinstance(body, str):
            continue
        tokens = re.findall(r'\b\w+\b', body.lower())    # same tokenization as the labeler
        for w in word_set.intersection(tokens):
            counts[(g, w)][pid] += tokens.count(w)

    rows = []
    for w in words:
        row = {"word": w}
        for g, label in ((female_group, "female"), (male_group, "male")):
            c = counts[(g, w)]
            total = sum(c.values())
            row[f"{label}_count"] = total
            row[f"{label}_n_posts"] = len(c)
            row[f"{label}_top_post_share_%"] = (max(c.values()) / total * 100) if total else 0.0
        rows.append(row)
    res = pd.DataFrame(rows)
    print(res.round(1).to_string(index=False))
    return res


def two_by_two(df):
    """
    Post level, all 4 groups: mean_vader ~ female * hard.
    The 'female:hard' term tests whether the gender gap in Hard AI differs from the gender gap in Soft AI.
    Robust (HC3) standard errors, because the groups have different sizes and spreads.
    """
    groups = [g for _, f, m in TOPICS for g in (f, m)]
    data = df[df['parent_research_group'].isin(groups)]
    posts = data.groupby(['post_id', 'parent_research_group'])['sentiment_vader'].mean().reset_index(name='mean_vader')
    posts['female'] = posts['parent_research_group'].str.startswith('female').astype(int)
    posts['hard'] = posts['parent_research_group'].str.contains('hard').astype(int)

    cells = posts.groupby('parent_research_group')['mean_vader'].agg(['count', 'mean', 'median']).loc[groups]
    print("   Mean VADER of the comments, per post (one row per group):")
    print(cells.round(4).to_string())

    model = smf.ols("mean_vader ~ female * hard", data=posts).fit(cov_type='HC3')
    print("\n   Model: mean_vader ~ female * hard  (posts as units, HC3 standard errors)")
    for term, meaning in [("female", "gender gap in Soft AI"),
                          ("hard", "Hard vs Soft, male-context posts"),
                          ("female:hard", "extra gender gap in Hard AI (interaction)")]:
        print(f"   {term:12s} coef = {model.params[term]:+.4f}  p = {model.pvalues[term]:.4f}   ({meaning})")

    print("\n   Within each gender: Hard AI vs Soft AI posts")
    for gender in ("female", "male"):
        hard_vals = posts[posts['parent_research_group'] == f"{gender}_hard_ai"]['mean_vader']
        soft_vals = posts[posts['parent_research_group'] == f"{gender}_soft_ai"]['mean_vader']
        t_p = ttest_ind(hard_vals, soft_vals, equal_var=False).pvalue
        mw_p = mannwhitneyu(hard_vals, soft_vals, alternative='two-sided').pvalue
        print(f"   {gender:6s}: Hard {hard_vals.mean():.4f} (n={len(hard_vals)}) vs Soft {soft_vals.mean():.4f} "
              f"(n={len(soft_vals)}) | Welch p = {t_p:.4f} | Mann-Whitney p = {mw_p:.4f}")
    return posts, model


def plot_two_by_two(posts, results_dir):
    """Each dot = one post; black line = group mean. Hard AI on the left, Soft AI on the right."""
    order = [("female_hard_ai", '#e48aa8'), ("male_hard_ai", '#6fa8dc'),
             ("female_soft_ai", '#e48aa8'), ("male_soft_ai", '#6fa8dc')]
    plt.figure(figsize=(9, 5))
    for x, (g, color) in enumerate(order):
        vals = posts[posts['parent_research_group'] == g]['mean_vader']
        plt.scatter([x] * len(vals), vals, color=color, alpha=0.6, s=35)
        plt.hlines(vals.mean(), x - 0.25, x + 0.25, color='black', linewidth=2)
    plt.xticks(range(4), [f"{g}\n(n={(posts['parent_research_group'] == g).sum()})" for g, _ in order])
    plt.axvline(1.5, color='grey', linestyle='--', alpha=0.5)
    plt.ylabel("Mean VADER of the post's comments")
    plt.title("Gender x topic: comment sentiment per post (black line = mean)", fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(results_dir, "RQ2_Robustness_2x2.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved chart to: {save_path}")
    plt.show()


def plot_post_means(all_posts, results_dir):
    """Each dot = one post (mean VADER of its comments)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (topic, female_group, male_group) in zip(axes, TOPICS):
        posts = all_posts[topic]
        for x, (g, color) in enumerate([(female_group, '#e48aa8'), (male_group, '#6fa8dc')]):
            vals = posts[posts['parent_research_group'] == g]['mean_vader']
            ax.scatter([x] * len(vals), vals, color=color, alpha=0.7, s=40)
            ax.hlines(vals.mean(), x - 0.25, x + 0.25, color='black', linewidth=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Female posts (n={len(posts[posts['parent_research_group'] == female_group])})",
                            f"Male posts (n={len(posts[posts['parent_research_group'] == male_group])})"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(topic)
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel("Mean VADER of the post's comments")
    plt.suptitle("Robustness: comment sentiment per post (black line = mean)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(results_dir, "RQ2_Robustness_Post_Means.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved chart to: {save_path}")
    plt.show()


def analyze_robustness():

    base_dir = "/content/gdrive/MyDrive/Data mining/text mining"
    results_dir = os.path.join(base_dir, "visualization")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print("Loading comments and posts (v3)...")
    df = rq2_regression.load_data(base_dir)
    posts_file = pd.read_csv(os.path.join(base_dir, "data/processed/final_posts_dataset_v3.csv"),
                             usecols=['id', 'title'])
    titles = posts_file.drop_duplicates(subset=['id']).set_index('id')['title'].to_dict()

    all_posts, all_loo = {}, []
    for topic, female_group, male_group in TOPICS:
        print("\n" + "=" * 50)
        print(f"{topic}: 1. post-level comparison (VADER)")
        print("=" * 50)
        all_posts[topic] = post_level(df, female_group, male_group)

        print(f"\n{topic}: 2. leave one female-context post out")
        loo = leave_one_out(df, female_group, male_group, titles)
        loo.insert(0, "topic", topic)
        all_loo.append(loo)

    all_spread = []
    for topic, female_group, male_group in TOPICS:
        print("\n" + "=" * 50)
        print(f"{topic}: 3. word spread across posts")
        print("=" * 50)
        sp = word_spread(df, female_group, male_group, KEY_WORDS)
        sp.insert(0, "topic", topic)
        all_spread.append(sp)
    spread = pd.concat(all_spread, ignore_index=True)

    print("\n" + "=" * 50)
    print("4. Gender x topic (2x2), post level")
    print("=" * 50)
    posts_2x2, model_2x2 = two_by_two(df)

    pd.concat(all_loo, ignore_index=True).to_csv(
        os.path.join(results_dir, "RQ2_Robustness_leave_one_out.csv"), index=False)
    spread.to_csv(os.path.join(results_dir, "RQ2_Robustness_word_spread.csv"), index=False)
    print(f"\nSaved tables to: {results_dir}")

    plot_post_means(all_posts, results_dir)
    plot_two_by_two(posts_2x2, results_dir)
    return all_posts, all_loo, spread, model_2x2


def main():
    print("=" * 60)
    print("STARTING RQ2 robustness check (VADER, post level)")
    print("=" * 60)
    return analyze_robustness()


if __name__ == "__main__":
    main()
