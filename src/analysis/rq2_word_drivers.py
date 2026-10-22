"""
RQ2 extension - check: which words create the differences?
-----------------------------------------------------------
For a lexicon, compares how often each of its words appears in comments on Female vs Male posts
(per 1,000 words), and shows the words that push the difference most in each direction.
Also prints example comments, so we can read in what context the words are used.
Same comments and deduplication as rq2_liwc_v3_analysis.py.
"""
import os
import re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "/content/gdrive/MyDrive/Data mining/text mining"
DATA_PATH = os.path.join(BASE_DIR, "data/processed/final_comments_dataset_v3.csv")
LEXICONS_V3_DIR = os.path.join(BASE_DIR, "data/lexicons_v3")
RESULTS_DIR = os.path.join(BASE_DIR, "visualization")

# the lexicons that were significant in Hard AI
DEFAULT_LEXICONS = ["situational_failure", "negative_emotions", "positive_emotions",
                    "physical_adversity", "achievement_excellence", "reward_success"]


def load_lexicon(name):
    with open(os.path.join(LEXICONS_V3_DIR, f"{name}.txt"), encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())


def tokenize(text):
    """Same tokenization as the labeler."""
    return re.findall(r'\b\w+\b', text.lower()) if isinstance(text, str) else []


def count_words(texts, lexicon):
    """Total number of words, and how many times each lexicon word appears."""
    total, counts = 0, Counter()
    for text in texts:
        words = tokenize(text)
        total += len(words)
        counts.update(w for w in words if w in lexicon)
    return total, counts


def word_drivers(df, lexicon_name, female_group="female_hard_ai", male_group="male_hard_ai", top_n=15):
    """
    For each word: rate per 1,000 words in each group, and the gap (female - male).
    The gaps of all words add up to the lexicon's overall gap, so the biggest gaps are the drivers.
    """
    lexicon = load_lexicon(lexicon_name)
    total_f, counts_f = count_words(df.loc[df['parent_research_group'] == female_group, 'body'], lexicon)
    total_m, counts_m = count_words(df.loc[df['parent_research_group'] == male_group, 'body'], lexicon)

    rows = []
    for w in set(counts_f) | set(counts_m):
        rate_f = counts_f[w] / total_f * 1000
        rate_m = counts_m[w] / total_m * 1000
        rows.append({"word": w, "female_count": counts_f[w], "male_count": counts_m[w],
                     "female_per_1000": rate_f, "male_per_1000": rate_m, "gap": rate_f - rate_m})
    words = pd.DataFrame(rows).sort_values("gap", ascending=False)

    overall_gap = sum(counts_f.values()) / total_f * 1000 - sum(counts_m.values()) / total_m * 1000
    print(f"\n=== {lexicon_name}: {female_group} vs {male_group} ===")
    print(f"Overall: {sum(counts_f.values()) / total_f * 1000:.2f} vs "
          f"{sum(counts_m.values()) / total_m * 1000:.2f} words per 1,000 (gap {overall_gap:+.2f})")
    cols = ["word", "female_count", "male_count", "female_per_1000", "male_per_1000", "gap"]
    print(f"\nTop {top_n} words MORE common on Female posts:")
    print(words[words["gap"] > 0].head(top_n)[cols].round(3).to_string(index=False))
    print(f"\nTop {top_n} words MORE common on Male posts:")
    print(words[words["gap"] < 0].tail(top_n).iloc[::-1][cols].round(3).to_string(index=False))
    return words


def plot_word_drivers(words, lexicon_name, top_n=15):
    """Bar chart: words pushing toward Female posts (right, pink) and Male posts (left, blue)."""
    top = pd.concat([words[words["gap"] > 0].head(top_n), words[words["gap"] < 0].tail(top_n)])
    top = top.sort_values("gap")
    colors = ['#d5a6bd' if g > 0 else '#a4c2f4' for g in top["gap"]]
    plt.figure(figsize=(8, max(6, len(top) * 0.3)))
    plt.barh(top["word"], top["gap"], color=colors)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title(f"{lexicon_name}: which words create the gap (Hard AI)", fontweight='bold')
    plt.xlabel("Gap in uses per 1,000 words  (> 0 = more in comments on Female posts)")
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f"RQ2_LIWCV3_words_{lexicon_name}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved chart to: {save_path}")
    plt.show()


def sample_comments(df, lexicon_name, words, group="female_hard_ai", n=5, seed=42):
    """Random comments from the group that contain one of the given words (to read the context)."""
    words = set(words)
    subset = df[df['parent_research_group'] == group]
    has_word = subset['body'].apply(lambda t: bool(words & set(tokenize(t))))
    sample = subset[has_word].sample(min(n, has_word.sum()), random_state=seed)
    print(f"\n--- {n} example comments ({group}) with: {', '.join(sorted(words))} ---")
    for _, row in sample.iterrows():
        found = sorted(words & set(tokenize(row['body'])))
        print(f"[{', '.join(found)}] {str(row['body'])[:300]}")
        print()


def main(lexicons=DEFAULT_LEXICONS, top_n=15):
    print("Loading comments dataset (v3)...")
    df = pd.read_csv(DATA_PATH, usecols=['body', 'post_id', 'parent_research_group'])
    df = df.drop_duplicates(subset=['body', 'post_id'])     # same as sentiment_analysis.py
    df = df[df['parent_research_group'].isin(['female_hard_ai', 'male_hard_ai'])]

    all_words = {}
    for lexicon_name in lexicons:
        words = word_drivers(df, lexicon_name, top_n=top_n)
        plot_word_drivers(words, lexicon_name, top_n=top_n)
        # read comments with the 3 words that push most toward Female posts
        sample_comments(df, lexicon_name, words.head(3)["word"])
        all_words[lexicon_name] = words
    return all_words


if __name__ == "__main__":
    main()
