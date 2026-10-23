"""
LIWC v3 Labeler (affect lexicons)
----------------------------------------------
1. Reads the existing final datasets (created by algorithmic_labeler.py).
2. Adds one LIWCV3_<lexicon> column per v3 lexicon (same scoring as stage 1: % of words in the lexicon).
3. Saves new files (*_v3.csv). The existing final datasets are not changed.
"""

import re
from pathlib import Path
from typing import Dict, Set
import pandas as pd

# Paths
DATA_DIR = Path("/content/gdrive/MyDrive/Data mining/text mining/data")
PROCESSED_DIR = DATA_DIR / "processed"
LEXICONS_V3_DIR = DATA_DIR / "lexicons_v3"

# v3 lexicons created by lexical_collector_v3
LIWC_V3_CATEGORIES = [
    # positive
    "achievement_excellence", "cognitive_insight", "positive_emotions", "certainty_confidence",
    "affiliation_support", "optimism_vitality", "praise_admiration", "reward_success",
    # negative
    "physical_adversity", "social_legal_negativity", "situational_failure",
    "negative_traits", "existential_negativity", "negative_emotions",
]


NOT_BEFORE_OF = {"kind", "sort"}


def load_v3_lexicons() -> Dict[str, Set[str]]:
    """Loads the v3 lexicon files into a dictionary of sets."""
    print(f"Loading v3 lexicons from: {LEXICONS_V3_DIR}")
    lexicons = {}
    for category in LIWC_V3_CATEGORIES:
        file_path = LEXICONS_V3_DIR / f"{category}.txt"
        # stop instead of silently scoring 0: a missing file would make a whole column wrong
        with open(file_path, "r", encoding="utf-8") as f:
            lexicons[category] = set(line.strip().lower() for line in f if line.strip())
        print(f"   > Loaded '{category}': {len(lexicons[category])} words")
    return lexicons


def get_liwc_v3_scores(text: str, lexicons: Dict[str, Set[str]]) -> Dict[str, float]:
    """
    Percentage of words belonging to each v3 lexicon.
    Same tokenization and formula as get_liwc_scores in algorithmic_labeler.py.
    """
    if not isinstance(text, str) or not text.strip():
        return {f"LIWCV3_{cat}": 0.0 for cat in lexicons}

    words = re.findall(r'\b\w+\b', text.lower())
    total_words = len(words)
    if total_words == 0:
        return {f"LIWCV3_{cat}": 0.0 for cat in lexicons}

    # "kind of" / "sort of" are not emotion: drop 'kind' and 'sort' when 'of' comes right after them
    words = [w for i, w in enumerate(words)
             if not (w in NOT_BEFORE_OF and i + 1 < len(words) and words[i + 1] == "of")]

    scores = {}
    for category, lexicon_words in lexicons.items():
        match_count = sum(1 for word in words if word in lexicon_words)
        scores[f"LIWCV3_{category}"] = (match_count / total_words) * 100
    return scores


def add_v3_columns(df: pd.DataFrame, text: pd.Series, lexicons: Dict[str, Set[str]]) -> pd.DataFrame:
    """Scores every row and adds the LIWCV3_ columns next to the existing ones."""
    scores = pd.DataFrame([get_liwc_v3_scores(t, lexicons) for t in text], index=df.index)
    return pd.concat([df, scores], axis=1)


def process_v3_labeling():
    lexicons = load_v3_lexicons()

    # posts: same text as stage 1 (title + selftext)
    posts_path = PROCESSED_DIR / "final_posts_dataset.csv"
    print(f"Loading posts from {posts_path}...")
    df_posts = pd.read_csv(posts_path)
    posts_text = df_posts["title"].fillna("").astype(str) + " " + df_posts["selftext"].fillna("").astype(str)
    df_posts = add_v3_columns(df_posts, posts_text, lexicons)
    out_posts = PROCESSED_DIR / "final_posts_dataset_v3.csv"
    df_posts.to_csv(out_posts, index=False, encoding="utf-8")
    print(f"Saved {len(df_posts)} posts to: {out_posts}")

    # comments: same text as stage 1 (body); parent_research_group etc. are already in the file
    comments_path = PROCESSED_DIR / "final_comments_dataset.csv"
    print(f"Loading comments from {comments_path}...")
    df_comments = pd.read_csv(comments_path)
    df_comments = add_v3_columns(df_comments, df_comments["body"], lexicons)
    out_comments = PROCESSED_DIR / "final_comments_dataset_v3.csv"
    df_comments.to_csv(out_comments, index=False, encoding="utf-8")
    print(f"Saved {len(df_comments)} comments to: {out_comments}")

    # quick look: how often each lexicon appears at all (share of texts with a score above 0)
    v3_cols = [f"LIWCV3_{c}" for c in LIWC_V3_CATEGORIES]
    summary = pd.DataFrame({
        "posts_mean_%": df_posts[v3_cols].mean(),
        "posts_nonzero_%": (df_posts[v3_cols] > 0).mean() * 100,
        "comments_mean_%": df_comments[v3_cols].mean(),
        "comments_nonzero_%": (df_comments[v3_cols] > 0).mean() * 100,
    }).round(2)
    print("\n--- Summary of the new columns ---")
    print(summary.to_string())
    return df_posts, df_comments


def main():
    print("=" * 60)
    print("LIWC v3 LABELING (affect lexicons)")
    print("=" * 60)
    return process_v3_labeling()


if __name__ == "__main__":
    main()
