"""
Direction 1 - stage 2: BERTopic
-------------------------------
Clusters the posts into topics, once for each embedding file from stage 1b:
    minilm_masked   main run
    minilm_plain    robustness: masking
    mpnet_masked    robustness: model

- The topics are built from the ready-made embeddings (masked text, except in minilm_plain).
- The words that describe each topic (c-TF-IDF) come from the ORIGINAL text, with the gender words
  as stopwords, so neither gender words nor replacement words (person, parent) describe a topic.
- UMAP uses the plan's parameters plus random_state=42, so every run gives the same topics.
- HDBSCAN: min_cluster_size=100, min_samples=10, max_cluster_size=3000 (see topic_stage2_tuning.py).
- Main run only: reduce_outliers assigns the outliers (-1) to their nearest topic -> column topic_all.
"""
import os
import time

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from umap import UMAP

from topic_stage1_masking import KNOWN, MASK_MAP

BASE_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
OUT_DIR = os.path.join(BASE_DIR, "topic_discovery")
POSTS_MASKED = os.path.join(OUT_DIR, "posts_masked.pkl")

RUNS = {
    # run name -> embedding file
    "minilm_masked": "emb_minilm_masked.npy",
    "minilm_plain": "emb_minilm_plain.npy",
    "mpnet_masked": "emb_mpnet_masked.npy",
}
MAIN_RUN = "minilm_masked"

MIN_TOPIC_SIZE = 100        # decision 5
MIN_SAMPLES = 10            # chosen with topic_stage2_tuning.py
MAX_TOPIC_SIZE = 3000       # chosen with topic_stage2_tuning.py: no single topic may take over the data
SEED = 42

# custom stopwords from full_cleaner.py, plus all gender words of the masking list
CUSTOM_STOPS = ['https', 'http', 'com', 'www', 'reddit', 'post', 'comment', 'deleted', 'removed',
                'would', 'could', 'should', 'like', 'one', 'get', 'know', 'think', 'people', 'amp', 'x200b']
# BERTopic deletes every non-alphanumeric character before counting words, so "he's" becomes "hes" and
# "woman's" becomes "womans". These forms are added too, so they do not show up as topic words.
GENDER_NO_APOSTROPHE = {w + "s" for w in KNOWN} | {"hes", "shes", "hed"}
STOPWORDS = sorted(set(ENGLISH_STOP_WORDS) | set(CUSTOM_STOPS) | KNOWN | set(MASK_MAP) | GENDER_NO_APOSTROPHE)


def build_model():
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED)
    # min_topic_size is set here directly: when a HDBSCAN model is passed, BERTopic ignores its own min_topic_size
    # min_samples=10 and max_cluster_size=3000 were chosen in topic_stage2_tuning.py: with the default
    # settings one topic took 95% of the posts, and min_samples alone still collapsed the unmasked run
    hdbscan_model = HDBSCAN(min_cluster_size=MIN_TOPIC_SIZE, min_samples=MIN_SAMPLES,
                            max_cluster_size=MAX_TOPIC_SIZE, metric="euclidean",
                            cluster_selection_method="eom", prediction_data=True)
    # no min_df: BERTopic fits the vectorizer on one joined document per TOPIC, so min_df=5 would mean
    # "the word appears in at least 5 topics" and would drop exactly the words that are specific to a topic
    vectorizer = CountVectorizer(stop_words=STOPWORDS)
    return BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model,
                    vectorizer_model=vectorizer, verbose=True)


def topic_words_table(model, n_words=10):
    """Topic info with the top words of each topic as one string."""
    info = model.get_topic_info()[["Topic", "Count", "Name"]].copy()
    info["top_words"] = [", ".join(w for w, _ in model.get_topic(t)[:n_words]) if t != -1 else ""
                         for t in info["Topic"]]
    return info


def run_one(name, df, docs):
    print(f"\n{'=' * 60}\nRUN: {name}\n{'=' * 60}")
    emb = np.load(os.path.join(OUT_DIR, RUNS[name]))
    assert emb.shape[0] == len(df), "embeddings and posts are not in the same order / size"

    start = time.time()
    model = build_model()
    topics, _ = model.fit_transform(docs, embeddings=emb)
    print(f"   fitted in {(time.time() - start) / 60:.1f} min")

    out = pd.DataFrame({"id": df["id"], "topic": topics})
    n_topics = out.loc[out["topic"] != -1, "topic"].nunique()
    outlier_share = (out["topic"] == -1).mean()
    print(f"   topics: {n_topics} | outliers (-1): {(out['topic'] == -1).sum()} posts ({outlier_share:.1%})")

    if name == MAIN_RUN:
        print("   reduce_outliers (main run only): assigning -1 posts to their nearest topic...")
        out["topic_all"] = model.reduce_outliers(docs, topics, strategy="embeddings", embeddings=emb)
        print(f"   outliers left after reduce_outliers: {(out['topic_all'] == -1).sum()}")

    info = topic_words_table(model)
    out.to_csv(os.path.join(OUT_DIR, f"topics_{name}.csv"), index=False)
    info.to_csv(os.path.join(OUT_DIR, f"topic_info_{name}.csv"), index=False)
    try:
        model.save(os.path.join(OUT_DIR, f"bertopic_{name}"), serialization="pickle")
    except Exception as e:      # the csv files above are what the next stages use
        print(f"   (model not saved: {e})")
    print(f"   saved topics_{name}.csv and topic_info_{name}.csv to {OUT_DIR}")

    print(f"\n   20 largest topics ({name}):")
    with pd.option_context("display.max_colwidth", 90, "display.width", 200):
        print(info[info["Topic"] != -1].head(20)[["Topic", "Count", "top_words"]].to_string(index=False))
    return {"run": name, "n_topics": n_topics, "outlier_share": outlier_share}


def main(runs=None):
    """runs: list of run names, e.g. ["minilm_masked"]. Default: all three."""
    print("=" * 60)
    print("DIRECTION 1 - stage 2: BERTopic")
    print("=" * 60)
    df = pd.read_pickle(POSTS_MASKED)
    docs = df["text_clean"].tolist()          # original text -> topic words
    print(f"Loaded {len(df)} posts | stopwords for topic words: {len(STOPWORDS)}")

    summary = [run_one(name, df, docs) for name in (runs or list(RUNS))]
    summary = pd.DataFrame(summary)
    print("\n--- Summary ---")
    print(summary.round(3).to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
