"""
Direction 1 - stage 2 tuning: HDBSCAN settings
----------------------------------------------
With the default HDBSCAN settings one topic took 95% of the posts. min_samples=10 fixed this for the masked
MiniLM and mpnet runs, but the unmasked MiniLM run still collapsed into one topic. So the settings are tried
on ALL THREE embeddings, to find one setting that splits every run into topics (each robustness run should
differ from the main run only in the embeddings, not in the clustering settings).

For each run, UMAP is computed ONCE (same parameters and seed as stage 2) and saved; HDBSCAN takes seconds.
min_cluster_size stays 100 (decision 5). What changes:
    min_samples               how dense a region must be to count as a topic core (default = min_cluster_size).
    cluster_selection_method  "eom" (default) prefers a few big stable clusters;
                              "leaf" takes the smallest clusters at the bottom of the tree -> more, finer topics.
    max_cluster_size          eom only: a cluster larger than this is never chosen, so one giant topic is
                              impossible; it is split into its sub-clusters instead.
Nothing here changes the stage 2 files.
"""
import os
import time

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from umap import UMAP

BASE_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
OUT_DIR = os.path.join(BASE_DIR, "topic_discovery")

RUNS = {
    "minilm_masked": "emb_minilm_masked.npy",
    "minilm_plain": "emb_minilm_plain.npy",
    "mpnet_masked": "emb_mpnet_masked.npy",
}

MIN_TOPIC_SIZE = 100
SEED = 42

SETTINGS = [
    # (cluster_selection_method, min_samples, max_cluster_size)   0 = no limit
    ("eom", None, 0),       # HDBSCAN default
    ("eom", 10, 0),         # what stage 2 uses now
    ("eom", 5, 0),
    ("eom", 10, 2000),      # no topic larger than 2,000 posts (~3.5%)
    ("eom", 10, 3000),      # no topic larger than 3,000 posts (~5%)
    ("leaf", 10, 0),
    ("leaf", 5, 0),
]


def reduce_once(run):
    """UMAP with the plan's parameters, saved so it is computed only once per run."""
    reduced_file = os.path.join(OUT_DIR, f"umap5_{run}.npy")
    if os.path.exists(reduced_file):
        print(f"   loading reduced embeddings: {reduced_file}")
        return np.load(reduced_file)
    emb = np.load(os.path.join(OUT_DIR, RUNS[run]))
    print(f"   UMAP on {emb.shape} ...")
    start = time.time()
    reduced = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                   random_state=SEED).fit_transform(emb)
    print(f"   done in {(time.time() - start) / 60:.1f} min")
    np.save(reduced_file, reduced)
    return reduced


def try_settings(run, reduced):
    rows = []
    for method, min_samples, max_size in SETTINGS:
        labels = HDBSCAN(min_cluster_size=MIN_TOPIC_SIZE, min_samples=min_samples, metric="euclidean",
                         cluster_selection_method=method, max_cluster_size=max_size).fit_predict(reduced)
        sizes = pd.Series(labels[labels != -1]).value_counts()
        rows.append({
            "run": run,
            "method": method,
            "min_samples": min_samples if min_samples is not None else "default",
            "max_size": max_size if max_size else "-",
            "n_topics": len(sizes),
            "outliers_%": round((labels == -1).mean() * 100, 1),
            "largest_topic_%": round(sizes.max() / len(labels) * 100, 1) if len(sizes) else 0,
            "median_topic_size": int(sizes.median()) if len(sizes) else 0,
        })
    return rows


def main(runs=None):
    print("=" * 60)
    print("DIRECTION 1 - stage 2 tuning: HDBSCAN settings on all runs")
    print("=" * 60)
    rows = []
    for run in (runs or list(RUNS)):
        print(f"\n--- {run} ---")
        rows += try_settings(run, reduce_once(run))
    res = pd.DataFrame(rows)

    print("\n--- Results ---")
    for run, part in res.groupby("run", sort=False):
        print(f"\n{run}")
        print(part.drop(columns="run").to_string(index=False))
    res.to_csv(os.path.join(OUT_DIR, "hdbscan_tuning.csv"), index=False)
    return res


if __name__ == "__main__":
    main()
