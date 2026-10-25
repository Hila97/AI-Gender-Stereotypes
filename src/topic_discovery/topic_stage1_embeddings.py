"""
Direction 1 - stage 1b: embeddings
----------------------------------
Turns every post into a vector with a sentence-transformers model. Three versions (plan, decision 3-4):
    emb_minilm_masked.npy   all-MiniLM-L6-v2 on the masked text      -> main run
    emb_minilm_plain.npy    all-MiniLM-L6-v2 on the unmasked text    -> robustness: masking
    emb_mpnet_masked.npy    all-mpnet-base-v2 on the masked text     -> robustness: model
Rows are in the same order as posts_masked.pkl. A file that already exists is not computed again.
Run on a GPU (Colab: Runtime -> Change runtime type -> GPU).
"""
import os
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

BASE_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
OUT_DIR = os.path.join(BASE_DIR, "topic_discovery")
POSTS_MASKED = os.path.join(OUT_DIR, "posts_masked.pkl")

RUNS = [
    # (file name, model, text column)
    ("emb_minilm_masked.npy", "all-MiniLM-L6-v2", "text_masked"),
    ("emb_minilm_plain.npy", "all-MiniLM-L6-v2", "text_clean"),
    ("emb_mpnet_masked.npy", "all-mpnet-base-v2", "text_masked"),
]


def compute_embeddings(texts, model_name, batch_size=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    print(f"   model: {model_name} | device: {device} | max tokens per post: {model.max_seq_length}")
    start = time.time()
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    print(f"   done in {(time.time() - start) / 60:.1f} min, shape {emb.shape}")
    return emb.astype(np.float32)


def run_embeddings():
    df = pd.read_pickle(POSTS_MASKED)
    print(f"Loaded {len(df)} posts from {POSTS_MASKED}")
    print(f"   empty texts: {(df['text_clean'].str.len() == 0).sum()}")
    if not torch.cuda.is_available():
        print("   WARNING: no GPU found, this will be very slow")

    for file_name, model_name, text_col in RUNS:
        path = os.path.join(OUT_DIR, file_name)
        print(f"\n--- {file_name} ({model_name} on {text_col}) ---")
        if os.path.exists(path):
            print(f"   already exists, skipping: {path}")
            continue
        emb = compute_embeddings(df[text_col].tolist(), model_name)
        np.save(path, emb)
        print(f"   saved: {path}")

    # sanity check: same number of rows as the posts, no empty vectors
    print("\n--- Check ---")
    for file_name, _, _ in RUNS:
        emb = np.load(os.path.join(OUT_DIR, file_name))
        n_zero = int((np.abs(emb).sum(axis=1) == 0).sum())
        print(f"   {file_name}: shape {emb.shape}, rows match posts: {emb.shape[0] == len(df)}, zero vectors: {n_zero}")


def main():
    print("=" * 60)
    print("DIRECTION 1 - stage 1b: embeddings")
    print("=" * 60)
    run_embeddings()


if __name__ == "__main__":
    main()
