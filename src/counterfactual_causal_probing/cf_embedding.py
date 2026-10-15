import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


# Experiment 2c (follow-up plan): effect of the gender swap on the embedding position of a post.
# This file runs BERT-base, saves the embeddings, and saves per-post shifts.
# All statistical tests are computed in cf_analysis.py.
#
# Two measures per post:
#  1. delta = cosine distance between the embedding of the original and of the swapped post
#     (HOW MUCH the post moves). Same for the control swap.
#  2. shift along the Hard <-> Soft axis (WHERE the post moves): the axis is the direction from
#     the mean embedding of Soft AI posts to the mean embedding of Hard AI posts. A positive shift
#     means the swapped post moved towards Hard AI. This is the causal version of the word2vec
#     finding of the original project ("male is closer to Hard AI").
#
# Why not word2vec: a word2vec post vector is the average of its word vectors, so swapping
# he -> she always moves it by exactly (v_she - v_he) / length. The shift is fixed by the word
# pair and the post length and does not depend on context, so it measures nothing new.
#
# Why mean pooling and not [CLS]: bert-base was not fine-tuned on a sentence task, and its
# [CLS] vector is known to be a poor sentence representation. Mean over all tokens is standard.

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 512

DATA_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
CF_DIR = os.path.join(DATA_DIR, "counterfactual")


def load_pairs():
    """Counterfactual pairs + control version, without the pairs flagged by LanguageTool."""
    df = pd.read_pickle(os.path.join(CF_DIR, "counterfactual_posts_checked.pkl"))
    ctrl = pd.read_pickle(os.path.join(CF_DIR, "control_posts.pkl"))
    df = df.join(ctrl[["text_ctrl", "n_ctrl_swaps", "has_control"]])
    print(f"All pairs: {len(df)}")
    df = df[~df["lt_flagged"]]
    print(f"After removing LanguageTool-flagged pairs: {len(df)}")
    return df


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return tokenizer, model, device


def tokenize(tokenizer, texts):
    # truncate once here, so the same token ids are used for embedding and for the truncation check
    return tokenizer([str(t) for t in texts], truncation=True, max_length=MAX_LEN)["input_ids"]


def embed(model, tokenizer, device, input_ids, batch_size=64):
    """Mean-pooled last hidden state for every text (padding tokens are excluded from the mean)."""
    order = np.argsort([len(x) for x in input_ids])   # similar lengths per batch -> less padding
    emb = np.zeros((len(input_ids), model.config.hidden_size), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = tokenizer.pad({"input_ids": [input_ids[i] for i in idx]}, return_tensors="pt").to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            mean = (hidden.float() * mask).sum(1) / mask.sum(1)
            emb[idx] = mean.cpu().numpy()

            if (start // batch_size) % 100 == 0:
                print(f"  {start + len(idx)}/{len(order)}")
    return emb


def cosine_distance(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return 1 - (a * b).sum(1)


def main():
    OUTPUT_PATH = os.path.join(CF_DIR, "cf_embedding_shifts.pkl")
    EMB_PATH = os.path.join(CF_DIR, "cf_embeddings_{}.npy")   # raw embeddings, in case we need them again

    # all pairs: bert-base was not trained on our data
    df = load_pairs()
    tokenizer, model, device = load_model()
    print(f"Model: {MODEL_NAME}, device: {device}")

    ids_orig = tokenize(tokenizer, df["full_text"])
    ids_cf = tokenize(tokenizer, df["text_cf"])
    has_ctrl = df["has_control"].values
    ids_ctrl = tokenize(tokenizer, df.loc[has_ctrl, "text_ctrl"])

    print("Embedding original texts ...")
    e_orig = embed(model, tokenizer, device, ids_orig)
    print("Embedding gender-swapped texts ...")
    e_cf = embed(model, tokenizer, device, ids_cf)
    print("Embedding control texts ...")
    e_ctrl = embed(model, tokenizer, device, ids_ctrl)

    # float16 halves the file size, precision is more than enough for distances
    for name, e in [("orig", e_orig), ("cf", e_cf), ("ctrl", e_ctrl)]:
        np.save(EMB_PATH.format(name), e.astype(np.float16))

    out = pd.DataFrame(index=df.index)
    out["id"] = df["id"]
    out["cf_truncated_same"] = [a == b for a, b in zip(ids_orig, ids_cf)]

    # 1. how much the post moves
    out["delta_gender"] = cosine_distance(e_orig, e_cf)
    out["ctrl_truncated_same"] = None
    out.loc[has_ctrl, "ctrl_truncated_same"] = [
        a == b for a, b in zip([x for x, h in zip(ids_orig, has_ctrl) if h], ids_ctrl)
    ]
    out["delta_ctrl"] = np.nan
    out.loc[has_ctrl, "delta_ctrl"] = cosine_distance(e_orig[has_ctrl], e_ctrl)

    # 2. where the post moves: projection on the Soft -> Hard axis (built from the ORIGINAL posts)
    cat = df["ai_category"].values
    axis = e_orig[cat == "hard_ai"].mean(0) - e_orig[cat == "soft_ai"].mean(0)
    axis = axis / np.linalg.norm(axis)
    print(f"Hard/Soft axis built from {(cat == 'hard_ai').sum()} Hard and {(cat == 'soft_ai').sum()} Soft posts")

    out["proj_orig"] = e_orig @ axis
    out["proj_cf"] = e_cf @ axis
    out["proj_ctrl"] = np.nan
    out.loc[has_ctrl, "proj_ctrl"] = e_ctrl @ axis

    # sanity check of the axis: Hard posts should be higher on it than Soft posts
    print(f"Mean projection: Hard {out.loc[cat == 'hard_ai', 'proj_orig'].mean():.3f}, "
          f"Soft {out.loc[cat == 'soft_ai', 'proj_orig'].mean():.3f}")

    for col in ["n_swaps", "gender_context", "ai_category", "has_control", "n_ctrl_swaps"]:
        out[col] = df[col]

    print(f"\nMean delta: gender {out['delta_gender'].mean():.5f}, control {out['delta_ctrl'].mean():.5f}")
    print(f"Pairs where the swap is beyond token 512: {out['cf_truncated_same'].sum()}")

    out.to_pickle(OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")
    return out
