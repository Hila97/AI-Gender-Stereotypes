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
# [CLS] vector is known to be a poor sentence representation. Mean over all tokens is standard
# (Reimers & Gurevych, Sentence-BERT, 2019). [CLS] is still saved in the same run, as a robustness
# check: if the conclusions hold with both vectors, they do not depend on this choice.

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
    """
    Two vectors per text from the same forward pass:
      mean = mean of the last hidden state over all tokens (padding excluded) -> main measure
      cls  = last hidden state of the [CLS] token (position 0)              -> robustness check
    """
    order = np.argsort([len(x) for x in input_ids])   # similar lengths per batch -> less padding
    emb = np.zeros((len(input_ids), model.config.hidden_size), dtype=np.float32)
    cls = np.zeros_like(emb)

    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = tokenizer.pad({"input_ids": [input_ids[i] for i in idx]}, return_tensors="pt").to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            mean = (hidden.float() * mask).sum(1) / mask.sum(1)
            emb[idx] = mean.cpu().numpy()
            cls[idx] = hidden[:, 0].float().cpu().numpy()

            if (start // batch_size) % 100 == 0:
                print(f"  {start + len(idx)}/{len(order)}")
    return emb, cls


def cosine_distance(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return 1 - (a * b).sum(1)


def add_shift_columns(out, e_orig, e_cf, e_ctrl, has_ctrl, cat, suffix=""):
    """delta (how much) and projection on the Soft -> Hard axis (where) for one kind of vector."""
    # 1. how much the post moves
    out["delta_gender" + suffix] = cosine_distance(e_orig, e_cf)
    out["delta_ctrl" + suffix] = np.nan
    out.loc[has_ctrl, "delta_ctrl" + suffix] = cosine_distance(e_orig[has_ctrl], e_ctrl)

    # 2. where the post moves: projection on the Soft -> Hard axis (built from the ORIGINAL posts)
    axis = e_orig[cat == "hard_ai"].mean(0) - e_orig[cat == "soft_ai"].mean(0)
    axis = axis / np.linalg.norm(axis)
    out["proj_orig" + suffix] = e_orig @ axis
    out["proj_cf" + suffix] = e_cf @ axis
    out["proj_ctrl" + suffix] = np.nan
    out.loc[has_ctrl, "proj_ctrl" + suffix] = e_ctrl @ axis

    # sanity check of the axis: Hard posts should be higher on it than Soft posts
    name = suffix.strip("_") or "mean"
    print(f"[{name}] mean projection: Hard {out.loc[cat == 'hard_ai', 'proj_orig' + suffix].mean():.3f}, "
          f"Soft {out.loc[cat == 'soft_ai', 'proj_orig' + suffix].mean():.3f}; "
          f"mean delta: gender {out['delta_gender' + suffix].mean():.5f}, control {out['delta_ctrl' + suffix].mean():.5f}")


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
    e_orig, c_orig = embed(model, tokenizer, device, ids_orig)
    print("Embedding gender-swapped texts ...")
    e_cf, c_cf = embed(model, tokenizer, device, ids_cf)
    print("Embedding control texts ...")
    e_ctrl, c_ctrl = embed(model, tokenizer, device, ids_ctrl)

    # float16 halves the file size, precision is more than enough for distances
    for name, e in [("orig", e_orig), ("cf", e_cf), ("ctrl", e_ctrl),
                    ("orig_cls", c_orig), ("cf_cls", c_cf), ("ctrl_cls", c_ctrl)]:
        np.save(EMB_PATH.format(name), e.astype(np.float16))

    out = pd.DataFrame(index=df.index)
    out["id"] = df["id"]
    out["cf_truncated_same"] = [a == b for a, b in zip(ids_orig, ids_cf)]
    out["ctrl_truncated_same"] = None
    out.loc[has_ctrl, "ctrl_truncated_same"] = [
        a == b for a, b in zip([x for x, h in zip(ids_orig, has_ctrl) if h], ids_ctrl)
    ]

    cat = df["ai_category"].values
    print(f"Hard/Soft axis built from {(cat == 'hard_ai').sum()} Hard and {(cat == 'soft_ai').sum()} Soft posts")
    # same column names as before for mean pooling (so cf_analysis.py works unchanged), *_cls for [CLS]
    add_shift_columns(out, e_orig, e_cf, e_ctrl, has_ctrl, cat)
    add_shift_columns(out, c_orig, c_cf, c_ctrl, has_ctrl, cat, suffix="_cls")

    for col in ["n_swaps", "gender_context", "ai_category", "has_control", "n_ctrl_swaps"]:
        out[col] = df[col]

    print(f"Pairs where the swap is beyond token 512: {out['cf_truncated_same'].sum()}")

    out.to_pickle(OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")
    return out
