import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Experiment 2a (follow-up plan): effect of the gender swap on the Hard/Soft AI classifier.
# This file only RUNS the model and saves raw predictions per post.
# All metrics and statistical tests (CFR, delta P, McNemar ...) are computed in cf_analysis.py.

MAX_LEN = 512          # bert_large_finetuned was trained with max_length=512
HARD_LABEL = 0         # label_map in data_classificationn_preprocessing.py: hard_ai=0, soft_ai=1


def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    return tokenizer, model, device


def tokenize(tokenizer, texts):
    # truncate once here, so the same token ids are used for prediction and for the truncation check
    return tokenizer(list(texts), truncation=True, max_length=MAX_LEN)["input_ids"]


def predict(model, tokenizer, device, input_ids, batch_size=32):
    """Returns (predicted label, probability of Hard AI) for every text."""
    # sort by length so each batch has similar lengths -> much less padding, much faster
    order = np.argsort([len(x) for x in input_ids])
    p_hard = np.zeros(len(input_ids))

    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = tokenizer.pad({"input_ids": [input_ids[i] for i in idx]}, return_tensors="pt").to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(**batch).logits
            p_hard[idx] = torch.softmax(logits.float(), dim=1)[:, HARD_LABEL].cpu().numpy()

            if (start // batch_size) % 50 == 0:
                print(f"  {start + len(idx)}/{len(order)}")

    pred = np.where(p_hard >= 0.5, HARD_LABEL, 1 - HARD_LABEL)
    return pred, p_hard


def run_classification(df, tokenizer, model, device):
    out = pd.DataFrame(index=df.index)
    out["id"] = df["id"]
    out["label"] = df["label"]   # true label from test_data.pkl

    ids_orig = tokenize(tokenizer, df["full_text"])
    ids_cf = tokenize(tokenizer, df["text_cf"])

    # if all gender swaps were after token 512, the model sees the exact same input -> no real counterfactual
    out["cf_truncated_same"] = [a == b for a, b in zip(ids_orig, ids_cf)]

    print("Predicting original texts ...")
    out["pred_orig"], out["p_hard_orig"] = predict(model, tokenizer, device, ids_orig)
    print("Predicting gender-swapped texts ...")
    out["pred_cf"], out["p_hard_cf"] = predict(model, tokenizer, device, ids_cf)

    # control texts only exist for part of the posts
    has_ctrl = df["has_control"].values
    ids_ctrl = tokenize(tokenizer, df.loc[has_ctrl, "text_ctrl"])
    out["ctrl_truncated_same"] = None
    out.loc[has_ctrl, "ctrl_truncated_same"] = [
        a == b for a, b in zip([x for x, h in zip(ids_orig, has_ctrl) if h], ids_ctrl)
    ]

    print("Predicting control texts ...")
    pred_ctrl, p_ctrl = predict(model, tokenizer, device, ids_ctrl)
    out["pred_ctrl"] = np.nan
    out["p_hard_ctrl"] = np.nan
    out.loc[has_ctrl, "pred_ctrl"] = pred_ctrl
    out.loc[has_ctrl, "p_hard_ctrl"] = p_ctrl

    # columns needed later for the breakdowns in cf_analysis.py
    for col in ["n_swaps", "gender_context", "ai_category", "has_control", "n_ctrl_swaps"]:
        if col in df.columns:
            out[col] = df[col]
    return out


def main():
    DATA_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
    MODEL_DIR = "/content/gdrive/MyDrive/text mining/models"
    CF_DIR = os.path.join(DATA_DIR, "counterfactual")
    RESULTS_DIR = os.path.join(DATA_DIR, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    OUTPUT_PATH = os.path.join(RESULTS_DIR, "cf_classification_preds.pkl")

    # 1. counterfactual pairs + control version, without the pairs flagged by LanguageTool
    df = pd.read_pickle(os.path.join(CF_DIR, "counterfactual_posts_checked.pkl"))
    ctrl = pd.read_pickle(os.path.join(CF_DIR, "control_posts.pkl"))
    df = df.join(ctrl[["text_ctrl", "n_ctrl_swaps", "has_control"]])
    print(f"All pairs: {len(df)}")
    df = df[~df["lt_flagged"]]
    print(f"After removing LanguageTool-flagged pairs: {len(df)}")

    # 2. keep only posts from the BERT test set (the model never saw them in training)
    test_df = pd.read_pickle(os.path.join(DATA_DIR, "processed", "test_data.pkl"))
    df = df.drop(columns=["label"], errors="ignore").merge(test_df[["id", "label"]], on="id", how="inner")
    print(f"Pairs in the BERT test set: {len(df)} (test set size: {len(test_df)})")

    # 3. run the model
    tokenizer, model, device = load_model(os.path.join(MODEL_DIR, "bert_large_finetuned"))
    print(f"Device: {device}")
    out = run_classification(df, tokenizer, model, device)

    # 4. quick sanity checks only (the real analysis is in cf_analysis.py)
    acc = (out["pred_orig"] == out["label"]).mean()
    print(f"\nAccuracy on the original texts: {acc:.3f}  (full test set was 0.966)")
    print(f"Pairs where the swap is beyond token 512: {out['cf_truncated_same'].sum()}")
    print(f"Pairs where the gender swap changed the predicted label: {(out['pred_orig'] != out['pred_cf']).sum()}")

    out.to_pickle(OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")
