import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Experiment 2b (follow-up plan): effect of the gender swap on sentiment.
# This file only RUNS the sentiment models and saves raw scores per post.
# All metrics and statistical tests (delta S, Wilcoxon, Cohen's d ...) are computed in cf_analysis.py.
#
# Why not VADER: VADER is a lexicon model and gender words (he/she, man/woman ...) are not in
# its lexicon, so swapping them cannot change the score -> delta S = 0 by construction.
# vader_sanity_check() shows this on a sample. The main experiment uses a contextual model.

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"   # trained on social media text
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


# ---------------------------------------------------------------- VADER sanity check

def vader_sanity_check(df, n=500, seed=42):
    """Run VADER on a random sample of pairs and show that the gender swap almost never changes its score."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    # 1. are the gender words in VADER's lexicon at all?
    gender_words = ["he", "she", "him", "her", "his", "hers", "man", "woman", "men", "women",
                    "boy", "girl", "guy", "gal", "mother", "father", "husband", "wife",
                    "boyfriend", "girlfriend", "brother", "sister", "son", "daughter"]
    in_lexicon = [w for w in gender_words if w in analyzer.lexicon]
    print(f"Gender words in VADER's lexicon: {len(in_lexicon)}/{len(gender_words)} {in_lexicon}")

    # 2. score a sample of pairs
    sample = df.sample(n=min(n, len(df)), random_state=seed)
    out = pd.DataFrame(index=sample.index)
    out["id"] = sample["id"]
    out["vader_orig"] = [analyzer.polarity_scores(str(t))["compound"] for t in sample["full_text"]]
    out["vader_cf"] = [analyzer.polarity_scores(str(t))["compound"] for t in sample["text_cf"]]
    out["dS_vader"] = out["vader_cf"] - out["vader_orig"]

    same = (out["dS_vader"] == 0).mean()
    print(f"\nVADER on {len(out)} random pairs:")
    print(f"  delta S = 0 exactly in {same:.1%} of pairs")
    print(f"  mean |delta S| = {out['dS_vader'].abs().mean():.4f}")

    changed = out[out["dS_vader"] != 0]
    if len(changed):
        # when VADER does change, it is because of a non-gender word that the swap touched
        # (e.g. 'miss' or 'Ms'), or because of sentence-level rules, not because of gender itself
        print(f"  pairs with a change: {len(changed)} (max |delta S| = {changed['dS_vader'].abs().max():.4f})")

    path = os.path.join(CF_DIR, "cf_vader_sanity.pkl")
    out.to_pickle(path)
    print(f"Saved to {path}")
    return out


# ---------------------------------------------------------------- RoBERTa sentiment

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return tokenizer, model, device


def tokenize(tokenizer, texts):
    # truncate once here, so the same token ids are used for prediction and for the truncation check
    return tokenizer([str(t) for t in texts], truncation=True, max_length=MAX_LEN)["input_ids"]


def predict_sentiment(model, tokenizer, device, input_ids, batch_size=64):
    """
    Returns a sentiment score per text: S = P(positive) - P(negative), in [-1, 1]
    (same range as VADER's compound score).
    """
    label2id = {k.lower(): v for k, v in model.config.label2id.items()}
    pos, neg = label2id["positive"], label2id["negative"]

    # sort by length so each batch has similar lengths -> much less padding, much faster
    order = np.argsort([len(x) for x in input_ids])
    scores = np.zeros(len(input_ids))

    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = tokenizer.pad({"input_ids": [input_ids[i] for i in idx]}, return_tensors="pt").to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(**batch).logits
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            scores[idx] = probs[:, pos] - probs[:, neg]

            if (start // batch_size) % 100 == 0:
                print(f"  {start + len(idx)}/{len(order)}")
    return scores


def run_sentiment(df, tokenizer, model, device):
    out = pd.DataFrame(index=df.index)
    out["id"] = df["id"]

    ids_orig = tokenize(tokenizer, df["full_text"])
    ids_cf = tokenize(tokenizer, df["text_cf"])

    # if all gender swaps were after token 512, the model sees the exact same input -> no real counterfactual
    out["cf_truncated_same"] = [a == b for a, b in zip(ids_orig, ids_cf)]

    print("Scoring original texts ...")
    out["S_orig"] = predict_sentiment(model, tokenizer, device, ids_orig)
    print("Scoring gender-swapped texts ...")
    out["S_cf"] = predict_sentiment(model, tokenizer, device, ids_cf)

    # control texts only exist for part of the posts
    has_ctrl = df["has_control"].values
    ids_ctrl = tokenize(tokenizer, df.loc[has_ctrl, "text_ctrl"])
    out["ctrl_truncated_same"] = None
    out.loc[has_ctrl, "ctrl_truncated_same"] = [
        a == b for a, b in zip([x for x, h in zip(ids_orig, has_ctrl) if h], ids_ctrl)
    ]

    print("Scoring control texts ...")
    out["S_ctrl"] = np.nan
    out.loc[has_ctrl, "S_ctrl"] = predict_sentiment(model, tokenizer, device, ids_ctrl)

    # columns needed later for the breakdowns in cf_analysis.py (H2b: Hard vs Soft, swap direction)
    for col in ["n_swaps", "gender_context", "ai_category", "has_control", "n_ctrl_swaps"]:
        if col in df.columns:
            out[col] = df[col]
    return out


# ---------------------------------------------------------------- run

def main_vader():
    df = load_pairs()
    return vader_sanity_check(df)


def main():
    OUTPUT_PATH = os.path.join(CF_DIR, "cf_sentiment_scores.pkl")

    # all pairs, not only the BERT test set: this model was not trained on our data
    df = load_pairs()

    tokenizer, model, device = load_model()
    print(f"Model: {MODEL_NAME}, device: {device}")
    out = run_sentiment(df, tokenizer, model, device)

    # quick sanity checks only (the real analysis is in cf_analysis.py)
    print(f"\nMean sentiment: original {out['S_orig'].mean():+.3f}, swapped {out['S_cf'].mean():+.3f}")
    print(f"Pairs where the swap is beyond token 512: {out['cf_truncated_same'].sum()}")
    print(f"Pairs where the score changed by more than 0.1: {((out['S_cf'] - out['S_orig']).abs() > 0.1).sum()}")

    out.to_pickle(OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")
    return out
