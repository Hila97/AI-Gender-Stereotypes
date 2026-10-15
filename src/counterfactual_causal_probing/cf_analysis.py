import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats


# All metrics and statistical tests for direction 2 (counterfactual causal probing).
# The experiment files (cf_bert_classification.py, cf_sentiment.py, cf_embedding.py)
# only save raw model outputs; everything is computed here, one function per experiment.
# Every p-value is stored in the results dict, so a Holm correction over ALL tests
# can be done at the end (step 25).

DATA_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
CF_DIR = os.path.join(DATA_DIR, "counterfactual")


# ---------------------------------------------------------------- helpers

def paired_test(diff):
    """Wilcoxon signed-rank test (H0: median diff = 0) + Cohen's d for paired samples (d_z)."""
    diff = np.asarray(diff, dtype=float)
    res = {"n": int(len(diff)), "mean": float(diff.mean()), "median": float(np.median(diff)),
           "std": float(diff.std(ddof=1)) if len(diff) > 1 else float("nan")}
    if len(diff) < 2 or np.all(diff == 0):
        res.update({"wilcoxon_p": float("nan"), "cohens_d": 0.0})
        return res
    res["wilcoxon_p"] = float(stats.wilcoxon(diff).pvalue)
    res["cohens_d"] = float(diff.mean() / diff.std(ddof=1))
    return res


def mcnemar_exact(correct_a, correct_b):
    """
    Exact McNemar test on two paired correctness vectors.
    b = correct before, wrong after; c = wrong before, correct after.
    Odds ratio = b / c (with 0.5 correction so it is defined when c = 0).
    """
    a, b_ = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    b = int((a & ~b_).sum())
    c = int((~a & b_).sum())
    p = float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c > 0 else 1.0
    return {"b_correct_to_wrong": b, "c_wrong_to_correct": c,
            "mcnemar_p": p, "odds_ratio": (b + 0.5) / (c + 0.5)}


def wilson_ci(k, n, z=1.96):
    """95% confidence interval for a proportion (works well also when k is tiny, like 3 flips)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (float(centre - half), float(centre + half))


def changed_sentences(original, flipped):
    orig_s, cf_s = re.split(r"(?<=[.!?])\s+", original), re.split(r"(?<=[.!?])\s+", flipped)
    if len(orig_s) != len(cf_s):
        return [(original, flipped)]
    return [(o, c) for o, c in zip(orig_s, cf_s) if o != c]


def fmt_p(p):
    return "nan" if p != p else (f"{p:.2e}" if p < 0.001 else f"{p:.3f}")


# ---------------------------------------------------------------- 2a: Hard/Soft classification

def analyze_2a(preds, texts=None, n_examples=5):
    """
    preds: output of cf_bert_classification.py (counterfactual/cf_classification_preds.pkl)
    texts: optional df with id, full_text, text_cf to print the flipped examples
    """
    r = {}
    n_all = len(preds)

    # pairs where every swap was after token 512 -> the model saw the same input, not a real test
    df = preds[~preds["cf_truncated_same"].astype(bool)].copy()
    r["n_pairs_total"] = n_all
    r["n_pairs_truncated_removed"] = int(n_all - len(df))
    r["n_pairs"] = len(df)

    df["flip_cf"] = df["pred_orig"] != df["pred_cf"]
    df["dP_cf"] = df["p_hard_cf"] - df["p_hard_orig"]        # > 0 : the swap made the post look more "Hard AI"

    print("=" * 60)
    print("  2a. Effect of the gender swap on Hard/Soft classification")
    print("=" * 60)
    print(f"Pairs: {n_all}, removed (swap after token 512): {n_all - len(df)}, analysed: {len(df)}")

    # --- 1. Counterfactual Flip Rate
    k = int(df["flip_cf"].sum())
    r["CFR"] = k / len(df)
    r["CFR_ci95"] = wilson_ci(k, len(df))
    print(f"\n[CFR] label flipped in {k}/{len(df)} pairs = {r['CFR']:.2%} "
          f"(95% CI {r['CFR_ci95'][0]:.2%} - {r['CFR_ci95'][1]:.2%})")

    # --- 2. McNemar: does the swap change ACCURACY (correct -> wrong more than wrong -> correct)?
    r["mcnemar"] = mcnemar_exact(df["pred_orig"] == df["label"], df["pred_cf"] == df["label"])
    acc_o = (df["pred_orig"] == df["label"]).mean()
    acc_c = (df["pred_cf"] == df["label"]).mean()
    r["accuracy_orig"], r["accuracy_cf"] = float(acc_o), float(acc_c)
    m = r["mcnemar"]
    print(f"\n[Accuracy] original {acc_o:.3f} -> swapped {acc_c:.3f}")
    print(f"[McNemar] correct->wrong {m['b_correct_to_wrong']}, wrong->correct {m['c_wrong_to_correct']}, "
          f"p = {fmt_p(m['mcnemar_p'])}, OR = {m['odds_ratio']:.2f}")

    # --- 3. delta P: does the swap shift the probability of Hard AI, even without flipping the label?
    r["dP_all"] = paired_test(df["dP_cf"])
    d = r["dP_all"]
    print(f"\n[dP = P(hard|cf) - P(hard|orig)] mean {d['mean']:+.4f}, median {d['median']:+.4f}, "
          f"mean |dP| {df['dP_cf'].abs().mean():.4f}")
    print(f"   Wilcoxon p = {fmt_p(d['wilcoxon_p'])}, Cohen's d = {d['cohens_d']:+.3f}")

    # --- 4. by swap direction: male->female vs female->male (H2a: stereotype = male <-> Hard AI)
    print("\n[dP by swap direction]  (positive = more Hard AI after the swap)")
    direction = {"male": "male->female", "female": "female->male", "both": "both (mixed)"}
    r["dP_by_direction"] = {}
    for ctx, name in direction.items():
        sub = df[df["gender_context"] == ctx]
        if len(sub) < 2:
            continue
        res = paired_test(sub["dP_cf"])
        r["dP_by_direction"][name] = res
        print(f"   {name:15s} n={res['n']:5d}  mean {res['mean']:+.4f}  "
              f"Wilcoxon p = {fmt_p(res['wilcoxon_p'])}  d = {res['cohens_d']:+.3f}")

    m2f = df.loc[df["gender_context"] == "male", "dP_cf"]
    f2m = df.loc[df["gender_context"] == "female", "dP_cf"]
    if len(m2f) > 1 and len(f2m) > 1:
        p = float(stats.mannwhitneyu(m2f, f2m).pvalue)
        r["direction_mannwhitney_p"] = p
        print(f"   male->female vs female->male (Mann-Whitney): p = {fmt_p(p)}")

    # --- 5. by true label: is the effect different for Hard vs Soft posts?
    print("\n[dP by true label]")
    r["dP_by_label"] = {}
    for lab, name in [(0, "Hard AI"), (1, "Soft AI")]:
        sub = df[df["label"] == lab]
        res = paired_test(sub["dP_cf"])
        r["dP_by_label"][name] = res
        print(f"   {name:8s} n={res['n']:5d}  mean {res['mean']:+.4f}  "
              f"Wilcoxon p = {fmt_p(res['wilcoxon_p'])}  d = {res['cohens_d']:+.3f}")

    # --- 6. by number of swaps: more swapped words -> bigger effect?
    print("\n[|dP| by number of gender swaps]")
    bins = pd.cut(df["n_swaps"], [0, 1, 3, 10, np.inf], labels=["1", "2-3", "4-10", ">10"])
    by_n = df.groupby(bins, observed=True).agg(n=("dP_cf", "size"),
                                               mean_abs_dP=("dP_cf", lambda x: x.abs().mean()),
                                               CFR=("flip_cf", "mean"))
    r["by_n_swaps"] = by_n.reset_index().astype({"n_swaps": str}).to_dict("records")
    print(by_n.round(4).to_string())
    rho = stats.spearmanr(df["n_swaps"], df["dP_cf"].abs())
    r["n_swaps_spearman"] = {"rho": float(rho.statistic), "p": float(rho.pvalue)}
    print(f"   Spearman(n_swaps, |dP|): rho = {rho.statistic:.3f}, p = {fmt_p(rho.pvalue)}")

    # --- 7. gender vs control: is the gender effect bigger than swapping 'any' word? (step 24a)
    r["gender_vs_control"] = compare_with_control(df)

    # --- 8. the flipped pairs
    flips = df[df["flip_cf"]]
    if texts is not None and len(flips):
        print(f"\n[Flipped examples] ({min(n_examples, len(flips))} of {len(flips)})")
        ex = flips.merge(texts[["id", "full_text", "text_cf"]], on="id", how="left")
        for _, row in ex.head(n_examples).iterrows():
            print(f"- true={'Hard' if row['label'] == 0 else 'Soft'} | P(hard) {row['p_hard_orig']:.2f} -> {row['p_hard_cf']:.2f}"
                  f" | {row['gender_context']} | n_swaps={row['n_swaps']}")
            # show only the sentences that the swap changed (the start of the post is often unchanged)
            for o, c in changed_sentences(str(row["full_text"]), str(row["text_cf"]))[:3]:
                print(f"  ORIG: {o[:300]}")
                print(f"  CF:   {c[:300]}")
    return r


def compare_with_control(df):
    """Gender swap vs control swap on the SAME posts (only posts that have a control version)."""
    r = {}
    sub = df[df["has_control"].astype(bool) & df["ctrl_truncated_same"].eq(False)].copy()
    sub["dP_ctrl"] = sub["p_hard_ctrl"] - sub["p_hard_orig"]
    sub["flip_ctrl"] = sub["pred_orig"] != sub["pred_ctrl"]

    print("\n[Gender vs control]  (same posts; |dP| = size of the change, ignoring direction)")
    for name, s in [("all posts with control", sub),
                    # sensitivity: same number of swapped words in both versions
                    ("same n of swaps", sub[sub["n_ctrl_swaps"] == sub["n_swaps"]])]:
        if len(s) < 2:
            continue
        res = paired_test(s["dP_cf"].abs() - s["dP_ctrl"].abs())   # > 0 : gender swap moves the model more
        res.update({"CFR_gender": float(s["flip_cf"].mean()), "CFR_control": float(s["flip_ctrl"].mean()),
                    "mean_abs_dP_gender": float(s["dP_cf"].abs().mean()),
                    "mean_abs_dP_control": float(s["dP_ctrl"].abs().mean())})
        r[name] = res
        print(f"   {name:24s} n={res['n']:5d}  |dP| gender {res['mean_abs_dP_gender']:.4f} vs control "
              f"{res['mean_abs_dP_control']:.4f}  CFR {res['CFR_gender']:.2%} vs {res['CFR_control']:.2%}  "
              f"Wilcoxon p = {fmt_p(res['wilcoxon_p'])}  d = {res['cohens_d']:+.3f}")
    return r


# ---------------------------------------------------------------- run

def main_2a():
    preds = pd.read_pickle(os.path.join(CF_DIR, "cf_classification_preds.pkl"))
    texts = pd.read_pickle(os.path.join(CF_DIR, "counterfactual_posts_checked.pkl"))
    results = analyze_2a(preds, texts)

    out_path = os.path.join(CF_DIR, "cf_2a_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")
    return results
