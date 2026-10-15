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


# ---------------------------------------------------------------- 2b: sentiment

DIRECTION = {"male": "male->female", "female": "female->male", "both": "both (mixed)"}


def analyze_2b(scores, texts=None, vader=None, n_examples=5):
    """
    scores: output of cf_sentiment.py (counterfactual/cf_sentiment_scores.pkl)
    texts:  optional df with id, full_text, text_cf to print the largest shifts
    vader:  optional output of vader_sanity_check (counterfactual/cf_vader_sanity.pkl)
    """
    r = {}
    print("=" * 60)
    print("  2b. Effect of the gender swap on sentiment")
    print("=" * 60)

    # --- 0. VADER: why a contextual model is needed
    if vader is not None:
        r["vader"] = {"n": len(vader), "share_dS_zero": float((vader["dS_vader"] == 0).mean()),
                      "mean_abs_dS": float(vader["dS_vader"].abs().mean())}
        print(f"[VADER sanity check] dS = 0 in {r['vader']['share_dS_zero']:.1%} of {len(vader)} sampled pairs "
              f"-> VADER cannot measure the effect, using RoBERTa instead")

    n_all = len(scores)
    df = scores[~scores["cf_truncated_same"].astype(bool)].copy()
    df["dS_cf"] = df["S_cf"] - df["S_orig"]        # > 0 : the swapped post is more positive
    r["n_pairs_total"], r["n_pairs"] = n_all, len(df)
    r["n_pairs_truncated_removed"] = int(n_all - len(df))
    print(f"\nPairs: {n_all}, removed (swap after token 512): {n_all - len(df)}, analysed: {len(df)}")

    # --- 1. overall shift
    r["dS_all"] = paired_test(df["dS_cf"])
    d = r["dS_all"]
    r["mean_abs_dS"] = float(df["dS_cf"].abs().mean())
    r["share_abs_dS_gt_0.1"] = float((df["dS_cf"].abs() > 0.1).mean())
    print(f"\n[dS = S(cf) - S(orig)] mean {d['mean']:+.4f}, median {d['median']:+.4f}, "
          f"mean |dS| {r['mean_abs_dS']:.4f}, |dS| > 0.1 in {r['share_abs_dS_gt_0.1']:.1%} of pairs")
    print(f"   Wilcoxon p = {fmt_p(d['wilcoxon_p'])}, Cohen's d = {d['cohens_d']:+.3f}")

    # --- 2. by swap direction
    print("\n[dS by swap direction]  (positive = more positive after the swap)")
    r["dS_by_direction"] = {}
    for ctx, name in DIRECTION.items():
        sub = df[df["gender_context"] == ctx]
        if len(sub) < 2:
            continue
        res = paired_test(sub["dS_cf"])
        r["dS_by_direction"][name] = res
        print(f"   {name:15s} n={res['n']:6d}  mean {res['mean']:+.4f}  "
              f"Wilcoxon p = {fmt_p(res['wilcoxon_p'])}  d = {res['cohens_d']:+.3f}")
    m2f = df.loc[df["gender_context"] == "male", "dS_cf"]
    f2m = df.loc[df["gender_context"] == "female", "dS_cf"]
    if len(m2f) > 1 and len(f2m) > 1:
        r["direction_mannwhitney_p"] = float(stats.mannwhitneyu(m2f, f2m).pvalue)
        print(f"   male->female vs female->male (Mann-Whitney): p = {fmt_p(r['direction_mannwhitney_p'])}")

    # --- 3. H2b: male->female in Hard AI posts increases sentiment (benevolence paradox),
    #            smaller or absent in Soft AI posts
    print("\n[H2b: dS by direction x AI category]")
    r["dS_by_direction_category"] = {}
    for ctx in ["male", "female"]:
        for cat in ["hard_ai", "soft_ai", "general"]:
            sub = df[(df["gender_context"] == ctx) & (df["ai_category"] == cat)]
            if len(sub) < 2:
                continue
            res = paired_test(sub["dS_cf"])
            key = f"{DIRECTION[ctx]} | {cat}"
            r["dS_by_direction_category"][key] = res
            print(f"   {key:26s} n={res['n']:6d}  mean {res['mean']:+.4f}  "
                  f"Wilcoxon p = {fmt_p(res['wilcoxon_p'])}  d = {res['cohens_d']:+.3f}")

    hard = df.loc[(df["gender_context"] == "male") & (df["ai_category"] == "hard_ai"), "dS_cf"]
    soft = df.loc[(df["gender_context"] == "male") & (df["ai_category"] == "soft_ai"), "dS_cf"]
    if len(hard) > 1 and len(soft) > 1:
        r["H2b_hard_vs_soft_mannwhitney_p"] = float(stats.mannwhitneyu(hard, soft).pvalue)
        print(f"   male->female: Hard AI vs Soft AI (Mann-Whitney): p = {fmt_p(r['H2b_hard_vs_soft_mannwhitney_p'])}")

    # --- 4. feminization effect: both directions on one scale.
    # dS_fem = change in sentiment when the post becomes MORE female
    # (male->female: dS, female->male: -dS). Benevolence paradox -> dS_fem > 0.
    one_dir = df[df["gender_context"].isin(["male", "female"])].copy()
    one_dir["dS_fem"] = np.where(one_dir["gender_context"] == "male", one_dir["dS_cf"], -one_dir["dS_cf"])
    print("\n[Feminization effect]  dS_fem = sentiment change when the post becomes more female")
    r["dS_fem"] = {}
    for cat, s in [("all", one_dir)] + [(c, one_dir[one_dir["ai_category"] == c]) for c in ["hard_ai", "soft_ai", "general"]]:
        if len(s) < 2:
            continue
        res = paired_test(s["dS_fem"])
        r["dS_fem"][cat] = res
        print(f"   {cat:8s} n={res['n']:6d}  mean {res['mean']:+.4f}  "
              f"Wilcoxon p = {fmt_p(res['wilcoxon_p'])}  d = {res['cohens_d']:+.3f}")

    # --- 5. by number of swaps
    print("\n[|dS| by number of gender swaps]")
    bins = pd.cut(df["n_swaps"], [0, 1, 3, 10, np.inf], labels=["1", "2-3", "4-10", ">10"])
    by_n = df.groupby(bins, observed=True).agg(n=("dS_cf", "size"),
                                               mean_abs_dS=("dS_cf", lambda x: x.abs().mean()))
    r["by_n_swaps"] = by_n.reset_index().astype({"n_swaps": str}).to_dict("records")
    print(by_n.round(4).to_string())
    rho = stats.spearmanr(df["n_swaps"], df["dS_cf"].abs())
    r["n_swaps_spearman"] = {"rho": float(rho.statistic), "p": float(rho.pvalue)}
    print(f"   Spearman(n_swaps, |dS|): rho = {rho.statistic:.3f}, p = {fmt_p(rho.pvalue)}")

    # --- 6. gender vs control (step 24a)
    sub = df[df["has_control"].astype(bool) & df["ctrl_truncated_same"].eq(False)].copy()
    sub["dS_ctrl"] = sub["S_ctrl"] - sub["S_orig"]
    print("\n[Gender vs control]  (same posts; |dS| = size of the change, ignoring direction)")
    r["gender_vs_control"] = {}
    for name, s in [("all posts with control", sub),
                    ("same n of swaps", sub[sub["n_ctrl_swaps"] == sub["n_swaps"]])]:
        if len(s) < 2:
            continue
        res = paired_test(s["dS_cf"].abs() - s["dS_ctrl"].abs())   # > 0 : gender swap moves sentiment more
        res.update({"mean_abs_dS_gender": float(s["dS_cf"].abs().mean()),
                    "mean_abs_dS_control": float(s["dS_ctrl"].abs().mean())})
        r["gender_vs_control"][name] = res
        print(f"   {name:24s} n={res['n']:6d}  |dS| gender {res['mean_abs_dS_gender']:.4f} vs control "
              f"{res['mean_abs_dS_control']:.4f}  Wilcoxon p = {fmt_p(res['wilcoxon_p'])}  d = {res['cohens_d']:+.3f}")

    # --- 7. largest shifts (qualitative examples)
    if texts is not None:
        top = df.reindex(df["dS_cf"].abs().sort_values(ascending=False).index).head(n_examples)
        ex = top.merge(texts[["id", "full_text", "text_cf"]], on="id", how="left")
        print(f"\n[Largest sentiment shifts] (top {len(ex)})")
        for _, row in ex.iterrows():
            print(f"- S {row['S_orig']:+.2f} -> {row['S_cf']:+.2f} | {DIRECTION.get(row['gender_context'], row['gender_context'])}"
                  f" | {row['ai_category']} | n_swaps={row['n_swaps']}")
            for o, c in changed_sentences(str(row["full_text"]), str(row["text_cf"]))[:2]:
                print(f"  ORIG: {o[:300]}")
                print(f"  CF:   {c[:300]}")
    return r


# ---------------------------------------------------------------- 2c: embedding position

def analyze_2c(shifts, texts=None, n_examples=5):
    """
    shifts: output of cf_embedding.py (counterfactual/cf_embedding_shifts.pkl)
    delta_* = cosine distance between original and swapped embedding (HOW MUCH the post moved)
    proj_*  = position on the Soft -> Hard axis (WHERE the post moved)
    """
    r = {}
    print("=" * 60)
    print("  2c. Effect of the gender swap on embedding position")
    print("=" * 60)

    n_all = len(shifts)
    df = shifts[~shifts["cf_truncated_same"].astype(bool)].copy()
    r["n_pairs_total"], r["n_pairs"] = n_all, len(df)
    r["n_pairs_truncated_removed"] = int(n_all - len(df))
    print(f"Pairs: {n_all}, removed (swap after token 512): {n_all - len(df)}, analysed: {len(df)}")

    # the Hard-Soft gap is used as a unit, so a shift can be read as "% of the way from Soft to Hard"
    hard_c = df.loc[df["ai_category"] == "hard_ai", "proj_orig"].mean()
    soft_c = df.loc[df["ai_category"] == "soft_ai", "proj_orig"].mean()
    gap = hard_c - soft_c
    r["axis"] = {"hard_mean": float(hard_c), "soft_mean": float(soft_c), "gap": float(gap)}
    df["dProj"] = (df["proj_cf"] - df["proj_orig"]) / gap      # > 0 : moved towards Hard AI
    print(f"Axis check: Hard posts at {hard_c:.3f}, Soft posts at {soft_c:.3f} (gap {gap:.3f}). "
          f"Shifts below are in % of this gap.")

    # --- 1. how much the post moves
    d = df["delta_gender"]
    r["delta_gender"] = {"mean": float(d.mean()), "median": float(d.median())}
    print(f"\n[delta = cosine distance orig vs swapped] mean {d.mean():.5f}, median {d.median():.5f}")

    print("\n[delta by number of gender swaps]")
    bins = pd.cut(df["n_swaps"], [0, 1, 3, 10, np.inf], labels=["1", "2-3", "4-10", ">10"])
    by_n = df.groupby(bins, observed=True).agg(n=("delta_gender", "size"), mean_delta=("delta_gender", "mean"))
    r["by_n_swaps"] = by_n.reset_index().astype({"n_swaps": str}).to_dict("records")
    print(by_n.round(5).to_string())
    rho = stats.spearmanr(df["n_swaps"], df["delta_gender"])
    r["n_swaps_spearman"] = {"rho": float(rho.statistic), "p": float(rho.pvalue)}
    print(f"   Spearman(n_swaps, delta): rho = {rho.statistic:.3f}, p = {fmt_p(rho.pvalue)}")

    # --- 2. H2c: gender swaps move the embedding more than control swaps (same posts)
    sub = df[df["has_control"].astype(bool) & df["ctrl_truncated_same"].eq(False)].copy()
    print("\n[H2c: delta gender vs delta control]  (same posts)")
    r["H2c"] = {}
    for name, s in [("all posts with control", sub),
                    ("same n of swaps", sub[sub["n_ctrl_swaps"] == sub["n_swaps"]])]:
        if len(s) < 2:
            continue
        res = paired_test(s["delta_gender"] - s["delta_ctrl"])   # > 0 : gender moves the post more
        res.update({"mean_delta_gender": float(s["delta_gender"].mean()),
                    "mean_delta_control": float(s["delta_ctrl"].mean()),
                    "ratio": float(s["delta_gender"].mean() / s["delta_ctrl"].mean())})
        r["H2c"][name] = res
        print(f"   {name:24s} n={res['n']:6d}  delta gender {res['mean_delta_gender']:.5f} vs control "
              f"{res['mean_delta_control']:.5f} (x{res['ratio']:.1f})  Wilcoxon p = {fmt_p(res['wilcoxon_p'])}  "
              f"d = {res['cohens_d']:+.3f}")

    # --- 3. direction on the Hard/Soft axis
    print("\n[Shift on the Soft->Hard axis by swap direction]  (positive = towards Hard AI, % of gap)")
    r["dProj_by_direction"] = {}
    for ctx, name in DIRECTION.items():
        s = df[df["gender_context"] == ctx]
        if len(s) < 2:
            continue
        res = paired_test(s["dProj"])
        r["dProj_by_direction"][name] = res
        print(f"   {name:15s} n={res['n']:6d}  mean {res['mean']:+.3%}  "
              f"Wilcoxon p = {fmt_p(res['wilcoxon_p'])}  d = {res['cohens_d']:+.3f}")
    m2f = df.loc[df["gender_context"] == "male", "dProj"]
    f2m = df.loc[df["gender_context"] == "female", "dProj"]
    if len(m2f) > 1 and len(f2m) > 1:
        r["direction_mannwhitney_p"] = float(stats.mannwhitneyu(m2f, f2m).pvalue)
        print(f"   male->female vs female->male (Mann-Whitney): p = {fmt_p(r['direction_mannwhitney_p'])}")

    # masculinization effect: both directions on one scale.
    # dProj_masc = shift towards Hard AI when the post becomes MORE male
    # (female->male: dProj, male->female: -dProj). Stereotype "male = Hard AI" -> dProj_masc > 0.
    one_dir = df[df["gender_context"].isin(["male", "female"])].copy()
    one_dir["dProj_masc"] = np.where(one_dir["gender_context"] == "female", one_dir["dProj"], -one_dir["dProj"])
    print("\n[Masculinization effect]  shift towards Hard AI when the post becomes more male")
    r["dProj_masc"] = {}
    for cat, s in [("all", one_dir)] + [(c, one_dir[one_dir["ai_category"] == c]) for c in ["hard_ai", "soft_ai", "general"]]:
        if len(s) < 2:
            continue
        res = paired_test(s["dProj_masc"])
        r["dProj_masc"][cat] = res
        print(f"   {cat:8s} n={res['n']:6d}  mean {res['mean']:+.3%}  "
              f"Wilcoxon p = {fmt_p(res['wilcoxon_p'])}  d = {res['cohens_d']:+.3f}")

    # control: does a random word swap also push posts along the axis?
    sub["dProj_ctrl"] = (sub["proj_ctrl"] - sub["proj_orig"]) / gap
    res = paired_test(sub["dProj"].abs() - sub["dProj_ctrl"].abs())
    res.update({"mean_abs_dProj_gender": float(sub["dProj"].abs().mean()),
                "mean_abs_dProj_control": float(sub["dProj_ctrl"].abs().mean())})
    r["dProj_gender_vs_control"] = res
    print(f"\n[|axis shift| gender vs control] n={res['n']}  {res['mean_abs_dProj_gender']:.3%} vs "
          f"{res['mean_abs_dProj_control']:.3%}  Wilcoxon p = {fmt_p(res['wilcoxon_p'])}  d = {res['cohens_d']:+.3f}")

    # --- 4. examples: largest moves on the axis
    if texts is not None:
        top = df.reindex(df["dProj"].abs().sort_values(ascending=False).index).head(n_examples)
        ex = top.merge(texts[["id", "full_text", "text_cf"]], on="id", how="left")
        print(f"\n[Largest shifts on the Soft->Hard axis] (top {len(ex)})")
        for _, row in ex.iterrows():
            print(f"- shift {row['dProj']:+.1%} of gap | delta {row['delta_gender']:.4f} | "
                  f"{DIRECTION.get(row['gender_context'], row['gender_context'])} | {row['ai_category']} | n_swaps={row['n_swaps']}")
            for o, c in changed_sentences(str(row["full_text"]), str(row["text_cf"]))[:2]:
                print(f"  ORIG: {o[:300]}")
                print(f"  CF:   {c[:300]}")
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


def main_2b():
    scores = pd.read_pickle(os.path.join(CF_DIR, "cf_sentiment_scores.pkl"))
    texts = pd.read_pickle(os.path.join(CF_DIR, "counterfactual_posts_checked.pkl"))
    vader_path = os.path.join(CF_DIR, "cf_vader_sanity.pkl")
    vader = pd.read_pickle(vader_path) if os.path.exists(vader_path) else None
    results = analyze_2b(scores, texts, vader)

    out_path = os.path.join(CF_DIR, "cf_2b_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")
    return results


def main_2c():
    shifts = pd.read_pickle(os.path.join(CF_DIR, "cf_embedding_shifts.pkl"))
    texts = pd.read_pickle(os.path.join(CF_DIR, "counterfactual_posts_checked.pkl"))
    results = analyze_2c(shifts, texts)

    out_path = os.path.join(CF_DIR, "cf_2c_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")
    return results
