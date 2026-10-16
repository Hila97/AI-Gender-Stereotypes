import difflib
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


def holm(pvals):
    """Holm-Bonferroni correction. Returns adjusted p-values in the original order."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def fmt_p(p):
    return "nan" if p != p else (f"{p:.2e}" if p < 0.001 else f"{p:.3f}")


DIRECTION = {"male": "male->female", "female": "female->male", "both": "both (mixed)"}

# masculinization sign: +1 when the swap makes the post MORE male (female->male), -1 for male->female.
# Multiplying a change by it puts both directions on one scale ("what happens when a post becomes more male").
def masc_sign(gender_context):
    return np.where(np.asarray(gender_context) == "female", 1.0, -1.0)


# ---------------------------------------------------------------- extra checks (all experiments)

MIN_WORDS = 20   # posts shorter than this are "short": one swapped word is a big part of the text


def swapped_pairs(original, flipped):
    """Which word pairs the swap changed in one post, e.g. {'he/she', 'his/her'} (order-free)."""
    a = re.findall(r"[a-z]+", str(original).lower())
    b = re.findall(r"[a-z]+", str(flipped).lower())
    if len(a) == len(b):          # the usual case: one word replaced by one word
        diffs = [(x, y) for x, y in zip(a, b) if x != y]
    else:                         # rare: a swap changed the number of words -> align with difflib
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        diffs = [(" ".join(a[i1:i2]), " ".join(b[j1:j2]))
                 for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag == "replace"]
    return {"/".join(sorted(d)) for d in diffs}


def text_features(texts, ids):
    """Per post: number of words and the swapped word pair ('mixed' if the post has more than one pair type)."""
    t = texts.drop_duplicates("id")
    t = t[t["id"].isin(set(ids))]
    pairs = [swapped_pairs(o, c) for o, c in zip(t["full_text"], t["text_cf"])]
    return pd.DataFrame({
        "id": t["id"].values,
        "n_words": [len(str(x).split()) for x in t["full_text"]],
        "pair": [next(iter(p)) if len(p) == 1 else ("mixed" if p else "none") for p in pairs],
    })


def _line(name, res, pct=False, width=22):
    m = f"{res['mean']:+.3%}" if pct else f"{res['mean']:+.4f}"
    return (f"   {name:{width}s} n={res['n']:6d}  mean {m}  "
            f"Wilcoxon p = {fmt_p(res['wilcoxon_p'])}  d = {res['cohens_d']:+.3f}")


def length_check(df, col, title, pct=False):
    """Same test on long posts only: is the effect driven by very short posts?"""
    print(f"\n[Length check: {title}]  (short = fewer than {MIN_WORDS} words)")
    r = {}
    for name, s in [("all", df), (f">= {MIN_WORDS} words", df[df["n_words"] >= MIN_WORDS]),
                    (f"< {MIN_WORDS} words", df[df["n_words"] < MIN_WORDS])]:
        if len(s) < 2:
            continue
        r[name] = paired_test(s[col])
        print(_line(name, r[name], pct))
    return r


def pair_breakdown(df, col, title, min_n=100, pct=False):
    """
    Effect per swapped word pair (he/she, his/her, man/woman ...).
    Only posts with ONE pair type are assigned to it, posts with several are 'mixed'.
    Shows whether the effect comes from all gender words or from a few specific ones.
    """
    print(f"\n[Word-pair breakdown: {title}]  (pairs with n >= {min_n})")
    r = {}
    counts = df["pair"].value_counts()
    for pair in counts[counts >= min_n].index:
        r[pair] = paired_test(df.loc[df["pair"] == pair, col])
        print(_line(pair, r[pair], pct))
    return r


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
    r["dP_by_direction"] = {}
    for ctx, name in DIRECTION.items():
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

    # --- 4b. exact H2a, as written in the plan: the swap should push the post TOWARDS the other class.
    #   female->male in posts the model calls Soft -> P(hard) should go UP   (expected dP > 0)
    #   male->female in posts the model calls Hard -> P(hard) should go DOWN (expected dP < 0)
    # Split by the PREDICTED class of the original (what the model "thinks" before the swap).
    # The other two cells are shown too: there P(hard) is already near 0 or 1, so it can hardly move.
    print("\n[H2a exact: dP by swap direction x predicted class of the original]")
    r["H2a_exact"] = {}
    for ctx, expected in [("female", "+"), ("male", "-")]:
        for pc, pname in [(1, "pred Soft"), (0, "pred Hard")]:
            sub = df[(df["gender_context"] == ctx) & (df["pred_orig"] == pc)]
            if len(sub) < 2:
                continue
            key = f"{DIRECTION[ctx]} | {pname}"
            res = paired_test(sub["dP_cf"])
            main_cell = (ctx == "female" and pc == 1) or (ctx == "male" and pc == 0)
            res["expected_sign"] = expected if main_cell else None
            r["H2a_exact"][key] = res
            print(_line(key, res, width=26) + (f"   <- H2a, expected {expected}" if main_cell else ""))

    # both directions on one scale: change in P(hard) when the post becomes MORE male (stereotype -> > 0)
    one_dir = df[df["gender_context"].isin(["male", "female"])].copy()
    one_dir["dP_masc"] = masc_sign(one_dir["gender_context"]) * one_dir["dP_cf"]
    r["dP_masc"] = paired_test(one_dir["dP_masc"])
    print("\n[Masculinization effect]  change in P(hard) when the post becomes more male")
    print(_line("all", r["dP_masc"]))

    # --- 4c. extra checks: text length and word pair (need the texts)
    if texts is not None:
        one_dir = one_dir.merge(text_features(texts, one_dir["id"]), on="id", how="left")
        r["length_check"] = length_check(one_dir, "dP_masc", "masculinization effect on P(hard)")
        r["pair_breakdown"] = pair_breakdown(one_dir, "dP_masc", "masculinization effect on P(hard)", min_n=50)

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

    # --- 4b. extra checks: text length and word pair
    # (the largest shifts were short idioms, so: does the effect survive without short posts?)
    if texts is not None:
        one_dir = one_dir.merge(text_features(texts, one_dir["id"]), on="id", how="left")
        r["length_check"] = length_check(one_dir, "dS_fem", "feminization effect on sentiment")
        r["length_check_hard_ai"] = length_check(one_dir[one_dir["ai_category"] == "hard_ai"], "dS_fem",
                                                 "feminization effect, Hard AI posts only (H2b)")
        r["pair_breakdown"] = pair_breakdown(one_dir, "dS_fem", "feminization effect on sentiment")

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


def analyze_2b_emotions(emo):
    """
    GoEmotions part of 2b: which emotions change when a post becomes MORE female?
    dE_fem = change in emotion probability when the post becomes more female
    (male->female: cf - orig, female->male: orig - cf), same idea as dS_fem.
    Holm correction over the 28 emotions.
    """
    r = {}
    df = emo[~emo["cf_truncated_same"].astype(bool) & emo["gender_context"].isin(["male", "female"])]
    sign = np.where(df["gender_context"] == "male", 1.0, -1.0)
    emotions = [c[2:-5] for c in df.columns if c.startswith("E_") and c.endswith("_orig")]

    print("\n[GoEmotions]  dE_fem = change in emotion probability when the post becomes more female")
    rows = []
    for e in emotions:
        d_fem = sign * (df[f"E_{e}_cf"].values - df[f"E_{e}_orig"].values)
        res = paired_test(d_fem)
        row = {"emotion": e, "base_rate": float(df[f"E_{e}_orig"].mean()),
               "mean_dE_fem": res["mean"], "d": res["cohens_d"], "p": res["wilcoxon_p"]}
        for cat in ["hard_ai", "soft_ai"]:
            m = (df["ai_category"] == cat).values
            row[f"d_{cat}"] = paired_test(d_fem[m])["cohens_d"] if m.sum() > 1 else float("nan")
        rows.append(row)

    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p"].fillna(1.0))
    table = table.sort_values("d", ascending=False)
    r["table"] = table.to_dict("records")
    r["n"] = len(df)

    show = table[["emotion", "base_rate", "mean_dE_fem", "d", "p_holm", "d_hard_ai", "d_soft_ai"]].copy()
    show["p_holm"] = show["p_holm"].map(fmt_p)
    print(f"   n = {len(df)} pairs (male->female and female->male), sorted by Cohen's d")
    print("   positive d = the emotion is stronger when the post is more female")
    print(show.round(4).to_string(index=False))

    # gender vs control: total emotional change = sum over the 28 emotions of |change|
    if f"E_{emotions[0]}_ctrl" in emo.columns:
        sub = emo[~emo["cf_truncated_same"].astype(bool) & emo["has_control"].astype(bool)
                  & emo["ctrl_truncated_same"].eq(False)]
        tot_g = sum((sub[f"E_{e}_cf"] - sub[f"E_{e}_orig"]).abs() for e in emotions)
        tot_c = sum((sub[f"E_{e}_ctrl"] - sub[f"E_{e}_orig"]).abs() for e in emotions)
        print("\n[Total emotional change: gender vs control]  (sum of |change| over the 28 emotions, same posts)")
        r["gender_vs_control"] = {}
        same_n = (sub["n_ctrl_swaps"] == sub["n_swaps"]).values
        for name, m in [("all posts with control", np.ones(len(sub), bool)), ("same n of swaps", same_n)]:
            if m.sum() < 2:
                continue
            res = paired_test(tot_g.values[m] - tot_c.values[m])   # > 0 : gender changes emotions more
            res.update({"mean_total_gender": float(tot_g.values[m].mean()),
                        "mean_total_control": float(tot_c.values[m].mean())})
            r["gender_vs_control"][name] = res
            print(f"   {name:24s} n={res['n']:6d}  gender {res['mean_total_gender']:.4f} vs control "
                  f"{res['mean_total_control']:.4f}  Wilcoxon p = {fmt_p(res['wilcoxon_p'])}  d = {res['cohens_d']:+.3f}")
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

    # --- 3b. extra checks: text length and word pair
    if texts is not None:
        feats = text_features(texts, df["id"])
        one_dir = one_dir.merge(feats, on="id", how="left")
        r["length_check"] = length_check(one_dir, "dProj_masc", "masculinization shift on the axis", pct=True)
        same_n = sub[sub["n_ctrl_swaps"] == sub["n_swaps"]].merge(feats, on="id", how="left")
        same_n["delta_diff"] = same_n["delta_gender"] - same_n["delta_ctrl"]
        r["length_check_H2c"] = length_check(same_n, "delta_diff", "H2c delta gender - delta control, same n of swaps")
        r["pair_breakdown"] = pair_breakdown(one_dir, "dProj_masc", "masculinization shift on the axis", pct=True)

    # --- 3c. robustness: the same tests with the [CLS] vector instead of mean pooling
    if "proj_orig_cls" in df.columns:
        r["cls"] = cls_check_2c(df, one_dir)

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


def cls_check_2c(df, one_dir):
    """
    Robustness check: repeat the two main 2c tests with the [CLS] vector instead of mean pooling.
    Mean pooling is the main measure (raw [CLS] of bert-base is a weaker sentence vector),
    but if the conclusions hold with both, they do not depend on how the post vector was built.
    """
    r = {}
    print("\n[Robustness: [CLS] vector instead of mean pooling]")
    hard_c = df.loc[df["ai_category"] == "hard_ai", "proj_orig_cls"].mean()
    soft_c = df.loc[df["ai_category"] == "soft_ai", "proj_orig_cls"].mean()
    gap = hard_c - soft_c
    r["axis_gap"] = float(gap)

    # H2c with CLS: same posts, same n of swaps
    s = df[df["has_control"].astype(bool) & df["ctrl_truncated_same"].eq(False)
           & (df["n_ctrl_swaps"] == df["n_swaps"])]
    if len(s) > 1:
        res = paired_test(s["delta_gender_cls"] - s["delta_ctrl_cls"])
        res.update({"mean_delta_gender": float(s["delta_gender_cls"].mean()),
                    "mean_delta_control": float(s["delta_ctrl_cls"].mean()),
                    "ratio": float(s["delta_gender_cls"].mean() / s["delta_ctrl_cls"].mean())})
        r["H2c_same_n"] = res
        print(f"   H2c same n of swaps: delta gender {res['mean_delta_gender']:.5f} vs control "
              f"{res['mean_delta_control']:.5f} (x{res['ratio']:.1f})  Wilcoxon p = {fmt_p(res['wilcoxon_p'])}  "
              f"d = {res['cohens_d']:+.3f}")

    # masculinization on the CLS axis
    o = one_dir[["id", "dProj_masc"]].merge(
        df[["id", "proj_orig_cls", "proj_cf_cls", "gender_context", "ai_category"]], on="id")
    o["dProj_masc_cls"] = masc_sign(o["gender_context"]) * (o["proj_cf_cls"] - o["proj_orig_cls"]) / gap
    r["dProj_masc"] = {}
    for cat, x in [("all", o)] + [(c, o[o["ai_category"] == c]) for c in ["hard_ai", "soft_ai", "general"]]:
        if len(x) < 2:
            continue
        r["dProj_masc"][cat] = paired_test(x["dProj_masc_cls"])
        print(_line(f"masculinization {cat}", r["dProj_masc"][cat], pct=True, width=24))

    # do the two vectors agree post by post?
    rho = stats.spearmanr(o["dProj_masc"], o["dProj_masc_cls"])
    r["spearman_mean_vs_cls"] = {"rho": float(rho.statistic), "p": float(rho.pvalue)}
    print(f"   Spearman(mean-pooling shift, CLS shift) per post: rho = {rho.statistic:.3f}, p = {fmt_p(rho.pvalue)}")
    return r


def axis_check_2c(shifts, e_orig, e_cf):
    """
    Sensitivity check for the Soft->Hard axis.
    Problem: the axis is (mean of Hard posts) - (mean of Soft posts). If Hard posts contain more
    male words than Soft posts, the axis partly points in the "male" direction, and a gender swap
    would move posts along it just because of that (data composition, not a stereotype in BERT).
    Fix: a gender-balanced axis, built separately inside male posts and inside female posts:
        axis = average of (Hard_male - Soft_male) and (Hard_female - Soft_female)
    If the masculinization effect stays with this axis, it is not an artifact of the composition.
    e_orig, e_cf: the saved embeddings (cf_embeddings_orig.npy / _cf.npy), same row order as shifts.
    """
    r = {}
    e_orig = e_orig.astype(np.float32)
    e_cf = e_cf.astype(np.float32)
    cat = shifts["ai_category"].values
    ctx = shifts["gender_context"].values
    valid = ~shifts["cf_truncated_same"].astype(bool).values

    def unit(v):
        return v / np.linalg.norm(v)

    def centroid(mask):
        return e_orig[mask].mean(0)

    axis_orig = unit(centroid(cat == "hard_ai") - centroid(cat == "soft_ai"))
    parts = [centroid((cat == "hard_ai") & (ctx == g)) - centroid((cat == "soft_ai") & (ctx == g))
             for g in ["male", "female"]]
    axis_bal = unit(np.mean(parts, axis=0))
    gender_dir = unit(centroid(ctx == "male") - centroid(ctx == "female"))

    print("=" * 60)
    print("  2c. Axis sensitivity check (gender-balanced Soft->Hard axis)")
    print("=" * 60)
    r["cos_axis_orig_vs_gender"] = float(axis_orig @ gender_dir)
    r["cos_axis_balanced_vs_gender"] = float(axis_bal @ gender_dir)
    r["cos_axis_orig_vs_balanced"] = float(axis_orig @ axis_bal)
    print(f"cos(original axis, male-female direction) = {r['cos_axis_orig_vs_gender']:+.3f}")
    print(f"cos(balanced axis, male-female direction) = {r['cos_axis_balanced_vs_gender']:+.3f}")
    print(f"cos(original axis, balanced axis)         = {r['cos_axis_orig_vs_balanced']:+.3f}")

    for name, axis in [("original axis", axis_orig), ("balanced axis", axis_bal)]:
        p_o, p_c = e_orig @ axis, e_cf @ axis
        gap = p_o[cat == "hard_ai"].mean() - p_o[cat == "soft_ai"].mean()
        d_proj = (p_c - p_o) / gap
        # more male -> positive (female->male as is, male->female flipped)
        masc = np.where(ctx == "female", d_proj, -d_proj)
        print(f"\n[{name}] Hard-Soft gap {gap:.3f}")
        r[name] = {"gap": float(gap)}
        for c in ["all", "hard_ai", "soft_ai", "general"]:
            m = valid & np.isin(ctx, ["male", "female"]) & ((cat == c) if c != "all" else True)
            res = paired_test(masc[m])
            r[name][c] = res
            print(f"   masculinization {c:8s} n={res['n']:6d}  mean {res['mean']:+.3%}  "
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


def main_2b():
    scores = pd.read_pickle(os.path.join(CF_DIR, "cf_sentiment_scores.pkl"))
    texts = pd.read_pickle(os.path.join(CF_DIR, "counterfactual_posts_checked.pkl"))
    vader_path = os.path.join(CF_DIR, "cf_vader_sanity.pkl")
    vader = pd.read_pickle(vader_path) if os.path.exists(vader_path) else None
    results = analyze_2b(scores, texts, vader)

    # third stage of 2b: emotion profile (only if cf_sentiment.main_goemotions() was run)
    emo_path = os.path.join(CF_DIR, "cf_goemotions_scores.pkl")
    if os.path.exists(emo_path):
        results["goemotions"] = analyze_2b_emotions(pd.read_pickle(emo_path))

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


def main_2c_axis_check():
    shifts = pd.read_pickle(os.path.join(CF_DIR, "cf_embedding_shifts.pkl"))
    e_orig = np.load(os.path.join(CF_DIR, "cf_embeddings_orig.npy"))
    e_cf = np.load(os.path.join(CF_DIR, "cf_embeddings_cf.npy"))
    results = axis_check_2c(shifts, e_orig, e_cf)

    out_path = os.path.join(CF_DIR, "cf_2c_axis_check.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")
    return results


# ---------------------------------------------------------------- summary of direction 2

def _load_json(name):
    path = os.path.join(CF_DIR, name)
    if not os.path.exists(path):
        print(f"(missing {name}, skipped)")
        return None
    with open(path) as f:
        return json.load(f)


def _d_ci95(d, n):
    """Approximate 95% CI of a paired Cohen's d (d_z): se = sqrt(1/n + d^2 / 2n)."""
    if n is None or n < 2 or d != d:
        return (float("nan"), float("nan"))
    se = np.sqrt(1 / n + d ** 2 / (2 * n))
    return (float(d - 1.96 * se), float(d + 1.96 * se))


def collect_main_tests(r2a, r2b, r2c, rax):
    """
    One row per main test of direction 2. 'expected' is the sign the stereotype predicts
    (None = no direction predicted, e.g. gender vs control on |change|).
    """
    rows = []

    def add(exp, test, res, expected=None, hyp=None, p_key="wilcoxon_p"):
        if not res:
            return
        n, d = res.get("n"), res.get("cohens_d", float("nan"))
        rows.append({"experiment": exp, "test": test, "hypothesis": hyp, "expected_sign": expected,
                     "n": n, "mean": res.get("mean", float("nan")), "d": d,
                     "d_ci95": _d_ci95(d, n), "p": res.get(p_key, float("nan"))})

    g = lambda r, *keys: (None if r is None else _get(r, keys))

    add("2a", "McNemar: accuracy orig vs swapped", g(r2a, "mcnemar") and
        {"n": r2a["n_pairs"], "cohens_d": float("nan"), "mean": r2a["CFR"], "mcnemar_p": r2a["mcnemar"]["mcnemar_p"]},
        p_key="mcnemar_p")
    add("2a", "dP overall", g(r2a, "dP_all"))
    add("2a", "H2a: female->male in predicted Soft (dP)", g(r2a, "H2a_exact", "female->male | pred Soft"), "+", "H2a")
    add("2a", "H2a: male->female in predicted Hard (dP)", g(r2a, "H2a_exact", "male->female | pred Hard"), "-", "H2a")
    add("2a", "masculinization dP", g(r2a, "dP_masc"), "+")
    add("2a", "|dP| gender - control, same n", g(r2a, "gender_vs_control", "same n of swaps"))

    add("2b", "dS overall", g(r2b, "dS_all"))
    add("2b", "feminization dS, all", g(r2b, "dS_fem", "all"), "+")
    add("2b", "H2b: feminization dS, Hard AI", g(r2b, "dS_fem", "hard_ai"), "+", "H2b")
    add("2b", "feminization dS, Soft AI", g(r2b, "dS_fem", "soft_ai"))
    add("2b", "|dS| gender - control, same n", g(r2b, "gender_vs_control", "same n of swaps"))
    add("2b", "GoEmotions total change gender - control, same n",
        g(r2b, "goemotions", "gender_vs_control", "same n of swaps"))

    add("2c", "H2c: delta gender - control, same n", g(r2c, "H2c", "same n of swaps"), "+", "H2c")
    add("2c", "masculinization axis shift (original axis)", g(r2c, "dProj_masc", "all"), "+")
    add("2c", "masculinization axis shift (balanced axis)", g(rax, "balanced axis", "all"), "+")
    add("2c", "|axis shift| gender - control", g(r2c, "dProj_gender_vs_control"))
    add("2c", "masculinization axis shift (CLS)", g(r2c, "cls", "dProj_masc", "all"), "+")
    return rows


def _get(d, keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def hypothesis_verdicts(rows, r2b=None, alpha=0.05):
    """
    Supported            = every test of the hypothesis is significant after Holm, in the expected
                           direction, with at least a small effect (|d| >= 0.2).
    Negligible           = significant and in the expected direction, but |d| < 0.2.
    Not supported        = anything else (not significant, or the opposite direction).
    """
    out = {}
    for hyp in ["H2a", "H2b", "H2c"]:
        hr = [r for r in rows if r["hypothesis"] == hyp]
        if not hr:
            continue
        right = [np.sign(r["mean"]) == (1 if r["expected_sign"] == "+" else -1) for r in hr]
        sig = [r["p_holm"] < alpha for r in hr]
        big = [abs(r["d"]) >= 0.2 for r in hr]
        if all(right) and all(sig) and all(big):
            verdict = "supported"
        elif all(right) and all(sig):
            verdict = "expected direction, but negligible effect (|d| < 0.2)"
        elif any(s and not ok for s, ok in zip(sig, right)):
            verdict = "not supported (significant in the OPPOSITE direction)"
        else:
            verdict = "not supported"
        out[hyp] = {"verdict": verdict, "tests": [r["test"] for r in hr]}
    # H2b also says the effect is stronger in Hard than in Soft posts
    if r2b and "H2b" in out and "dS_fem" in r2b:
        h, s = r2b["dS_fem"].get("hard_ai"), r2b["dS_fem"].get("soft_ai")
        if h and s:
            out["H2b"]["hard_stronger_than_soft"] = bool(h["mean"] > s["mean"])
            out["H2b"]["hard_vs_soft_mannwhitney_p"] = r2b.get("H2b_hard_vs_soft_mannwhitney_p")
    return out


def cross_experiment_links():
    """
    Do the three experiments agree post by post? For every one-direction post (male->female or
    female->male), the masculinization change in each experiment, and Spearman between them.
    2a only has the test set, so links with 2a use those posts; 2b-2c use all posts.
    """
    r = {}
    paths = {k: os.path.join(CF_DIR, f) for k, f in [("2a", "cf_classification_preds.pkl"),
                                                     ("2b", "cf_sentiment_scores.pkl"),
                                                     ("2c", "cf_embedding_shifts.pkl")]}
    if not all(os.path.exists(p) for p in paths.values()):
        print("(cross-experiment links skipped: a results pkl is missing)")
        return r

    def prep(df):
        df = df[~df["cf_truncated_same"].astype(bool) & df["gender_context"].isin(["male", "female"])]
        return df.drop_duplicates("id")

    a = prep(pd.read_pickle(paths["2a"]))
    b = prep(pd.read_pickle(paths["2b"]))
    c = pd.read_pickle(paths["2c"])
    c = c[~c["cf_truncated_same"].astype(bool)]
    gap = (c.loc[c["ai_category"] == "hard_ai", "proj_orig"].mean()
           - c.loc[c["ai_category"] == "soft_ai", "proj_orig"].mean())   # same unit as in analyze_2c
    c = prep(c)

    a = pd.DataFrame({"id": a["id"], "dP_masc": masc_sign(a["gender_context"]) * (a["p_hard_cf"] - a["p_hard_orig"]),
                      "abs_dP": (a["p_hard_cf"] - a["p_hard_orig"]).abs()})
    b = pd.DataFrame({"id": b["id"], "dS_masc": masc_sign(b["gender_context"]) * (b["S_cf"] - b["S_orig"]),
                      "abs_dS": (b["S_cf"] - b["S_orig"]).abs()})
    c = pd.DataFrame({"id": c["id"], "dProj_masc": masc_sign(c["gender_context"]) * (c["proj_cf"] - c["proj_orig"]) / gap,
                      "delta": c["delta_gender"]})

    print("\n[Cross-experiment links]  Spearman per post (one-direction posts)")
    links = [("2a-2c", a, c, "dP_masc", "dProj_masc", "classifier shift vs embedding axis shift"),
             ("2a-2b", a, b, "dP_masc", "dS_masc", "classifier shift vs sentiment shift"),
             ("2b-2c", b, c, "dS_masc", "dProj_masc", "sentiment shift vs embedding axis shift"),
             ("2a-2c size", a, c, "abs_dP", "delta", "|dP| vs embedding distance"),
             ("2b-2c size", b, c, "abs_dS", "delta", "|dS| vs embedding distance")]
    for key, x, y, cx, cy, desc in links:
        m = x.merge(y, on="id")
        if len(m) < 3:
            continue
        rho = stats.spearmanr(m[cx], m[cy])
        r[key] = {"description": desc, "n": len(m), "rho": float(rho.statistic), "p": float(rho.pvalue)}
        print(f"   {desc:42s} n={len(m):6d}  rho = {rho.statistic:+.3f}  p = {fmt_p(rho.pvalue)}")
    return r


def main_summary():
    """
    Final step of direction 2:
      1. all main tests in one table, with Holm correction over ALL of them (not per experiment)
      2. a verdict for H2a, H2b, H2c
      3. post-by-post links between the experiments
    Saves counterfactual/direction2_results.json (also used by cf_visualization.py).
    """
    r2a, r2b = _load_json("cf_2a_results.json"), _load_json("cf_2b_results.json")
    r2c, rax = _load_json("cf_2c_results.json"), _load_json("cf_2c_axis_check.json")

    rows = collect_main_tests(r2a, r2b, r2c, rax)
    valid = [i for i, row in enumerate(rows) if row["p"] == row["p"]]
    adj = holm([rows[i]["p"] for i in valid])
    for row in rows:
        row["p_holm"] = float("nan")
    for i, p in zip(valid, adj):
        rows[i]["p_holm"] = float(p)

    print("=" * 60)
    print("  Direction 2: all main tests (Holm over all tests)")
    print("=" * 60)
    table = pd.DataFrame(rows)
    show = table[["experiment", "test", "expected_sign", "n", "mean", "d", "p", "p_holm"]].copy()
    show["expected_sign"] = show["expected_sign"].fillna("")
    show["p"], show["p_holm"] = show["p"].map(fmt_p), show["p_holm"].map(fmt_p)
    print(show.round(4).to_string(index=False))

    verdicts = hypothesis_verdicts(rows, r2b)
    print("\n[Hypotheses]")
    for hyp, v in verdicts.items():
        print(f"   {hyp}: {v['verdict']}")

    links = cross_experiment_links()

    results = {"tests": rows, "hypotheses": verdicts, "cross_experiment": links}
    out_path = os.path.join(CF_DIR, "direction2_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")
    return results
