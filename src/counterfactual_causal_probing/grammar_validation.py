import math
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import language_tool_python


# Stage 1 (rule-based corrections) is done inside swap_gender() in
# counterfactual_generation.py: her -> his/him and his -> her/hers are chosen
# by spaCy POS/dependency tags (possessive vs. objective case).
#
# Stage 2 (this file): LanguageTool validation pass.
# We only FLAG errors, we do not auto-correct, because LanguageTool would also
# "fix" errors that already exist in the original post, and then the original
# and the counterfactual would differ in more than gender.


# split after . ! ? -- the swap never changes punctuation, so the original and
# the counterfactual split into the same number of sentences
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def get_rule_id(match):
    # language_tool_python >= 3 uses rule_id, older versions use ruleId
    return getattr(match, "rule_id", None) or getattr(match, "ruleId", None)


def changed_sentence_pairs(original, flipped):
    """Return only the (original, flipped) sentence pairs that were changed by the swap."""
    orig_sents = SENT_SPLIT.split(original)
    cf_sents = SENT_SPLIT.split(flipped)

    if len(orig_sents) != len(cf_sents):
        # should not happen, fall back to the whole text
        return [(original, flipped)]

    return [(o, c) for o, c in zip(orig_sents, cf_sents) if o != c]


def find_new_errors(tool, original, flipped):
    """
    Errors that LanguageTool finds in the counterfactual but NOT in the original,
    i.e. errors introduced by the swap. Only changed sentences are checked (much faster).
    """
    new_errors = []

    for orig_sent, cf_sent in changed_sentence_pairs(original, flipped):
        orig_rules = Counter(get_rule_id(m) for m in tool.check(orig_sent))
        cf_matches = tool.check(cf_sent)
        cf_rules = Counter(get_rule_id(m) for m in cf_matches)

        for m in cf_matches:
            rule = get_rule_id(m)
            # a rule that fires more times in the counterfactual than in the original = new error
            if cf_rules[rule] > orig_rules[rule]:
                new_errors.append({
                    "rule_id": rule,
                    "message": m.message,
                    "original_sentence": orig_sent,
                    "flipped_sentence": cf_sent,
                })
                cf_rules[rule] -= 1

    return new_errors


def run_grammar_validation(df, text_col="full_text", cf_col="text_cf", n_threads=4):
    tool = language_tool_python.LanguageTool("en-US")

    pairs = list(zip(df[text_col], df[cf_col]))
    all_errors = []

    # LanguageTool runs as a local server, so several threads can query it at once (~3x faster)
    with ThreadPoolExecutor(n_threads) as ex:
        for i, errs in enumerate(ex.map(lambda p: find_new_errors(tool, *p), pairs)):
            all_errors.append(errs)
            if (i + 1) % 1000 == 0:
                print(f"  checked {i + 1}/{len(pairs)}")

    tool.close()

    df = df.copy()
    df["lt_new_errors"] = all_errors
    df["lt_n_new_errors"] = df["lt_new_errors"].apply(len)
    df["lt_flagged"] = df["lt_n_new_errors"] > 0
    return df


def main():
    BASE_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
    CF_DIR = os.path.join(BASE_DIR, "counterfactual")
    INPUT_PATH = os.path.join(CF_DIR, "counterfactual_posts.pkl")
    OUTPUT_PATH = os.path.join(CF_DIR, "counterfactual_posts_checked.pkl")
    REVIEW_PATH = os.path.join(CF_DIR, "manual_review_sample.csv")

    SAMPLE_FRAC = 0.05   # share of flagged pairs to review manually
    SEED = 42

    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_pickle(INPUT_PATH)
    print(f"Loaded {len(df)} counterfactual pairs.")

    print("Running LanguageTool on changed sentences ...")
    df = run_grammar_validation(df)

    n_flagged = df["lt_flagged"].sum()
    print(f"\nFlagged pairs (at least one new error): {n_flagged} / {len(df)} ({n_flagged / len(df):.1%})")

    # which rules fire most often -> shows what kind of errors the swap creates
    rule_counts = Counter(e["rule_id"] for errs in df["lt_new_errors"] for e in errs)
    print("\n--- Top 15 new error rules ---")
    for rule, count in rule_counts.most_common(15):
        print(f"{rule:40s} {count}")

    df.to_pickle(OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}")

    # random 5% of the flagged pairs for manual review
    flagged = df[df["lt_flagged"]]
    n_sample = min(len(flagged), max(1, math.ceil(SAMPLE_FRAC * len(flagged))))
    sample = flagged.sample(n=n_sample, random_state=SEED)

    rows = []
    for idx, errs in zip(sample.index, sample["lt_new_errors"]):
        for e in errs:
            rows.append({
                "row_index": idx,
                "rule_id": e["rule_id"],
                "message": e["message"],
                "original_sentence": e["original_sentence"],
                "flipped_sentence": e["flipped_sentence"],
                "is_real_error": "",   # fill manually: yes / no
                "notes": "",
            })

    pd.DataFrame(rows).to_csv(REVIEW_PATH, index=False, encoding="utf-8")
    print(f"Saved {len(sample)} flagged pairs ({len(rows)} errors) for manual review to {REVIEW_PATH}")
