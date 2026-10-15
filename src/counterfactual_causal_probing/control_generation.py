import os
import re
import random

import pandas as pd


# Control condition (follow-up plan, section 3.2.4):
# swap NON-gendered, semantically matched word pairs, to show that the effects of the
# gender swap are specific to gender and not to "any word substitution".
#
# Rules for choosing the pairs:
#  - not a gender word
#  - not in hard_ai.txt / soft_ai.txt (otherwise the swap could change the true Hard/Soft label)
#  - both words of a pair come from the same category (language <-> language, OS <-> OS ...)
CONTROL_PAIRS = [
    # companies, languages and brands
    ("python", "java"),
    ("windows", "linux"),
    ("google", "microsoft"),
    ("chrome", "firefox"),
    ("apple", "samsung"),
    ("ios", "android"),
    ("github", "gitlab"),
    ("mac", "pc"),

    # Reddit and forum terms
    ("post", "comment"),
    ("posts", "comments"),
    ("thread", "forum"),
    ("website", "app"),
    ("link", "url"),
    ("article", "blog"),

    # generic hardware and devices (not AI hardware like GPUs)
    ("laptop", "desktop"),
    ("phone", "tablet"),
    ("keyboard", "mouse"),
    ("screen", "monitor"),

    # generic professional / academic terms
    ("company", "startup"),
    ("companies", "startups"),
    ("team", "group"),
    ("university", "college"),
    ("student", "researcher"),
    ("paper", "study"),

    # time
    ("monday", "friday"),
    ("morning", "evening"),
    ("week", "month"),
    ("weeks", "months"),
    ("hours", "minutes"),

    # everyday objects
    ("city", "town"),
    ("coffee", "tea"),
    ("car", "bike"),
]

control_swaps = {}
for a, b in CONTROL_PAIRS:
    control_swaps[a] = b
    control_swaps[b] = a

CONTROL_PATTERN = re.compile(
    # not inside hyphenated words: "post-training" or "phone-based" stay as they are
    r"(?<![\w-])(" + "|".join(sorted(map(re.escape, control_swaps), key=len, reverse=True)) + r")(?![\w-])",
    flags=re.IGNORECASE
)


def match_case(new_word, old_word):
    if old_word.isupper() and len(old_word) > 1:
        return new_word.upper()
    if old_word[0].isupper():
        return new_word.capitalize()
    return new_word


def control_swap(text, n_target, rng):
    """
    Swap up to n_target control words in the text (n_target = number of gender swaps
    in the same post, so both versions change a similar number of words).
    Returns (control_text, n_ctrl_swaps).
    """
    matches = list(CONTROL_PATTERN.finditer(text))
    if not matches:
        return text, 0

    # choose which occurrences to swap (random but reproducible)
    k = min(n_target, len(matches))
    chosen = sorted(rng.sample(matches, k), key=lambda m: m.start())

    parts, last = [], 0
    for m in chosen:
        old = m.group(0)
        parts.append(text[last:m.start()])
        parts.append(match_case(control_swaps[old.lower()], old))
        last = m.end()
    parts.append(text[last:])

    return "".join(parts), k


def generate_control(df, text_col="full_text", n_col="n_swaps", seed=42):
    rng = random.Random(seed)

    results = [control_swap(t, n, rng) for t, n in zip(df[text_col], df[n_col])]

    out = pd.DataFrame(index=df.index)
    if "id" in df.columns:
        out["id"] = df["id"]
    out["text_ctrl"] = [r[0] for r in results]
    out["n_ctrl_swaps"] = [r[1] for r in results]
    out["has_control"] = out["n_ctrl_swaps"] > 0
    return out


def main():
    BASE_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
    CF_DIR = os.path.join(BASE_DIR, "counterfactual")
    INPUT_PATH = os.path.join(CF_DIR, "counterfactual_posts.pkl")
    OUTPUT_PATH = os.path.join(CF_DIR, "control_posts.pkl")

    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_pickle(INPUT_PATH)
    print(f"Loaded {len(df)} counterfactual pairs.")

    ctrl = generate_control(df)

    n_ctrl = ctrl["has_control"].sum()
    print(f"\nPosts with a control version: {n_ctrl} / {len(ctrl)} ({n_ctrl / len(ctrl):.1%})")

    with_ctrl = ctrl[ctrl["has_control"]]
    exact = (with_ctrl["n_ctrl_swaps"] == df.loc[with_ctrl.index, "n_swaps"]).mean()
    print(f"Same number of swaps as the gender version: {exact:.1%} of them")

    print("\n--- n_ctrl_swaps distribution ---")
    print(with_ctrl["n_ctrl_swaps"].describe())

    print("\n--- 5 random examples ---")
    for idx in with_ctrl.sample(min(5, len(with_ctrl)), random_state=42).index:
        print(f"ORIGINAL: {df.at[idx, 'full_text'][:300]}")
        print(f"CONTROL:  {ctrl.at[idx, 'text_ctrl'][:300]}")
        print(f"n_ctrl_swaps: {ctrl.at[idx, 'n_ctrl_swaps']} (gender swaps: {df.at[idx, 'n_swaps']})\n")

    # separate file, same index as counterfactual_posts.pkl -> join with df.join(ctrl)
    ctrl.to_pickle(OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")
