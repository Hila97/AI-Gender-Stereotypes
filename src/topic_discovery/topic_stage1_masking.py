"""
Direction 1 - stage 1a: masking gender words
--------------------------------------------
Before computing embeddings, gender words are replaced with neutral words (she/he -> they,
mother/father -> parent), so that topics are not built around the gender words themselves.
GFI is NOT affected: it is counted later on the original text with the original gender lexicon.

This file:
  1. holds the masking list (MASK_MAP), based on direct_swaps from Direction 2 (counterfactual_generation.py)
     plus the words added after the manual check;
  2. finds candidate gender words that are not in the list (find_candidates, check_known_gender_words);
  3. masks the posts (spaCy, then a regex pass for leftovers) and checks the result (run_masking = main).
"""
import os
import re
import html
from collections import Counter

import pandas as pd
import spacy
from gensim.models import Word2Vec

BASE_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
MODEL_DIR = "/content/gdrive/MyDrive/Data mining/text mining/models"
POSTS_PKL = os.path.join(BASE_DIR, "processed/word2vec_posts.pkl")
W2V_MODEL = os.path.join(MODEL_DIR, "gender_ai_w2v_optimized.model")
OUT_DIR = os.path.join(BASE_DIR, "topic_discovery")

# gender word -> neutral word. None = the word is deleted.
# her / his are not here: they need the part of speech (her book -> their book, told her -> told them),
# and are handled with spaCy in the masking step, as in Direction 2.
MASK_MAP = {
    # pronouns
    "she": "they", "he": "they",
    "him": "them", "hers": "theirs",
    "herself": "themselves", "himself": "themselves",

    # people
    "woman": "person", "man": "person", "women": "people", "men": "people",
    "lady": "person", "gentleman": "person", "ladies": "people", "gentlemen": "people",
    "girl": "person", "boy": "person", "girls": "people", "boys": "people",
    "gal": "person", "guy": "person", "gals": "people", "guys": "people",
    "female": None, "male": None,              # as an adjective: "female engineer" -> "engineer"
    "females": "people", "males": "people",

    # family
    "mother": "parent", "father": "parent", "mothers": "parents", "fathers": "parents",
    "mom": "parent", "dad": "parent", "moms": "parents", "dads": "parents",
    "mum": "parent", "mommy": "parent", "daddy": "parent",
    "daughter": "child", "son": "child", "daughters": "children", "sons": "children",
    "sister": "sibling", "brother": "sibling", "sisters": "siblings", "brothers": "siblings",
    "wife": "spouse", "husband": "spouse", "wives": "spouses", "husbands": "spouses",
    "girlfriend": "partner", "boyfriend": "partner", "girlfriends": "partners", "boyfriends": "partners",
    "fiancee": "partner", "fiance": "partner",
    "bride": "spouse", "groom": "spouse",
    "aunt": "relative", "uncle": "relative", "aunts": "relatives", "uncles": "relatives",
    "niece": "relative", "nephew": "relative", "nieces": "relatives", "nephews": "relatives",
    "grandmother": "grandparent", "grandfather": "grandparent",
    "grandma": "grandparent", "grandpa": "grandparent",
    "granddaughter": "grandchild", "grandson": "grandchild",
    "stepmother": "stepparent", "stepfather": "stepparent",

    # other
    "queen": "monarch", "king": "monarch",
    "princess": "royal", "prince": "royal",
    "actress": "actor", "waitress": "server",
    "sis": "friend", "bro": "friend",
    "sir": None, "madam": None, "ma'am": None,

    # added after the manual check (check_known_gender_words)
    "dude": "person", "dudes": "people", "lad": "person", "lads": "people", "chap": "person",
    "fellas": "people", "gents": "people", "bros": "friends",
    "gf": "partner", "bf": "partner", "hubby": "spouse", "wifey": "spouse",
    "mama": "parent", "papa": "parent", "granny": "grandparent",
    "grandmothers": "grandparents", "grandfathers": "grandparents", "auntie": "relative",
    "godfather": "godparent", "godmother": "godparent", "widow": "person", "housewife": "homemaker",
    "chairman": "chair", "businessman": "businessperson", "businesswoman": "businessperson",
    "salesman": "salesperson", "congressman": "representative", "fireman": "firefighter",
    "maid": "housekeeper", "heroine": "hero", "actresses": "actors",
    "kings": "monarchs", "queens": "monarchs", "emperor": "monarch", "empress": "monarch",
    "princesses": "royals", "duke": "noble", "goddess": "deity", "goddesses": "deities",
    "monk": "cleric", "monks": "clerics", "nun": "cleric", "mistress": "partner",
    "mankind": "humankind",
    # gender as a concept: the topic stays, the direction is removed
    "feminine": "gendered", "masculine": "gendered", "manly": "gendered", "girly": "gendered",
    "femininity": "gender", "masculinity": "gender",
    "maternity": "parental", "paternity": "parental", "maternal": "parental", "paternal": "parental",
    "motherhood": "parenthood", "sisterhood": "solidarity", "brotherhood": "solidarity",
    # left out on purpose (mostly not about gender here): bachelor (degree), lord (oh lord),
    # cowboy (cowboy coding), maiden (maiden voyage)
}

# titles are deleted only before a name ("Ms. Lee"), because MS is also a degree or Microsoft
TITLES = ["mr", "mrs", "ms", "miss"]

KNOWN = set(MASK_MAP) | {"her", "his"} | set(TITLES)

MIN_COUNT = 20      # a candidate must appear at least this many times in the corpus
TOP_N = 100         # how many candidates to print


def clean_text(text):
    """Minimal cleaning for the embeddings: links, deleted/removed markers and HTML entities."""
    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\[(deleted|removed)\]", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def word_counts(texts):
    """How many times each lowercase word appears in the corpus."""
    counts = Counter()
    for t in texts:
        counts.update(re.findall(r"[a-z]+(?:'[a-z]+)?", t.lower()))
    return counts


def suffix_candidates(counts):
    """Words like chairman / businesswomen / hostess, whose stem is also a word in the corpus."""
    found = {}
    for w, c in counts.items():
        if c < MIN_COUNT or w in KNOWN:
            continue
        for suf in ("woman", "women", "man", "men"):
            stem = w[:-len(suf)]
            if w.endswith(suf) and len(stem) >= 3 and stem in counts:
                found[w] = f"suffix -{suf}"
                break
        else:
            stem = w[:-3]
            if w.endswith("ess") and len(stem) >= 3 and (stem in counts or stem + "er" in counts or stem + "or" in counts):
                found[w] = "suffix -ess"
    return found


def w2v_candidates(counts, model_path, topn=20):
    """Nearest neighbours of each gender-lexicon word in the project's word2vec model."""
    wv = Word2Vec.load(model_path).wv
    seeds = [w for w in KNOWN if w in wv]
    print(f"   word2vec seeds found in the model: {len(seeds)} of {len(KNOWN)}")
    found = {}
    for seed in seeds:
        for w, sim in wv.most_similar(seed, topn=topn):
            if w in KNOWN or counts.get(w, 0) < MIN_COUNT:
                continue
            if w not in found or sim > found[w][1]:
                found[w] = (seed, sim)
    return {w: f"w2v near '{seed}' ({sim:.2f})" for w, (seed, sim) in found.items()}


def find_candidates():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading posts...")
    df = pd.read_pickle(POSTS_PKL)
    texts = df["full_text"].fillna("").map(clean_text)
    counts = word_counts(texts)
    print(f"   {len(df)} posts, {len(counts)} distinct words")

    print("Searching by suffix...")
    by_suffix = suffix_candidates(counts)
    print("Searching by word2vec neighbours...")
    by_w2v = w2v_candidates(counts, W2V_MODEL)

    rows = []
    for w in set(by_suffix) | set(by_w2v):
        source = " | ".join(s for s in [by_suffix.get(w), by_w2v.get(w)] if s)
        rows.append({"word": w, "count": counts[w], "source": source})
    cand = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)

    save_path = os.path.join(OUT_DIR, "mask_candidates.csv")
    cand.to_csv(save_path, index=False)
    print(f"\nSaved all {len(cand)} candidates to: {save_path}")

    print(f"\nTop {TOP_N} candidates by frequency (not in the masking list yet):")
    with pd.option_context("display.max_rows", TOP_N, "display.max_colwidth", 60):
        print(cand.head(TOP_N).to_string())
    return cand


# Common English words that mark gender by their meaning, collected by hand
# (family, address, roles and job titles, -hood words). Only those in the corpus are printed.
EXTRA_GENDER_WORDS = [
    # address / informal
    "dude", "dudes", "bros", "fella", "fellas", "lad", "lads", "lass", "lassie",
    "gf", "bf", "wifey", "hubby", "missus", "boyo", "chap", "chaps", "gent", "gents",
    # family
    "mama", "mamma", "papa", "granny", "nana", "gramps", "auntie",
    "stepdaughter", "stepson", "stepsister", "stepbrother", "stepmom", "stepdad",
    "godmother", "godfather", "grandmothers", "grandfathers", "grandsons", "granddaughters",
    "widow", "widower", "widows", "housewife", "housewives", "fiancé", "fiancée", "brides", "grooms",
    "bridesmaid", "groomsmen", "bachelor", "bachelorette", "spinster", "maiden",
    "motherhood", "fatherhood", "sisterhood", "brotherhood", "mankind",
    "maternal", "paternal", "maternity", "paternity",
    # roles and titles
    "queens", "kings", "princesses", "princes", "duchess", "duke", "lord",
    "empress", "emperor", "heiress", "goddess", "goddesses", "heroine", "mistress", "maid", "maids",
    "nun", "nuns", "monk", "monks", "actresses", "waitresses", "hostess", "stewardess",
    "chairman", "chairwoman", "spokesman", "spokeswoman", "businessman", "businesswoman",
    "businessmen", "businesswomen", "congressman", "congresswoman", "salesman", "saleswoman",
    "policeman", "policewoman", "fireman", "firemen", "cameraman", "craftsman", "sportsman",
    "schoolgirl", "schoolboy", "cowboy", "cowboys", "landlady", "manly", "womanly",
    "feminine", "masculine", "femininity", "masculinity", "girly", "boyish",
]


def check_known_gender_words(min_count=5):
    """Counts the hand-made list above in the corpus (only words not already in the masking list)."""
    df = pd.read_pickle(POSTS_PKL)
    counts = word_counts(df["full_text"].fillna("").map(clean_text))
    rows = [{"word": w, "count": counts[w]} for w in EXTRA_GENDER_WORDS
            if w not in KNOWN and counts[w] >= min_count]
    res = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
    print(f"Hand-made gender words found at least {min_count} times (not in the masking list):")
    with pd.option_context("display.max_rows", None):
        print(res.to_string())
    return res


# ---------------------------------------------------------------- masking

# punctuation after "his" that means it stands alone ("the book is his.") - same as Direction 2
END_PUNCT = {".", ",", ";", ":", "!", "?", ")", "]"}

# fast pre-filter: only texts with at least one gender word go through spaCy
GENDER_PATTERN = re.compile(
    r"\b(" + "|".join(sorted((re.escape(w) for w in KNOWN), key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE
)
# for the check after masking: every listed word except titles ("miss" is also a verb)
LEFTOVER_PATTERN = re.compile(
    r"\b(" + "|".join(sorted((re.escape(w) for w in KNOWN - set(TITLES)), key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE
)


def match_case(new_word, token):
    """Copy the case of the original word to the new word."""
    if token.is_upper and len(token.text) > 1:
        return new_word.upper()
    if token.text[0].isupper():
        return new_word[0].upper() + new_word[1:]
    return new_word


def mask_doc(doc):
    """Mask gender words in a spaCy doc. Returns (masked_text, n_masked)."""
    parts = []
    n_masked = 0
    for token in doc:
        w = token.lower_

        if w == "her":
            # "her book" -> "their book", "I told her" -> "I told them"
            new = "their" if (token.tag_ == "PRP$" or token.dep_ == "poss") else "them"
        elif w == "his":
            next_is_end = token.i + 1 < len(doc) and doc[token.i + 1].text in END_PUNCT
            new = "their" if (token.dep_ == "poss" and not next_is_end) else "theirs"
        elif w.rstrip(".") in TITLES:
            # only before a name, and not "MS" in capitals (degree / Microsoft); "I miss her" stays
            next_is_name = token.i + 1 < len(doc) and doc[token.i + 1].pos_ == "PROPN"
            new = "" if (next_is_name and not token.is_upper) else token.text
        elif w in ("female", "male"):
            # "a female" (noun) -> "a person", "female engineer" (adjective) -> "engineer"
            new = "person" if token.pos_ == "NOUN" else ""
        elif w in MASK_MAP:
            new = MASK_MAP[w] or ""
        else:
            new = token.text

        if new.lower() != w:
            n_masked += 1
            if new:
                new = match_case(new, token)

        # a deleted word is dropped together with the space after it
        if new:
            parts.append(new + token.whitespace_)
    return "".join(parts).strip(), n_masked


def mask_texts(texts, batch_size=256, n_process=1):
    """Mask a list of texts. Texts without any gender word are returned as they are."""
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    masked = list(texts)
    n_masked = [0] * len(texts)
    idx = [i for i, t in enumerate(texts) if GENDER_PATTERN.search(t)]
    print(f"   texts with at least one gender word: {len(idx)} of {len(texts)}")

    docs = nlp.pipe((texts[i] for i in idx), batch_size=batch_size, n_process=n_process)
    for k, doc in enumerate(docs):
        masked[idx[k]], n_masked[idx[k]] = mask_doc(doc)
        if (k + 1) % 5000 == 0:
            print(f"   masked {k + 1}/{len(idx)}")
    return masked, n_masked


# words that spaCy did not split off (e.g. "men&women", "~~man~~") are replaced in a second, regex pass.
# her / his here have no part of speech, so they get the most common form ("their").
FALLBACK_MAP = {**MASK_MAP, "her": "their", "his": "their"}


def regex_fallback(text):
    """Replace listed words that are still in the text after spaCy. Returns (text, n_replaced)."""
    n = 0

    def replace(m):
        nonlocal n
        n += 1
        w = m.group(0)
        new = FALLBACK_MAP[w.lower()] or ""
        if new and w.isupper() and len(w) > 1:
            return new.upper()
        if new and w[0].isupper():
            return new[0].upper() + new[1:]
        return new

    text = LEFTOVER_PATTERN.sub(replace, text)
    text = re.sub(r"(\*\*|__)\s*\1", "", text)      # empty bold left after deleting a word: "**male**" -> "****"
    return re.sub(r"  +", " ", text), n


def leftover_check(df, n_examples=8):
    """Check 1: listed gender words left after masking (should be 0), with a few examples in context."""
    left = Counter()
    examples = []
    for t in df["text_masked"]:
        for m in LEFTOVER_PATTERN.finditer(t):
            left[m.group(0).lower()] += 1
            if len(examples) < n_examples:
                examples.append(t[max(0, m.start() - 40): m.end() + 40].replace("\n", " "))
    print(f"   total: {sum(left.values())}")
    if left:
        print(f"   most common: {left.most_common(15)}")
        print("   examples in context:")
        for e in examples:
            print(f"      ...{e}...")
    return left


def check_masking(df):
    """Check 2: examples before and after, to see that the text is still readable."""
    changed = df[df["n_masked"] > 0]
    print(f"\n--- Check 2: examples ({len(changed)} posts were changed, "
          f"{changed['n_masked'].sum()} words masked) ---")
    print("\n5 random posts:")
    for _, row in changed.sample(min(5, len(changed)), random_state=42).iterrows():
        print(f"BEFORE: {row['text_clean'][:300]}")
        print(f"AFTER:  {row['text_masked'][:300]}\n")
    print("5 posts with the most masked words:")
    for _, row in changed.nlargest(5, "n_masked").iterrows():
        print(f"[{row['n_masked']} words] BEFORE: {row['text_clean'][:300]}")
        print(f"[{row['n_masked']} words] AFTER:  {row['text_masked'][:300]}\n")


def run_masking(n_process=1, from_saved=False):
    """
    from_saved=False: mask from the start (spaCy + regex pass).
    from_saved=True:  load posts_masked.pkl that was already masked with spaCy, and run only the regex pass.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    save_path = os.path.join(OUT_DIR, "posts_masked.pkl")

    if from_saved:
        print(f"Loading the spaCy-masked posts from {save_path} ...")
        df = pd.read_pickle(save_path)
    else:
        print("Loading posts...")
        cols = ["id", "subreddit", "full_text", "cleaned_tokens", "ai_category", "research_group",
                "gender_context", "female_term_count", "male_term_count"]
        df = pd.read_pickle(POSTS_PKL)[cols].reset_index(drop=True)
        print(f"   {len(df)} posts")
        df["text_clean"] = df["full_text"].fillna("").map(clean_text)
        print("Masking (spaCy, this takes a while)...")
        df["text_masked"], df["n_masked"] = mask_texts(df["text_clean"].tolist(), n_process=n_process)

    print("\n--- Check 1a: listed gender words left after spaCy ---")
    leftover_check(df)

    print("\nRegex pass on the leftovers...")
    fixed = df["text_masked"].map(regex_fallback)
    df["text_masked"] = fixed.str[0]
    df["n_masked"] = df["n_masked"] + fixed.str[1]
    print(f"   replaced {fixed.str[1].sum()} more words")

    print("\n--- Check 1b: listed gender words left after the regex pass (should be 0) ---")
    left = leftover_check(df)

    check_masking(df)

    df.to_pickle(save_path)
    print(f"Saved {len(df)} posts to: {save_path}")
    return df, left


def main(n_process=1, from_saved=False):
    print("=" * 60)
    print("DIRECTION 1 - stage 1a: masking gender words")
    print("=" * 60)
    return run_masking(n_process=n_process, from_saved=from_saved)


if __name__ == "__main__":
    main()
