import os
import re
import pandas as pd
import spacy

# load spacy
nlp = spacy.load("en_core_web_sm")

#swap based on gender lexicons
direct_swaps = {
    "she": "he", 
    "he": "she",
    "herself": "himself", 
    "himself": "herself",
    "him": "her",
    "hers": "his",

    "woman": "man", 
    "man": "woman",
    "women": "men", 
    "men": "women",
    "female": "male", 
    "male": "female",
    "females": "males", 
    "males": "females",
    "lady": "gentleman", 
    "gentleman": "lady",
    "ladies": "gentlemen", 
    "gentlemen": "ladies",

    "girl": "boy", 
    "boy": "girl",
    "girls": "boys", 
    "boys": "girls",

    "gal": "guy", 
    "guy": "gal",
    "gals": "guys", 
    "guys": "gals",

    "mother": "father", 
    "father": "mother",
    "mom": "dad", 
    "dad": "mom",
    "daughter": "son", 
    "son": "daughter",
    "sister": "brother", 
    "brother": "sister",
    "wife": "husband", 
    "husband": "wife",
    "aunt": "uncle", 
    "uncle": "aunt",
    "niece": "nephew", 
    "nephew": "niece",
    "grandmother": "grandfather", 
    "grandfather": "grandmother",

    # extended lexicon (not in the original gender lexicons)
    "sir": "madam",
    "madam": "sir",
    "ma'am": "sir",

    "girlfriend": "boyfriend", 
    "boyfriend": "girlfriend",
    "girlfriends": "boyfriends", 
    "boyfriends": "girlfriends",
    "fiancee": "fiance", 
    "fiance": "fiancee",
    "bride": "groom", 
    "groom": "bride",

    "mothers": "fathers", 
    "fathers": "mothers",
    "moms": "dads", 
    "dads": "moms",
    "mum": "dad",
    "mommy": "daddy", 
    "daddy": "mommy",
    "daughters": "sons", 
    "sons": "daughters",
    "sisters": "brothers", 
    "brothers": "sisters",
    "wives": "husbands", 
    "husbands": "wives",
    "aunts": "uncles", 
    "uncles": "aunts",
    "nieces": "nephews", 
    "nephews": "nieces",
    "grandma": "grandpa", 
    "grandpa": "grandma",
    "granddaughter": "grandson", 
    "grandson": "granddaughter",
    "stepmother": "stepfather", 
    "stepfather": "stepmother",

    "queen": "king", 
    "king": "queen",
    "princess": "prince", 
    "prince": "princess",
    "actress": "actor",
    "waitress": "waiter",
    "sis": "bro", 
    "bro": "sis"
}


# titles are swapped only before a name ("Ms. Lee"), because "MS" is also
# a degree (MS in data science) or Microsoft. Mr. -> Ms. (neutral form), Mrs./Ms./Miss -> Mr.
title_swaps = {
    "mr.": "ms.", "mr": "ms",
    "mrs.": "mr.", "mrs": "mr",
    "ms.": "mr.", "ms": "mr",
    "miss": "mr",
}

# punctuation after "his" that means it stands alone ("the book is his.")
# quotes and opening brackets are NOT here: in 'his "ideas"' it is still possessive
END_PUNCT = {".", ",", ";", ":", "!", "?", ")", "]"}


def match_case(new_word, token):
    """Copy the case of the original word to the swapped word."""
    if token.is_upper and len(token.text) > 1:
        return new_word.upper()          # "GUYS" -> "GALS"
    if token.text[0].isupper():
        return new_word[0].upper() + new_word[1:]   # "She" -> "He"
    return new_word


def swap_gender_doc(doc):
    """Swap gender words in a spaCy doc. Returns (flipped_text, n_swaps)."""
    new_words = []
    n_swaps = 0

    for token in doc:
        word_lower = token.lower_

        # edge cases
        if word_lower == "her":
            # PRP$ = Possessive Pronoun
            if token.tag_ == "PRP$" or token.dep_ == "poss":
                new_word = "his"  # "her book" -> "his book"
            else:
                new_word = "him"  # "I saw her" -> "I saw him"

        elif word_lower == "his":
            next_token_is_end_punct = False
            if token.i + 1 < len(doc):
                if doc[token.i + 1].text in END_PUNCT:
                    next_token_is_end_punct = True

            if token.dep_ == "poss" and not next_token_is_end_punct:
                new_word = "her"  # "his car" -> "her car"
            else:
                new_word = "hers" # "the car is his," -> "the car is hers,"

        # titles: only before a name, and not "MS" in capitals (degree / Microsoft)
        # also separates "Miss Lee" from the verb "I miss her"
        elif word_lower in title_swaps:
            next_is_name = token.i + 1 < len(doc) and doc[token.i + 1].pos_ == "PROPN"
            if next_is_name and not token.is_upper:
                new_word = title_swaps[word_lower]
            else:
                new_word = token.text

        # basic lexicon
        elif word_lower in direct_swaps:
            new_word = direct_swaps[word_lower]

        else:
            new_word = token.text

        # case is changed ONLY for swapped words; every other word stays exactly as written
        # (capitalize() on all words turned "Q&A" -> "Q&a", "U.S." -> "U.s.", "D&D" -> "D&d")
        if new_word.lower() != word_lower:
            n_swaps += 1
            new_word = match_case(new_word, token)

        new_words.append(new_word)

    # back to text
    flipped_text = "".join([new_words[i] + doc[i].whitespace_ for i in range(len(doc))])

    return flipped_text, n_swaps


def swap_gender(text):
    flipped_text, _ = swap_gender_doc(nlp(text))
    return flipped_text


# regex for a fast pre-filter: does the text contain any gender word at all?
_gender_words = set(k.rstrip(".") for k in list(direct_swaps) + list(title_swaps)) | {"her", "his"}
GENDER_PATTERN = re.compile(
    r"\b(" + "|".join(sorted((re.escape(w) for w in _gender_words), key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE
)


def has_gender_word(text):
    return bool(GENDER_PATTERN.search(text))


def generate_counterfactuals(df, text_col="full_text", batch_size=256, n_process=1):
    """
    Adds 'text_cf' (gender-swapped text) and 'n_swaps' (number of swapped words).
    Keeps only rows where the swap actually changed the text.
    """
    df = df.copy()
    df[text_col] = df[text_col].fillna("").astype(str)

    # 1. keep only texts with at least one gender word
    mask = df[text_col].apply(has_gender_word)
    df = df[mask].copy()
    print(f"Texts with at least one gender word: {len(df)} / {len(mask)} ({mask.mean():.1%})")

    # 2. swap with nlp.pipe (much faster than nlp(text) in a loop)
    flipped, swaps = [], []
    texts = df[text_col].tolist()
    for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size, n_process=n_process)):
        t, n = swap_gender_doc(doc)
        flipped.append(t)
        swaps.append(n)
        if (i + 1) % 5000 == 0:
            print(f"  processed {i + 1}/{len(texts)}")

    df["text_cf"] = flipped
    df["n_swaps"] = swaps

    # 3. drop rows where nothing changed (e.g. "his" matched by regex but text identical)
    before = len(df)
    df = df[(df["n_swaps"] > 0) & (df[text_col] != df["text_cf"])].copy()
    print(f"Removed {before - len(df)} rows with no change after swapping.")
    print(f"Final counterfactual pairs: {len(df)}")

    return df


def main():
    BASE_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
    # deduplicated posts; train/val/test were split from this file
    INPUT_PATH = os.path.join(BASE_DIR, "processed/word2vec_posts.pkl")
    OUTPUT_DIR = os.path.join(BASE_DIR, "counterfactual")
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, "counterfactual_posts.pkl")

    print(f"Loading original data from {INPUT_PATH} ...")
    df = pd.read_pickle(INPUT_PATH)
    print(f"Loaded {len(df)} posts.")

    df_cf = generate_counterfactuals(df, text_col="full_text")

    # summary
    print("\n--- n_swaps distribution ---")
    print(df_cf["n_swaps"].describe())
    if "gender_context" in df_cf.columns:
        print("\n--- gender_context of swapped posts ---")
        print(df_cf["gender_context"].value_counts())

    print("\n--- 5 random examples ---")
    for _, row in df_cf.sample(min(5, len(df_cf)), random_state=42).iterrows():
        print(f"ORIGINAL: {row['full_text'][:300]}")
        print(f"FLIPPED:  {row['text_cf'][:300]}")
        print(f"n_swaps:  {row['n_swaps']}\n")

    # save a NEW file, the original csv is not touched
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_cf.to_pickle(OUTPUT_PATH)
    print(f"Saved {len(df_cf)} rows to {OUTPUT_PATH}")
