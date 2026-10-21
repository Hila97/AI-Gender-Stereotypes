"""
    Build the v3 affective lexicons (after Barbara's review):
    1. Seed words (Barbara's combined lists + the positive words she picked from each v2 file)
    2. Empath (Expansion)
    3. NRC (emotion columns only)
    4. Harvard General Inquirer (a word is added only if GI also marks it Positiv / Negativ)
    5. Cleaning (valence filter, Barbara's flagged words, gender and AI words)
    6. Saving + report

    The v2 lexicons (lexical_collector.py, data/lexicons) are not changed.
"""
import csv
import os
import re
from collections import defaultdict

from empath import Empath
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

POSITIVE = [
    "achievement_excellence", "cognitive_insight", "positive_emotions", "certainty_confidence",
    "affiliation_support", "optimism_vitality", "praise_admiration", "reward_success",
]
NEGATIVE = [
    "physical_adversity", "social_legal_negativity", "situational_failure",
    "negative_traits", "existential_negativity", "negative_emotions",
]

# ---------------------------------------------------------
# 1. Seed words
# ---------------------------------------------------------
# Barbara's combined positive list ("One positive seeds list from all the files")
SEEDS = {
    "achievement_excellence": {"success", "achievement", "triumph", "attain", "excel", "goal", "victory", "better"},
    "cognitive_insight": {"wisdom", "insight", "understand", "clear", "reason", "logic", "think", "know"},
    "positive_emotions": {"joy", "happy", "excited", "love", "good", "nice", "great", "delight", "trust", "hope"},
    "certainty_confidence": {"confidence", "assurance", "certain", "definite", "absolute", "surely", "stable", "firm"},
    "affiliation_support": {"friend", "support", "community", "team", "share", "ally", "acceptance", "affection"},
    "optimism_vitality": {"optimism", "hope", "positive", "future", "upbeat", "gratitude", "positivity", "eager"},
    "praise_admiration": {"admire", "bless", "appreciate", "praise", "agree", "honor", "cherish", "valuable"},
    "reward_success": {"prize", "benefit", "award", "gain", "reward", "earn", "expect", "value"},
}

# Barbara's positive words from each v2 file, added to the matching v3 category
SEEDS_FROM_V2_FILES = {
    "affiliation_support": {"acceptance", "admire", "affection", "amiable", "brotherhood", "cooperation",
                            "friendship", "harmony", "sincere", "sympathetic"},            # affiliation_social.txt
    "certainty_confidence": {"absolute", "assurance", "assure", "confident", "sturdy", "supreme",
                             "surely", "trustworthy", "unwavering", "validation"},          # certainty.txt
    "cognitive_insight": {"ascertain", "brainy", "certainty", "clear", "comprehend", "genius",
                          "insight", "rational", "resourceful", "wisdom"},                  # cognitive_processes.txt
    "optimism_vitality": {"eagerness", "enthusiastic", "esteem", "exuberance", "gratitude", "optimism",
                          "positiveness", "positivity", "upbeat"},                          # perceptual_processes.txt
    "achievement_excellence": {"ability", "accomplish", "achievement", "advantage", "champion", "excel",
                               "excellence", "success", "triumph", "victory"},              # power_achievement.txt
    "positive_emotions": {"abundance", "acclaim", "beautiful", "blissful", "cheerful", "delightful",
                          "happiness", "joyful", "marvelous", "prosperous"},                # positive_emotions.txt
}
for _cat, _words in SEEDS_FROM_V2_FILES.items():
    SEEDS[_cat] |= _words

# Barbara's combined negative list: 5 themes, each theme = all of its sub-groups
SEEDS.update({
    "physical_adversity": {
        # Disasters & Hazards
        "accident", "ambush", "avalanche", "crisis", "cyclone", "disaster", "flood", "hurricane", "peril",
        "storm", "tempest", "toxic", "volcano",
        # Disease & Medical
        "ailment", "alcoholism", "amnesia", "bacteria", "cancer", "dementia", "fatality", "infection",
        "lesion", "parasite", "plague", "poison", "symptom", "virus",
        # Damage & Decay
        "broken", "corrode", "damage", "decay", "debris", "defunct", "deteriorate", "eroded", "filth",
        "junk", "ruin", "tarnish", "waste",
    },
    "social_legal_negativity": {
        # Crime & Wrongdoing
        "abduction", "arson", "bribery", "corruption", "crime", "felony", "fraud", "homicide", "perjury",
        "robbery", "theft", "treason", "villain",
        # Conflict & Aggression
        "adversary", "assault", "battle", "combat", "dispute", "hostility", "invasion", "riot", "strife",
        "warfare", "weapon",
        # Oppression & Power
        "autocracy", "burden", "captive", "domination", "enslave", "exploitation", "oppression", "tyranny",
        "usurp", "veto",
    },
    "situational_failure": {
        # Lack & Deficiency
        "absence", "barren", "bankruptcy", "debt", "deficit", "empty", "famine", "inadequate", "lack",
        "loss", "poverty", "scarcity", "shortage", "void",
        # Error & Confusion
        "ambiguity", "amiss", "blunder", "confusion", "error", "falsehood", "flaw", "illusion", "mistake",
        "oversight", "paradox",
        # Impediment
        "barrier", "blockade", "delay", "difficulty", "hinder", "obstacle", "restriction", "snag", "thwart",
    },
    "negative_traits": {
        # Deceptive / Dishonest
        "deceitful", "evasive", "fake", "hypocrisy", "insincere", "liar", "phony", "sly", "treacherous",
        # Hostile / Cold
        "aloof", "apathetic", "callous", "cynical", "harsh", "malicious", "rude", "sinister", "spiteful", "vulgar",
        # Incompetent
        "awkward", "clumsy", "feeble", "idle", "ignorant", "inept", "lazy", "shallow", "useless", "weak",
    },
    "existential_negativity": {
        # Endings
        "cessation", "death", "demise", "doom", "expiration", "extinct", "fatal", "finale", "grave",
        "mortality", "termination",
        # Burden
        "chores", "fatigue", "hardship", "ordeal", "pressure", "strain", "stress", "toil", "trouble",
    },
    # our v2 seeds + three basic emotions that were missing (anger, disgust, anxiety)
    "negative_emotions": {"scared", "fail", "worried", "hate", "nervous", "sadness", "fear",
                          "anger", "disgust", "anxiety"},
})

# Words Barbara found in the v2 files with the OPPOSITE valence ("Negative seeds lists from each
# of the files separately", and the contrast seeds from negative_emotions.txt).
# Removed in step 5 (rule 2); they are examples of the noise the expansion added.
BARBARA_NEGATIVE_IN_POSITIVE_FILES = {
    # affiliation_social.txt
    "alienate", "argument", "boycott", "controversy", "disagreement", "dispute", "excommunication", "exile",
    "expel", "ridicule", "shun", "threat", "traitor",
    # certainty.txt
    "abolish", "abuse", "aggression", "aggressive", "ambush", "antagonism", "assail", "assault", "attack",
    "autocrat", "brutality", "condemn", "crush", "destroy", "dictator", "fight", "gun", "harm", "harsh",
    "invade", "subvert", "suppress", "tyranny", "undermine",
    # cognitive_processes.txt
    "baffle", "bewilderment", "confuse", "consternation", "delirium", "delusion", "devious", "diabolical",
    "illusion", "insane", "senseless", "stress", "stupor", "unaware",
    # perceptual_processes.txt
    "aloof", "antipathy", "apathetic", "apathy", "cranky", "cringe", "objection", "prejudicial", "repress",
    "rigid", "selfishness", "shameful",
    # positive_emotions.txt
    "abandonment", "abduction", "accident", "anxiety", "betrayal", "carnage", "catastrophe", "death", "deceit",
    "devil", "dictatorship", "disaster", "doomsday", "dread", "earthquake", "falsehood", "horror", "insult",
    "kidnap", "violate", "violent",
    # power_achievement.txt
    "blackmail", "capture", "censor", "chastise", "coerce", "coercion", "condescending", "confiscate",
    "defeat", "demolish", "denounce", "depose", "deprive", "dictate", "dictatorial", "dominate", "enslave",
    "eradicate", "evict", "exterminate", "force", "imprison", "usurp", "veto",
}
BARBARA_POSITIVE_IN_NEGATIVE_FILES = {"validity", "verification", "volunteer"}  # from negative_emotions.txt

# AI terms Barbara marked in hard_ai.txt as carrying valence (removed with the AI lexicons, rule 3)
BARBARA_AI_TERMS = {"fine-tuning", "intelligence", "learning", "optimization",
                    "bias", "dropout", "overfitting", "latency", "pruning"}


# ---------------------------------------------------------
# Source mappings
# ---------------------------------------------------------
# NRC: only emotion columns add words, and only if NRC's general 'positive' / 'negative' column
# agrees with the lexicon's valence. Those two columns are also the valence filter in step 5. 'anticipation' and 'surprise' are not used at all:
# they have no valence (in v2 they brought words like 'death' and 'accident' into positive_emotions).
# 'trust' is not used either: the word 'trust' itself is removed as a soft_ai word (rule 3), and the
# column adds many words with no emotion. The words it would add are saved in
# report_nrc_trust_excluded.csv, to check this decision.
NRC_ADD = {
    "positive_emotions": {"joy"},
    "negative_emotions": {"anger", "fear", "sadness", "disgust"},
}

# Harvard GI: a word from these columns is added only if GI also marks it Positiv (positive
# lexicons) or Negativ (negative lexicons).
# CHECK each column's definition in the GI documentation before the final run; step 4 prints
# which columns were found in the file and how many words each one added.
GI_ADD = {
    "achievement_excellence": ["Complet", "Goal", "PowGain"],
    "cognitive_insight": ["Know", "Solve", "Think"],
    "positive_emotions": ["Pleasur", "PosAff", "EMOT"],
    "certainty_confidence": ["SureLw"],
    "affiliation_support": ["AffTot", "SocRel"],
    "optimism_vitality": [],                      # no matching GI column; seeds + Empath only
    "praise_admiration": ["Virtue", "RspGain"],
    "reward_success": ["Goal", "Try"],
    "physical_adversity": ["WlbPhys", "WlbLoss"],
    "social_legal_negativity": ["Milit", "Legal", "PowCon"],
    "situational_failure": ["Fail"],
    "negative_traits": ["Vice", "Hostile", "Weak"],
    "existential_negativity": ["Finish"],
    "negative_emotions": ["Pain", "NegAff", "EMOT"],
}

WORD_RE = re.compile(r"^[a-z][a-z'-]*$")   # single words only; the labeler matches single tokens

# Function words Empath returns next to real words ("yet", "though", "truly"...): not affect, so not added
STOP_WORDS = ENGLISH_STOP_WORDS | {
    "actually", "although", "certainly", "honestly", "however", "ultimately", "truly", "really",
    "regardless", "necessarily", "inevitably", "nevertheless", "moreover", "perhaps", "therefore",
    "thus", "though", "yet", "like", "much", "wholly", "reasonably", "surely", "own", "others",
}


def _add(lexicons, sources, category, word, source):
    word = word.strip().lower()
    if WORD_RE.match(word):
        lexicons[category].add(word)
        sources[category][word].add(source)


def load_word_file(path):
    with open(path, encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


def build_lexicons(nrc_path, gi_path, old_lexicons_dir, output_dir, empath_size=50):

    os.makedirs(output_dir, exist_ok=True)
    lexicons = {cat: set() for cat in POSITIVE + NEGATIVE}
    sources = {cat: defaultdict(set) for cat in lexicons}    # word -> {'seed', 'empath', 'nrc', 'gi'}

    print("--- Step 1: Seed Words ---")
    for cat in lexicons:
        for w in SEEDS[cat]:
            _add(lexicons, sources, cat, w, "seed")
        print(f"   {cat:26s} {len(lexicons[cat]):3d} seeds")

    # ---------------------------------------------------------
    # 2. Empath Expansion
    # ---------------------------------------------------------
    print("--- Step 2: Expanding with Empath ---")
    lexicon_generator = Empath()
    for cat in lexicons:
        try:
            # create_category() returns None: it only prints the words and keeps them in .cats[name]
            lexicon_generator.create_category(cat, sorted(SEEDS[cat]), size=empath_size)
            before = len(lexicons[cat])
            for w in lexicon_generator.cats.get(cat, []):
                if w.lower() not in STOP_WORDS:
                    _add(lexicons, sources, cat, w, "empath")
            print(f"   > '{cat}': +{len(lexicons[cat]) - before} words from Empath")
        except Exception as e:
            print(f"   > Warning: Empath expansion failed for {cat}: {e}")

    # ---------------------------------------------------------
    # 3. NRC Emotion Lexicon
    # ---------------------------------------------------------
    print("--- Step 3: Processing NRC Lexicon ---")
    nrc_valence = {"positive": set(), "negative": set()}
    nrc_trust_excluded = set()   # words NRC 'trust' would have added to positive_emotions (for the report)
    if os.path.exists(nrc_path):
        nrc_emotions = []   # (word, emotion); added after the whole file is read, so the valence is known
        with open(nrc_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) != 3 or parts[2] != "1":
                    continue
                word, emotion = parts[0].lower(), parts[1]
                if emotion in nrc_valence:
                    nrc_valence[emotion].add(word)
                nrc_emotions.append((word, emotion))
        # like GI: an emotion word is added only if NRC also marks it with the lexicon's valence
        # (e.g. 'ambulance' is fear + trust but neither positive nor negative -> not added)
        for word, emotion in nrc_emotions:
            for cat, emotions in NRC_ADD.items():
                valence = "positive" if cat in POSITIVE else "negative"
                if emotion in emotions and word in nrc_valence[valence]:
                    _add(lexicons, sources, cat, word, "nrc")
            if emotion == "trust" and word in nrc_valence["positive"] and WORD_RE.match(word):
                nrc_trust_excluded.add(word)
        for cat in NRC_ADD:
            n = sum("nrc" in s for s in sources[cat].values())
            print(f"   > '{cat}': {n} words from NRC")
    else:
        print(f"   > Error: NRC file not found at: {nrc_path}")

    # ---------------------------------------------------------
    # 4. Harvard General Inquirer
    # ---------------------------------------------------------
    print("--- Step 4: Processing Harvard General Inquirer ---")
    gi_valence = {"positive": set(), "negative": set()}
    if os.path.exists(gi_path):
        with open(gi_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f, delimiter="\t")
            reader.fieldnames = [name.upper() for name in reader.fieldnames] if reader.fieldnames else []
            columns = set(reader.fieldnames)
            for col in sorted({c.upper() for cols in GI_ADD.values() for c in cols} | {"POSITIV", "NEGATIV"}):
                if col not in columns:
                    print(f"   > Warning: GI column '{col}' not found in the file - it adds nothing")
            gi_counts = defaultdict(int)
            for row in reader:
                word = row.get("ENTRY", "").split("#")[0].lower()
                if not word:
                    continue
                is_pos = bool(row.get("POSITIV")) and row.get("POSITIV") != "0"
                is_neg = bool(row.get("NEGATIV")) and row.get("NEGATIV") != "0"
                if is_pos:
                    gi_valence["positive"].add(word)
                if is_neg:
                    gi_valence["negative"].add(word)
                for cat, cols in GI_ADD.items():
                    if not (is_pos if cat in POSITIVE else is_neg):
                        continue
                    for col in cols:
                        val = row.get(col.upper())
                        if val and val != "0":
                            _add(lexicons, sources, cat, word, "gi")
                            gi_counts[(cat, col)] += 1
                            break
        for (cat, col), n in sorted(gi_counts.items()):
            print(f"   > '{cat}' <- {col}: {n} words")
    else:
        print(f"   > Error: GI file not found at: {gi_path}")

    # ---------------------------------------------------------
    # 5. Cleaning
    # ---------------------------------------------------------
    print("--- Step 5: Cleaning ---")
    neg_marked = nrc_valence["negative"] | gi_valence["negative"]
    pos_marked = nrc_valence["positive"] | gi_valence["positive"]

    gender_ai = set()
    for fn in ["gender_female.txt", "gender_male.txt", "hard_ai.txt", "soft_ai.txt"]:
        path = os.path.join(old_lexicons_dir, fn)
        if os.path.exists(path):
            gender_ai |= load_word_file(path)
        else:
            print(f"   > Warning: {path} not found - its words are not removed")
    gender_ai |= BARBARA_AI_TERMS

    removed = []   # (category, word, rule)
    for cat in lexicons:
        positive = cat in POSITIVE
        for w in sorted(lexicons[cat]):
            is_seed = "seed" in sources[cat][w]
            rule = None
            # rule 3 first: gender / AI words leave every lexicon, seeds included
            if w in gender_ai:
                rule = "3_gender_or_ai_word"
            # rules 1-2 never remove Barbara's seeds (she chose them by hand)
            elif not is_seed and positive and w in BARBARA_NEGATIVE_IN_POSITIVE_FILES:
                rule = "2_barbara_flagged"
            elif not is_seed and not positive and w in BARBARA_POSITIVE_IN_NEGATIVE_FILES:
                rule = "2_barbara_flagged"
            elif not is_seed and positive and w in neg_marked:
                rule = "1_opposite_valence"
            elif not is_seed and not positive and w in pos_marked:
                rule = "1_opposite_valence"
            if rule:
                lexicons[cat].discard(w)
                removed.append((cat, w, rule))
    # rule 4: a word that is still in both a positive and a negative lexicon probably carries no
    # emotion (e.g. 'leadership', 'necessity' from Empath). It leaves every lexicon where it is not a seed.
    in_pos = set().union(*(lexicons[c] for c in POSITIVE))
    in_neg = set().union(*(lexicons[c] for c in NEGATIVE))
    for w in sorted(in_pos & in_neg):
        for cat in lexicons:
            if w in lexicons[cat] and "seed" not in sources[cat][w]:
                lexicons[cat].discard(w)
                removed.append((cat, w, "4_in_positive_and_negative"))
    for cat in lexicons:
        r = defaultdict(int)
        for c, _, rule in removed:
            if c == cat:
                r[rule] += 1
        print(f"   {cat:26s} kept {len(lexicons[cat]):5d}  removed: " +
              ", ".join(f"{k}={v}" for k, v in sorted(r.items())))

    # ---------------------------------------------------------
    # 6. Saving Results + report
    # ---------------------------------------------------------
    print("--- Step 6: Saving Lexicons ---")
    for cat, words in lexicons.items():
        with open(os.path.join(output_dir, f"{cat}.txt"), "w", encoding="utf-8") as f:
            for w in sorted(words):
                f.write(w + "\n")
        print(f"   Saved '{cat}.txt' with {len(words)} words.")

    with open(os.path.join(output_dir, "report_words_and_sources.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lexicon", "valence", "word", "sources"])
        for cat in lexicons:
            for w in sorted(lexicons[cat]):
                writer.writerow([cat, "positive" if cat in POSITIVE else "negative", w,
                                 "+".join(sorted(sources[cat][w]))])
    with open(os.path.join(output_dir, "report_removed_words.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lexicon", "word", "rule", "sources"])
        for cat, w, rule in removed:
            writer.writerow([cat, w, rule, "+".join(sorted(sources[cat][w]))])

    # Words NRC 'trust' would have added: is the decision to leave it out a good one?
    # in_v3_anyway = the word still got into some lexicon from another source.
    in_any = set().union(*lexicons.values())
    with open(os.path.join(output_dir, "report_nrc_trust_excluded.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "in_v3_anyway"])
        for w in sorted(nrc_trust_excluded):
            writer.writerow([w, w in in_any])
    trust_only = sorted(nrc_trust_excluded - in_any)
    print(f"\n--- NRC 'trust' (not used): {len(nrc_trust_excluded)} words, "
          f"{len(trust_only)} of them are in no v3 lexicon ---")
    step = max(1, len(trust_only) // 40)
    print("   sample of words we lost: " + ", ".join(trust_only[::step][:40]))

    print("\n--- Overlap check: positive vs negative lexicons (should be empty) ---")
    clash = False
    for p in POSITIVE:
        for n in NEGATIVE:
            common = lexicons[p] & lexicons[n]
            if common:
                clash = True
                print(f"   {p} & {n}: {len(common)} -> {sorted(common)[:15]}")
    if not clash:
        print("   none")

    print("\n--- Sample of expanded (non-seed) words per lexicon, for a manual look ---")
    for cat in lexicons:
        added = sorted(w for w in lexicons[cat] if "seed" not in sources[cat][w])
        step = max(1, len(added) // 30)
        print(f"   {cat} ({len(added)} added): " + ", ".join(added[::step][:30]))

    print(f"\nDone! All lexicons saved to: {output_dir}")
    return lexicons


# --------------------------------------------------
# Execution
# --------------------------------------------------

def main():
    base = "/content/gdrive/MyDrive/Data mining/text mining"
    nrc_path = f"{base}/NRC emotion.txt"
    gi_path = f"{base}/Harvard General Inquirer.txt"
    old_lexicons_dir = f"{base}/data/lexicons"       # gender / AI lexicons are read from here
    output_dir = f"{base}/data/lexicons_v3"
    build_lexicons(nrc_path, gi_path, old_lexicons_dir, output_dir)


if __name__ == "__main__":
    main()
