"""
    First-stage version, replaced by lexical_collector_v3.py
    Build lexicons using:
    1. Seed words (Lecturer's list - Always included)
    2. Empath (Expansion)
    3. NRC (Text file parsing)
    4. Harvard General Inquirer (CSV/Tab parsing via csv module)
"""
import os
import csv
from empath import Empath

def build_lexicons(nrc_path, gi_path, output_dir):
 
    os.makedirs(output_dir, exist_ok=True)
    lexicon_generator = Empath()

    # ---------------------------------------------------------
    # 1. Seed Words 
    # ---------------------------------------------------------
    print("--- Step 1: Initializing Seed Words ---")
    lexicons = {
        "power_achievement": {
            "control", "superior", "goal", "success", "better", "win", "power", "achievement"
        },
        "cognitive_processes": {
            "think", "know", "analyze", "cause", "determine", "logically", "cognition", "reason"
        },
        "negative_emotions": {
            "scared", "fail", "worried", "hate", "nervous", "sadness", "fear"
        },
        "certainty": {
            "always", "never", "obviously", "definitely", "surely", "certain", "confidence"
        },
        "affiliation_social": {
            "friend", "talk", "community", "we", "team", "share", "social", "affiliation"
        },
        "perceptual_processes": {
            "see", "hear", "feel", "look", "sound", "perceive", "perception"
        },
        "positive_emotions": {
            "good", "nice", "great", "joy", "trust", "hope", "excited", "amazed", "love", "happy"
        },
        "reward_anticipation": {
            "take", "prize", "benefit", "award", "earn", "gain", "reward", "expect", "eager"
        }
    }


    # ---------------------------------------------------------
    # 2. Empath Expansion
    # ---------------------------------------------------------
    print("--- Step 2: Expanding with Empath ---")
    for category in lexicons:
        try:
            seeds = list(lexicons[category])
            new_words = lexicon_generator.create_category(category, seeds, size=50)
            
            if new_words:
                for w in new_words:
                    lexicons[category].add(w.lower())
                print(f"   > Expanded '{category}': added related words via Empath.")
        except Exception as e:
            print(f"   > Warning: Empath expansion failed for {category}: {e}")


    # ---------------------------------------------------------
    # 3. NRC Emotion Lexicon
    # ---------------------------------------------------------
    print("--- Step 3: Processing NRC Lexicon ---")
    # NRC
    nrc_negative_emotions = {"anger", "fear", "sadness", "disgust", "negative"}
    nrc_positive_emotions = {"joy", "trust", "anticipation", "surprise", "positive"}

    if os.path.exists(nrc_path):
        with open(nrc_path, 'r', encoding='utf-8') as f:
            for line in f:
                # empty line or #
                if not line.strip() or line.startswith('#'):
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    word = parts[0]
                    emotion = parts[1]
                    score = parts[2]
                    
                    if score == '1':
                        if emotion in nrc_negative_emotions:
                            lexicons["negative_emotions"].add(word.lower())
                        elif emotion in nrc_positive_emotions:
                            lexicons["positive_emotions"].add(word.lower())

        print("   > NRC processed successfully.")
    else:
        print(f"   > Error: NRC file not found at: {nrc_path}")


    # ---------------------------------------------------------
    # 4. Harvard General Inquirer
    # ---------------------------------------------------------
    print("--- Step 4: Processing Harvard General Inquirer (as CSV) ---")
    
    gi_mapping = {
        "power_achievement": ["Power", "Achieve", "Pwr", "Strong"],
        "cognitive_processes": ["Cognit", "Think", "Know", "Solve", "Reason"],
        "negative_emotions": ["Negativ", "Ngtv", "Hostile", "Pain"],
        "certainty": ["Sure", "If", "Strong", "Definite"],
        "affiliation_social": ["Affil", "Social", "Com"],
        "perceptual_processes": ["Percpt", "See", "Hear", "Feel", "Sense"],
        "positive_emotions": ["Positiv", "Pstv", "Virtue", "Pleasur"],
        "reward_anticipation": ["Goal", "Try"]
    }

    if os.path.exists(gi_path):
        try:
            with open(gi_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f, delimiter='\t')
              
                reader.fieldnames = [name.upper() for name in reader.fieldnames] if reader.fieldnames else []
                
                print(f"   > HGI Columns detected: {reader.fieldnames[:7]}...") 

                for row in reader:
                  
                    raw_word = row.get("ENTRY", "")
                    if not raw_word:
                        continue
                    
                    word = raw_word.split("#")[0].lower()

                    for my_cat, gi_cols in gi_mapping.items():
                        for col_name in gi_cols:
                            col_upper = col_name.upper()
                            if col_upper in row:
                                val = row[col_upper]
                                if val and val != "0":
                                    lexicons[my_cat].add(word)
                                    break 
                                    
            print("   > HGI processed successfully.")
        except Exception as e:
            print(f"   > Error reading HGI file: {e}")
    else:
        print(f"   > Error: HGI file not found at: {gi_path}")

    # ---------------------------------------------------------
    # 5. Saving Results
    # ---------------------------------------------------------
    print("--- Step 5: Saving Lexicons ---")
    for category, words in lexicons.items():
        file_path = os.path.join(output_dir, f"{category}.txt")
        sorted_words = sorted(list(words))
        
        with open(file_path, "w", encoding="utf-8") as f:
            for w in sorted_words:
                f.write(w + "\n")
        
        print(f"   Saved '{category}.txt' with {len(words)} words.")

    print(f"\nDone! All lexicons saved to: {output_dir}")

# --------------------------------------------------
# Execution
# --------------------------------------------------

def main():

  nrc_path = "/content/gdrive/MyDrive/Data mining/text mining/NRC emotion.txt"
  gi_path = "/content/gdrive/MyDrive/Data mining/text mining/Harvard General Inquirer.txt"
  output_dir = "/content/gdrive/MyDrive/Data mining/text mining/data/lexicons_v2"

  print("Starting Lexical Collection...")
  build_lexicons(nrc_path, gi_path, output_dir)
  print("Done.")

if __name__ == "__main__":
    main()
