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
    # titles: Mr. -> Ms. (neutral form), Mrs./Ms. -> Mr.
    "mr.": "ms.",
    "mr": "ms",
    "mrs.": "mr.",
    "mrs": "mr",
    "ms.": "mr.",
    "ms": "mr",
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


def swap_gender(text):
    #spacy split to tokens
    doc = nlp(text)
    new_words = []
    
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
            next_token_is_punct = False
            if token.i + 1 < len(doc): 
                if doc[token.i + 1].is_punct:
                    next_token_is_punct = True
            
            if token.dep_ == "poss" and not next_token_is_punct:
                new_word = "her"  # "his car" -> "her car"
            else:
                new_word = "hers" # "the car is his," -> "the car is hers,"

        # "Miss" as a title ("Miss Lee") vs. the verb ("I miss her")
        elif word_lower == "miss":
            new_word = "mr" if token.pos_ == "PROPN" else token.text

        # basic lexicon
        elif word_lower in direct_swaps:
            new_word = direct_swaps[word_lower]
            
        else:
            new_word = token.text
            
        if token.is_title:
            new_word = new_word.capitalize()
            
        new_words.append(new_word)
        
    # back to text
    flipped_text = "".join([new_words[i] + doc[i].whitespace_ for i in range(len(doc))])
    
    return flipped_text

def main():

  print("Starting counterfactual generation...")
  test_sentences = [
    # 1. בדיקת המילה her: פעם אחת כמושא ופעם שנייה כשייכות
    "I saw her driving her new car with her mother.",
    
    # 2. בדיקת המילה his: פעם אחת כשייכות צמודה ופעם שנייה כשייכות עצמאית
    "His uncle told the teacher that the missing book is his.",
    
    # 3. בדיקת המילים שחזרו למילון (him ו-hers)
    "This bag is hers, so please give it back to her.",
    
    # 4. בדיקת מילים מהלקסיקון (משפחה, תארים) ושמירה על אות גדולה בתחילת משפט
    "The gentle woman and her husband walked the girls to school.",
    
    # 5. משפט האולטימטיבי: הכל מהכל בבת אחת
    "She told her brother that the car is his, but she still loves her car."
  ]

  print("--- Starting Sanity Check ---\n")

  for i, sentence in enumerate(test_sentences, 1):
      print(f"Test {i}:")
      print(f"Original: {sentence}")
      print(f"Flipped:  {swap_gender(sentence)}\n")
