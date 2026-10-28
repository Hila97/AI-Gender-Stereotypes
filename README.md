# Gender Stereotypes in AI Discourse on Reddit

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hila97/AI-Gender-Stereotypes/blob/main/Final_Project_AI_Gender_Stereotypes.ipynb)

## Goal

This project studies whether AI discourse on Reddit reproduces a gender stereotype that links women to **Soft AI** (ethics, policy, UX, art) and men to **Hard AI** (models, algorithms, hardware), and whether the climate of the discussion changes with the gender context.

The unit of analysis is **discourse framing, not speaker identity**. A "female-context post" is a post whose gender words are mostly female, whoever wrote it. This is a deliberate choice: the question is whether discourse that mentions women is framed differently, not whether women write differently.

Each research question is first answered with a simple, transparent method (lexicons, VADER, cosine similarity), and then examined in depth with a method that addresses that method's main weakness.

## Research Questions and Hypotheses

### RQ1 – Division and bias
*Is the mention of women associated with Soft AI more than the mention of men?*

**H1:** female context is associated with Soft AI and male context with Hard AI. H1 is tested at three levels:

| Level | Method | Why this level |
|---|---|---|
| Words | Hard/Soft lexicons, chi-square test | Direct test of the association |
| Semantic space | Word2Vec, cosine similarity, t-SNE, then **WEAT** (Hard vs Soft terms, technical vs governance roles, competence vs warmth) | Cosine is descriptive; WEAT adds a permutation test and robustness checks |
| Topics | **BERTopic** with gender words masked, and a **Gendered Framing Index** (GFI, −1 male to +1 female) per topic | The lexicons were fixed in advance; topics discovered from the data test whether the split appears bottom-up |

### RQ2 – Sentiment and climate
*Do comments on female-context posts in Hard AI topics have a more hostile tone?*

**H2:** comments on female-context Hard AI posts are more negative than comments on male-context Hard AI posts.

| Step | Method |
|---|---|
| Overall tone | VADER compound score of the comments, Welch's t-test. VADER scores the tone of the whole comment; gender words are not in its lexicon, so they do not affect the score. |
| Kind of affect | 14 affect lexicons (Welch's t-test, Cohen's d) |
| Mirror or reaction | The tone of the posts themselves: do comments only mirror their post? |
| Robustness | Comments come from few posts, so: regression with standard errors clustered by post and a control for the post's own tone, post-level tests, and leave-one-post-out |

### RQ3 – Gender intervention
*If only the gender words in a post are changed, do language models treat the post differently?*

RQ1 and RQ2 show associations. RQ3 is an intervention at the lexical level: counterfactual pairs swap every gender word in a post (she↔he, woman↔man, …) and keep the rest of the text unchanged. A control condition swaps non-gender word pairs.

**H3:** the gender swap changes how models treat the post:

| | Linked to | Model | Measure |
|---|---|---|---|
| **H3a** | RQ1 | The fine-tuned BERT Large Hard/Soft classifier | Decision flips (CFR), change in P(Hard), McNemar |
| **H3b** | RQ2 | RoBERTa sentiment model trained on social media text | Change in sentiment (ΔS). VADER cannot be used here: the only words that change are gender words, which are not in its lexicon. |
| **H3c** | RQ1 | Pre-trained BERT-base (not fine-tuned) | Embedding shift compared with control swaps, and its direction on a Soft→Hard axis |

A question on women's self-talk (LIWC) was part of the original plan and was dropped, because it is about the writer, whose gender cannot be verified on Reddit. The questions above are all about discourse framing.

## Data

Posts and comments were collected with the Reddit API from 33 subreddits. Posts were found by searching for the gender words; comments were collected for the 500 most-commented posts.

| Category | Subreddits |
|---|---|
| Technical Core | r/MachineLearning, r/OpenAI, r/deeplearning, r/LanguageTechnology, r/learnmachinelearning, r/MLQuestions, r/datascience, r/LocalLLaMA, r/Oobabooga, r/LLMDevs |
| Creative / Usage | r/Midjourney, r/StableDiffusion, r/AIArt, r/comfyui, r/dalle |
| Career | r/CSCareerQuestions, r/ExperiencedDevs, r/cscareerquestionsEU, r/MLjobs |
| Ethics & Soft AI | r/artificial, r/singularity, r/ChatGPT, r/AIethics, r/ControlProblem, r/AItechnology |
| AI Tools | r/ClaudeAI, r/Bard, r/bing, r/Perplexity, r/GPT3 |
| General | r/ArtificialInteligence, r/technology, r/FutureTechnology |

Three more subreddits were in the collection list (r/Leonardo_AI, r/NotionAI, r/datasciencecareer) but returned no posts.

| Data | Count |
|---|---|
| Posts | 57,392 |
| Posts with a female context | 13,757 |
| Posts with a male context | 24,314 |
| Posts with a Hard AI label | 11,628 |
| Posts with a Soft AI label | 11,681 |
| Comments | 942,108 |

<details>
<summary>Gender-context posts per subreddit</summary>

| Category | Subreddit | Female | Male | Total |
|---|---|---|---|---|
| Technical Core | MachineLearning | 228 | 575 | 803 |
| Technical Core | OpenAI | 310 | 724 | 1,034 |
| Technical Core | deeplearning | 65 | 568 | 633 |
| Technical Core | LanguageTechnology | 56 | 457 | 513 |
| Technical Core | learnmachinelearning | 214 | 623 | 837 |
| Technical Core | MLQuestions | 52 | 553 | 605 |
| Technical Core | datascience | 231 | 567 | 798 |
| Technical Core | LocalLLaMA | 206 | 580 | 786 |
| Technical Core | Oobabooga | 21 | 120 | 141 |
| Technical Core | LLMDevs | 24 | 343 | 367 |
| Creative / Usage | Midjourney | 1,228 | 1,136 | 2,364 |
| Creative / Usage | StableDiffusion | 1,146 | 954 | 2,100 |
| Creative / Usage | AIArt | 2,160 | 1,997 | 4,157 |
| Creative / Usage | comfyui | 422 | 657 | 1,079 |
| Creative / Usage | dalle | 96 | 287 | 383 |
| Career | CSCareerQuestions | 1,357 | 2,336 | 3,693 |
| Career | ExperiencedDevs | 298 | 853 | 1,151 |
| Career | cscareerquestionsEU | 466 | 837 | 1,303 |
| Career | MLjobs | 15 | 88 | 103 |
| Ethics & Soft AI | artificial | 420 | 922 | 1,342 |
| Ethics & Soft AI | singularity | 360 | 1,077 | 1,437 |
| Ethics & Soft AI | ChatGPT | 1,768 | 2,229 | 3,997 |
| Ethics & Soft AI | AIethics | 2 | 11 | 13 |
| Ethics & Soft AI | ControlProblem | 41 | 217 | 258 |
| Ethics & Soft AI | AItechnology | 3 | 12 | 15 |
| AI Tools | ClaudeAI | 287 | 940 | 1,227 |
| AI Tools | Bard | 244 | 704 | 948 |
| AI Tools | bing | 257 | 430 | 687 |
| AI Tools | Perplexity | 0 | 9 | 9 |
| AI Tools | GPT3 | 102 | 467 | 569 |
| General | ArtificialInteligence | 589 | 1,294 | 1,883 |
| General | technology | 1,089 | 1,742 | 2,831 |
| General | FutureTechnology | 0 | 5 | 5 |
| **Total** | — | **13,757** | **24,314** | **38,071** |

</details>

### Lexicons
| Lexicon | Words | Examples | Used for |
|---|---|---|---|
| Female | 21 | woman, female, girl, she, her, hers | Gender context, GFI, counterfactual swaps |
| Male | 19 | man, male, guy, he, his, him | Gender context, GFI, counterfactual swaps |
| Hard AI | 132 | LLM, transformer, PyTorch, tensor, GPU, optimization | AI domain label |
| Soft AI | 148 | ethics, bias, fairness, policy, regulation, design, art | AI domain label |
| Affect (14 lexicons) | 43 to 2,340 each | see below | RQ2 |

The affect lexicons follow a linguist's affect model (affect as an umbrella term for emotions, appraisals and their causes, each with a valence). They were built from the linguist's seed words, expanded with Empath, NRC and the Harvard General Inquirer, and cleaned of opposite-valence, gender and AI words:

| Positive (8) | Negative (6) |
|---|---|
| achievement_excellence, cognitive_insight, positive_emotions, certainty_confidence, affiliation_support, optimism_vitality, praise_admiration, reward_success | physical_adversity (disasters, disease, damage), social_legal_negativity (crime, conflict, oppression), situational_failure (lack, error, impediment), negative_traits (dishonest, hostile, incompetent), existential_negativity (endings, burden), negative_emotions |

They replace the first, LIWC-style lexicons (power & achievement, cognitive processes, certainty, affiliation, perceptual processes, negative emotion), which mixed neutral, positive and negative words.

## Multiple Comparisons

| Test family | Correction |
|---|---|
| RQ2: 14 affect lexicons + VADER, per topic | Bonferroni (α / 15) |
| RQ1 topics: pairs of topics (Dunn) and each topic vs the mean (permutation test) | Bonferroni |
| RQ1 semantic space: the 3 WEAT tests | None; the significant result (p = 0.004) also passes Bonferroni (α / 3) |
| RQ3: the 17 main tests of H3a–H3c | Holm |
| RQ3: 28 GoEmotions emotions | Holm |

With tens of thousands of texts almost every test is significant, so effect sizes (Cohen's d, r, ε²) are reported for every test.

## Key Findings

### RQ1 – Division and bias: supported at word and semantic level, partly at topic level
- **Words:** female-context posts are 63.6% Soft AI; male-context posts are 52.1% Hard AI (chi-square p < 10⁻⁷⁵).
- **Semantic space (WEAT):**
  - On the full lexicons, Hard AI terms are closer to male words than Soft AI terms (d = 0.93–1.05, p < 0.001). With the plan's 10 + 10 words the effect is in the same direction but not significant (d = 0.44, p = 0.18).
  - Competence words are closer to male words and warmth words to female words (d = 1.33, p = 0.004), the strongest and most stable result.
  - Technical vs governance roles: not interpretable (rare words).
- **Topics (BERTopic and GFI):**
  - Topics differ in GFI, with a small-to-medium effect (ε² = 0.068).
  - Hard AI topics are more male only when "guys" and "man" are counted (p = 0.003; without them p = 0.18). On Reddit these words are often address terms ("hey guys", "thanks man"), so the topic-level split is weak and depends on the lexicon.
  - The male lean that does not depend on these words is in topics about people and power (industry, Musk, OpenAI, careers).

### RQ2 – Sentiment and climate: not supported, the effect goes the other way
- Comments on female-context Hard AI posts are *more* positive (VADER 0.186 vs 0.109).
- The comments come from 42 Hard AI posts (11 female-context, 31 male-context).
- The gap holds with standard errors clustered by post, after controlling for the post's own tone, at post level (p = 0.002) and when any single post is left out.
- No single type of affect explains it.

### RQ3 – Gender intervention: not supported after Holm correction
- **H3a:** gender swaps almost never change the classifier's decision (3 of 2,368 pairs).
- **H3b:** sentiment shifts are negligible (|d| < 0.11).
- **H3c:** in the pre-trained BERT-base embedding, masculinizing a post moves it towards Hard AI (d = 0.86): a general-purpose language model links male words with Hard AI.

## Limitations

- Gender context is inferred from word counts. It describes the discourse, not the author.
- Posts were collected by searching for gender words, so only differences between groups or topics can be interpreted, not absolute levels.
- Comment-level results rest on a small number of threads (comments were collected for the 500 most-commented posts).
- RQ3: a control swap exists for 45.4% of the posts, so gender vs control comparisons use that subset. First names are not swapped, so some gender cues remain. LanguageTool checks grammar, not whether the swapped text is coherent.
- Topic names and the Hard/Soft grouping of the topics were drafted with an AI tool and reviewed manually.
- Future work: SEAT and DC-SEAT on contextual models, the embedding test (H3c) on the fine-tuned classifier itself, so that representation and decision are measured in the same model, and linking the counterfactual results to the topics.

## Repository Structure

```
AI-Gender-Stereotypes/
├── Final_Project_AI_Gender_Stereotypes.ipynb   # main notebook, all analyses with their outputs
├── AI_Gender_Stereotypes.ipynb                 # first-stage notebook
├── requirements.txt
├── data/
│   ├── raw/          # collected posts and comments (Git LFS)
│   ├── processed/    # cleaned and labeled datasets (Git LFS)
│   └── lexicons/     # gender and Hard/Soft AI lexicons
└── src/
    ├── collection/                    # Reddit collection, LIWC-style and affect lexicon building
    ├── preprocessing/                 # cleaning, implicit and algorithmic labeling
    ├── analysis/                      # RQ1 chi-square, RQ2 sentiment, Word2Vec, affect-lexicon analyses
    ├── classification/                # Logistic Regression, DistilBERT, BERT Large
    ├── visualization/                 # EDA, t-SNE, model comparison
    ├── topic_discovery/               # RQ1 topics: masking, BERTopic, GFI, tests, comments
    ├── counterfactual_causal_probing/ # RQ3: generation, control, grammar check, experiments, analysis
    └── weat/                          # RQ1 semantic space: WEAT
```

## Code and Notebook

- All analyses were run in **Google Colab** from `Final_Project_AI_Gender_Stereotypes.ipynb`. Each section adds a `src/` folder to the path and calls a script's `main()`. All outputs (tables, figures, test results) are saved in the notebook, so the results can be read without running it. A map at the top of the notebook shows which section answers which research question.
- Paths are hard-coded to a Google Drive folder (`BASE_DIR = /content/gdrive/MyDrive/Data mining/text mining`). The datasets in `data/` are stored with Git LFS; intermediate files, trained models and embeddings were kept on Drive and are not part of the repository.
- Install with `pip install -r requirements.txt`; the one-time downloads (spaCy model, NLTK data) are listed at the end of that file.

## Authors

Hila Pinker

M.Sc. final project, 2026. Supervised by Dr. Chaya Liebeskind, Department of Computer Science, Jerusalem College of Technology.
