"""
RQ2 robustness check - the VADER sentiment test, repeated with RoBERTa
---------------------------------------------------------------------
Same two steps as the VADER pipeline, only the model is different:
  1. Labeling (like src/preprocessing/algorithmic_labeler.py):
     each Hard AI comment gets a RoBERTa score in a NEW column 'sentiment_roberta'
     S = P(positive) - P(negative), in [-1, 1] (same range as VADER's compound score,
     same score as in RQ3 - src/counterfactual_causal_probing/cf_sentiment.py).
     'sentiment_vader' is not changed, and final_comments_dataset.csv is not overwritten:
     the scores are saved to a separate file.
  2. Statistical test (like src/analysis/sentiment_analysis.py):
     same duplicate removal, same boxplot, same Welch t-test, female_hard_ai vs male_hard_ai.

Only the Hard AI comments are labeled, because the test compares only these two groups.
Run in Colab with a GPU.
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from scipy.stats import ttest_ind
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"   # same model as RQ3
MAX_LEN = 512
HARD_AI_GROUPS = ['female_hard_ai', 'male_hard_ai']

base_dir = "/content/gdrive/MyDrive/Data mining/text mining"
data_path = os.path.join(base_dir, "data/processed/final_comments_dataset.csv")
output_path = os.path.join(base_dir, "data/processed/hard_ai_comments_roberta.csv")
results_dir = os.path.join(base_dir, "visualization")


# ---------------------------------------------------------------- 1. labeling

def load_model(name=MODEL_NAME):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(device)
    model.eval()
    return tokenizer, model, device


def get_roberta_scores(texts, tokenizer, model, device, batch_size=64):
    """S = P(positive) - P(negative) for each text, in [-1, 1]."""
    label2id = {k.lower(): v for k, v in model.config.label2id.items()}
    pos, neg = label2id["positive"], label2id["negative"]

    input_ids = tokenizer([str(t) for t in texts], truncation=True, max_length=MAX_LEN)["input_ids"]
    # sort by length so each batch has similar lengths -> less padding, faster
    order = np.argsort([len(x) for x in input_ids])
    scores = np.zeros(len(input_ids))

    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = tokenizer.pad({"input_ids": [input_ids[i] for i in idx]}, return_tensors="pt").to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(**batch).logits
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            scores[idx] = probs[:, pos] - probs[:, neg]

            if (start // batch_size) % 200 == 0:
                print(f"  {start + len(idx)}/{len(order)}")
    return scores


def label_hard_ai_comments():
    """Labels the Hard AI comments with RoBERTa and saves them. If the file exists, it is loaded instead."""
    if os.path.exists(output_path):
        print(f"Loading existing RoBERTa labels from {output_path}")
        return pd.read_csv(output_path)

    print("Loading comments dataset...")
    df = pd.read_csv(data_path)
    df = df.drop_duplicates(subset=['body', 'post_id'])          # same as sentiment_analysis.py
    df = df[df['parent_research_group'].isin(HARD_AI_GROUPS)].copy()
    print(f"Hard AI comments to label: {len(df)}")

    tokenizer, model, device = load_model()
    print(f"Model: {MODEL_NAME}, device: {device}")
    df['sentiment_roberta'] = get_roberta_scores(df['body'].fillna(""), tokenizer, model, device)

    df.to_csv(output_path, index=False)
    print(f"Saved RoBERTa labels to: {output_path}")
    return df


# ---------------------------------------------------------------- 2. statistical test

def analyze_rq2_sentiment_roberta(df_hard):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    palette_focused = {'female_hard_ai': '#d5a6bd', 'male_hard_ai': '#a4c2f4'}

    # --- Focused Comparison: Sentiment in Hard AI (Female vs Male) ---
    plt.figure(figsize=(7, 6))
    sns.boxplot(x='parent_research_group', y='sentiment_roberta', data=df_hard,
                palette=palette_focused, order=HARD_AI_GROUPS, width=0.5)

    plt.title('Focused Comparison: RoBERTa Sentiment in Hard AI (Female vs Male)')
    plt.ylabel('RoBERTa Sentiment Score (-1 to 1)')
    plt.xlabel('Gender Context')
    plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
    plt.xticks(ticks=[0, 1], labels=['Female Hard AI', 'Male Hard AI'])

    save_path_focused = os.path.join(results_dir, "RQ2_Sentiment_Boxplot_Focused_RoBERTa.png")
    plt.savefig(save_path_focused, bbox_inches='tight', dpi=300)
    print(f"Saved Focused Graph to: {save_path_focused}")
    plt.show()

    # --- STATISTICAL TEST ---
    alpha = 0.05

    print("\n" + "=" * 50)
    print("STATISTICAL TEST: Hostility in Hard AI (Female vs Male) - RoBERTa")
    print("=" * 50)

    group_female = df_hard[df_hard['parent_research_group'] == 'female_hard_ai']['sentiment_roberta']
    group_male = df_hard[df_hard['parent_research_group'] == 'male_hard_ai']['sentiment_roberta']

    t_stat, p_val = ttest_ind(group_female, group_male, equal_var=False)

    # Cohen's d (pooled SD), to compare the effect size with the VADER result
    n1, n2 = len(group_female), len(group_male)
    pooled_sd = np.sqrt(((n1 - 1) * group_female.var() + (n2 - 1) * group_male.var()) / (n1 + n2 - 2))
    cohen_d = (group_female.mean() - group_male.mean()) / pooled_sd

    print(f"Female Hard AI (Mean): {group_female.mean():.4f}")
    print(f"Male Hard AI   (Mean): {group_male.mean():.4f}")
    print(f"Difference:            {group_female.mean() - group_male.mean():.4f}")
    print("-" * 30)
    print(f"T-Statistic: {t_stat:.4f}")
    print(f"P-Value:     {p_val:.4e}")
    print(f"Cohen's d:   {cohen_d:.4f}")

    if p_val < alpha:
        print("\n=> RESULT: SIGNIFICANT DIFFERENCE FOUND.")
        if t_stat < 0:
            print("Women received significantly MORE NEGATIVE (lower) sentiment than men.")
        else:
            print("Women received significantly MORE POSITIVE (higher) sentiment than men (Surprisingly!).")
    else:
        print("\n=> RESULT: NO SIGNIFICANT DIFFERENCE.")
        print("We cannot prove that women receive different sentiment in Hard AI topics.")



# ---------------------------------------------------------------- 3. VADER vs RoBERTa

def compare_vader_roberta(df_hard):
    """Same test for both models, on exactly the same comments. Returns the comparison table."""
    rows = []
    for name, col in [("VADER", "sentiment_vader"), ("RoBERTa", "sentiment_roberta")]:
        f = df_hard[df_hard['parent_research_group'] == 'female_hard_ai'][col]
        m = df_hard[df_hard['parent_research_group'] == 'male_hard_ai'][col]
        t_stat, p_val = ttest_ind(f, m, equal_var=False)
        pooled_sd = np.sqrt(((len(f) - 1) * f.var() + (len(m) - 1) * m.var()) / (len(f) + len(m) - 2))
        rows.append({"Model": name,
                     "Female Hard AI (Mean)": round(f.mean(), 4),
                     "Male Hard AI (Mean)": round(m.mean(), 4),
                     "Difference": round(f.mean() - m.mean(), 4),
                     "T-Statistic": round(t_stat, 4),
                     "P-Value": f"{p_val:.4e}",
                     "Cohen's d": round((f.mean() - m.mean()) / pooled_sd, 4),
                     "N female": len(f), "N male": len(m)})
    table = pd.DataFrame(rows).set_index("Model")

    print("\n" + "=" * 50)
    print("COMPARISON: VADER vs RoBERTa (same Hard AI comments)")
    print("=" * 50)
    print(table.to_string())
    print("-" * 30)
    print(f"Correlation VADER vs RoBERTa (same comments): "
          f"{df_hard['sentiment_vader'].corr(df_hard['sentiment_roberta']):.4f}")
    return table


def main():
    print("=" * 60)
    print("STARTING Sentiment Analysis with RoBERTa (Hard AI, Female vs Male)")
    print("=" * 60)
    df_hard = label_hard_ai_comments()
    analyze_rq2_sentiment_roberta(df_hard)
    compare_vader_roberta(df_hard)


if __name__ == "__main__":
    main()
