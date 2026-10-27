"""
Direction 3 - WEAT on the existing Word2Vec model (plan, sections 3.3.2, 3.3.3 and 3.3.6)
-----------------------------------------------------------------------------------------
For each of the three tests of section 3.3.3 (X, Y = target words, A = male, B = female):

    s(w, A, B)    = mean cos(w, a) - mean cos(w, b)        > 0: w is closer to the male words
    statistic     = sum over X of s(x) - sum over Y of s(y)
    d             = (mean over X of s(x) - mean over Y of s(y)) / std over X and Y of s(w)
    p (one-sided) = share of 10,000 random X/Y relabelings with a statistic at least as large

d > 0 means X (hard / technical / competence) is closer to male and Y (soft / governance / warmth) closer to female.

Word lists (decisions made after weat_check_cell.py):
  - gender: 7 + 7 nouns; he/him/his and she/her/hers are not in the model (removed as stopwords before training);
  - two-word terms (neural network, gradient descent, data scientist): mean of the two unit vectors;
  - fine-tuning and nurturing are in the model as fine-tune and nurture; caring is in the model only as "care",
    which also covers "I don't care" and "health care", so it is left out;
  - test 2: policymaker is missing and 4 of the 5 governance words appear fewer than 60 times, so the test is
    reported as low reliability.

Robustness (SD-WEAT, section 3.3.6): d without each word in turn (leave-one-out), and d on 1,000 random subsets
(80% of every list); we report the mean, the standard deviation and the 95% range of d.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "/content/gdrive/MyDrive/Data mining/text mining"
MODEL_PATH = os.path.join(BASE_DIR, "models/gender_ai_w2v_optimized.model")
OUT_DIR = os.path.join(BASE_DIR, "data/weat")
FIG_DIR = os.path.join(OUT_DIR, "figures")
N_PERM = 10_000
N_SUB = 1_000
SUB_SHARE = 0.8
RARE = 100                  # words that appear fewer times than this are marked in the word table
SEED = 42

MALE = ["man", "male", "boy", "father", "son", "brother", "husband"]
FEMALE = ["woman", "female", "girl", "mother", "daughter", "sister", "wife"]
TESTS = {
    "1 Hard vs Soft": {
        "X": ["neural network", "transformer", "backpropagation", "gradient descent", "cnn", "lstm", "gpu",
              "tensor", "optimizer", "fine-tuning"],
        "Y": ["ethics", "fairness", "bias", "governance", "regulation", "privacy", "accountability",
              "transparency", "deepfake", "consent"],
        "X_name": "Hard AI", "Y_name": "Soft AI"},
    "2 Technical vs Governance roles": {
        "X": ["engineer", "developer", "programmer", "researcher", "architect", "data scientist"],
        "Y": ["ethicist", "advocate", "regulator", "auditor", "coordinator"],
        "X_name": "technical roles", "Y_name": "governance roles"},
    "3 Competence vs Warmth": {
        "X": ["competent", "skilled", "capable", "expert", "proficient", "technical", "analytical", "logical"],
        "Y": ["warm", "supportive", "collaborative", "empathetic", "inclusive", "nurturing", "helpful"],
        "X_name": "competence", "Y_name": "warmth"},
}
LOW_RELIABILITY = {"2 Technical vs Governance roles"}
IN_MODEL_AS = {"fine-tuning": "fine-tune", "nurturing": "nurture"}

COLOR = {"X": "#2a78d6", "Y": "#eb6834"}
INK2, MUTED, GRID, AXIS = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "font.size": 9})


def load_model(path=MODEL_PATH):
    from gensim.models import Word2Vec
    return Word2Vec.load(path).wv


def unit(v):
    return v / np.linalg.norm(v)


def vector(wv, term):
    """Unit vector of a term; a two-word term is the mean of its two unit vectors."""
    parts = IN_MODEL_AS.get(term, term).split()
    return unit(np.mean([unit(wv[p]) for p in parts], axis=0))


def count(wv, term):
    """How often the term appeared in the corpus (for two words: the rarer one)."""
    return min(wv.get_vecattr(p, "count") for p in IN_MODEL_AS.get(term, term).split())


def association(W, A, B):
    """s(w, A, B) for every row of W (all vectors are unit vectors, so cos = dot product)."""
    return (W @ A.T).mean(axis=1) - (W @ B.T).mean(axis=1)


def effect_size(sx, sy):
    return (sx.mean() - sy.mean()) / np.concatenate([sx, sy]).std(ddof=1)


def weat(X, Y, A, B, n_perm=N_PERM, rng=None):
    rng = rng or np.random.default_rng(SEED)
    sx, sy = association(X, A, B), association(Y, A, B)
    stat = sx.sum() - sy.sum()
    s_all, nx = np.concatenate([sx, sy]), len(sx)
    perm = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(len(s_all))
        perm[i] = s_all[idx[:nx]].sum() - s_all[idx[nx:]].sum()
    p = ((perm >= stat).sum() + 1) / (n_perm + 1)
    return {"statistic": stat, "d": effect_size(sx, sy), "p": p, "sx": sx, "sy": sy}


def robustness(X, Y, A, B, rng=None):
    """SD-WEAT: d without each word, and d on random 80% subsets of every list."""
    rng = rng or np.random.default_rng(SEED)
    loo = []
    for name, M in [("X", X), ("Y", Y), ("A", A), ("B", B)]:
        for i in range(len(M)):
            lists = {"X": X, "Y": Y, "A": A, "B": B}
            lists[name] = np.delete(M, i, axis=0)
            loo.append(effect_size(association(lists["X"], lists["A"], lists["B"]),
                                   association(lists["Y"], lists["A"], lists["B"])))
    sub = []
    for _ in range(N_SUB):
        pick = {n: M[rng.choice(len(M), max(2, round(SUB_SHARE * len(M))), replace=False)]
                for n, M in [("X", X), ("Y", Y), ("A", A), ("B", B)]}
        sub.append(effect_size(association(pick["X"], pick["A"], pick["B"]),
                               association(pick["Y"], pick["A"], pick["B"])))
    return np.array(loo), np.array(sub)


def run_tests(wv):
    A = np.array([vector(wv, w) for w in MALE])
    B = np.array([vector(wv, w) for w in FEMALE])
    rows, words, loo_rows = [], [], []
    for name, t in TESTS.items():
        X = np.array([vector(wv, w) for w in t["X"]])
        Y = np.array([vector(wv, w) for w in t["Y"]])
        r = weat(X, Y, A, B)
        loo, sub = robustness(X, Y, A, B)
        rows.append({"test": name, "n_X": len(X), "n_Y": len(Y), "statistic": r["statistic"], "d": r["d"],
                     "p": r["p"], "loo_d_min": loo.min(), "loo_d_max": loo.max(),
                     "sd_weat_mean_d": sub.mean(), "sd_weat_std_d": sub.std(ddof=1),
                     "sd_weat_d_2.5%": np.percentile(sub, 2.5), "sd_weat_d_97.5%": np.percentile(sub, 97.5),
                     "share_subsets_d_above_0": (sub > 0).mean(),
                     "reliability": "low" if name in LOW_RELIABILITY else "ok"})
        for side in ["X", "Y"]:
            for w, s in zip(t[side], r["s" + side.lower()]):
                words.append({"test": name, "side": side, "group": t[f"{side}_name"], "word": w,
                              "s_male_minus_female": s, "count": count(wv, w), "rare": count(wv, w) < RARE})
        # which single word changes d the most when it is left out
        names = t["X"] + t["Y"] + MALE + FEMALE
        for w, d in zip(names, loo):
            loo_rows.append({"test": name, "left_out": w, "d": d, "change": d - r["d"]})
    return pd.DataFrame(rows).set_index("test"), pd.DataFrame(words), pd.DataFrame(loo_rows)


def graph_words(words, name):
    """One bar per target word: s(w) > 0 means the word is closer to the male words."""
    t = TESTS[name]
    d = words[words["test"] == name].sort_values("s_male_minus_female")
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(d) + 1.2))
    y = np.arange(len(d))
    ax.barh(y, d["s_male_minus_female"], color=[COLOR[s] for s in d["side"]], height=0.7)
    labels = [f"{w}{' *' if r else ''}" for w, r in zip(d["word"], d["rare"])]
    ax.set_yticks(y, labels)
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("closer to female words  ←  s(w)  →  closer to male words")
    ax.set_title(f"WEAT test {name}")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[s]) for s in ["X", "Y"]]
    ax.legend(handles, [t["X_name"], t["Y_name"]], frameon=False, loc="lower right")
    if d["rare"].any():
        ax.text(1, -0.12 - 0.3 / len(d), f"* fewer than {RARE} occurrences in the corpus", transform=ax.transAxes,
                ha="right", fontsize=7, color=MUTED)
    save(fig, f"weat_words_test{name[0]}.png")


def graph_effects(res):
    """d of every test, with the 95% range of the random subsets and the leave-one-out range."""
    fig, ax = plt.subplots(figsize=(7, 2.6))
    y = np.arange(len(res))[::-1]
    for yi, (name, r) in zip(y, res.iterrows()):
        ax.plot([r["sd_weat_d_2.5%"], r["sd_weat_d_97.5%"]], [yi, yi], color=COLOR["X"], linewidth=2,
                solid_capstyle="round", label="95% of random subsets" if yi == y[0] else None)
        ax.plot([r["loo_d_min"], r["loo_d_max"]], [yi - 0.18] * 2, color=MUTED, linewidth=1.2,
                label="leave one word out" if yi == y[0] else None)
        ax.scatter(r["d"], yi, s=60, color=COLOR["X"], edgecolor="white", linewidth=1.5, zorder=3,
                   label="d (all words)" if yi == y[0] else None)
        p_txt = "p < 0.001" if r["p"] < 0.001 else f"p = {r['p']:.3f}"
        ax.annotate(f"d = {r['d']:.2f}, {p_txt}", (r["d"], yi + 0.22), ha="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [f"{n}{' (low reliability)' if n in LOW_RELIABILITY else ''}" for n in res.index])
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.set_ylim(-0.6, len(res) - 0.4)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("effect size d  (> 0: X closer to male, Y closer to female)")
    ax.set_title("WEAT on Word2Vec: effect size and robustness")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    save(fig, "weat_effect_sizes.png")


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name), dpi=200, bbox_inches="tight")
    plt.show()


def main(draw_graphs=True, wv=None):
    print("=" * 60)
    print("DIRECTION 3 - WEAT on the existing Word2Vec model")
    print("=" * 60)
    wv = wv if wv is not None else load_model()
    res, words, loo = run_tests(wv)
    os.makedirs(OUT_DIR, exist_ok=True)
    res.to_csv(os.path.join(OUT_DIR, "weat_results.csv"))
    words.to_csv(os.path.join(OUT_DIR, "weat_word_scores.csv"), index=False)
    loo.to_csv(os.path.join(OUT_DIR, "weat_leave_one_out.csv"), index=False)

    print(f"\n--- WEAT results ({N_PERM:,} permutations, {N_SUB:,} random subsets) ---")
    print(res.T.round(3).to_string())
    print("\n--- Words that change d the most when left out ---")
    top = loo.reindex(loo["change"].abs().sort_values(ascending=False).index).groupby("test").head(3)
    print(top.round(3).to_string(index=False))
    if draw_graphs:
        for name in TESTS:
            graph_words(words, name)
        graph_effects(res)
    print(f"\nSaved weat_results.csv, weat_word_scores.csv and weat_leave_one_out.csv to {OUT_DIR}")
    return {"res": res, "words": words, "loo": loo, "wv": wv}


if __name__ == "__main__":
    main()
