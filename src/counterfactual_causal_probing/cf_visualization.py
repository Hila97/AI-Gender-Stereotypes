import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# Plots for direction 2 (counterfactual causal probing).
# Every plot reads only files that the experiments and cf_analysis.py already saved,
# so nothing here runs a model. One function per plot; main() draws all of them and
# saves PNGs to counterfactual/figures/.

DATA_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
CF_DIR = os.path.join(DATA_DIR, "counterfactual")
FIG_DIR = os.path.join(CF_DIR, "figures")

# one fixed color per meaning, used the same way in every plot
COLOR = {
    "male->female": "#2a78d6",   # blue
    "female->male": "#eb6834",   # orange
    "gender": "#2a78d6",
    "control": "#9a9893",        # gray: the control swap is the "neutral" reference
    "flip": "#e34948",           # red: only for the 3 flipped pairs in 2a
    "other": "#b9c9dc",          # light blue-gray: background points
}
EXP_COLOR = {"2a": "#2a78d6", "2b": "#eb6834", "2c": "#1baf7a"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2a27", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False,
})


def _save(fig, name, save):
    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, name)
        fig.savefig(path)
        print(f"Saved {path}")


def _load_json(name):
    with open(os.path.join(CF_DIR, name)) as f:
        return json.load(f)


def _ci95(x):
    """Mean and 95% confidence half-width (normal approximation, fine for n in the hundreds+)."""
    x = np.asarray(x, float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


# ---------------------------------------------------------------- 2a

def plot_2a_scatter(save=True):
    """
    P(hard) before vs after the gender swap, one point per test post.
    Almost all points sit on the diagonal (no change); the 3 flipped posts are highlighted.
    """
    preds = pd.read_pickle(os.path.join(CF_DIR, "cf_classification_preds.pkl"))
    df = preds[~preds["cf_truncated_same"].astype(bool)]
    flip = (df["pred_orig"] != df["pred_cf"]).values

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color="#8a8984", lw=1, ls="--", zorder=1)
    ax.axhline(0.5, color="#d6d5d0", lw=0.8, zorder=1)
    ax.axvline(0.5, color="#d6d5d0", lw=0.8, zorder=1)
    ax.scatter(df.loc[~flip, "p_hard_orig"], df.loc[~flip, "p_hard_cf"], s=10,
               color=COLOR["other"], alpha=0.7, lw=0, label=f"no flip (n = {(~flip).sum():,})", zorder=2)
    ax.scatter(df.loc[flip, "p_hard_orig"], df.loc[flip, "p_hard_cf"], s=70, color=COLOR["flip"],
               edgecolor="white", lw=1.5, label=f"label flipped (n = {flip.sum()})", zorder=3)
    for _, r in df[flip].iterrows():
        ax.annotate(f"{r['p_hard_orig']:.2f} → {r['p_hard_cf']:.2f}", (r["p_hard_orig"], r["p_hard_cf"]),
                    xytext=(8, -4), textcoords="offset points", fontsize=8, color="#52514e")

    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), aspect="equal",
           xlabel="P(Hard AI) – original post", ylabel="P(Hard AI) – gender-swapped post",
           title="2a. Gender swaps almost never change the classifier")
    ax.legend(loc="upper left")
    _save(fig, "2a_scatter_p_hard.png", save)
    return fig


# ---------------------------------------------------------------- 2b

def plot_2b_hist(save=True):
    """
    Distribution of the sentiment change dS for VADER vs RoBERTa (same x axis, two panels).
    VADER is a single bar at 0: it cannot see gender words at all.
    """
    vader = pd.read_pickle(os.path.join(CF_DIR, "cf_vader_sanity.pkl"))
    scores = pd.read_pickle(os.path.join(CF_DIR, "cf_sentiment_scores.pkl"))
    scores = scores[~scores["cf_truncated_same"].astype(bool)]
    ds = (scores["S_cf"] - scores["S_orig"]).values

    lim = 0.15
    bins = np.linspace(-lim, lim, 61)
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.6), sharex=True)
    for ax, x, name, color in [(axes[0], vader["dS_vader"].values, "VADER (lexicon)", COLOR["control"]),
                               (axes[1], ds, "RoBERTa (contextual)", COLOR["gender"])]:
        w = np.ones(len(x)) / len(x) * 100
        ax.hist(np.clip(x, -lim, lim), bins=bins, weights=w, color=color, edgecolor="white", lw=0.5)
        ax.set_ylabel("% of pairs")
        ax.text(0.99, 0.9, f"{name}\nn = {len(x):,}, ΔS = 0 in {np.mean(x == 0):.0%}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#2b2a27")
    axes[1].set_xlabel("ΔS = sentiment(swapped) − sentiment(original)   (values beyond ±0.15 shown at the edge)")
    axes[0].set_title("2b. VADER cannot see the swap; RoBERTa sees small changes")
    fig.tight_layout()
    _save(fig, "2b_hist_vader_vs_roberta.png", save)
    return fig


def plot_2b_bars(save=True):
    """Mean dS by swap direction x AI category, with 95% CI (H2b expected male->female > 0 in Hard AI)."""
    scores = pd.read_pickle(os.path.join(CF_DIR, "cf_sentiment_scores.pkl"))
    df = scores[~scores["cf_truncated_same"].astype(bool)]
    df = df.assign(dS=df["S_cf"] - df["S_orig"])

    cats = [("hard_ai", "Hard AI"), ("soft_ai", "Soft AI"), ("general", "General")]
    dirs = [("male", "male->female"), ("female", "female->male")]
    x = np.arange(len(cats))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.2, 4))
    for j, (ctx, name) in enumerate(dirs):
        stats_ = [_ci95(df.loc[(df["gender_context"] == ctx) & (df["ai_category"] == c), "dS"]) for c, _ in cats]
        means, errs = zip(*stats_)
        ax.bar(x + (j - 0.5) * (width + 0.02), means, width, yerr=errs, color=COLOR[name],
               label=name.replace("->", " → "), error_kw={"lw": 1, "ecolor": "#52514e", "capsize": 3})
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xticks(x, [label for _, label in cats])
    ax.set_ylabel("mean ΔS  (+ = more positive after swap)")
    ax.set_title("2b. Feminizing lowers sentiment slightly; weakest in Hard AI")
    ax.legend(loc="best")
    _save(fig, "2b_bars_direction_category.png", save)
    return fig


# ---------------------------------------------------------------- 2c

def plot_2c_arrows(n_arrows=120, seed=42, save=True):
    """
    The "map" view of 2c. x = position on the Soft -> Hard axis (0 = Soft centre, 1 = Hard centre),
    y = position on the female -> male direction (0 = female centre, 1 = male centre).
    Thin arrows: a random sample of posts, original -> swapped. Thick arrows: the mean move per direction.
    """
    shifts = pd.read_pickle(os.path.join(CF_DIR, "cf_embedding_shifts.pkl"))
    e_orig = np.load(os.path.join(CF_DIR, "cf_embeddings_orig.npy")).astype(np.float32)
    e_cf = np.load(os.path.join(CF_DIR, "cf_embeddings_cf.npy")).astype(np.float32)
    cat, ctx = shifts["ai_category"].values, shifts["gender_context"].values
    valid = ~shifts["cf_truncated_same"].astype(bool).values

    def unit(v):
        return v / np.linalg.norm(v)

    def scaled(axis, low_mask, high_mask):
        """Projection on `axis`, rescaled so the low group's centre is 0 and the high group's is 1."""
        p_o, p_c = e_orig @ axis, e_cf @ axis
        lo, hi = p_o[low_mask].mean(), p_o[high_mask].mean()
        return (p_o - lo) / (hi - lo), (p_c - lo) / (hi - lo)

    c = lambda m: e_orig[m].mean(0)
    hs_axis = unit(c(cat == "hard_ai") - c(cat == "soft_ai"))
    mf_axis = unit(c(ctx == "male") - c(ctx == "female"))
    x_o, x_c = scaled(hs_axis, cat == "soft_ai", cat == "hard_ai")
    y_o, y_c = scaled(mf_axis, ctx == "female", ctx == "male")

    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    handles, shown = [], []
    for g, name in [("male", "male->female"), ("female", "female->male")]:
        idx = np.where(valid & (ctx == g))[0]
        sample = rng.choice(idx, size=min(n_arrows, len(idx)), replace=False)
        shown.append(sample)
        for i in sample:
            ax.annotate("", xy=(x_c[i], y_c[i]), xytext=(x_o[i], y_o[i]),
                        arrowprops={"arrowstyle": "-|>", "color": COLOR[name], "lw": 0.6, "alpha": 0.35,
                                    "mutation_scale": 6})
        # mean move, drawn from the group's mean original position
        mx0, my0 = x_o[idx].mean(), y_o[idx].mean()
        mx1, my1 = x_c[idx].mean(), y_c[idx].mean()
        ax.annotate("", xy=(mx1, my1), xytext=(mx0, my0),
                    arrowprops={"arrowstyle": "-|>", "color": COLOR[name], "lw": 3, "mutation_scale": 18})
        ax.scatter([mx0], [my0], s=36, color=COLOR[name], edgecolor="white", lw=1.5, zorder=4)
        handles.append(Line2D([0], [0], color=COLOR[name], lw=3,
                              label=f"{name.replace('->', ' → ')}  (mean Δx = {np.mean(x_c[idx] - x_o[idx]):+.1%})"))

    # arrows do not set the axis limits by themselves: use the central 96% of the drawn points
    shown = np.concatenate(shown)
    xs, ys = np.r_[x_o[shown], x_c[shown], 0, 1], np.r_[y_o[shown], y_c[shown], 0, 1]
    (x0, x1), (y0, y1) = np.percentile(xs, [2, 98]), np.percentile(ys, [2, 98])
    ax.set_xlim(min(x0, -0.1) - 0.1, max(x1, 1.1) + 0.1)
    ax.set_ylim(min(y0, -0.1) - 0.1, max(y1, 1.1) + 0.1)

    for v, lab in [(0, "Soft centre"), (1, "Hard centre")]:
        ax.axvline(v, color="#c8c7c2", lw=0.8, ls="--")
        ax.text(v, 1.0, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8,
                color="#52514e")
    ax.set(xlabel="Soft AI  →  Hard AI   (0 = Soft centre, 1 = Hard centre)",
           ylabel="female  →  male   (0 = female centre, 1 = male centre)")
    ax.set_title("2c. Gender swaps move posts along the Soft–Hard axis", pad=18)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, fontsize=8.5)
    _save(fig, "2c_arrows_map.png", save)
    return fig


def plot_2c_bars(save=True):
    """
    Two panels, gender (blue) vs control (gray), same posts:
    left  = HOW MUCH the post moves (cosine distance, same n of swaps) -> equal
    right = how much it moves ALONG the Soft-Hard axis               -> gender ~x2
    """
    r = _load_json("cf_2c_results.json")
    h2c = r["H2c"]["same n of swaps"]
    ax_ = r["dProj_gender_vs_control"]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
    panels = [(axes[0], [h2c["mean_delta_gender"], h2c["mean_delta_control"]],
               "Distance moved (cosine)", "How much? (same n of swaps)", "{:.4f}"),
              (axes[1], [ax_["mean_abs_dProj_gender"] * 100, ax_["mean_abs_dProj_control"] * 100],
               "|shift on Soft–Hard axis| (% of gap)", "In which direction?", "{:.1f}%")]
    for ax, vals, ylabel, title, fmt in panels:
        bars = ax.bar(["gender swap", "control swap"], vals, width=0.55,
                      color=[COLOR["gender"], COLOR["control"]])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.2)
    fig.suptitle("2c. Gender moves posts as far as any word, but in a fixed direction", fontweight="bold")
    fig.tight_layout()
    _save(fig, "2c_bars_gender_vs_control.png", save)
    return fig


# ---------------------------------------------------------------- summary

def plot_summary_forest(save=True):
    """
    Forest plot of Cohen's d with 95% CI for every main test of direction 2.
    Filled marker = significant after Holm (over all tests), hollow = not significant.
    Dashed lines at |d| = 0.2: the usual "small effect" threshold.
    """
    res = _load_json("direction2_results.json")
    rows = [t for t in res["tests"] if t["d"] == t["d"]]   # McNemar has no d
    rows = rows[::-1]                                       # first test at the top

    fig, ax = plt.subplots(figsize=(7.4, 0.36 * len(rows) + 1.4))
    for i, t in enumerate(rows):
        color = EXP_COLOR[t["experiment"]]
        lo, hi = t["d_ci95"]
        sig = t["p_holm"] < 0.05
        ax.plot([lo, hi], [i, i], color=color, lw=2, solid_capstyle="round")
        ax.scatter([t["d"]], [i], s=46, color=color if sig else "white", edgecolor=color, lw=1.8, zorder=3)
    ax.axvline(0, color="#52514e", lw=0.9)
    for v in (-0.2, 0.2):
        ax.axvline(v, color="#b4b3ae", lw=0.8, ls="--")
    ax.set_yticks(range(len(rows)), [f"{t['experiment']}  {t['test']}" for t in rows], fontsize=8.5)
    ax.set_xlabel("Cohen's d (95% CI)   — dashed lines: |d| = 0.2")
    ax.set_title("Direction 2: effect sizes of all main tests")
    ax.grid(axis="y", visible=False)

    handles = [Line2D([0], [0], marker="o", color=EXP_COLOR[e], lw=2, label=lab)
               for e, lab in [("2a", "2a classification"), ("2b", "2b sentiment"), ("2c", "2c embedding")]]
    handles += [Line2D([0], [0], marker="o", ls="", color="#52514e", label="significant after Holm"),
                Line2D([0], [0], marker="o", ls="", markerfacecolor="white", color="#52514e",
                       label="not significant")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.35, -0.1), ncol=3, fontsize=8)
    _save(fig, "summary_forest_cohens_d.png", save)
    return fig


# ---------------------------------------------------------------- run

def main(save=True):
    figs = {}
    for name, fn in [("2a_scatter", plot_2a_scatter), ("2b_hist", plot_2b_hist), ("2b_bars", plot_2b_bars),
                     ("2c_arrows", plot_2c_arrows), ("2c_bars", plot_2c_bars), ("summary_forest", plot_summary_forest)]:
        print(f"\n--- {name}")
        figs[name] = fn(save=save)
        plt.show()
    return figs
