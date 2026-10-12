"""Paper figures from metrics/ (run aggregate.py first). PDF (vector, for Overleaf) + PNG (preview).

Main figures use the threshold-free metrics (AUROC, AUPRC); "_thresholded" versions show recall, precision and F1
at each model's fixed validation threshold (deployment view, appendix).

  fig_compact_<crit>.pdf                 per classifier: zero-shot prompt-only / zero-shot prompt+response / best
                                         fine-tuned; AUROC, AUPRC, recall, precision, F1 with 95% bootstrap CIs
  fig_sweep_<crit>.pdf                   truncation sweeps, one column per classifier, zero-shot vs best fine-tuned (AUROC, AUPRC)
  fig_sweep_thresholded_<crit>.pdf       same, recall / precision / F1 at the fixed threshold
  fig_sweep_overlay_<crit>.pdf           best fine-tuned model of every classifier on one axis (AUROC, AUPRC)
  fig_sweep_overlay_thresholded_<crit>.pdf  same for recall / precision / F1
  fig_sweep_combined_<crit>.pdf          all classifiers overlaid, solid fine-tuned + dashed zero-shot (AUROC, AUPRC)
  fig_sweep_combined_thresholded_<crit>.pdf  same for recall / precision / F1
  fig_sweep_finetuned_4metrics[_row]_<crit>.pdf  (2x2 or 1x4) fine-tuned only (no baselines), all classifiers overlaid: AUPRC, precision, recall, F1
  fig_sweep_all_strategies.pdf           per classifier, all six strategies + zero-shot (AUROC, AUPRC)
  fig_sweep_all_strategies_thresholded.pdf   same for recall / precision / F1
  fig_strategy_heatmap.pdf               all strategies x classifiers at 100% response (appendix)

Colours: reference categorical palette with distinct markers (identity never relies on colour alone); greys for
zero-shot baselines.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C
from aggregate import CLASSIFIERS

M, FIG = C.NEW_RESULTS / "metrics", C.NEW_RESULTS / "figures"
CLF_COLOR = {"deberta": "#2a78d6", "promptguard": "#eb6834", "llamaguard3": "#1baf7a", "wildguard": "#eda100", "laya": "#e87ba4"}
SHORT = {"deberta": "DeBERTa-v3", "promptguard": "PromptGuard-2", "llamaguard3": "LlamaGuard-3", "wildguard": "WildGuard", "laya": "LAYA"}
CLF_MARK = {"deberta": "o", "promptguard": "s", "llamaguard3": "^", "wildguard": "D", "laya": "v"}
FT_COLOR, ZS_COLOR, ZS_PROMPT_COLOR, FT_PROMPT_COLOR = "#2a78d6", "#52514e", "#b9b8b3", "#9cc3ee"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "lines.linewidth": 1.6, "lines.markersize": 4, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


from matplotlib.lines import Line2D

AUC_METRICS = [("roc_auc", "AUROC"), ("auprc", "AUPRC")]
THR_METRICS = [("recall", "Recall"), ("precision", "Precision"), ("f1", "F1 (harmful)")]
STRAT_COLOR = dict(zip(C.STRATEGIES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
STRAT_MARK = dict(zip(C.STRATEGIES, ["o", "s", "^", "D", "v", "P"]))


def ylim_for(m):
    return (0.4, 1.0) if m == "roc_auc" else (0, 1.0)


def compact_bars(crit):
    df = pd.read_csv(C.NEW_RESULTS / "compact" / f"compact_{crit}.csv")
    metrics = AUC_METRICS + THR_METRICS
    clfs = [c for c in CLASSIFIERS if c in set(df.classifier)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.2, 2.2), sharey=False)
    w = 0.2
    kinds = [("Zero-shot, prompt only", ZS_PROMPT_COLOR, "Zero-shot, prompt only"),
             ("Zero-shot, prompt + response", ZS_COLOR, "Zero-shot, prompt + response"),
             ("Fine-tuned, prompt only", FT_PROMPT_COLOR, "Fine-tuned, prompt only"),
             ("Fine-tuned (", FT_COLOR, "Fine-tuned, prompt + response")]
    for ax, (m, title) in zip(axes, metrics):
        for k, (prefix, color, label) in enumerate(kinds):
            vals, lo, hi = [], [], []
            for c in clfs:
                r = df[(df.classifier == c) & df.setting.str.startswith(prefix)]
                v = r[m].iloc[0] if len(r) else np.nan
                vals.append(v)
                lo.append(v - r[f"{m}_lo"].iloc[0] if len(r) else 0); hi.append(r[f"{m}_hi"].iloc[0] - v if len(r) else 0)
            x = np.arange(len(clfs)) + (k - 1.5) * w
            ax.bar(x, vals, w * 0.92, color=color, label=label, linewidth=0)
            ax.errorbar(x, vals, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.2)
        ax.set_title(title, color=INK)
        ax.set_xticks(np.arange(len(clfs)))
        ax.set_xticklabels([SHORT[c] for c in clfs], rotation=40, ha="right", fontsize=6)
        ax.set_ylim(0, 1.0); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Score (test)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.10))
    save(fig, f"fig_compact_{crit}")


def sweeps(crit, metrics, suffix):
    sel = json.load(open(M / "selection.json"))["selected"][crit]
    t = pd.read_csv(M / "test_all.csv")
    clfs = [c for c in CLASSIFIERS if c in sel]
    fig, axes = plt.subplots(len(metrics), len(clfs), figsize=(7.0, 1.55 * len(metrics) + 0.9), sharex=True, sharey="row", squeeze=False)
    for j, c in enumerate(clfs):
        for i, (m, ylabel) in enumerate(metrics):
            ax = axes[i, j]
            for var, color, label, ls in [("zero_shot", ZS_COLOR, "Zero-shot", "--"), (f"ft_{sel[c]}", FT_COLOR, "Fine-tuned", "-")]:
                d = t[(t.classifier == c) & (t.variant == var)].sort_values("frac")
                if len(d) > 1:
                    ax.plot(d.frac * 100, d[m], ls, color=color, marker="o" if var != "zero_shot" else "s",
                            markersize=3, label=label)
            ax.set_ylim(*ylim_for(m)); ax.set_xticks([0, 50, 100])
            if i == 0:
                ax.set_title(f"{CLASSIFIERS[c][0]}\n({sel[c].replace('_', ' ')})", color=INK)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == len(metrics) - 1:
                ax.set_xlabel("Response shown (%)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, f"fig_sweep{suffix}_{crit}")


def overlay(crit, metrics, suffix):
    sel = json.load(open(M / "selection.json"))["selected"][crit]
    t = pd.read_csv(M / "test_all.csv")
    clfs = [c for c in CLASSIFIERS if c in sel]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.0, 2.5), sharey=False)
    for k, (ax, (m, title)) in enumerate(zip(axes, metrics)):
        for c in clfs:
            d = t[(t.classifier == c) & (t.variant == f"ft_{sel[c]}")].sort_values("frac")
            if len(d) > 1:
                ax.plot(d.frac * 100, d[m], color=CLF_COLOR[c], marker=CLF_MARK[c], label=CLASSIFIERS[c][0])
        ax.set_title(title, color=INK); ax.set_ylim(*ylim_for(m)); ax.set_xticks(range(0, 101, 20))
        ax.set_xlabel("Response shown (%)")
        if k == 0:
            ax.set_ylabel("Score (test)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, f"fig_sweep_overlay{suffix}_{crit}")


def combined(crit, metrics, suffix):
    """Every classifier overlaid in each panel: solid = fine-tuned (selected strategy), dashed = zero-shot."""
    sel = json.load(open(M / "selection.json"))["selected"][crit]
    t = pd.read_csv(M / "test_all.csv")
    clfs = [c for c in CLASSIFIERS if c in sel]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.0, 2.8), sharey=False)
    for k, (ax, (m, title)) in enumerate(zip(axes, metrics)):
        for c in clfs:
            for var, ls, alpha in [("zero_shot", "--", 0.55), (f"ft_{sel[c]}", "-", 1.0)]:
                d = t[(t.classifier == c) & (t.variant == var)].sort_values("frac")
                if len(d) > 1:
                    ax.plot(d.frac * 100, d[m], ls, color=CLF_COLOR[c], marker=CLF_MARK[c], markersize=3, linewidth=1.4, alpha=alpha)
        ax.set_title(title, color=INK); ax.set_ylim(*ylim_for(m)); ax.set_xticks(range(0, 101, 20))
        ax.set_xlabel("Response shown (%)")
        if k == 0:
            ax.set_ylabel("Score (test)")
    clf_h = [Line2D([], [], color=CLF_COLOR[c], marker=CLF_MARK[c], markersize=4, label=CLASSIFIERS[c][0]) for c in clfs]
    sty_h = [Line2D([], [], color=MUTED, ls="-", label="Fine-tuned (selected strategy)"),
             Line2D([], [], color=MUTED, ls="--", alpha=0.7, label="Zero-shot")]
    fig.add_artist(fig.legend(handles=clf_h, loc="upper center", ncol=len(clf_h), frameon=False, bbox_to_anchor=(0.5, 1.10)))
    fig.legend(handles=sty_h, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, f"fig_sweep_combined{suffix}_{crit}")


def finetuned_four(crit, row=False):
    """Fine-tuned (selected strategy) models only, all classifiers overlaid: AUPRC, precision, recall, F1 (no baselines)."""
    sel = json.load(open(M / "selection.json"))["selected"][crit]
    t = pd.read_csv(M / "test_all.csv")
    clfs = [c for c in CLASSIFIERS if c in sel]
    panels = [("auprc", "AUPRC (threshold-free)"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1 (harmful)")]
    if row:
        fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=True)
        axes = np.atleast_2d(axes)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True, sharey=True)
    for k, (ax, (m, title)) in enumerate(zip(axes.ravel(), panels)):
        for c in clfs:
            d = t[(t.classifier == c) & (t.variant == f"ft_{sel[c]}")].sort_values("frac")
            if len(d) > 1:
                ax.plot(d.frac * 100, d[m], color=CLF_COLOR[c], marker=CLF_MARK[c], markersize=3.5,
                        label=f"{CLASSIFIERS[c][0]} ({sel[c].replace('_', ' ')})")
        if row:
            ax.set_title(title.replace(" (threshold-free)", ""), color=INK); ax.set_xticks(range(0, 101, 25))
            ax.set_xlabel("Response shown (%)")
            if k == 0:
                ax.set_ylabel("Score (test)")
        else:
            ax.set_title(title if m == "auprc" else f"{title}, fixed validation threshold", color=INK)
            ax.set_xticks(range(0, 101, 20))
            if k >= 2:
                ax.set_xlabel("Response shown (%)")
            if k % 2 == 0:
                ax.set_ylabel("Score (test)")
        ax.set_ylim(0, 1.0)
    h, l = axes[0, 0].get_legend_handles_labels()
    if row:
        fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.10), fontsize=6.3)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        save(fig, f"fig_sweep_finetuned_4metrics_row_{crit}")
    else:
        fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        save(fig, f"fig_sweep_finetuned_4metrics_{crit}")


def all_strategies(metrics, suffix, crit="recall"):
    """Per classifier, every imbalance strategy (bold = recall-selected) plus zero-shot (dashed grey)."""
    sel = json.load(open(M / "selection.json"))["selected"][crit]
    t = pd.read_csv(M / "test_all.csv")
    clfs = [c for c in CLASSIFIERS if c in set(t.classifier)]
    fig, axes = plt.subplots(len(metrics), len(clfs), figsize=(7.0, 1.6 * len(metrics) + 1.2), sharex=True, sharey="row", squeeze=False)
    for j, c in enumerate(clfs):
        for i, (m, ylabel) in enumerate(metrics):
            ax = axes[i, j]
            z = t[(t.classifier == c) & (t.variant == "zero_shot")].sort_values("frac")
            ax.plot(z.frac * 100, z[m], "--", color=ZS_COLOR, linewidth=1.1, alpha=0.8)
            for s in C.STRATEGIES:
                d = t[(t.classifier == c) & (t.variant == f"ft_{s}")].sort_values("frac")
                lw = 2.0 if sel.get(c) == s else 1.0
                if len(d) > 1:
                    ax.plot(d.frac * 100, d[m], color=STRAT_COLOR[s], marker=STRAT_MARK[s], markersize=2.2, linewidth=lw)
                elif len(d) == 1:
                    ax.plot(d.frac * 100, d[m], linestyle="none", color=STRAT_COLOR[s], marker=STRAT_MARK[s], markersize=4)
            ax.set_ylim(*ylim_for(m)); ax.set_xticks([0, 50, 100])
            if i == 0:
                ax.set_title(f"{CLASSIFIERS[c][0]}\n(bold: {sel.get(c, '-').replace('_', ' ')})", color=INK, fontsize=7.5)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == len(metrics) - 1:
                ax.set_xlabel("Response shown (%)")
    handles = [Line2D([], [], color=STRAT_COLOR[s], marker=STRAT_MARK[s], markersize=4, label=s.replace("_", " ")) for s in C.STRATEGIES]
    handles.append(Line2D([], [], color=ZS_COLOR, ls="--", label="zero-shot"))
    fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.03), fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, f"fig_sweep_all_strategies{suffix}")


def strategy_heatmap():
    t = pd.read_csv(M / "test_all.csv")
    t = t[(t.frac == 1.0) & t.variant.str.startswith("ft_")]
    clfs = [c for c in CLASSIFIERS if c in set(t.classifier)]
    panels = [("roc_auc", "AUROC"), ("auprc", "AUPRC"), ("recall", "Recall"), ("f1", "F1 (harmful)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 2.4), sharey=True)
    for ax, (m, title) in zip(axes, panels):
        grid = np.array([[t[(t.classifier == c) & (t.strategy == s)][m].mean() for c in clfs] for s in C.STRATEGIES])
        ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for (i, j), v in np.ndenumerate(grid):
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5, color="white" if v > 0.6 else INK)
        ax.set_xticks(range(len(clfs))); ax.set_xticklabels([SHORT[c] for c in clfs], rotation=40, ha="right")
        ax.set_yticks(range(len(C.STRATEGIES))); ax.set_yticklabels([s.replace("_", " ") for s in C.STRATEGIES])
        ax.set_title(title, color=INK); ax.grid(False)
    save(fig, "fig_strategy_heatmap")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--criteria", nargs="+", default=["recall", "f1", "f1_macro"])
    a = ap.parse_args()
    for crit in a.criteria:
        compact_bars(crit)
        finetuned_four(crit)
        finetuned_four(crit, row=True)
        for metrics, suffix in [(AUC_METRICS, ""), (THR_METRICS, "_thresholded")]:
            sweeps(crit, metrics, suffix)
            overlay(crit, metrics, suffix)
            combined(crit, metrics, suffix)
    for metrics, suffix in [(AUC_METRICS, ""), (THR_METRICS, "_thresholded")]:
        all_strategies(metrics, suffix)
    strategy_heatmap()
    print("figures:", len(list(FIG.glob("*.pdf"))), "PDFs")
