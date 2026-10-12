"""Write report/REPORT.md and compact/README.md from metrics/ (re-run any time; it lists what is still missing)."""
import json
from datetime import datetime, timezone

import pandas as pd

import common as C
from aggregate import CLASSIFIERS

M, R, COMPACT = C.NEW_RESULTS / "metrics", C.NEW_RESULTS / "report", C.NEW_RESULTS / "compact"
EXPECTED_FT_SWEEP = {"deberta": "all", "promptguard": "all", "laya": "all", "llamaguard3": "selected", "wildguard": "selected"}


def fmt(x, d=3):
    return "–" if pd.isna(x) else f"{x:.{d}f}"


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(fmt(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def missing():
    t = pd.read_csv(M / "test_all.csv")
    v = pd.read_csv(M / "thresholds_and_val.csv")
    sel = json.load(open(M / "selection.json"))["selected"]["recall"]
    todo = []
    for c in CLASSIFIERS:
        for var in ["zero_shot", "zs_native_prompt"] + [f"ft_{s}" for s in C.STRATEGIES]:
            has_val = len(v[(v.classifier == c) & (v.variant == var)]) > 0
            fr = set(t[(t.classifier == c) & (t.variant == var)].frac.round(1))
            need = {1.0} if var == "zs_native_prompt" else set(C.TRUNCATION_STEPS)
            if var.startswith("ft_") and EXPECTED_FT_SWEEP[c] == "selected" and sel.get(c) != var[3:]:
                need = {1.0}
            if var.startswith("ft_") and EXPECTED_FT_SWEEP[c] == "selected" and c not in sel:
                need = {1.0}
            miss = sorted(need - fr)
            if not has_val or miss:
                todo.append(f"- `{c}/{var}`: " + ("validation, " if not has_val else "")
                            + (f"test at {', '.join(f'{int(x * 100)}%' for x in miss)}" if miss else "").rstrip(", "))
    return todo


def main():
    R.mkdir(exist_ok=True)
    sel = json.load(open(M / "selection.json"))
    t = pd.read_csv(M / "test_all.csv")
    todo = missing()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = [f"# Results report (generated {now})", ""]
    L += ["**Status:** " + ("complete." if not todo else f"{len(todo)} prediction blocks still running (listed at the end)."), ""]
    L += ["Protocol: AUROC and AUPRC are the primary, threshold-free metrics. Every row also has its own decision threshold "
          "(best harmful-class F1 on the prompt-grouped validation split, in that row's input mode): zero-shot prompt-only, "
          "zero-shot prompt + full response and each fine-tuned strategy are calibrated separately. In truncation sweeps the "
          "full-response threshold is held fixed (deployment view; secondary to AUROC/AUPRC). The strategy is selected on "
          "validation (recall-priority: within 0.02 of the best val recall, then best val F1). CIs: 95% prompt-clustered "
          "bootstrap (1000x). Details in `../CHANGES.md`.", ""]

    for crit, title in [("recall", "Compact results: recall-priority selection (main)"), ("f1", "Compact results: F1 selection"),
                        ("f1_macro", "Compact results: macro-F1 selection")]:
        p = COMPACT / f"compact_{crit}.csv"
        if not p.exists():
            continue
        c = pd.read_csv(p)
        c["recall [95% CI]"] = [f"{fmt(a)} [{fmt(b, 2)}, {fmt(d, 2)}]" for a, b, d in zip(c.recall, c.recall_lo, c.recall_hi)]
        c["F1 [95% CI]"] = [f"{fmt(a)} [{fmt(b, 2)}, {fmt(d, 2)}]" for a, b, d in zip(c.f1, c.f1_lo, c.f1_hi)]
        c["AUROC [95% CI]"] = [f"{fmt(a)} [{fmt(b, 2)}, {fmt(d, 2)}]" for a, b, d in zip(c.roc_auc, c.roc_auc_lo, c.roc_auc_hi)]
        c["AUPRC [95% CI]"] = [f"{fmt(a)} [{fmt(b, 2)}, {fmt(d, 2)}]" for a, b, d in zip(c.auprc, c.auprc_lo, c.auprc_hi)]
        c["classifier"] = c.classifier.map(lambda x: CLASSIFIERS[x][0])
        L += [f"## {title}", "", md(c[["classifier", "setting", "AUROC [95% CI]", "AUPRC [95% CI]", "threshold", "precision",
                                      "recall [95% CI]", "F1 [95% CI]", "f1_macro"]]), ""]
        if crit == "recall":
            L += ["![compact](../figures/fig_compact_recall.png)", ""]

    L += ["## Selected strategy per criterion (validation only)", ""]
    L += ["| classifier | recall-priority | F1 | macro-F1 |", "|---|---|---|---|"]
    for k in CLASSIFIERS:
        L.append(f"| {CLASSIFIERS[k][0]} | " + " | ".join(sel["selected"][c].get(k, "pending") for c in ("recall", "f1", "f1_macro")) + " |")
    L.append("")

    a = t[t.frac == 1.0].copy()
    a["classifier"] = a.classifier.map(lambda x: CLASSIFIERS[x][0])
    L += ["## Every variant at 100% response (test)", "",
          md(a[["classifier", "variant", "threshold", "precision", "recall", "f1", "f1_macro", "auprc", "roc_auc"]]), ""]

    L += ["## Truncation sweep: AUROC / AUPRC (test, zero-shot vs recall-selected fine-tuned)", ""]
    for stem in ("fig_sweep_recall", "fig_sweep_overlay_recall", "fig_sweep_combined_recall", "fig_sweep_all_strategies",
                 "fig_sweep_thresholded_recall", "fig_sweep_combined_thresholded_recall", "fig_sweep_all_strategies_thresholded"):
        L += [f"![{stem}](../figures/{stem}.png)", ""]
    rows = []
    for k, s in sel["selected"]["recall"].items():
        for var in ["zero_shot", f"ft_{s}"]:
            d = t[(t.classifier == k) & (t.variant == var)].set_index("frac")
            if len(d):
                rows.append([CLASSIFIERS[k][0], var] + [f"{fmt(d.roc_auc.get(x), 2)} / {fmt(d.auprc.get(x), 2)}" for x in C.TRUNCATION_STEPS])
    L += [md(pd.DataFrame(rows, columns=["classifier", "variant"] + [f"{int(x * 100)}%" for x in C.TRUNCATION_STEPS])), ""]

    for name, title in [("harmful_prompt_subset.csv", "Harmful-prompt subset (test @100%): response understanding only"),
                        ("legacy_vs_new.csv", "Legacy vs re-run (old best-F1 rows vs new protocol, test @100%)")]:
        p = M / name
        if p.exists():
            d = pd.read_csv(p)
            keep = [c for c in d.columns if c not in ("n", "n_pos", "tn", "fp", "fn", "tp", "accuracy")]
            L += [f"## {title}", "", md(d[keep]), ""]

    L += ["## Still to come" if todo else "## Completeness", ""]
    L += todo if todo else ["All planned prediction blocks are present."]
    (R / "REPORT.md").write_text("\n".join(L) + "\n")

    cl = [f"# Compact results (generated {now})", "",
          "Five classifiers x {zero-shot prompt-only, zero-shot prompt+response, best fine-tuned strategy}; every row has its",
          "own validation-calibrated threshold. AUROC/AUPRC are primary; `*_thresholded` figures show recall/precision/F1.",
          "`*_recall` = recall-priority selection (main), `*_f1` / `*_f1_macro` = alternatives.", "",
          "- `compact_<crit>.csv`: numbers with 95% CIs (and paired fine-tuned minus zero-shot deltas for F1/recall)",
          "- `tab_compact_<crit>.tex`, `tab_sweep_<crit>.tex`: LaTeX for Overleaf (booktabs, multirow, graphicx)",
          "- `figures/`: compact bar chart and truncation sweeps (best strategy only), PDF + PNG", ""]
    (COMPACT / "README.md").write_text("\n".join(cl) + "\n" + ("\n**Pending:**\n" + "\n".join(todo) + "\n" if todo else ""))
    figs = COMPACT / "figures"
    figs.mkdir(exist_ok=True)
    for src in (C.NEW_RESULTS / "figures").glob("fig_*"):
        if not src.name.startswith("fig_strategy_heatmap"):
            (figs / src.name).write_bytes(src.read_bytes())
    print("report written;", len(todo), "blocks pending")


if __name__ == "__main__":
    main()
