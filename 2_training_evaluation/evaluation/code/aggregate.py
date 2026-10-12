"""Turn stored probabilities into every reported number (no GPU needed; re-run any time).

Protocol (identical for all classifiers and variants):
  1. threshold = argmax harmful-class F1 on the prompt-grouped VALIDATION split, computed separately for every
     classifier/variant in its own input mode (prompt only: zs_native_prompt; prompt + full response: everything else);
  2. that threshold is applied unchanged to the test split at every truncation level;
  3. the "best strategy" is chosen on VALIDATION (default: recall-priority), never on test;
  4. 95% CIs come from a 1000x cluster bootstrap over test prompts.

Outputs (evaluation/metrics/ and evaluation/compact/):
  thresholds_and_val.csv       val metrics + threshold for every classifier/variant
  test_all.csv                 test metrics for every classifier/variant/truncation level
  main_results.csv             test @100% for every variant, with bootstrap CIs
  selection.json               selected strategy per classifier for each criterion
  harmful_prompt_subset.csv    test @100% restricted to harmful prompts (isolates response understanding)
  per_generator.csv            selected models broken down by response-generating LLM
  legacy_vs_new.csv            (only if a legacy_results/ folder with the training scripts' own CSVs exists)
  thresholds_ft_prompt_only.csv validation@0% threshold of every fine-tuned variant (prompt-only fine-tuned setting)
  compact/compact_<criterion>.csv  prompt-only zero-shot / prompt+response zero-shot / prompt-only best fine-tuned /
                                   prompt+response best fine-tuned, each with its own validation threshold; AUROC and AUPRC are the primary, threshold-free metrics
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import common as C

CLASSIFIERS = {  # key -> (display name, zero-shot model shown for it, fine-tuned base)
    "deberta": ("DeBERTa-v3", "ProtectAI DeBERTa-v3 (PI-v2)", "microsoft/deberta-v3-base"),
    "promptguard": ("Prompt-Guard-2-86M", "Llama-Prompt-Guard-2-86M", "Llama-Prompt-Guard-2-86M"),
    "llamaguard3": ("Llama-Guard-3-8B", "Llama-Guard-3-8B", "Llama-Guard-3-8B + QLoRA"),
    "wildguard": ("WildGuard-7B", "WildGuard", "WildGuard + QLoRA"),
    "laya": ("LAYA", "LAYA (ModernBERT-large)", "LAYA full fine-tune"),
}
CRITERIA = ("recall", "f1", "f1_macro")
M = C.NEW_RESULTS / "metrics"
COMPACT = C.NEW_RESULTS / "compact"


def variants_of(clf):
    d = C.PRED_DIR / clf
    return sorted(p.name[: -len(".csv.gz")] for p in d.glob("*.csv.gz")) if d.exists() else []


def strategy_of(variant):
    return variant[3:] if variant.startswith("ft_") else variant


def paired_delta_ci(pa, pb, ta, tb, y, groups, metric="f1", n_boot=1000, seed=C.SEED):
    """CI of metric(A) - metric(B) on the same test rows, cluster-bootstrapped over prompts."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    def m(p, t, yy):
        pred = p >= t; tp = (pred & (yy == 1)).sum(); fp = (pred & (yy == 0)).sum(); fn = (~pred & (yy == 1)).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
        return {"recall": rec, "precision": prec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}[metric]
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        out.append(m(pa[idx], ta, y[idx]) - m(pb[idx], tb, y[idx]))
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recall-tolerance", type=float, default=0.02)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    M.mkdir(exist_ok=True); COMPACT.mkdir(exist_ok=True)

    val_rows, test_rows, store = [], [], {}
    for clf in CLASSIFIERS:
        for var in variants_of(clf):
            d = C.load_preds(clf, var)
            v = d[(d.split == "val") & (d.frac == 1.0)].sort_values("pair_id")
            if not len(v):
                continue
            th = C.best_f1_threshold(v.prob.to_numpy(), v.label.to_numpy())
            val_rows.append({"classifier": clf, "variant": var, "strategy": strategy_of(var),
                             **C.metrics_at(v.prob.to_numpy(), v.label.to_numpy(), th)})
            for f, t in d[d.split == "test"].groupby("frac"):
                t = t.sort_values("pair_id")
                store[(clf, var, round(f, 1))] = t
                test_rows.append({"classifier": clf, "variant": var, "strategy": strategy_of(var), "frac": round(f, 1),
                                  **C.metrics_at(t.prob.to_numpy(), t.label.to_numpy(), th)})
    val_df, test_df = pd.DataFrame(val_rows), pd.DataFrame(test_rows)
    val_df.to_csv(M / "thresholds_and_val.csv", index=False)
    test_df.sort_values(["classifier", "variant", "frac"]).to_csv(M / "test_all.csv", index=False)
    th_of = {(r.classifier, r.variant): r.threshold for r in val_df.itertuples()}

    # ---- strategy selection on validation
    selection = {}
    for crit in CRITERIA:
        selection[crit] = {}
        for clf in CLASSIFIERS:
            vt = val_df[(val_df.classifier == clf) & val_df.variant.str.startswith("ft_")]
            if len(vt):
                selection[crit][clf] = C.select_strategy(vt, crit, args.recall_tolerance)
    C.save_json({"rule": {"recall": f"max val recall (tolerance {args.recall_tolerance}), then max val F1",
                          "f1": "max val F1 (harmful class)", "f1_macro": "max val macro-F1"},
                 "selected": selection}, M / "selection.json")

    # ---- main results with CIs (test @100%, plus zero-shot @0%)
    # Prompt-only fine-tuned setting: the fine-tuned model on the prompt with an empty response (sweep 0%), with its
    # OWN threshold calibrated on validation at 0% (val@0% scores come from infer_ft_prompt_only.py).
    th0_of = {}
    for clf in CLASSIFIERS:
        for var in variants_of(clf):
            d = C.load_preds(clf, var)
            v = d[(d.split == "val") & (d.frac == 0.0)].sort_values("pair_id")
            if var.startswith("ft_") and len(v):
                th0_of[(clf, var)] = C.best_f1_threshold(v.prob.to_numpy(), v.label.to_numpy())
    pd.DataFrame([{"classifier": c, "variant": v, "threshold_prompt_only": t} for (c, v), t in th0_of.items()]).to_csv(
        M / "thresholds_ft_prompt_only.csv", index=False)

    def row(clf, var, frac, setting, th=None):
        t = store.get((clf, var, frac))
        if t is None:
            return None
        th = th_of[(clf, var)] if th is None else th
        p, y, g = t.prob.to_numpy(), t.label.to_numpy(), t.prompt_id.to_numpy()
        base = C.metrics_at(p, y, th)
        ci = C.bootstrap_ci(p, y, g, th, n_boot=args.n_boot)
        out = {"classifier": clf, "classifier_name": CLASSIFIERS[clf][0], "variant": var, "strategy": strategy_of(var),
               "setting": setting, "frac": frac, **base}
        for k, (lo, hi) in ci.items():
            out[f"{k}_lo"], out[f"{k}_hi"] = lo, hi
        return out

    main_rows = []
    for clf in CLASSIFIERS:
        for var in variants_of(clf):
            if (clf, var, 1.0) in store:
                main_rows.append(row(clf, var, 1.0, "prompt+response"))
            if (clf, var) in th0_of:
                main_rows.append(row(clf, var, 0.0, "prompt only (own threshold)", th0_of[(clf, var)]))
        if (clf, "zero_shot", 0.0) in store:
            main_rows.append(row(clf, "zero_shot", 0.0, "empty response (sweep 0%)"))
    main_df = pd.DataFrame([r for r in main_rows if r])
    main_df.to_csv(M / "main_results.csv", index=False)

    # ---- compact tables (one per criterion) with paired deltas vs zero-shot prompt+response
    for crit in CRITERIA:
        rows = []
        for clf in CLASSIFIERS:
            s = selection[crit].get(clf)
            # Each baseline row has its OWN threshold, calibrated on validation in that same input mode:
            # prompt only = zs_native_prompt (prompt alone), prompt + response = zero_shot at 100%.
            for setting, var, frac in [("Zero-shot, prompt only", "zs_native_prompt", 1.0),
                                       ("Zero-shot, prompt + response", "zero_shot", 1.0),
                                       ("Fine-tuned, prompt only", f"ft_{s}", 0.0),
                                       (f"Fine-tuned ({s})", f"ft_{s}", 1.0)]:
                r = main_df[(main_df.classifier == clf) & (main_df.variant == var) & (main_df.frac == frac)]
                if var.startswith("ft_") and frac == 0.0:
                    r = r[r.setting == "prompt only (own threshold)"]
                if len(r):
                    r = r.iloc[0].to_dict(); r["setting"] = setting; rows.append(r)
            if s and (clf, f"ft_{s}", 1.0) in store and (clf, "zero_shot", 1.0) in store:
                a, b = store[(clf, f"ft_{s}", 1.0)], store[(clf, "zero_shot", 1.0)]
                for met in ("f1", "recall"):
                    lo, hi = paired_delta_ci(a.prob.to_numpy(), b.prob.to_numpy(), th_of[(clf, f"ft_{s}")],
                                             th_of[(clf, "zero_shot")], a.label.to_numpy(), a.prompt_id.to_numpy(), met, args.n_boot)
                    rows[-1][f"delta_{met}_vs_zs_lo"], rows[-1][f"delta_{met}_vs_zs_hi"] = lo, hi
        pd.DataFrame(rows).to_csv(COMPACT / f"compact_{crit}.csv", index=False)

    # ---- harmful-prompt subset and per-generator breakdown (recall-selected model + zero-shot)
    sub_rows, gen_rows = [], []
    for clf in CLASSIFIERS:
        s = selection["recall"].get(clf)
        for var in ["zero_shot"] + ([f"ft_{s}"] if s else []):
            t = store.get((clf, var, 1.0))
            if t is None:
                continue
            th = th_of[(clf, var)]
            h = t[t.prompt_label == 1]
            sub_rows.append({"classifier": clf, "variant": var, **C.metrics_at(h.prob.to_numpy(), h.label.to_numpy(), th)})
            for gname, gt in t.groupby("model"):
                gen_rows.append({"classifier": clf, "variant": var, "generator": gname,
                                 **C.metrics_at(gt.prob.to_numpy(), gt.label.to_numpy(), th)})
    pd.DataFrame(sub_rows).to_csv(M / "harmful_prompt_subset.csv", index=False)
    pd.DataFrame(gen_rows).to_csv(M / "per_generator.csv", index=False)

    # ---- legacy vs new (best-F1 rows of the original runs)
    legacy = C.NEW_RESULTS / "legacy_results"  # optional: CSV/JSON summaries written by the training scripts themselves
    if not legacy.exists():
        print("aggregate: wrote", sorted(p.name for p in M.glob("*")), "and", sorted(p.name for p in COMPACT.glob("*.csv")))
        return
    files = {"deberta": "deberta_metrics.csv", "promptguard": "promptguard_metrics.csv",
             "llamaguard3": "llamaguard_metrics.csv", "wildguard": "wildguard_metrics.csv"}
    lv = []
    for clf, fn in files.items():
        old = pd.read_csv(legacy / fn)
        if "operating_threshold" not in old.columns:  # resume_wildguard.py wrote "threshold"
            old = old.rename(columns={"threshold": "operating_threshold"})
        old = old[old.experiment.str.startswith("agnostic_") & old.experiment.str.contains("best_f1|BestF1", regex=True)]
        for r in old.itertuples():
            s = next(x for x in sorted(C.STRATEGIES, key=len, reverse=True) if x in r.experiment)
            new = test_df[(test_df.classifier == clf) & (test_df.variant == f"ft_{s}") & (test_df.frac == 1.0)]
            if len(new):
                n = new.iloc[0]
                lv.append({"classifier": clf, "strategy": s, "old_threshold": r.operating_threshold, "new_threshold": n.threshold,
                           "old_recall": r.recall_pos, "new_recall": n.recall, "old_precision": r.precision_pos,
                           "new_precision": n.precision, "old_f1": r.f1_pos, "new_f1": n.f1, "old_roc_auc": r.roc_auc,
                           "new_roc_auc": n.roc_auc, "new_auprc": n.auprc})
    laya_json = legacy / "LAYA" / "summary_metrics.json"
    if laya_json.exists():
        import json
        for item in json.load(open(laya_json))["base_and_finetuned_metrics"][1:]:
            s, o = item["strategy"], item["test"]["best_f1"]
            new = test_df[(test_df.classifier == "laya") & (test_df.variant == f"ft_{s}") & (test_df.frac == 1.0)]
            if len(new):
                n = new.iloc[0]
                lv.append({"classifier": "laya", "strategy": s, "old_threshold": o["threshold"], "new_threshold": n.threshold,
                           "old_recall": o["recall_harmful"], "new_recall": n.recall, "old_precision": o["precision_harmful"],
                           "new_precision": n.precision, "old_f1": o["f1_harmful"], "new_f1": n.f1, "old_roc_auc": o["roc_auc"],
                           "new_roc_auc": n.roc_auc, "new_auprc": n.auprc})
    pd.DataFrame(lv).to_csv(M / "legacy_vs_new.csv", index=False)
    print("aggregate: wrote", sorted(p.name for p in M.glob("*")), "and", sorted(p.name for p in COMPACT.glob("*.csv")))


if __name__ == "__main__":
    main()
