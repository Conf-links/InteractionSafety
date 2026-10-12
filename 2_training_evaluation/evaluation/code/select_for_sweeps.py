"""Print the fine-tuned variants that need a full truncation sweep for one classifier.

Uses validation predictions only. Returns the union of the strategies chosen by the requested criteria (default: recall-priority only;
pass --criteria recall f1 f1_macro to sweep every selectable "best strategy"). Used for the 8B guards,
where sweeping all six adapters would cost ~6x more GPU time; encoders and LAYA are swept for every strategy.
"""
import argparse

import pandas as pd

import common as C


def val_table(classifier: str) -> pd.DataFrame:
    rows = []
    for s in C.STRATEGIES:
        d = C.load_preds(classifier, f"ft_{s}")
        v = d[(d["split"] == "val") & (d["frac"] == 1.0)]
        if len(v):
            th = C.best_f1_threshold(v["prob"].to_numpy(), v["label"].to_numpy())
            rows.append({"strategy": s, **C.metrics_at(v["prob"].to_numpy(), v["label"].to_numpy(), th)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--recall-tolerance", type=float, default=0.02)
    ap.add_argument("--criteria", nargs="+", default=["recall"], choices=["recall", "f1", "f1_macro"])
    a = ap.parse_args()
    t = val_table(a.classifier)
    picks = []
    for crit in a.criteria:
        s = C.select_strategy(t, crit, a.recall_tolerance)
        if s not in picks:
            picks.append(s)
    print(" ".join(f"ft_{s}" for s in picks))
