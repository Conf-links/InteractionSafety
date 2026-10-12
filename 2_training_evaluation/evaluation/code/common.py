"""Shared utilities for the evaluation pipeline (2_training_evaluation/evaluation/).

Everything that must be identical across classifiers lives here:
  * data loading and the prompt-grouped train/validation split (same as the original training scripts),
  * input formatting and character-level response truncation,
  * prediction storage (one CSV.gz per classifier/variant, holding val + test probabilities),
  * the single thresholding rule (best harmful-class F1 on validation at 100% response, exact PR-curve search),
  * metrics, prompt-clustered bootstrap confidence intervals, and best-strategy selection.
"""
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

SEED = 42
VAL_FRACTION = 0.15
TRUNCATION_STEPS = [round(i / 10, 1) for i in range(11)]
STRATEGIES = ["standard", "weighted_loss", "oversampling", "undersampling", "weighted_oversampling", "focal_loss"]

CODE_DIR = Path(__file__).resolve().parent
NEW_RESULTS = CODE_DIR.parent
DATA_DIR = NEW_RESULTS.parent  # Dataset_Training/
PRED_DIR = NEW_RESULTS / "predictions"


# ----------------------------------------------------------------------------- data
def load_splits(train_file: Optional[str] = None, test_file: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (train_part, val_part, test). Identical split to FineTune*.py: prompt_id grouped, seed 42, 15%."""
    train = pd.read_csv(train_file or DATA_DIR / "train_set.csv")
    test = pd.read_csv(test_file or DATA_DIR / "test_set.csv")
    train = train[train["label"].isin([0, 1])].copy()
    test = test[test["label"].isin([0, 1])].copy().reset_index(drop=True)
    ids = np.array(sorted(train["prompt_id"].unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(ids)
    val_ids = set(ids[: int(len(ids) * VAL_FRACTION)])
    is_val = train["prompt_id"].isin(val_ids)
    return train[~is_val].reset_index(drop=True), train[is_val].reset_index(drop=True), test


def truncate_response(r, frac: float) -> str:
    r = "" if pd.isna(r) else str(r)
    if frac <= 0.0:
        return ""
    if frac >= 1.0:
        return r
    return r[: int(len(r) * frac)]


def as_text(x) -> str:
    return "" if pd.isna(x) else str(x)


def format_prompt_response(p, r, frac: float = 1.0) -> str:
    """Input format used by every encoder classifier during fine-tuning."""
    return f"Prompt: {as_text(p)}\nResponse: {truncate_response(r, frac)}"


# ----------------------------------------------------------------------------- prediction storage
PRED_COLUMNS = ["split", "frac", "pair_id", "prompt_id", "model", "prompt_label", "label", "prob"]


def pred_path(classifier: str, variant: str) -> Path:
    return PRED_DIR / classifier / f"{variant}.csv.gz"


def load_preds(classifier: str, variant: str) -> pd.DataFrame:
    p = pred_path(classifier, variant)
    if not p.exists():
        return pd.DataFrame(columns=PRED_COLUMNS)
    return pd.read_csv(p)


def done_conditions(classifier: str, variant: str) -> set:
    d = load_preds(classifier, variant)
    return set(zip(d["split"], d["frac"].round(1))) if len(d) else set()


def append_preds(classifier: str, variant: str, split: str, frac: float, df: pd.DataFrame, probs: np.ndarray) -> None:
    """Adds one (split, frac) block of probabilities; replaces an existing block with the same key."""
    assert len(df) == len(probs)
    if np.isnan(probs).any():
        raise ValueError(f"{classifier}/{variant}/{split}/{frac}: {np.isnan(probs).sum()} NaN probabilities")
    block = pd.DataFrame({
        "split": split, "frac": round(float(frac), 1),
        "pair_id": df["pair_id"].to_numpy(), "prompt_id": df["prompt_id"].to_numpy(), "model": df["model"].to_numpy(),
        "prompt_label": df["prompt_label"].to_numpy(), "label": df["label"].astype(int).to_numpy(),
        "prob": np.asarray(probs, dtype=np.float64),
    })
    old = load_preds(classifier, variant)
    if len(old):
        old = old[~((old["split"] == split) & (old["frac"].round(1) == round(float(frac), 1)))]
        block = pd.concat([old, block], ignore_index=True)
    p = pred_path(classifier, variant)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.gz")
    block.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, p)


# ----------------------------------------------------------------------------- thresholding & metrics
def best_f1_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """Exact search over all distinct scores (PR curve). Ties resolve to the lowest threshold (higher recall)."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def metrics_at(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Dict:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    preds = (probs >= threshold).astype(int)
    two = len(np.unique(labels)) > 1
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "n": int(len(labels)), "n_pos": int(labels.sum()),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)) if two else float("nan"),
        "auprc": float(average_precision_score(labels, probs)) if two else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_ci(probs: np.ndarray, labels: np.ndarray, groups: np.ndarray, threshold: float,
                 metrics: Iterable[str] = ("precision", "recall", "f1", "f1_macro", "auprc", "roc_auc"),
                 n_boot: int = 1000, seed: int = SEED, alpha: float = 0.05) -> Dict[str, Tuple[float, float]]:
    """Percentile CIs from a cluster bootstrap that resamples prompts (all 7 responses of a prompt move together)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    probs, labels = np.asarray(probs, float), np.asarray(labels, int)
    vals = {m: [] for m in metrics}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        y, p = labels[idx], probs[idx]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        pred = (p >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        prec0 = tn / (tn + fn) if tn + fn else 0.0
        rec0 = tn / (tn + fp) if tn + fp else 0.0
        f10 = 2 * prec0 * rec0 / (prec0 + rec0) if prec0 + rec0 else 0.0
        row = {"precision": prec, "recall": rec, "f1": f1, "f1_macro": (f1 + f10) / 2}
        if "auprc" in vals:
            row["auprc"] = average_precision_score(y, p)
        if "roc_auc" in vals:
            row["roc_auc"] = roc_auc_score(y, p)
        for m in metrics:
            vals[m].append(row[m])
    return {m: (float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))) for m, v in vals.items()}


def select_strategy(val_rows: pd.DataFrame, criterion: str = "recall", recall_tolerance: float = 0.02) -> str:
    """Pick one fine-tuning strategy per classifier using VALIDATION metrics only (test never used for selection).

    criterion="recall"  : keep strategies whose val recall is within `recall_tolerance` of the best val recall,
                          then take the highest val F1 among them (recall first, then F1 / precision).
    criterion="f1"      : highest val F1 (harmful class); ties -> higher recall.
    criterion="f1_macro": highest val macro-F1; ties -> higher recall.
    """
    r = val_rows.copy()
    if criterion == "recall":
        cand = r[r["recall"] >= r["recall"].max() - recall_tolerance - 1e-12]
        cand = cand.sort_values(["f1", "recall", "precision"], ascending=False)
    elif criterion in ("f1", "f1_macro"):
        cand = r.sort_values([criterion, "recall"], ascending=False)
    else:
        raise ValueError(criterion)
    return str(cand.iloc[0]["strategy"])


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def length_sorted_batches(lengths: List[int], max_tokens: int, max_batch: int) -> List[List[int]]:
    """Index batches sorted by length under a padded-token budget (inference speed; order is restored by caller)."""
    order = np.argsort(lengths, kind="stable")[::-1]
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = lengths[i]
        new_max = max(cur_max, L)
        if cur and (new_max * (len(cur) + 1) > max_tokens or len(cur) >= max_batch):
            batches.append(cur)
            cur, new_max = [], L
        cur.append(int(i))
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches
