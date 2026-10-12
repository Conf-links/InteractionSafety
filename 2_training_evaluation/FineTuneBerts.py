"""
Systematic Imbalance Benchmark & Truncation Sweeps for Injection Detection
==========================================================================
Models supported:
  - Raw base encoder:           microsoft/deberta-v3-base
  - Pre-finetuned detector:     protectai/deberta-v3-base-prompt-injection-v2
  - Dedicated Guardrail:        meta-llama/Llama-Prompt-Guard-2-86M

Pipelines:
  1. Baselines (Zero-Shot):
     - Prompt-Only (0% Response)
     - Prompt + Full Response (100% Response)
  2. Imbalance Training Strategies (TRAIN set only; VAL & TEST remain natural):
     - standard              : Standard Cross-Entropy Loss
     - weighted_loss         : Balanced Inverse-Class Frequency Weights
     - oversampling          : Random Minority Oversampling
     - undersampling         : Random Majority Undersampling
     - weighted_oversampling : Oversampling + Inverse Frequency Weights
     - focal_loss            : Alpha-balanced Focal Loss (gamma=2.0)
  3. Operating Threshold Calibration:
     - Default Threshold     : 0.50
     - Best F1-Positive      : Optimal threshold derived on Validation set
     - High-Recall Target    : Optimal threshold achieving >=85% Recall
  4. Response Truncation Sweep:
     - 0% -> 100% in 10% increments evaluated with calibrated thresholds.
"""

import argparse
import inspect
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import hf_hub_download
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline,
)
from transformers.pipelines.pt_utils import KeyDataset

ROOT = Path(__file__).resolve().parent
SEED = 42
TRUNCATION_STEPS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# ---------------------------------------------------------------------------
# Formatting & Splitting
# ---------------------------------------------------------------------------
def format_prompt_response(p: str, r: str, frac: float = 1.0) -> str:
    p_str = str(p) if pd.notna(p) else ""
    r_str = str(r) if pd.notna(r) else ""
    if frac <= 0.0:
        r_part = ""
    elif frac >= 1.0:
        r_part = r_str
    else:
        r_part = r_str[: int(len(r_str) * frac)]
    return f"Prompt: {p_str}\nResponse: {r_part}"


def to_hf_dataset(df: pd.DataFrame, tokenizer, max_length: int, frac: float = 1.0) -> Dataset:
    texts = [format_prompt_response(p, r, frac=frac) for p, r in zip(df["prompt"], df["response"])]
    enc = tokenizer(texts, truncation=True, max_length=max_length)
    enc["labels"] = df["label"].astype(int).tolist()
    return Dataset.from_dict(enc)


def split_train_val_by_prompt_id(train_df: pd.DataFrame, val_size: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_pids = np.array(sorted(train_df["prompt_id"].unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique_pids)

    n_val = int(len(unique_pids) * val_size)
    val_pids = set(unique_pids[:n_val])
    train_pids = set(unique_pids[n_val:])

    tr = train_df[train_df["prompt_id"].isin(train_pids)].reset_index(drop=True)
    va = train_df[train_df["prompt_id"].isin(val_pids)].reset_index(drop=True)
    return tr, va


# ---------------------------------------------------------------------------
# Resampling Techniques (Applied strictly to the Training set)
# ---------------------------------------------------------------------------
def apply_resampling(train_df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    df_pos = train_df[train_df["label"] == 1]
    df_neg = train_df[train_df["label"] == 0]
    n_pos, n_neg = len(df_pos), len(df_neg)

    if strategy in ("oversampling", "weighted_oversampling"):
        # Oversample minority class to 25% of the total dataset size (stabilizes gradients)
        target_pos = max(n_pos, int(n_neg * 0.25))
        df_pos_over = df_pos.sample(n=target_pos, replace=True, random_state=SEED)
        resampled = pd.concat([df_neg, df_pos_over], ignore_index=True)
        print(f"  [Oversampling] Minority class boosted from {n_pos} to {len(df_pos_over)} rows.")
        return resampled.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    elif strategy == "undersampling":
        # Undersample majority class down to 4x the minority count
        target_neg = max(n_pos, int(n_pos * 4.0))
        df_neg_under = df_neg.sample(n=target_neg, replace=False, random_state=SEED)
        resampled = pd.concat([df_neg_under, df_pos], ignore_index=True)
        print(f"  [Undersampling] Majority class reduced from {n_neg} to {len(df_neg_under)} rows.")
        return resampled.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    return train_df


# ---------------------------------------------------------------------------
# Custom Losses: Focal Loss & Weighted Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, (1.0 - self.alpha))
        focal_loss = alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class ImbalanceTrainer(Trainer):
    def __init__(self, loss_type: str = "standard", class_weights: Optional[torch.Tensor] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.class_weights = class_weights
        if self.loss_type == "focal_loss":
            self.focal_loss_fn = FocalLoss(alpha=0.75, gamma=2.0)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.loss_type in ("weighted_loss", "weighted_oversampling") and self.class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fn(logits.view(-1, 2), labels.view(-1))
        elif self.loss_type == "focal_loss":
            loss = self.focal_loss_fn(logits.view(-1, 2), labels.view(-1))
        else:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, 2), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Model Loader with DeBERTa-v3 LayerNorm Fix
# ---------------------------------------------------------------------------
def load_classification_model(model_name: str, num_labels: int = 2, token: Optional[str] = None):
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels, token=token)

    if "deberta-v3" in model_name.lower():
        model = AutoModelForSequenceClassification.from_config(config)
        try:
            ckpt = hf_hub_download(model_name, "model.safetensors", token=token)
            from safetensors.torch import load_file
            sd = load_file(ckpt)
        except Exception:
            ckpt = hf_hub_download(model_name, "pytorch_model.bin", token=token)
            sd = torch.load(ckpt, map_location="cpu", weights_only=True)

        renamed = {k.replace(".gamma", ".weight").replace(".beta", ".bias"): v for k, v in sd.items()}
        model.load_state_dict(renamed, strict=False)
        return model

    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, token=token, ignore_mismatched_sizes=True
    )


# ---------------------------------------------------------------------------
# Threshold Calibration & Multi-Threshold Evaluation
# ---------------------------------------------------------------------------
def compute_scores_at_threshold(probs: np.ndarray, labels: np.ndarray, thresh: float) -> Dict:
    preds = (probs >= thresh).astype(int)
    return {
        "threshold": float(thresh),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_pos": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "precision_pos": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "recall_pos": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
    }


def find_optimal_thresholds(val_probs: np.ndarray, val_labels: np.ndarray) -> Dict[str, float]:
    """Finds optimal operating thresholds on validation probabilities."""
    candidate_thresholds = np.linspace(0.05, 0.95, 91)
    best_f1, best_f1_th = -1.0, 0.50
    high_rec_th, target_rec = 0.50, 0.85
    min_prec_at_rec = -1.0

    for th in candidate_thresholds:
        preds = (val_probs >= th).astype(int)
        f1 = f1_score(val_labels, preds, pos_label=1, zero_division=0)
        rec = recall_score(val_labels, preds, pos_label=1, zero_division=0)
        prec = precision_score(val_labels, preds, pos_label=1, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_f1_th = float(th)

        if rec >= target_rec and prec > min_prec_at_rec:
            min_prec_at_rec = prec
            high_rec_th = float(th)

    return {"th_default": 0.50, "th_best_f1": best_f1_th, "th_high_recall": high_rec_th}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_pos": f1_score(labels, preds, pos_label=1, zero_division=0),
        "precision_pos": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall_pos": recall_score(labels, preds, pos_label=1, zero_division=0),
    }

# ---------------------------------------------------------------------------
# STAGE A: Zero-Shot Baseline Evaluator
# ---------------------------------------------------------------------------
def run_zeroshot_baseline(test_df: pd.DataFrame, name: str, model_name: str, batch_size: int, output_root: Path, token: Optional[str] = None) -> List[Dict]:
    print(f"\n{'#' * 75}\n# BASELINE (Zero-Shot): {name} [{model_name}]\n{'#' * 75}")
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("text-classification", model=model_name, device=device, truncation=True, batch_size=batch_size, token=token)
    labels = test_df["label"].astype(int).to_numpy()
    results = []

    for frac, mode in [(0.0, "prompt_only_0%"), (1.0, "full_response_100%")]:
        exp_name = f"baseline_{name}_{mode}"
        texts = [format_prompt_response(p, r, frac=frac) for p, r in zip(test_df["prompt"], test_df["response"])]
        ds = Dataset.from_dict({"text": texts})

        raw_preds = []
        for out in clf(KeyDataset(ds, "text")):
            lbl = str(out["label"]).upper()
            raw_preds.append(1 if lbl in ["INJECTION", "LABEL_1", "LABEL_2", "JAILBREAK", "UNSAFE"] else 0)

        raw_preds = np.array(raw_preds)
        metrics = {
            "experiment": exp_name,
            "strategy": "zero_shot",
            "threshold_used": 0.50,
            "accuracy": float(accuracy_score(labels, raw_preds)),
            "f1_macro": float(f1_score(labels, raw_preds, average="macro", zero_division=0)),
            "f1_pos": float(f1_score(labels, raw_preds, pos_label=1, zero_division=0)),
            "precision_pos": float(precision_score(labels, raw_preds, pos_label=1, zero_division=0)),
            "recall_pos": float(recall_score(labels, raw_preds, pos_label=1, zero_division=0)),
            "confusion_matrix": confusion_matrix(labels, raw_preds, labels=[0, 1]).tolist(),
        }
        results.append(metrics)
        print(f"[{mode:18s}] Acc: {metrics['accuracy']:.4f} | F1-Macro: {metrics['f1_macro']:.4f} | F1-Pos: {metrics['f1_pos']:.4f} | Rec: {metrics['recall_pos']:.4f} | Prec: {metrics['precision_pos']:.4f}")

    return results


# ---------------------------------------------------------------------------
# STAGE B: Imbalance Fine-Tuning Pipeline
# ---------------------------------------------------------------------------
def train_and_eval_imbalance_experiment(
    strategy: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    exp_prefix: str,
    model_name: str,
    cfg,
    output_root: Path,
) -> Tuple[Dict, str]:
    exp_name = f"{exp_prefix}_{strategy}"
    out_dir = output_root / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'-' * 70}\n# EXPERIMENT: {exp_name} | Base: {model_name}\n{'-' * 70}")

    # 1. Resample TRAIN only
    tr_df_resampled = apply_resampling(train_df, strategy)

    # 2. Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=cfg.hf_token)
    model = load_classification_model(model_name, num_labels=2, token=cfg.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    train_ds = to_hf_dataset(tr_df_resampled, tokenizer, cfg.max_length)
    val_ds = to_hf_dataset(val_df, tokenizer, cfg.max_length)
    test_ds = to_hf_dataset(test_df, tokenizer, cfg.max_length)

    # 3. Class Weights for Loss
    class_weights = None
    if strategy in ("weighted_loss", "weighted_oversampling"):
        labels_tr = tr_df_resampled["label"].to_numpy()
        n_tot, n0, n1 = len(labels_tr), np.sum(labels_tr == 0), np.sum(labels_tr == 1)
        w0 = n_tot / (2.0 * max(1, n0))
        w1 = (n_tot / (2.0 * max(1, n1))) * cfg.pos_weight_mult
        class_weights = torch.tensor([w0, w1], dtype=torch.float)

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        learning_rate=cfg.lr,
        weight_decay=0.01,
        warmup_steps=int(0.1 * (len(train_ds) // cfg.batch_size) * cfg.epochs),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=SEED,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    )

    trainer = ImbalanceTrainer(
        loss_type=strategy,
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # 4. Calibration on Validation Set
    val_out = trainer.predict(val_ds)
    val_probs = F.softmax(torch.tensor(val_out.predictions), dim=-1)[:, 1].numpy()
    val_labels = val_df["label"].astype(int).to_numpy()
    thresholds = find_optimal_thresholds(val_probs, val_labels)

    # 5. Untouched Test Set Evaluation Across Calibrated Thresholds
    test_out = trainer.predict(test_ds)
    test_probs = F.softmax(torch.tensor(test_out.predictions), dim=-1)[:, 1].numpy()
    test_labels = test_df["label"].astype(int).to_numpy()

    # Save predictions
    pd.DataFrame({
        "pair_id": test_df["pair_id"] if "pair_id" in test_df.columns else np.arange(len(test_df)),
        "true_label": test_labels,
        "prob_positive": test_probs,
        "pred_default_th": (test_probs >= 0.50).astype(int),
        "pred_best_f1_th": (test_probs >= thresholds["th_best_f1"]).astype(int),
    }).to_csv(out_dir / "test_predictions.csv", index=False)

    eval_results = {
        "experiment": exp_name,
        "strategy": strategy,
        "thresholds_calibrated": thresholds,
        "eval_at_th_0.50": compute_scores_at_threshold(test_probs, test_labels, 0.50),
        "eval_at_th_best_f1": compute_scores_at_threshold(test_probs, test_labels, thresholds["th_best_f1"]),
        "eval_at_th_high_rec": compute_scores_at_threshold(test_probs, test_labels, thresholds["th_high_recall"]),
        "roc_auc": float(roc_auc_score(test_labels, test_probs)),
    }

    print(f"\n[Test Scores: {exp_name}]")
    for th_key, tag in [("eval_at_th_0.50", "Th=0.50"), ("eval_at_th_best_f1", f"BestF1 (Th={thresholds['th_best_f1']:.2f})"), ("eval_at_th_high_rec", f"HighRec (Th={thresholds['th_high_recall']:.2f})")]:
        e = eval_results[th_key]
        print(f"  [{tag:18s}] Acc: {e['accuracy']:.4f} | F1-Macro: {e['f1_macro']:.4f} | F1-Pos: {e['f1_pos']:.4f} | Prec: {e['precision_pos']:.4f} | Rec: {e['recall_pos']:.4f}")

    return eval_results, str(out_dir)


# ---------------------------------------------------------------------------
# STAGE C: Truncation Sweeps (Inference-Only with Optimal Threshold)
# ---------------------------------------------------------------------------
def run_truncation_sweep_imbalance(
    model_dir: str,
    test_df: pd.DataFrame,
    exp_name: str,
    optimal_thresh: float,
    batch_size: int,
    max_length: int,
) -> List[Dict]:
    print(f"\n{'=' * 75}\n# SWEEP: {exp_name} (Using Operating Threshold: {optimal_thresh:.2f})\n{'=' * 75}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    sweep_records = []
    labels = test_df["label"].astype(int).to_numpy()
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    for frac in TRUNCATION_STEPS:
        pct = int(frac * 100)
        ds = to_hf_dataset(test_df, tokenizer, max_length, frac=frac)
        loader = DataLoader(ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collator)

        all_probs = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_probs.extend(probs)

        all_probs = np.array(all_probs)
        preds = (all_probs >= optimal_thresh).astype(int)

        sweep_records.append({
            "experiment": exp_name,
            "response_fraction": frac,
            "response_percentage": f"{pct}%",
            "threshold_used": optimal_thresh,
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "f1_pos": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
            "precision_pos": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
            "recall_pos": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        })
        print(f"[{pct:3d}% Response] Acc: {sweep_records[-1]['accuracy']:.4f} | F1-Macro: {sweep_records[-1]['f1_macro']:.4f} | F1-Pos: {sweep_records[-1]['f1_pos']:.4f} | Rec: {sweep_records[-1]['recall_pos']:.4f} | Prec: {sweep_records[-1]['precision_pos']:.4f}")

    return sweep_records


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Systematic Imbalance Benchmark & Sweeps")
    parser.add_argument("--train-file", default=str(ROOT / "train_set.csv"))
    parser.add_argument("--test-file", default=str(ROOT / "test_set.csv"))
    parser.add_argument("--model-type", choices=["deberta_base", "deberta_pretrained", "promptguard"], default="deberta_base",
                        help="Choose target architecture to benchmark")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--pos-weight-mult", type=float, default=1.2)
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN", None))
    parser.add_argument("--output-dir", default=str(ROOT / "imbalance_benchmark"))
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["standard", "weighted_loss", "oversampling", "undersampling", "weighted_oversampling", "focal_loss"],
        help="Imbalance strategies to run",
    )
    parser.add_argument("--skip-sweeps", action="store_true", help="Skip truncation sweeps to save time")
    cfg = parser.parse_args()

    # Model configuration router
    MODEL_MAP = {
        "deberta_base": {
            "model_name": "microsoft/deberta-v3-base",
            "zeroshot_model": "protectai/deberta-v3-base-prompt-injection-v2",
        },
        "deberta_pretrained": {
            "model_name": "protectai/deberta-v3-base-prompt-injection-v2",
            "zeroshot_model": "protectai/deberta-v3-base-prompt-injection-v2",
        },
        "promptguard": {
            "model_name": "meta-llama/Llama-Prompt-Guard-2-86M",
            "zeroshot_model": "meta-llama/Llama-Prompt-Guard-2-86M",
        },
    }

    selected = MODEL_MAP[cfg.model_type]
    fine_tune_model = selected["model_name"]
    zeroshot_model = selected["zeroshot_model"]

    output_root = Path(cfg.output_dir) / cfg.model_type
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Executing Imbalance Benchmark for [{cfg.model_type}]")
    print(f"  Fine-Tuning Architecture: {fine_tune_model}")
    print(f"  Zero-Shot Model Check:    {zeroshot_model}")

    train_df = pd.read_csv(cfg.train_file)
    test_df = pd.read_csv(cfg.test_file)
    train_df = train_df[train_df["label"].isin([0, 1])].copy()
    test_df = test_df[test_df["label"].isin([0, 1])].copy()

    response_models = sorted(train_df["model"].unique().tolist())
    all_metrics = []
    all_sweeps = []

    # ------------------- 1. ZERO-SHOT BASELINES -------------------
    # Agnostic (Full Test Set)
    all_metrics.extend(run_zeroshot_baseline(test_df, "model_agnostic", zeroshot_model, cfg.batch_size * 2, output_root, cfg.hf_token))
    # Specific (Per-Model Slices)
    for m in response_models:
        test_m = test_df[test_df["model"] == m].reset_index(drop=True)
        if not test_m.empty:
            all_metrics.extend(run_zeroshot_baseline(test_m, f"model_specific_{m}", zeroshot_model, cfg.batch_size * 2, output_root, cfg.hf_token))

    # ------------------- 2. MODEL-AGNOSTIC EXPERIMENTS ------------
    tr_df, va_df = split_train_val_by_prompt_id(train_df)
    for strat in cfg.strategies:
        res, model_ckpt = train_and_eval_imbalance_experiment(
            strategy=strat,
            train_df=tr_df, val_df=va_df, test_df=test_df,
            exp_prefix="agnostic",
            model_name=fine_tune_model,
            cfg=cfg, output_root=output_root
        )
        all_metrics.append(res)

        if not cfg.skip_sweeps:
            opt_th = res["thresholds_calibrated"]["th_best_f1"]
            all_sweeps.extend(run_truncation_sweep_imbalance(
                model_dir=model_ckpt,
                test_df=test_df,
                exp_name=f"agnostic_{strat}",
                optimal_thresh=opt_th,
                batch_size=cfg.batch_size,
                max_length=cfg.max_length
            ))

    # ------------------- 3. MODEL-SPECIFIC EXPERIMENTS ------------
    for m in []:#response_models:
        tr_m = train_df[train_df["model"] == m].reset_index(drop=True)
        te_m = test_df[test_df["model"] == m].reset_index(drop=True)
        tr_m, va_m = split_train_val_by_prompt_id(tr_m)

        for strat in cfg.strategies:
            res, model_ckpt = train_and_eval_imbalance_experiment(
                strategy=strat,
                train_df=tr_m, val_df=va_m, test_df=te_m,
                exp_prefix=f"specific_{m}",
                model_name=fine_tune_model,
                cfg=cfg, output_root=output_root
            )
            all_metrics.append(res)

            if not cfg.skip_sweeps:
                opt_th = res["thresholds_calibrated"]["th_best_f1"]
                all_sweeps.extend(run_truncation_sweep_imbalance(
                    model_dir=model_ckpt,
                    test_df=te_m,
                    exp_name=f"specific_{m}_{strat}",
                    optimal_thresh=opt_th,
                    batch_size=cfg.batch_size,
                    max_length=cfg.max_length
                ))

    # ------------------- 4. EXPORT SYSTEMATIC SUMMARY -------------
    with open(output_root / "all_imbalance_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Flatten for presentation-ready CSV
    flat_rows = []
    for m in all_metrics:
        if m.get("strategy") == "zero_shot":
            flat_rows.append({
                "experiment": m["experiment"],
                "strategy": "zero_shot",
                "operating_threshold": m["threshold_used"],
                "accuracy": m["accuracy"],
                "f1_macro": m["f1_macro"],
                "f1_pos": m["f1_pos"],
                "precision_pos": m["precision_pos"],
                "recall_pos": m["recall_pos"],
                "roc_auc": np.nan,
            })
        else:
            for th_mode in ["eval_at_th_0.50", "eval_at_th_best_f1", "eval_at_th_high_rec"]:
                sub = m[th_mode]
                flat_rows.append({
                    "experiment": f"{m['experiment']} [{th_mode.replace('eval_at_', '')}]",
                    "strategy": m["strategy"],
                    "operating_threshold": sub["threshold"],
                    "accuracy": sub["accuracy"],
                    "f1_macro": sub["f1_macro"],
                    "f1_pos": sub["f1_pos"],
                    "precision_pos": sub["precision_pos"],
                    "recall_pos": sub["recall_pos"],
                    "roc_auc": m["roc_auc"],
                })

    summary_df = pd.DataFrame(flat_rows)
    summary_df.to_csv(output_root / "all_imbalance_metrics.csv", index=False)

    if all_sweeps:
        pd.DataFrame(all_sweeps).to_csv(output_root / "imbalance_truncation_sweeps.csv", index=False)

    print("\n" + "=" * 90)
    print("SYSTEMATIC IMBALANCE BENCHMARK FINISHED")
    print(f"Results successfully archived in: {output_root.resolve()}")
    print("=" * 90)


if __name__ == "__main__":
    main()