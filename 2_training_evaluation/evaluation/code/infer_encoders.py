"""Inference for the encoder classifiers (DeBERTa-v3 family and Llama-Prompt-Guard-2-86M).

No training happens here: fine-tuned checkpoints come from FineTuneBerts.py runs (imbalance_runs/).
Produces predictions/<classifier>/<variant>.csv.gz with
  * val  @ 100% response  (used only for threshold calibration + strategy selection)
  * test @ 0%,10%,...,100% response (truncation sweep; 100% = headline result)

Variants
  zero_shot         off-the-shelf model on "Prompt: ...\\nResponse: ..." (same input format as fine-tuning)
  zs_native_prompt  off-the-shelf model on the raw prompt only (its native prompt-moderation use; supplementary)
  ft_<strategy>     fine-tuned checkpoint for each imbalance strategy

Changes vs FineTuneBerts.py: zero-shot now stores probabilities (so AUC/AUPRC and calibrated thresholds exist),
all variants get a full 11-point sweep, validation probabilities are saved, softmax is computed in fp32.
Input format, tokenizer, 512-token right truncation and bf16 compute are unchanged from training.
"""
import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import common as C

CLASSIFIERS = {
    # The raw microsoft/deberta-v3-base has a random classification head, so its zero-shot reference is the
    # ProtectAI DeBERTa-v3 detector (same backbone), exactly as in the original FineTuneBerts.py baseline.
    "deberta": {"zero_shot": "protectai/deberta-v3-base-prompt-injection-v2", "ft_root": "imbalance_runs/deberta_base"},
    "promptguard": {"zero_shot": "meta-llama/Llama-Prompt-Guard-2-86M", "ft_root": "imbalance_runs/promptguard"},
}


@torch.no_grad()
def predict(model, tok, texts, device, max_length=512, max_tokens=65536, max_batch=256):
    enc = tok(texts, truncation=True, max_length=max_length)
    ids = enc["input_ids"]
    probs = np.zeros(len(ids))
    for b in C.length_sorted_batches([len(x) for x in ids], max_tokens, max_batch):
        batch = tok.pad({"input_ids": [ids[i] for i in b], "attention_mask": [enc["attention_mask"][i] for i in b]},
                        return_tensors="pt")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
        probs[b] = torch.softmax(logits.float(), -1)[:, 1].cpu().numpy()
    return probs


def run_variant(clf, variant, model_path, val, test, device, native_prompt=False, fracs=C.TRUNCATION_STEPS):
    done = C.done_conditions(clf, variant)
    todo = [("val", 1.0)] + [("test", f) for f in (fracs if not native_prompt else [1.0])]
    todo = [t for t in todo if (t[0], round(t[1], 1)) not in done]
    if not todo:
        print(f"[{clf}/{variant}] already complete", flush=True)
        return
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
    for split, frac in todo:
        df = val if split == "val" else test
        t0 = time.time()
        if native_prompt:
            texts = [C.as_text(p) for p in df["prompt"]]
        else:
            texts = [C.format_prompt_response(p, r, frac) for p, r in zip(df["prompt"], df["response"])]
        probs = predict(model, tok, texts, device)
        C.append_preds(clf, variant, split, frac, df, probs)
        print(f"[{clf}/{variant}] {split} frac={frac:.1f} n={len(df)} {time.time() - t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifiers", nargs="+", default=list(CLASSIFIERS))
    ap.add_argument("--strategies", nargs="+", default=C.STRATEGIES)
    ap.add_argument("--train-file"); ap.add_argument("--test-file")
    args = ap.parse_args()
    device = torch.device("cuda")
    _, val, test = C.load_splits(args.train_file, args.test_file)
    for clf in args.classifiers:
        spec = CLASSIFIERS[clf]
        run_variant(clf, "zero_shot", spec["zero_shot"], val, test, device)
        run_variant(clf, "zs_native_prompt", spec["zero_shot"], val, test, device, native_prompt=True)
        for s in args.strategies:
            run_variant(clf, f"ft_{s}", str(C.DATA_DIR / spec["ft_root"] / f"agnostic_{s}"), val, test, device)


if __name__ == "__main__":
    main()
