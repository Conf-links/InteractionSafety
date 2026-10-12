"""Validation scores for every FINE-TUNED model on the prompt alone (0% response), so the prompt-only fine-tuned
setting gets its own threshold, calibrated on validation in that input mode (same rule as every other row).

"Prompt only" for a fine-tuned model = the training input with an empty response (the 0% point of the sweep):
  encoders / LAYA : "Prompt: <p>\\nResponse: "
  Llama-Guard-3   : LG3_TRAIN with an empty <assistant></assistant> block
  WildGuard       : WG_TRAIN with an empty "AI assistant:" field
The test 0% scores already exist in ft_<strategy>.csv.gz (truncation sweep). This script adds the matching
("val", 0.0) block to the same file; aggregate.py only reads val at 100%, so nothing else changes.
Every pair is scored exactly as in the sweep (same input builder, same length-sorted bf16 batching), so val and test
go through an identical pipeline. (Scoring each prompt once instead is cheaper but changes the batch padding, which
moves a few borderline bf16 scores by up to ~0.2; we keep the pair-level pipeline for exact consistency.)
It also re-scores test at 0% and reports the largest difference from the stored sweep scores (sanity check).
"""
import argparse

import numpy as np
import pandas as pd
import torch

import common as C


def encoder_scorer(clf):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import infer_encoders as E
    root = C.DATA_DIR / E.CLASSIFIERS[clf]["ft_root"]
    def score(strategy, df):
        path = str(root / f"agnostic_{strategy}")
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).to("cuda").eval()
        texts = [C.format_prompt_response(p, r, 0.0) for p, r in zip(df["prompt"], df["response"])]
        out = E.predict(model, tok, texts, torch.device("cuda"))
        del model; torch.cuda.empty_cache()
        return out
    return score


def laya_scorer():
    from safetensors.torch import load_file
    import laya
    from train_laya import make_items, predict
    agent = laya.load(str(C.Path.home() / "models" / "convaiinnovations-laya"), device="cpu")
    model, tok = agent.model.to("cuda"), agent.tok
    ck = C.NEW_RESULTS / "checkpoints" / "laya"  
    def score(strategy, df):
        model.load_state_dict(load_file(str(ck / f"ft_{strategy}" / "model.safetensors")))
        return predict(model, make_items(tok, df, 0.0), tok.pad_token_id, torch.device("cuda"))
    return score


def guard_scorer(guard):
    from peft import PeftModel
    import infer_guards as G
    spec = G.GUARDS[guard]
    tok, base = G.load_base(spec["hf"])
    neg_id = tok.encode(spec["words"][0], add_special_tokens=False)[-1]
    pos_id = tok.encode(spec["words"][1], add_special_tokens=False)[-1]
    state = {"peft": None}
    def score(strategy, df):
        name, adapter = f"ft_{strategy}", str(C.DATA_DIR / spec["ft_root"] / f"agnostic_{strategy}")
        if state["peft"] is None:
            state["peft"] = PeftModel.from_pretrained(base, adapter, adapter_name=name)
        elif name not in state["peft"].peft_config:
            state["peft"].load_adapter(adapter, adapter_name=name)
        state["peft"].set_adapter(name); state["peft"].eval()
        return G.last_token_scores(state["peft"], tok, G.ft_sequences(tok, guard, df, 0.0), neg_id, pos_id)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifiers", nargs="+", default=["deberta", "promptguard", "laya", "llamaguard3", "wildguard"])
    ap.add_argument("--strategies", nargs="+", default=C.STRATEGIES)
    args = ap.parse_args()
    _, val, test = C.load_splits()
    for clf in args.classifiers:
        score = (encoder_scorer(clf) if clf in ("deberta", "promptguard") else laya_scorer() if clf == "laya" else guard_scorer(clf))
        for s in args.strategies:
            var = f"ft_{s}"
            if ("val", 0.0) in C.done_conditions(clf, var):
                print(f"[{clf}/{var}] val@0% already present", flush=True); continue
            C.append_preds(clf, var, "val", 0.0, val, score(s, val))
            d = C.load_preds(clf, var)
            old = d[(d.split == "test") & (d.frac == 0.0)].set_index("pair_id").prob
            new = pd.Series(score(s, test), index=test["pair_id"].to_numpy())
            print(f"[{clf}/{var}] val@0% written ({len(val)} pairs); test@0% max |new-stored| = "
                  f"{(new - old.loc[new.index]).abs().max():.2e}", flush=True)


if __name__ == "__main__":
    main()
