"""Off-the-shelf LAYA on the prompt alone (no "Response:" field), validation + test.

The other four zero-shot classifiers already have a prompt-only mode in their native format (`zs_native_prompt`:
raw prompt for ProtectAI / Prompt-Guard-2, user turn only for Llama-Guard-3, the "Harmful request" answer for
WildGuard). LAYA has no native prompt mode, so it gets the same response-harm question with the state
"Prompt: <p>". Its threshold is calibrated on validation in this same mode, like every other prompt-only baseline.
"""
import torch

import common as C
import laya
from laya.common import QTYPES, build_sequence
from train_laya import MAX_LEN, HEAD_MAX_LEN, QUESTION, predict


def items(tok, df):
    out = []
    for p, y in zip(df["prompt"], df["label"].astype(int)):
        ids, markers = build_sequence(tok, f"Prompt: {C.as_text(p)}", QUESTION, MAX_LEN, HEAD_MAX_LEN)
        out.append({"ids": ids, "markers": markers, "qtype": QTYPES["choice"], "target": [1.0 - y, float(y)],
                    "label": int(y), "episode": 0, "ep_step": 0})
    return out


def main():
    _, val, test = C.load_splits()
    agent = laya.load(str(C.Path.home() / "models" / "convaiinnovations-laya"), device="cpu")
    model, tok, device = agent.model.to("cuda"), agent.tok, torch.device("cuda")
    for split, df in (("val", val), ("test", test)):
        C.append_preds("laya", "zs_native_prompt", split, 1.0, df, predict(model, items(tok, df), tok.pad_token_id, device))
        print(f"[laya/zs_native_prompt] {split} n={len(df)}", flush=True)


if __name__ == "__main__":
    main()
