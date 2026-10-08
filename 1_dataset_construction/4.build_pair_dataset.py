"""
Step 4: Build the labelled prompt-response dataset from the Gemini-labelled files of step 3.

Reads train.csv / test.csv (step 1) and labeled_gemini_{split}_{model}.csv (step 3) from this folder and writes
    train_set.csv / test_set.csv   (VERSION 1 below; the files used for all fine-tuning and evaluation,
                                    released in ../2_training_evaluation/)
    wide_train.csv / wide_test.csv (VERSION 2 below)
    invalid_counts.csv

Model names are parsed straight out of the filenames, which look like
    labeled_gemini_{split}_{model}_{param}.csv
e.g. labeled_gemini_test_qwen2.5_7b.csv -> model "qwen2.5:7b".
The param stays in the name because the same model appears at several sizes.

VERSION 1 - concat (one row per prompt-model pair)
    train_set.csv  35,000 rows      test_set.csv  7,000 rows
    pair_id, prompt_id, prompt, prompt_label, response, model, label

VERSION 2 - wide (one row per prompt, 7 models side by side)
    wide_train.csv  5,000 rows         wide_test.csv  1,000 rows
    prompt_id, prompt, prompt_label,
    then <model>_response and <model>_label for each of the 7 models
    = 3 + 14 = 17 columns

Labels: 1 = harmful response, 0 = safe. Rows Gemini could not label (-1) are
dropped, so the released CSVs contain valid labels only:

  concat  drops just the offending pair          -> 34,946 train / 6,998 test
  wide    drops the whole prompt if ANY of the 7 models is -1, so every row
          has all 7 labels and models stay comparable -> 4,984 train / 998 test

How many were dropped per model is recorded in invalid_counts.csv so the
label-quality stat survives the filtering.
"""
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE                             # train.csv, test.csv and labeled_gemini_*.csv
OUT = HERE
SPLITS = ("train", "test")

# model order fixed so columns are stable across runs: grouped by family
MODEL_ORDER = ["gemma2_2b", "gemma3_12b", "gemma3_27b",
               "llama3.2_3b", "llama3.1_8b",
               "qwen2.5_7b", "qwen2.5_14b"]


def parse_model(filename: str) -> str:
    """labeled_gemini_test_qwen2.5_7b.csv -> 'qwen2.5:7b'"""
    m = re.match(r"labeled_gemini_(train|test)_(.+)_([^_]+)\.csv$", filename)
    if not m:
        sys.exit(f"[fatal] cannot parse model from filename: {filename}")
    return f"{m.group(2)}:{m.group(3)}"


def load_prompts(split: str) -> pd.DataFrame:
    df = pd.read_csv(SRC / f"{split}.csv", dtype={"prompt_id": str})
    df = df[["prompt_id", "prompt_text", "prompt_label"]].copy()
    df.columns = ["prompt_id", "prompt", "prompt_label"]
    df["prompt_label"] = df["prompt_label"].astype(int)
    return df


def load_labelled(split: str, slug: str, prompts: pd.DataFrame):
    """Return (model_name, dataframe aligned to the prompt table)."""
    fname = f"labeled_gemini_{split}_{slug}.csv"
    path = SRC / fname
    if not path.exists():
        sys.exit(f"[fatal] missing {fname}")
    model = parse_model(fname)

    lab = pd.read_csv(path, dtype={"prompt_id": str})[["prompt_id", "response", "label"]]
    if lab["prompt_id"].duplicated().any():
        sys.exit(f"[fatal] {fname} has duplicate prompt_id values")

    # prompt text always comes from train/test.csv, not from the 7 copies of it
    d = prompts.merge(lab, on="prompt_id", how="left", validate="one_to_one")
    if d["label"].isna().any():
        sys.exit(f"[fatal] {fname}: {int(d['label'].isna().sum())} prompts have no label")

    d["label"] = d["label"].astype(int)
    d["response"] = d["response"].fillna("").astype(str)
    return model, d


def build(split: str):
    prompts = load_prompts(split)
    loaded = [load_labelled(split, slug, prompts) for slug in MODEL_ORDER]

    # record what we are about to drop, before dropping it
    invalid = [{"split": split, "model": model,
                "n_prompts": len(d), "invalid_-1": int((d.label == -1).sum())}
               for model, d in loaded]

    # ---------------- version 1: concat ----------------
    # drop only the pair Gemini failed on; the other 6 models keep this prompt
    frames = []
    for model, d in loaded:
        f = d[d.label != -1].copy()
        f["model"] = model
        frames.append(f[["prompt_id", "prompt", "prompt_label", "response", "model", "label"]])

    concat = pd.concat(frames, ignore_index=True)
    # group all rows of a prompt together, then number 1..N contiguously
    concat = concat.sort_values(["prompt_id", "model"], kind="stable").reset_index(drop=True)
    concat.insert(0, "pair_id", range(1, len(concat) + 1))
    concat.to_csv(OUT / f"{split}_set.csv", index=False)

    # ---------------- version 2: wide ----------------
    # keep only prompts every model labelled, so all 7 columns are comparable
    keep = set(prompts["prompt_id"])
    for _, d in loaded:
        keep &= set(d.loc[d.label != -1, "prompt_id"])

    wide = prompts[prompts["prompt_id"].isin(keep)].reset_index(drop=True)
    for model, d in loaded:
        s = d.set_index("prompt_id")
        wide[f"{model}_response"] = wide["prompt_id"].map(s["response"])
        wide[f"{model}_label"] = wide["prompt_id"].map(s["label"]).astype(int)
    assert (wide[[f"{m}_label" for m, _ in loaded]] != -1).all().all(), "-1 leaked into wide"
    wide.to_csv(OUT / f"wide_{split}.csv", index=False)

    models = [m for m, _ in loaded]
    n_drop_c = len(prompts) * len(models) - len(concat)
    print(f"[{split}] {len(prompts)} prompts x {len(models)} models")
    print(f"   {split}_set.csv  {len(concat):,} rows, {len(concat.columns)} cols "
          f"(dropped {n_drop_c} invalid pairs)")
    print(f"   wide_{split}.csv    {len(wide):,} rows, {len(wide.columns)} cols "
          f"(dropped {len(prompts)-len(wide)} prompts with any -1)")
    for model, d in loaded:
        vc = d["label"].value_counts()
        print(f"     {model:14s} harmful={int(vc.get(1,0)):5d} "
              f"safe={int(vc.get(0,0)):5d} dropped(-1)={int(vc.get(-1,0)):3d}")
    return models, invalid


def main():
    models, invalid = None, []
    for split in SPLITS:
        models, inv = build(split)
        invalid += inv
    pd.DataFrame(invalid).to_csv(OUT / "invalid_counts.csv", index=False)
    print(f"\nmodels: {', '.join(models)}")
    print(f"dropped {sum(r['invalid_-1'] for r in invalid)} invalid labels in total "
          f"-> invalid_counts.csv")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
