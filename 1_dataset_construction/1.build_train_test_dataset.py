"""
Step 1: Build a balanced binary-labelled prompt dataset with a fixed
train / test split (~5000 train, ~1000 test) from four sources.

Label convention: 1 = adversarial / malicious, 0 = benign.

Composition (per split), in the order the sources are drawn:

  xtram    -> xTRam1/safe-guard-prompt-injection   (has train + test splits)
              train part: 2000 adversarial + 1000 benign  (from its train split)
              test  part:  400 adversarial +  200 benign  (from its test split)

  advbench -> harmful_behaviors.csv from the llm-attacks GitHub repo
              train-only source, all rows adversarial (label 1)
              train part: 400   test part: 120   (disjoint, shuffled)

  deepset  -> deepset/prompt-injections            (has train + test splits)
              its train split -> our train part, its test split -> our test part
              rows keep their own labels (0 benign / 1 adversarial)

  arena    -> lmarena-ai/arena-human-preference-55k (benign user prompts)
              fills whatever is left so train hits ~5000 and test hits ~1000,
              all labelled 0

Every prompt is de-duplicated globally, and train / test never share a prompt,
so there is no leakage between the two splits.

Output (written next to this script):
  train.csv                  prompt_id, prompt_text, prompt_label, source_dataset, split
  test.csv                   same columns
  dataset_split_summary.csv  per-split / per-source label counts

Run:
    python3 1.build_train_test_dataset.py
    python3 1.build_train_test_dataset.py --train-total 5000 --test-total 1000
"""
import argparse
import ast
import csv
import io
import json
import random
import urllib.request
import warnings
from pathlib import Path

import pandas as pd
from datasets import load_dataset

SEED = 42

# --- how much each source contributes ------------------------------------
TRAIN_TOTAL = 5000
TEST_TOTAL = 1000

XTRAM_TRAIN_ADV = 2000
XTRAM_TRAIN_BEN = 1000
XTRAM_TEST_ADV = 400
XTRAM_TEST_BEN = 200

ADVBENCH_TRAIN = 400
ADVBENCH_TEST = 120

OUT_DIR = Path(__file__).parent
TRAIN_CSV = OUT_DIR / "train.csv"
TEST_CSV = OUT_DIR / "test.csv"
SUMMARY_CSV = OUT_DIR / "dataset_split_summary.csv"

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
    "main/data/advbench/harmful_behaviors.csv"
)


def clean(text) -> str:
    """Normalise whitespace and strip anything that cannot round-trip UTF-8."""
    if text is None:
        return ""
    s = str(text)
    s = s.encode("utf-8", "replace").decode("utf-8", "replace")
    s = s.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")
    return " ".join(s.split()).strip()


class Dedup:
    """Keeps every prompt unique across both splits and all four sources."""

    def __init__(self):
        self.seen = set()

    def keep(self, text: str) -> bool:
        key = text.lower()
        if not text or key in self.seen:
            return False
        self.seen.add(key)
        return True

    def filter(self, texts):
        return [t for t in texts if self.keep(t)]


# ---------------------------------------------------------------- xtram ---
def take_xtram(rng, dedup):
    print("[xtram] loading xTRam1/safe-guard-prompt-injection ...")
    ds = load_dataset("xTRam1/safe-guard-prompt-injection")

    def pool(split):
        adv, ben = [], []
        for r in ds[split]:
            t = clean(r["text"])
            if not t:
                continue
            (adv if int(r["label"]) == 1 else ben).append(t)
        rng.shuffle(adv)
        rng.shuffle(ben)
        return dedup.filter(adv), dedup.filter(ben)

    # train part comes from its train split, test part from its test split
    tr_adv, tr_ben = pool("train")
    te_adv, te_ben = pool("test")
    print(
        f"[xtram] train split available adv={len(tr_adv)} ben={len(tr_ben)} ; "
        f"test split available adv={len(te_adv)} ben={len(te_ben)}"
    )

    train = _take(tr_adv, XTRAM_TRAIN_ADV, 1, "xtram", "xtram train adv")
    train += _take(tr_ben, XTRAM_TRAIN_BEN, 0, "xtram", "xtram train benign")
    test = _take(te_adv, XTRAM_TEST_ADV, 1, "xtram", "xtram test adv")
    test += _take(te_ben, XTRAM_TEST_BEN, 0, "xtram", "xtram test benign")
    return train, test


# ------------------------------------------------------------- advbench ---
def take_advbench(rng, dedup):
    print("[advbench] fetching harmful_behaviors.csv from llm-attacks GitHub ...")
    with urllib.request.urlopen(ADVBENCH_URL, timeout=60) as r:
        data = r.read().decode()
    rows = [clean(row.get("goal", "")) for row in csv.DictReader(io.StringIO(data))]
    rows = dedup.filter([t for t in rows if t])
    rng.shuffle(rows)
    print(f"[advbench] usable rows={len(rows)} (all adversarial)")

    # train slice first, then a disjoint test slice from the remainder
    train = _take(rows, ADVBENCH_TRAIN, 1, "advbench", "advbench train")
    test = _take(rows, ADVBENCH_TEST, 1, "advbench", "advbench test")
    return train, test


# -------------------------------------------------------------- deepset ---
def take_deepset(rng, dedup):
    print("[deepset] loading deepset/prompt-injections ...")
    ds = load_dataset("deepset/prompt-injections")

    def rows_for(split):
        out = []
        for r in ds[split]:
            t = clean(r["text"])
            if not dedup.keep(t):
                continue
            out.append((t, int(r["label"]), "deepset"))
        rng.shuffle(out)
        return out

    # deepset's own splits map straight onto ours
    train = rows_for("train")
    test = rows_for("test")
    print(f"[deepset] train={len(train)} rows, test={len(test)} rows (own labels)")
    return train, test


# ---------------------------------------------------------------- arena ---
def take_arena(rng, dedup, n_train, n_test):
    """Benign filler so each split reaches its target size."""
    need = n_train + n_test
    if need <= 0:
        print("[arena] nothing left to fill, skipping")
        return [], []

    print("[arena] loading lmarena-ai/arena-human-preference-55k ...")
    ds = load_dataset("lmarena-ai/arena-human-preference-55k")
    pool = []
    for r in ds["train"]:
        raw = r.get("prompt")
        if raw is None:
            continue
        t = clean(first_turn(raw))
        if t:
            pool.append(t)

    pool = dedup.filter(pool)
    rng.shuffle(pool)
    print(f"[arena] candidate pool={len(pool)}, need train={n_train} test={n_test}")

    train = _take(pool, n_train, 0, "arena", "arena train benign")
    test = _take(pool, n_test, 0, "arena", "arena test benign")
    return train, test


def first_turn(raw) -> str:
    """Arena stores its prompt column as a JSON list of turns; keep turn 1.

    json.loads handles the JSON escapes those strings contain (e.g. \\/), which
    ast.literal_eval only warns about and fails on. ast stays as a fallback for
    the few rows written as Python literals, with its SyntaxWarning silenced.
    """
    s = str(raw).strip()
    if not (s.startswith("[") and s.endswith("]")):
        return s
    for parser in (json.loads, ast.literal_eval):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                parsed = parser(s)
            if isinstance(parsed, list):
                # empty list -> "", so the caller drops the row
                return str(parsed[0]) if parsed else ""
            return s
        except Exception:
            continue
    return s


def _take(pool, n, label, source, what):
    """Pop n items off pool (mutating it) and tag them; warn if short."""
    got = pool[:n]
    del pool[:n]
    if len(got) < n:
        print(f"  WARNING: {what} wanted {n} but only {len(got)} available")
    return [(t, label, source) for t in got]


def finalise(rows, split, rng, prefix):
    rng.shuffle(rows)
    df = pd.DataFrame(rows, columns=["prompt_text", "prompt_label", "source_dataset"])
    df.insert(0, "prompt_id", [f"{prefix}_{i:05d}" for i in range(1, len(df) + 1)])
    df["split"] = split
    return df


def summarise(train_df, test_df):
    rows = []
    for split, df in (("train", train_df), ("test", test_df)):
        for src, sub in df.groupby("source_dataset"):
            for label, n in sub["prompt_label"].value_counts().sort_index().items():
                rows.append({
                    "split": split,
                    "source_dataset": src,
                    "prompt_label": int(label),
                    "count": int(n),
                })
        for label in (0, 1):
            rows.append({
                "split": split,
                "source_dataset": "TOTAL",
                "prompt_label": label,
                "count": int((df["prompt_label"] == label).sum()),
            })
        rows.append({
            "split": split,
            "source_dataset": "TOTAL",
            "prompt_label": "ALL",
            "count": len(df),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-total", type=int, default=TRAIN_TOTAL)
    ap.add_argument("--test-total", type=int, default=TEST_TOTAL)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    dedup = Dedup()

    train_rows, test_rows = [], []

    x_tr, x_te = take_xtram(rng, dedup)
    train_rows += x_tr
    test_rows += x_te

    a_tr, a_te = take_advbench(rng, dedup)
    train_rows += a_tr
    test_rows += a_te

    d_tr, d_te = take_deepset(rng, dedup)
    train_rows += d_tr
    test_rows += d_te

    # arena tops both splits up to their targets with benign prompts
    fill_train = max(args.train_total - len(train_rows), 0)
    fill_test = max(args.test_total - len(test_rows), 0)
    ar_tr, ar_te = take_arena(rng, dedup, fill_train, fill_test)
    train_rows += ar_tr
    test_rows += ar_te

    train_df = finalise(train_rows, "train", rng, "train")
    test_df = finalise(test_rows, "test", rng, "test")

    train_df.to_csv(TRAIN_CSV, index=False, quoting=csv.QUOTE_ALL)
    test_df.to_csv(TEST_CSV, index=False, quoting=csv.QUOTE_ALL)
    print(f"\nWrote {TRAIN_CSV} with {len(train_df)} rows")
    print(f"Wrote {TEST_CSV} with {len(test_df)} rows")

    summary = summarise(train_df, test_df)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote {SUMMARY_CSV}")

    print("\n=== Summary (1 = adversarial, 0 = benign) ===")
    print(summary.to_string(index=False))
    for name, df in (("train", train_df), ("test", test_df)):
        pos = int((df["prompt_label"] == 1).sum())
        print(f"{name}: {len(df)} rows, {pos} adversarial "
              f"({pos / max(len(df), 1):.1%}), {len(df) - pos} benign")


if __name__ == "__main__":
    main()
