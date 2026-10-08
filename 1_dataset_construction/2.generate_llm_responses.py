"""
Step 2: Generate LLM responses for every prompt in train.csv and test.csv
using a local Ollama server.

One CSV per (model, split), written to responses/ next to this script:
    responses/responses_train_llama3.1_8b.csv
    responses/responses_test_llama3.1_8b.csv
with columns:
    prompt_id, split, prompt_text, prompt_label, source_dataset, model, response

Default models (run unless --models says otherwise):
    llama3.1:8b   llama3.2:3b   gemma3:27b   gemma3:12b   qwen2.5:7b   qwen2.5:14b

Optional models (NOT run by default, add with --include-optional or --models):
    gemma2:2b   qwen3:4b   llama3.1:70b

Run:
    python3 2.generate_llm_responses.py                       # the six defaults
    python3 2.generate_llm_responses.py --splits train        # train only
    python3 2.generate_llm_responses.py --models qwen3:4b     # one extra model
    python3 2.generate_llm_responses.py --include-optional    # defaults + extras
    python3 2.generate_llm_responses.py --limit 20            # smoke test

Notes:
    - Ollama must be serving on port 11434 (`ollama serve`), and each model
      must be pulled (`ollama pull gemma3:27b`).
    - Rows are appended to disk as soon as they are produced, so a re-run
      resumes: prompt_ids that already have a non-empty response are skipped.
    - Reasoning models (qwen3) are asked to skip their thinking block here;
      any stray <think>...</think> is stripped so this file holds the answer
      only. Chain-of-thought is collected separately by 3.generate_llm_cot.py.
"""
import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
TRAIN_CSV = BASE_DIR / "train.csv"
TEST_CSV = BASE_DIR / "test.csv"
OUT_DIR = BASE_DIR / "responses"

MODELS = [
    "llama3.1:8b",
    "llama3.2:3b",
    "gemma3:27b",
    "gemma3:12b",
    "qwen2.5:7b",
    "qwen2.5:14b",
    "gemma2:2b",
]

# kept out of the default run; enable with --include-optional or --models
OPTIONAL_MODELS = [
    "gemma2:2b",
    "qwen3:4b",
    "llama3.1:70b",
]

# models whose API call accepts a "think" flag
THINKING_PREFIXES = ("qwen3",)

OLLAMA_URL = "http://localhost:11434/api/generate"
NUM_PREDICT = 512        # cap response length
REQUEST_TIMEOUT = 300    # seconds per request (70b models are slow)
MAX_RETRIES = 3

FIELDNAMES = [
    "prompt_id",
    "split",
    "prompt_text",
    "prompt_label",
    "source_dataset",
    "model",
    "response",
]

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def safe_filename(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def output_path_for(model: str, split: str) -> Path:
    return OUT_DIR / f"responses_{split}_{safe_filename(model)}.csv"


def is_thinking_model(model: str) -> bool:
    return model.lower().startswith(THINKING_PREFIXES)


def strip_think(text: str) -> str:
    """Drop reasoning blocks so this file stores the final answer only."""
    out = THINK_BLOCK.sub("", text)
    # unclosed block: keep whatever follows the opening tag
    if "<think>" in out.lower():
        out = re.split(r"<think>", out, flags=re.IGNORECASE)[0]
    return out.strip()


def post(payload: dict) -> dict:
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def call_ollama(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": NUM_PREDICT, "temperature": 0.0},
    }
    if is_thinking_model(model):
        payload["think"] = False

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            body = post(payload)
            return strip_think(body.get("response", "") or "")
        except urllib.error.HTTPError as e:
            # older servers / non-reasoning builds reject the "think" flag
            if "think" in payload:
                payload.pop("think")
                continue
            last_err = e
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
            time.sleep(2 * attempt)
        except Exception as e:
            last_err = e
            break
    return f"<<ERROR: {type(last_err).__name__}: {last_err}>>"


def load_done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        return set()
    if "prompt_id" not in df.columns or "response" not in df.columns:
        return set()
    done = df.loc[df["response"].astype(str).str.len() > 0, "prompt_id"]
    return set(done.tolist())


def append_row(path: Path, row: dict, write_header: bool):
    with path.open("a" if path.exists() else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        if write_header:
            w.writeheader()
        w.writerow(row)


def run_model_split(model: str, split: str, prompts_df: pd.DataFrame):
    out_path = output_path_for(model, split)
    done = load_done_ids(out_path)
    write_header = not out_path.exists()
    total = len(prompts_df)
    todo = total - len(done)
    print(f"\n=== {model} | {split} ===")
    print(f"output: {out_path}")
    print(f"already done: {len(done)} / {total}; remaining: {todo}")

    started = time.time()
    n_done = 0
    for _, row in prompts_df.iterrows():
        pid = str(row["prompt_id"])
        if pid in done:
            continue
        text = str(row["prompt_text"])
        t0 = time.time()
        resp = call_ollama(model, text)
        dt = time.time() - t0
        append_row(
            out_path,
            {
                "prompt_id": pid,
                "split": split,
                "prompt_text": text,
                "prompt_label": str(row.get("prompt_label", "")),
                "source_dataset": str(row.get("source_dataset", "")),
                "model": model,
                "response": resp,
            },
            write_header,
        )
        write_header = False
        n_done += 1
        if n_done % 10 == 0 or n_done == 1:
            elapsed = time.time() - started
            rate = n_done / max(elapsed, 1e-6)
            eta_min = ((todo - n_done) / rate / 60) if rate > 0 else float("inf")
            print(
                f"  [{model}|{split}] {n_done}/{todo} "
                f"last={dt:.1f}s rate={rate:.2f}/s eta={eta_min:.1f}min",
                flush=True,
            )

    print(f"  [{model}|{split}] done. wrote {n_done} new rows.")


def load_split(split: str, limit: int) -> pd.DataFrame:
    path = TRAIN_CSV if split == "train" else TEST_CSV
    if not path.exists():
        print(f"ERROR: {path} not found. Run 1.build_train_test_dataset.py first.")
        sys.exit(1)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if limit > 0:
        df = df.head(limit)
    print(f"Loaded {len(df)} prompts from {path.name}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None,
                    help=f"Models to run (default: {' '.join(MODELS)}).")
    ap.add_argument("--include-optional", action="store_true",
                    help=f"Also run: {' '.join(OPTIONAL_MODELS)}")
    ap.add_argument("--splits", nargs="+", default=["train", "test"],
                    choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=0,
                    help="If > 0, only run the first N prompts of each split.")
    args = ap.parse_args()

    models = list(args.models) if args.models else list(MODELS)
    if args.include_optional and not args.models:
        models += OPTIONAL_MODELS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = {s: load_split(s, args.limit) for s in args.splits}
    print(f"Models: {', '.join(models)}")

    for model in models:
        for split in args.splits:
            run_model_split(model, split, splits[split])


if __name__ == "__main__":
    main()
