# Response-Aware Harm Classification

Code and dataset for evaluating and fine-tuning response-aware harm classifiers across multiple LLMs.

> **Content Warning:** Contains harmful prompts and LLM-generated responses. Intended strictly for safety research.

---

## Quick Start

### 1. Installation
```bash
conda create -n harm python=3.12 -y && conda activate harm
pip install -r requirements.txt
export HF_TOKEN="your_token_here"  # Required for gated Llama models
```
*Trained and evaluated on a single NVIDIA A100 (80GB).*

### 2. Fast Evaluation (No GPU Required)
Reproduce all reported figures, tables, and metrics directly from the precomputed scores:
```bash
cd 2_training_evaluation/evaluation/code
bash make_results.sh
```

---

## Dataset Overview

The interaction dataset is located in `2_training_evaluation/` (`train_set.csv` and `test_set.csv`). 

- **Total Pairs:** 41,944 prompt–response pairs (29,710 train / 5,236 val / 6,998 test).
- **Harmful Rate:** ~3.7% overall (`label = 1`).
- **Validation Split:** Deterministic 15% prompt-level split of `train_set.csv` (Seed: 42; all 7 responses per prompt remain together).
- **Generating Models:** `gemma2:2b`, `gemma3:12b`, `gemma3:27b`, `llama3.1:8b`, `llama3.2:3b`, `qwen2.5:7b`, `qwen2.5:14b`.
- **Key Columns:**
  - `prompt`: User input (benign or malicious).
  - `response`: Generated completion (up to 512 tokens).
  - `label`: Interaction ground truth (`1` = harmful response, `0` = safe).

---

## Repository Structure

```
├── 1_dataset_construction/       # Pipeline: prompts -> Ollama responses -> Gemini labeling -> pairs
├── 2_training_evaluation/
│   ├── train_set.csv, test_set.csv  # Final datasets
│   ├── FineTuneBerts.py             # Fine-tunes DeBERTa-v3 & Prompt-Guard-2
│   ├── FineTuneGuards.py            # Fine-tunes Llama-Guard-3 & WildGuard (QLoRA)
│   └── evaluation/
│       ├── code/                    # Inference, LAYA training, metrics, tables, plots
│       ├── predictions/             # Stored validation & test scores (all truncation steps)
│       └── metrics/, tables/, figures/ # Reproducible outputs
```

---

## Pipeline & Reproduction

### Step 1: Dataset Construction (Optional)
To regenerate the dataset from scratch:
```bash
cd 1_dataset_construction
python 1.build_train_test_dataset.py  # Collect & deduplicate base prompts
ollama serve &                        # Start local Ollama server
python 2.generate_llm_responses.py    # Generate model responses
export GEMINI_API_KEYS="key1,key2"
python 3.label_with_gemini.py         # Label pairs via gemini-3.1-flash-lite
python 4.build_pair_dataset.py        # Assemble train_set.csv & test_set.csv
```
*(Note: Re-running upstream steps can result in minor variations due to API updates. Use released CSVs for exact parity).*

---

### Step 2: Fine-Tuning
Train the 5 supported classifiers across 6 class-imbalance strategies (`standard`, `weighted_loss`, `oversampling`, `undersampling`, `weighted_oversampling`, `focal_loss`):

```bash
cd 2_training_evaluation

# Encoders (Full Fine-Tuning)
python FineTuneBerts.py --model-type deberta_base --output-dir imbalance_runs
python FineTuneBerts.py --model-type promptguard --output-dir imbalance_runs

# LLM Guards (QLoRA: 4-bit NF4, r=16, alpha=32)
python FineTuneGuards.py --model-type llamaguard3 --output-dir causal_llamaguard_runs
python FineTuneGuards.py --model-type wildguard   --output-dir causal_guard_runs

# LAYA (ModernBERT-large)
# Download convaiinnovations/laya first, then run:
python evaluation/code/train_laya.py
```

---

### Step 3: Full Evaluation & Analysis
Run model inference, compute truncation sweeps, and calculate deployment costs:

```bash
cd 2_training_evaluation/evaluation/code

# Full GPU evaluation (runs inference + LAYA training + result generation)
PY=$(which python) bash run_all.sh
```

**Evaluation Details:**
- **Threshold Calibration:** Thresholds maximize harmful-class F1 on the validation set, then transfer unchanged to the test set.
- **Partial Monitoring Sweeps:** Tests response lengths at 10% character increments (0% to 100%).
- **Deployment Costs Evaluated:** False rejection rate (safe interactions blocked), leakage rate (harmful completions missed), and token exposure (tokens leaked before early termination).
- **Statistical Significance:** 95% prompt-clustered bootstrap confidence intervals (1,000 resamples).