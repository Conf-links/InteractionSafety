#!/usr/bin/env bash
# End-to-end evaluation, after the fine-tuning runs of ../../README section "Fine-tuning" exist:
#   ../../imbalance_runs/{deberta_base,promptguard}/agnostic_<strategy>/        (FineTuneBerts.py)
#   ../../causal_llamaguard_runs/llamaguard3/agnostic_<strategy>/               (FineTuneGuards.py)
#   ../../causal_guard_runs/wildguard/agnostic_<strategy>/                      (FineTuneGuards.py / resume_wildguard.py)
# LAYA is fully fine-tuned here (train_laya.py) and needs the base model in ~/models/convaiinnovations-laya.
# Every step skips prediction blocks that already exist in ../predictions/, so the script can be re-run after a crash.
# The two GPU queues fit together on one 80 GB A100.
set -euo pipefail
cd "$(dirname "$0")"; source env.sh
mkdir -p ../logs
bash run_guards.sh        > ../logs/guards.log 2>&1 &
bash run_encoders_laya.sh > ../logs/encoders_laya.log 2>&1 &
wait
# zero-shot LAYA on the prompt alone, and every fine-tuned model on the prompt alone (validation@0% for its threshold)
$PY infer_laya_prompt_only.py
$PY infer_ft_prompt_only.py
# CPU only from here: numbers, tables, figures
bash make_results.sh
