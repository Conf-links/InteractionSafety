#!/usr/bin/env bash
# GPU queue 2: encoder inference (DeBERTa/ProtectAI, PromptGuard), then LAYA full fine-tuning + inference.
set -euo pipefail
cd "$(dirname "$0")"; source env.sh
$PY infer_encoders.py
$PY train_laya.py
echo "ENCODERS+LAYA DONE"
