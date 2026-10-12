#!/usr/bin/env bash
# GPU queue 1: generative guards (inference only; adapters are the original QLoRA runs).
set -euo pipefail
# SWEEP_CRITERIA="recall f1 f1_macro" bash run_guards.sh  -> also sweep the F1 / macro-F1 selected adapters
cd "$(dirname "$0")"; source env.sh
FT="ft_standard ft_weighted_loss ft_oversampling ft_undersampling ft_weighted_oversampling ft_focal_loss"
for G in llamaguard3 wildguard; do
  echo "##### $G phase 1: all adapters on val + test@100%"
  $PY infer_guards.py --guard $G --variants $FT --test-fracs 1.0
  SEL=$($PY select_for_sweeps.py --classifier $G --criteria ${SWEEP_CRITERIA:-recall})
  echo "##### $G phase 2: full sweep for selected adapters: $SEL"
  $PY infer_guards.py --guard $G --variants $SEL
  echo "##### $G phase 3: zero-shot (official format) sweep + native prompt moderation"
  $PY infer_guards.py --guard $G --variants zero_shot zs_native_prompt
done
echo "GUARDS DONE"
