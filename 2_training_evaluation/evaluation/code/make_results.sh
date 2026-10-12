#!/usr/bin/env bash
# Rebuilds every metric, table and figure from the stored scores in ../predictions/ (CPU only, a few minutes).
set -euo pipefail
cd "$(dirname "$0")"; source env.sh
export CUDA_VISIBLE_DEVICES=""
$PY dataset_stats.py      # metrics/dataset_*.csv, tables/tab_dataset_{sources,stats}.tex
$PY aggregate.py          # thresholds, test metrics at every truncation level, bootstrap CIs, strategy selection
$PY pipeline_costs.py     # false rejection / leakage / exposure of prompt gate, input-output guard, partial monitor
$PY make_figures.py       # figures/*.pdf|png
$PY make_tables.py        # tables/*.tex (booktabs)
$PY make_report.py        # report/REPORT.md, compact/README.md
echo "results rebuilt"
