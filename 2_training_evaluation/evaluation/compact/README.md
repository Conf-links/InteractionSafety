# Compact results (generated 2026-10-06 06:24 UTC)

Five classifiers x {zero-shot prompt-only, zero-shot prompt+response, best fine-tuned strategy}; every row has its
own validation-calibrated threshold. AUROC/AUPRC are primary; `*_thresholded` figures show recall/precision/F1.
`*_recall` = recall-priority selection (main), `*_f1` / `*_f1_macro` = alternatives.

- `compact_<crit>.csv`: numbers with 95% CIs (and paired fine-tuned minus zero-shot deltas for F1/recall)
- `tab_compact_<crit>.tex`, `tab_sweep_<crit>.tex`: LaTeX for Overleaf (booktabs, multirow, graphicx)
- `figures/`: compact bar chart and truncation sweeps (best strategy only), PDF + PNG

