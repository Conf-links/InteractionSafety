"""False rejection, leakage and exposure of three deployment pipelines on the TEST split (no GPU; uses stored scores).

Ground truth is the interaction label y. For every pipeline and classifier:
  false rejection rate (FRR) = blocked safe interactions / all safe interactions          (y = 0)
  leakage                    = harmful interactions allowed to finish / all harmful ones   (y = 1)
  exposure                   = response tokens generated before the system stops a harmful interaction, averaged over
                               the harmful interactions it stops (these tokens are produced but never served).
                               Also reported as a share of that response's full length.

Pipelines (each decision uses the classifier's own validation-calibrated best-F1 threshold, as in every other table):
  prompt gate      : classify the prompt before generation. Zero-shot = zs_native_prompt; fine-tuned = ft_<s> at 0%
                     response with its own threshold from validation at 0% (infer_ft_prompt_only.py). Exposure 0.
  input-output     : classify prompt + full response after generation (zero_shot / ft_<s> at 100%).
                     Every stopped harmful response is fully generated: exposure = full length.
  partial monitor  : classify prompt + response prefix at 10%, 20%, ..., 100% of the response characters while it is
                     generated; stop at the first checkpoint whose score reaches the threshold (the model's 100%
                     threshold, held fixed across checkpoints exactly as in the truncation sweep). Exposure = tokens in
                     the prefix at the stopping checkpoint. Blocks whenever any checkpoint fires, so its FRR is >= the
                     input-output guard's FRR and its leakage is <= the input-output guard's leakage.
Tokens are counted with the Llama-3 tokenizer (Llama-Guard-3-8B, the same tokenizer as llama3.1:8b / llama3.2:3b),
used for all seven generators so that exposure is comparable across them.

Outputs: metrics/pipeline_costs.csv, tables/tab_pipeline_costs.tex (fine-tuned, compact),
         tables/tab_pipeline_costs_full.tex (zero-shot and fine-tuned).
"""
import json

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

import common as C
from aggregate import CLASSIFIERS

M, T = C.NEW_RESULTS / "metrics", C.NEW_RESULTS / "tables"
CHECKPOINTS = [f for f in C.TRUNCATION_STEPS if f > 0]


def val_threshold(clf, var, frac):
    d = C.load_preds(clf, var)
    v = d[(d.split == "val") & (d.frac == frac)].sort_values("pair_id")
    return C.best_f1_threshold(v.prob.to_numpy(), v.label.to_numpy()) if len(v) else None


def test_scores(clf, var, frac, order):
    d = C.load_preds(clf, var)
    t = d[(d.split == "test") & (d.frac == frac)].set_index("pair_id")
    return t.prob.loc[order].to_numpy() if len(t) else None


def summarise(blocked, stop_tokens, y, full_tokens):
    """blocked: bool per pair; stop_tokens: tokens generated when stopped (only used where blocked & y==1)."""
    h = (y == 1) & blocked
    return {"frr": blocked[y == 0].mean(), "leakage": 1 - blocked[y == 1].mean(),
            "n_safe_blocked": int(blocked[y == 0].sum()), "n_harmful_leaked": int((~blocked[y == 1]).sum()),
            "exposure_tokens": float(stop_tokens[h].mean()) if h.any() else 0.0,
            "exposure_share": float((stop_tokens[h] / np.maximum(full_tokens[h], 1)).mean()) if h.any() else 0.0}


def main():
    _, _, test = C.load_splits()
    test = test.sort_values("pair_id").reset_index(drop=True)
    order, y = test.pair_id.to_numpy(), test.label.to_numpy()
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-Guard-3-8B")
    ntok = {f: np.array([len(tok.encode(C.truncate_response(r, f), add_special_tokens=False)) for r in test.response])
            for f in C.TRUNCATION_STEPS}
    full = ntok[1.0]
    sel = json.load(open(M / "selection.json"))["selected"]["recall"]

    rows = []
    for clf, s in sel.items():
        for kind, zs_var, var in [("Zero-shot", "zs_native_prompt", "zero_shot"), (f"Fine-tuned ({s.replace('_', ' ')})", None, f"ft_{s}")]:
            # prompt gate
            pv, pf = (zs_var, 1.0) if zs_var else (var, 0.0)
            th = val_threshold(clf, pv, pf)
            p = test_scores(clf, pv, pf, order)
            if th is not None and p is not None:
                b = p >= th
                rows.append({"classifier": clf, "model": kind, "pipeline": "Prompt gate", "threshold": th,
                             **summarise(b, np.zeros(len(y)), y, full)})
            else:
                print(f"skip prompt gate {clf}/{pv}@{pf}: no validation scores yet")
            # input-output guard and partial monitor share the 100% threshold
            th = val_threshold(clf, var, 1.0)
            p = test_scores(clf, var, 1.0, order)
            b = p >= th
            rows.append({"classifier": clf, "model": kind, "pipeline": "Input-output guard", "threshold": th,
                         **summarise(b, full.astype(float), y, full)})
            stop = np.full(len(y), np.nan)
            for f in CHECKPOINTS:
                fire = (test_scores(clf, var, f, order) >= th) & np.isnan(stop)
                stop[fire] = ntok[f][fire]
            b = ~np.isnan(stop)
            rows.append({"classifier": clf, "model": kind, "pipeline": "Partial monitor", "threshold": th,
                         **summarise(b, np.nan_to_num(stop), y, full)})
    df = pd.DataFrame(rows)
    df.insert(1, "classifier_name", df.classifier.map(lambda k: CLASSIFIERS[k][0]))
    df.to_csv(M / "pipeline_costs.csv", index=False)

    harm_tokens = full[y == 1]
    note = (f"Harmful test responses: {int((y == 1).sum())}, mean length {harm_tokens.mean():.0f} tokens "
            f"(median {np.median(harm_tokens):.0f}; Llama-3 tokenizer).")
    pct = lambda x: f"{100 * x:.1f}"
    PIPES = ["Prompt gate", "Input-output guard", "Partial monitor"]

    # compact: fine-tuned models, pipelines side by side
    L = ["% auto-generated by evaluation/code/pipeline_costs.py  (needs booktabs)",
         "\\begin{table*}[t]", "\\centering\\small",
         "\\caption{Deployment cost on the test split (fine-tuned models, recall-selected strategy). FRR: \\% of safe "
         "interactions blocked; Leak: \\% of harmful interactions allowed to finish; Exp.: mean response tokens generated "
         "before a stopped harmful interaction is cut off (never served to the user). The prompt gate scores the prompt "
         "with an empty response; the partial monitor checks the response every 10\\% of its length and stops at the first "
         "flag. Each setting uses its own validation-calibrated best-F1 threshold. " + note + "}",
         "\\label{tab:pipeline-costs}", "\\begin{tabular}{l" + "rrr" * 3 + "}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{p}}}" for p in PIPES) + " \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
         "Classifier" + " & FRR & Leak & Exp." * 3 + " \\\\", "\\midrule"]
    for clf in sel:
        r = df[(df.classifier == clf) & df.model.str.startswith("Fine")].set_index("pipeline")
        cells = []
        for p in PIPES:
            if p in r.index:
                cells += [pct(r.loc[p, "frr"]), pct(r.loc[p, "leakage"]), f"{r.loc[p, 'exposure_tokens']:.0f}"]
            else:
                cells += ["--"] * 3
        L.append(f"{CLASSIFIERS[clf][0]} ({sel[clf].replace('_', ' ')}) & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (T / "tab_pipeline_costs.tex").write_text("\n".join(L) + "\n")

    # full: zero-shot and fine-tuned, one row per pipeline
    L = ["% auto-generated by evaluation/code/pipeline_costs.py  (needs booktabs, multirow)",
         "\\begin{table}[t]", "\\centering\\small",
         "\\caption{False rejection, leakage and exposure for zero-shot and fine-tuned classifiers in each pipeline "
         "(test split). Exposure: mean tokens generated before a stopped harmful interaction is cut off, and its share "
         "of the full response. " + note + "}", "\\label{tab:pipeline-costs-full}",
         "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{lllrrr}", "\\toprule",
         "Classifier & Model & Pipeline & FRR (\\%) & Leakage (\\%) & Exposure tok. (\\%) \\\\", "\\midrule"]
    for i, clf in enumerate(sel):
        d = df[df.classifier == clf]
        if i:
            L.append("\\midrule")
        first = True
        for kind, g in d.groupby("model", sort=False):
            for j, r in enumerate(g.itertuples()):
                c0 = f"\\multirow{{{len(d)}}}{{*}}{{{CLASSIFIERS[clf][0]}}}" if first else ""
                c1 = f"\\multirow{{{len(g)}}}{{*}}{{{kind}}}" if j == 0 else ""
                first = False
                L.append(f"{c0} & {c1} & {r.pipeline} & {pct(r.frr)} & {pct(r.leakage)} & "
                         f"{r.exposure_tokens:.0f} ({pct(r.exposure_share)}) \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    (T / "tab_pipeline_costs_full.tex").write_text("\n".join(L) + "\n")
    pd.set_option("display.width", 220)
    print(note)
    print(df.drop(columns=["classifier"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
