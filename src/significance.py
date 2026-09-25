# src/significance.py
"""
Created with Claude on 20-09-2026, revised for consistency and personal modifications: 
McNemar's test for comparing two prediction conditions on the SAME
query set -- e.g. 0-shot vs 4-shot for the same model/prompt/language,
or Prompt A vs Prompt C for the same model/shot-count/language.

McNemar's test is for paired binary outcomes: it asks whether two
conditions disagree in a systematically lopsided way on the same
items, not just whether their overall accuracy differs. It only uses
the DISCORDANT pairs (items where the two conditions disagree); items
both conditions get right, or both get wrong, contribute nothing to
the test statistic.

Accuracy was aggregated from the samples in results/predictions

Usage (from the project root):
    # Compare two prediction files directly
    python -m src.significance results/predictions/<run_id>/DE_0shot/P1_Gemma-3-1B.tsv \
                                 results/predictions/<run_id>/DE_4shot/P1_Gemma-3-1B.tsv

    # Or build both paths from the predictions_dir layout automatically
    python -m src.significance --predictions-dir results/predictions/<run_id> \
        --language DE --prompt P1 --model Gemma-3-1B \
        --condition-a 0shot --condition-b 4shot
        
Tables in paper created with:
python -m src.significance --batch --predictions-dir results/predictions/20260916_131746 --language DE --condition-a 0shot --condition-b 4shot --models "Gemma-3-1B,Qwen2.5-1.5B,Phi-3.5-mini,Llama-3.2-1B" --prompts "P1,P3" --latex-out results/mcnemar_de.tex --tsv-out results/mcnemar_de.tsv
python -m src.significance --batch --predictions-dir results/predictions/20260916_131746 --language ES --condition-a 0shot --condition-b 4shot --models "Gemma-3-1B,Qwen2.5-1.5B,Phi-3.5-mini,Llama-3.2-1B" --prompts "P1,P3" --latex-out results/mcnemar_es.tex --tsv-out results/mcnemar_es.tsv
"""

import argparse
import os

import pandas as pd
from scipy.stats import binomtest

#Reused from previous loading functions
def load_predictions(filepath):
    df = pd.read_csv(filepath, sep="\t")
    required = {"statement", "prediction", "gold_label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{filepath} is missing required columns: {sorted(missing)}")
    return df


def build_condition_path(predictions_dir, language, condition, prompt, model):
    return os.path.join(
        predictions_dir, f"{language}_{condition}", f"{prompt}_{model}.tsv"
    )

#Created by Claude
def paired_correctness(df_a, df_b, invalid_as_incorrect=False):
    """
    Aligns two prediction DataFrames on `statement` and returns one row
    per statement present in both, with boolean "correct_a"/"correct_b"
    columns.

    A statement with no valid (parseable) prediction in either
    condition is:
      - treated as incorrect, if invalid_as_incorrect=True
      - dropped from the paired comparison otherwise (the default),
        since McNemar's test needs a definite correct/incorrect call in
        BOTH conditions for every paired item -- "no answer" isn't
        really "wrong" in the same sense a mis-classification is.

    A statement present in only one file is always dropped (with a
    warning), since the test requires the same items in both.
    """
    merged = df_a.merge(df_b, on="statement", suffixes=("_a", "_b"), how="inner")

    only_in_a = len(df_a) - len(merged)
    only_in_b = len(df_b) - len(merged)
    if only_in_a or only_in_b:
        print(
            f"Warning: {only_in_a} statement(s) only in A, "
            f"{only_in_b} only in B -- dropped (not present in both)."
        )

    if invalid_as_incorrect:
        merged["correct_a"] = (merged["prediction_a"] == merged["gold_label_a"]).fillna(False)
        merged["correct_b"] = (merged["prediction_b"] == merged["gold_label_b"]).fillna(False)
    else:
        valid_mask = merged["prediction_a"].notna() & merged["prediction_b"].notna()
        dropped = (~valid_mask).sum()
        if dropped:
            print(
                f"Dropping {dropped} statement(s) with an invalid "
                f"(unparseable) prediction in at least one condition. "
                f"Pass --invalid-as-incorrect to count these as wrong "
                f"instead of excluding them."
            )
        merged = merged[valid_mask].copy()
        merged["correct_a"] = merged["prediction_a"] == merged["gold_label_a"]
        merged["correct_b"] = merged["prediction_b"] == merged["gold_label_b"]

    return merged

#Eddited and corrected by Claude
def mcnemar_test(merged):
    """
    Runs McNemar's EXACT test (via the binomial distribution) on the
    paired correctness columns. The exact form is used rather than the
    chi-square approximation with continuity correction, since it is
    valid at any sample size.

    Returns a dict with the 2x2 counts and the two-sided p-value.
    """
    both_correct = int((merged["correct_a"] & merged["correct_b"]).sum())
    both_wrong = int((~merged["correct_a"] & ~merged["correct_b"]).sum())
    a_only = int((merged["correct_a"] & ~merged["correct_b"]).sum())
    b_only = int((~merged["correct_a"] & merged["correct_b"]).sum())

    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = binomtest(min(a_only, b_only), discordant, 0.5, alternative="two-sided").pvalue

    return {
        "n": len(merged),
        "accuracy_a": merged["correct_a"].mean(),
        "accuracy_b": merged["correct_b"].mean(),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "discordant_pairs": discordant,
        "p_value": p_value,
    }


def compare_conditions(path_a, path_b, invalid_as_incorrect=False, label_a=None, label_b=None):
    label_a = label_a or path_a
    label_b = label_b or path_b

    df_a = load_predictions(path_a)
    df_b = load_predictions(path_b)

    merged = paired_correctness(df_a, df_b, invalid_as_incorrect=invalid_as_incorrect)
    result = mcnemar_test(merged)

    print(f"McNemar's test: {label_a}  vs  {label_b}")
    print(f"  Paired items compared: {result['n']}")
    print(f"  Accuracy ({label_a}): {result['accuracy_a']:.3f}")
    print(f"  Accuracy ({label_b}): {result['accuracy_b']:.3f}")
    print(f"  Both correct:          {result['both_correct']}")
    print(f"  Both wrong:            {result['both_wrong']}")
    print(f"  Only {label_a} correct: {result['a_only_correct']}")
    print(f"  Only {label_b} correct: {result['b_only_correct']}")
    print(f"  Discordant pairs:      {result['discordant_pairs']}")
    print(f"  p-value (two-sided):   {result['p_value']:.4f}")

    return result

#Added by Claude
def batch_mcnemar(predictions_dir, language, condition_a, condition_b, models, prompts, invalid_as_incorrect=False):
    """
    Runs McNemar's test for every (prompt, model) combination, all
    comparing the same condition_a vs condition_b (e.g. "0shot" vs
    "4shot") within one language. Returns one row per combination that
    actually has both files on disk; missing combinations are skipped
    with a printed note rather than raising, since a single failed
    model/prompt shouldn't block a whole table.
    """
    rows = []

    for prompt in prompts:
        for model in models:
            path_a = build_condition_path(predictions_dir, language, condition_a, prompt, model)
            path_b = build_condition_path(predictions_dir, language, condition_b, prompt, model)

            if not (os.path.exists(path_a) and os.path.exists(path_b)):
                print(f"Skipping {model} / {prompt}: missing prediction file(s) for {condition_a} or {condition_b}.")
                continue

            df_a = load_predictions(path_a)
            df_b = load_predictions(path_b)
            merged = paired_correctness(df_a, df_b, invalid_as_incorrect=invalid_as_incorrect)
            result = mcnemar_test(merged)

            rows.append({"prompt": prompt, "model": model, **result})

    return pd.DataFrame(rows)


def _significance_stars(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def format_results_table(results_df, condition_a_label, condition_b_label):
    """
    Reshapes a batch_mcnemar() result into a display-ready table:
    Prompt, Model, N, accuracy under each condition, and the p-value
    with conventional significance stars (*/**/*** at .05/.01/.001)
    appended -- star thresholds and their meaning should still be
    stated in the table's caption/footnote, since stars alone aren't
    self-explanatory to a reader.
    """
    table = results_df.copy()
    table["p-value"] = table["p_value"].apply(lambda p: f"{p:.4f}{_significance_stars(p)}")

    table = table.rename(columns={
        "prompt": "Prompt",
        "model": "Model",
        "n": "N",
        "accuracy_a": f"Acc. ({condition_a_label})",
        "accuracy_b": f"Acc. ({condition_b_label})",
    })

    return table[["Prompt", "Model", "N", f"Acc. ({condition_a_label})", f"Acc. ({condition_b_label})", "p-value"]]


def to_latex_table(display_df, condition_a_label, condition_b_label, caption=None, label=None):
    caption = caption or (
        f"McNemar's test comparing {condition_a_label} and {condition_b_label} "
        f"per model and prompt. * p<.05, ** p<.01, *** p<.001."
    )
    label = label or "tab:mcnemar"

    return display_df.to_latex(
        index=False,
        escape=True,
        float_format="%.3f",
        caption=caption,
        label=label,
    )

def main():
    parser = argparse.ArgumentParser(
        description="McNemar's test between two paired prediction files, "
                     "or a batch table across models/prompts."
    )
    parser.add_argument("path_a", nargs="?", help="First prediction .tsv (single-comparison mode)")
    parser.add_argument("path_b", nargs="?", help="Second prediction .tsv (single-comparison mode)")
    parser.add_argument(
        "--predictions-dir",
        help="Base predictions dir; build paths from --language/--prompt/"
             "--model/--condition-a/--condition-b instead of giving paths directly.",
    )
    parser.add_argument("--language")
    parser.add_argument("--prompt", help="Single-comparison mode: one prompt.")
    parser.add_argument("--model", help="Single-comparison mode: one model.")
    parser.add_argument("--condition-a", required=False)
    parser.add_argument("--condition-b", required=False)
    parser.add_argument(
        "--invalid-as-incorrect",
        action="store_true",
        help="Count an unparseable response as incorrect instead of excluding "
             "that statement from the comparison (the default).",
    )

    parser.add_argument(
        "--batch", action="store_true",
        help="Batch mode: run every model x prompt combination for one "
             "condition_a/condition_b pair and print/save one summary table.",
    )
    parser.add_argument("--models", help="Comma-separated model names (batch mode).")
    parser.add_argument("--prompts", help="Comma-separated prompt names (batch mode).")
    parser.add_argument("--latex-out", help="Path to write the LaTeX table to (batch mode).")
    parser.add_argument("--tsv-out", help="Path to write the raw results table to as TSV (batch mode).")

    args = parser.parse_args()

    if args.batch:
        needed = ["predictions_dir", "language", "condition_a", "condition_b", "models", "prompts"]
        missing = [name for name in needed if getattr(args, name) is None]
        if missing:
            parser.error(
                "--batch requires: "
                + ", ".join("--" + name.replace("_", "-") for name in missing)
            )

        models = [m.strip() for m in args.models.split(",")]
        prompts = [p.strip() for p in args.prompts.split(",")]

        results_df = batch_mcnemar(
            args.predictions_dir, args.language, args.condition_a, args.condition_b,
            models, prompts, invalid_as_incorrect=args.invalid_as_incorrect,
        )

        if results_df.empty:
            print("No comparisons could be run -- check paths, models, and prompts.")
            return

        display_df = format_results_table(results_df, args.condition_a, args.condition_b)
        print(display_df.to_string(index=False))

        if args.tsv_out:
            results_df.to_csv(args.tsv_out, sep="\t", index=False)
            print(f"Raw results saved to: {args.tsv_out}")

        if args.latex_out:
            latex = to_latex_table(display_df, args.condition_a, args.condition_b)
            os.makedirs(os.path.dirname(args.latex_out) or ".", exist_ok=True)
            with open(args.latex_out, "w", encoding="utf-8") as f:
                f.write(latex)
            print(f"LaTeX table saved to: {args.latex_out}")

        return

    if args.predictions_dir:
        needed = ["language", "prompt", "model", "condition_a", "condition_b"]
        missing = [name for name in needed if getattr(args, name) is None]
        if missing:
            parser.error(
                "--predictions-dir requires also specifying: "
                + ", ".join("--" + name.replace("_", "-") for name in missing)
            )
        path_a = build_condition_path(args.predictions_dir, args.language, args.condition_a, args.prompt, args.model)
        path_b = build_condition_path(args.predictions_dir, args.language, args.condition_b, args.prompt, args.model)
        label_a, label_b = args.condition_a, args.condition_b
    else:
        if not args.path_a or not args.path_b:
            parser.error(
                "Provide either two file paths, or --predictions-dir together "
                "with --language/--prompt/--model/--condition-a/--condition-b, "
                "or --batch for a full table."
            )
        path_a, path_b = args.path_a, args.path_b
        label_a, label_b = path_a, path_b

    compare_conditions(path_a, path_b, invalid_as_incorrect=args.invalid_as_incorrect, label_a=label_a, label_b=label_b)


if __name__ == "__main__":
    main()