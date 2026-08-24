# src/evaluation.py

import os
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def calculate_metrics(df):
    """Calculate metrics for one prediction DataFrame."""

    total = len(df)

    # Exclude invalid predictions
    valid_df = df[df["prediction"].notna()].copy()

    invalid_count = len(df) - len(valid_df)

    if len(valid_df) == 0:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "invalid_count": invalid_count,
            "invalid_rate": 1.0,
        }

    y_true = valid_df["gold_label"].astype(int)
    y_pred = valid_df["prediction"].astype(int)
    
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "invalid_count": invalid_count,

        "invalid_rate": invalid_count / total,
        
        "confusion_matrix": cm
    }


def evaluate_file(
    filepath,
    model_name,
    prompt_name,
    dataset_name
):
    """Evaluate one prediction TSV."""

    df = pd.read_csv(
        filepath,
        sep="\t"
    )

    metrics = calculate_metrics(df)

    metrics.update({
        "dataset": dataset_name,
        "prompt": prompt_name,
        "model": model_name,
        "n": len(df),
    })

    return metrics


def evaluate_all_predictions(
    predictions_dir="results/predictions",
    output_path="results/metrics/all_metrics.tsv"
):
    """Evaluate every prediction TSV."""

    all_metrics = []

    for dataset_name in os.listdir(predictions_dir):

        dataset_dir = os.path.join(
            predictions_dir,
            dataset_name
        )

        if not os.path.isdir(dataset_dir):
            continue

        for filename in os.listdir(dataset_dir):

            if not filename.endswith(".tsv"):
                continue

            filepath = os.path.join(
                dataset_dir,
                filename
            )

            # Example:
            # P1_Qwen3-8B.tsv
            name = filename[:-4]

            prompt_name, model_name = name.split(
                "_",
                maxsplit=1
            )

            metrics = evaluate_file(
                filepath,
                model_name,
                prompt_name,
                dataset_name
            )

            all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)

    # Print confusion matrices before selecting the columns written to TSV.
    for metrics in all_metrics:
        print(
            f"Confusion Matrix for {metrics['model']} | "
            f"{metrics['prompt']} | {metrics['dataset']}:"
        )
        print(metrics["confusion_matrix"])

    metrics_df = metrics_df[
        [
            "dataset",
            "prompt",
            "model",
            "n",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "invalid_count",
            "invalid_rate",
        ]
    ]

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    metrics_df.to_csv(
        output_path,
        sep="\t",
        index=False
    )

    print(f"Metrics saved to: {output_path}")

    return metrics_df