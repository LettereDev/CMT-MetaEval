# src/inference.py
import os
import random

import pandas as pd
import torch

from .config import GENERATION_CONFIG, SPLIT_CONFIG
from .parser import parse_prediction
from .prompts import format_examples, build_prompt
from .datasets import load_lang_dataset


def generate_response(model, tokenizer, prompt):
    """
    Generate a response from the currently loaded model.
    """

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            inputs,
            **GENERATION_CONFIG,
        )

    generated_tokens = outputs[0][inputs.shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()


def get_support_and_query(test_lang, split_seed):
    """
    Loads the full dataset for a language (via load_lang_dataset, which
    already handles combining COMETA+MIST for DE or using CoMeta for ES)
    and splits it into:
      - a support pool, from which few-shot examples are drawn
      - a query set, which is what gets evaluated

    The split is deterministic given split_seed, so the same query set
    is used across all shot_count / support_seed conditions for a
    given language.
    """
    dataset = load_lang_dataset(test_lang)

    support_fraction = SPLIT_CONFIG["support_fraction"]
    support_size = int(len(dataset) * support_fraction)

    rng = random.Random(split_seed)
    support_indices = rng.sample(range(len(dataset)), support_size)

    support_df = dataset.iloc[support_indices].reset_index(drop=True)
    query_df = dataset.drop(index=support_indices).reset_index(drop=True)

    return support_df, query_df


def sample_support_examples(support_df, shot_count, support_seed):
    """
    Draws `shot_count` examples from the support pool using `support_seed`,
    so different seeds yield different few-shot example sets for the same
    shot_count. Returns an empty (but correctly-shaped) DataFrame for
    zero-shot.
    """
    if shot_count == 0 or len(support_df) == 0:
        return support_df.iloc[0:0]

    rng = random.Random(support_seed)
    indices = rng.sample(
        range(len(support_df)),
        min(shot_count, len(support_df))
    )

    return support_df.iloc[indices].reset_index(drop=True)


def run_experiment(
    model,
    tokenizer,
    prompt,
    prompt_name,
    model_name,
    test_lang,
    shot_count,
    split_seed,
    support_seed,
    output_dir="results/predictions"
):
    """
    Run one model + one prompt + one shot_count/support_seed condition,
    for one language (test_lang is a key in config.LANGUAGES, e.g. "DE"/"ES").
    """

    print(
        f"Running: "
        f"{model_name} | "
        f"{test_lang} | "
        f"{prompt_name} | "
        f"{shot_count}-shot"
        + (f" | seed={support_seed}" if shot_count > 0 else "")
    )

    # Build the support pool and the held-out query (evaluation) set
    support_df, query_df = get_support_and_query(test_lang, split_seed)

    # Draw few-shot examples from the support pool (empty for zero-shot)
    support_examples = sample_support_examples(
        support_df,
        shot_count,
        support_seed
    )

    results = []

    for row_index, row in enumerate(
        query_df.itertuples(index=False),
        start=1
    ):

        statement = row.statement
        gold_label = int(row.isMetaphor)  # type: ignore[arg-type]

        # Preserve which underlying dataset the row came from
        # (COMETA / MIST / CoMeta), added by datasets.load_dataset
        dataset_name = getattr(row, "source", test_lang)

        # Insert statement (and, for few-shot, examples) into prompt
        formatted_prompt = build_prompt(
            prompt,
            examples=support_examples,
            statement=statement
        )

        # Query model
        response = generate_response(
            model,
            tokenizer,
            formatted_prompt
        )

        # Convert response to 0/1/None
        prediction = parse_prediction(response)

        results.append({
            "statement": statement,
            "response": response,
            "prediction": prediction,
            "gold_label": gold_label,
            "dataset": dataset_name,
        })

        # Progress information
        if row_index % 10 == 0:
            print(
                f"  Processed "
                f"{row_index}/{len(query_df)}"
            )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Count invalid responses
    invalid_count = results_df["prediction"].isna().sum()
    total_count = len(results_df)

    print(
        f"Invalid responses: "
        f"{invalid_count}/{total_count} "
        f"({invalid_count / total_count:.2%})"
    )

    # One directory per experimental condition, so this lines up with
    # evaluation.evaluate_all_predictions(), which expects
    # predictions_dir/<experiment>/<prompt>_<model>.tsv
    condition_dir = f"{test_lang}_{shot_count}shot"
    if shot_count > 0:
        condition_dir += f"_seed{support_seed}"

    output_path = os.path.join(
        output_dir,
        condition_dir,
    )

    os.makedirs(
        output_path,
        exist_ok=True
    )

    # Save results
    filename = f"{prompt_name}_{model_name}.tsv"

    filepath = os.path.join(
        output_path,
        filename
    )

    results_df.to_csv(
        filepath,
        sep="\t",
        index=False
    )

    print(f"Saved: {filepath}")

    return results_df