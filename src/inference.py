import os
import pandas as pd
import torch

from config import GENERATION_CONFIG

from .parser import parse_prediction
from prompts import format_prompt


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


def run_experiment(
    model,
    tokenizer,
    dataset,
    dataset_name,
    prompt,
    prompt_name,
    model_name,
    experiment_name,
    output_dir="results/predictions"
):
    """
    Run one model + one prompt + one dataset combination.
    """

    print(
        f"Running: "
        f"{model_name} | "
        f"{prompt_name} | "
        f"{dataset_name}"
    )

    results = []

    for index, row in dataset.iterrows():

        statement = row["statement"]
        gold_label = int(row["isMetaphor"])

        # Insert statement into prompt
        formatted_prompt = format_prompt(
            prompt,
            statement
        )

        # Query model
        response = generate_response(
            model,
            tokenizer,
            formatted_prompt
        )

        # Convert response to 0/1
        prediction = parse_prediction(response)

        results.append({
            "statement": statement,
            "response": response,
            "prediction": prediction,
            "gold_label": gold_label,
        })

        # Progress information
        if (index + 1) % 10 == 0:
            print(
                f"  Processed {index + 1}/{len(dataset)}"
            )
            
    results_df = pd.DataFrame(results)

    invalid_count = results_df["prediction"].isna().sum()
    total_count = len(results_df)

    print(
        f"Invalid responses: "
        f"{invalid_count}/{total_count} "
        f"({invalid_count / total_count:.2%})"
    )

    # Create output directory
    output_path = os.path.join(
        output_dir,
        dataset_name
    )

    os.makedirs(
        output_path,
        exist_ok=True
    )

    # Save results
    filename = (
        f"{prompt_name}_{model_name}.tsv"
    )

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