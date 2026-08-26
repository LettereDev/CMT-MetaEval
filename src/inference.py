import os
import pandas as pd
import torch

from .config import GENERATION_CONFIG, EXPERIMENTS, DATASETS
from .parser import parse_prediction
from .prompts import format_prompt
from .datasets import load_dataset, combine_datasets


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


def select_dataset(experiment_name):
    '''
    Selects the corresponding dataset according to the current experiment settings, returns a combined dataset if the experiment requires it.
    '''
    dataset_names = EXPERIMENTS[experiment_name]

    # Single dataset
    if len(dataset_names) == 1:

        dataset_name = dataset_names[0]

        return load_dataset(
            DATASETS[dataset_name]
        )

    # Multiple datasets
    selected_paths = {
        name: DATASETS[name]
        for name in dataset_names
    }

    return combine_datasets(
        selected_paths
    )
    

def run_experiment(
    model,
    tokenizer,
    prompt,
    prompt_name,
    model_name,
    experiment_name,
    output_dir="results/predictions"
):
    """
    Run one model + one prompt + one experiment.
    """

    print(
        f"Running: "
        f"{model_name} | "
        f"{experiment_name} | "
        f"{prompt_name}"
    )

    # Select the appropriate dataset or dataset combination
    dataset = select_dataset(experiment_name)

    results = []

    for row_index, row in enumerate(
        dataset.itertuples(index=False),
        start=1
    ):

        statement = row.statement
        gold_label = int(row.isMetaphor) # type: ignore[arg-type]

        # Preserve source dataset if available
        dataset_name = getattr(
            row,
            "dataset",
            experiment_name
        )

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
                f"{row_index}/{len(dataset)}"
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

    # Create output directory
    output_path = os.path.join(
        output_dir,
        experiment_name
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