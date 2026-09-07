# src/inference.py
import os
import random

import pandas as pd
import torch

from .config import GENERATION_CONFIG, FEWSHOT_2SHOT_SEED, FEWSHOT_4SHOT_SEED
from .parser import parse_prediction
from .prompts import build_prompt
from .split import OUTPUT_DIR


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
        return_dict=True,
    )

    inputs = inputs.to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            **GENERATION_CONFIG,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()


def load_support_pool(language):
    """Loads the static support pool for one language, written by split.py."""
    support_path = os.path.join(OUTPUT_DIR, f"{language}_support.tsv")

    if not os.path.exists(support_path):
        raise FileNotFoundError(
            f"Missing support file for '{language}': expected {support_path}. "
            f"Run `python -m src.split` first to generate it."
        )

    return pd.read_csv(support_path, sep="\t")


def load_query_set(test_lang):
    """Loads the static query (evaluation) set for a language, written by split.py."""
    query_path = os.path.join(OUTPUT_DIR, f"{test_lang}_query.tsv")

    if not os.path.exists(query_path):
        raise FileNotFoundError(
            f"Missing query file for '{test_lang}': expected {query_path}. "
            f"Run `python -m src.split` first to generate it."
        )

    return pd.read_csv(query_path, sep="\t")

#Function made with Claude
def build_fewshot_examples(pair_seed=FEWSHOT_2SHOT_SEED, four_shot_seed=FEWSHOT_4SHOT_SEED):
    """
    Selects, once and deterministically, the fixed cross-lingual few-shot
    examples used identically across every model/prompt/language:

      - Two DE support examples (de_a, de_b) and two ES support examples
        (es_a, es_b) are drawn with `pair_seed`, forming the 2-shot pairs:
          "2shot_DE_ES": [de_a, es_a]  (DE example presented first)
          "2shot_ES_DE": [es_b, de_b]  (ES example presented first)
        4 distinct sentences total across the two pairs.
      - A separate draw of 2 more DE + 2 more ES examples is made with
        `four_shot_seed`, excluding the four sentences already used
        above, so the 4-shot condition is fully independent of (shares
        no sentences with) the 2-shot pairs:
          "4shot": [de_c1, es_c1, de_c2,  es_c2]  (2 DE + 2 ES)
      - 0-shot uses no examples.

    Returns a list of (shot_count, condition_label, examples_df) tuples,
    and prints exactly which sentences were selected for each condition.
    """
    de_pool = load_support_pool("DE")
    es_pool = load_support_pool("ES")

    pair_rng = random.Random(pair_seed)
    de_pair_indices = pair_rng.sample(range(len(de_pool)), 2)
    es_pair_indices = pair_rng.sample(range(len(es_pool)), 2)

    de_a = de_pool.iloc[[de_pair_indices[0]]]
    de_b = de_pool.iloc[[de_pair_indices[1]]]
    es_a = es_pool.iloc[[es_pair_indices[0]]]
    es_b = es_pool.iloc[[es_pair_indices[1]]]

    # Independent draw for 4-shot: exclude the indices already used
    # above so no sentence appears in both the 2-shot and 4-shot
    # conditions.
    four_shot_rng = random.Random(four_shot_seed)
    de_remaining = [i for i in range(len(de_pool)) if i not in de_pair_indices]
    es_remaining = [i for i in range(len(es_pool)) if i not in es_pair_indices]

    de_four_indices = four_shot_rng.sample(de_remaining, 2)
    es_four_indices = four_shot_rng.sample(es_remaining, 2)

    de_c1 = de_pool.iloc[[de_four_indices[0]]]
    de_c2 = de_pool.iloc[[de_four_indices[1]]]
    es_c1 = es_pool.iloc[[es_four_indices[0]]]
    es_c2 = es_pool.iloc[[es_four_indices[1]]]

    empty_examples = de_pool.iloc[0:0]

    conditions = [
        (0, "0shot", empty_examples),
        (2, "2shot_DE_ES", pd.concat([de_a, es_a], ignore_index=True)),
        (2, "2shot_ES_DE", pd.concat([es_b, de_b], ignore_index=True)),
        (4, "4shot", pd.concat([de_c1, es_c1, de_c2,  es_c2], ignore_index=True)),
    ]

    print(
        f"Fixed few-shot examples selected "
        f"(pair_seed={pair_seed}, four_shot_seed={four_shot_seed}):"
    )
    print(f"  2shot_DE_ES -> DE: \"{de_a.iloc[0].statement}\" | ES: \"{es_a.iloc[0].statement}\"")
    print(f"  2shot_ES_DE -> ES: \"{es_b.iloc[0].statement}\" | DE: \"{de_b.iloc[0].statement}\"")
    print(
        f"  4shot -> DE: \"{de_c1.iloc[0].statement}\", \"{de_c2.iloc[0].statement}\" | "
        f"ES: \"{es_c1.iloc[0].statement}\", \"{es_c2.iloc[0].statement}\""
    )

    return conditions


def run_experiment(
    model,
    tokenizer,
    prompt,
    prompt_name,
    model_name,
    test_lang,
    shot_count,
    condition_label,
    examples,
    output_dir="results/predictions"
):
    """
    Run one model + one prompt + one fixed few-shot condition
    (condition_label, e.g. "0shot", "2shot_DE_ES", "2shot_ES_DE", "4shot"),
    evaluated against test_lang's query set. `examples` is the fixed
    DataFrame of few-shot examples to insert into the prompt (empty for
    zero-shot) -- built once by build_fewshot_examples() and reused
    identically across every model/prompt/language.
    """

    print(
        f"Running: "
        f"{model_name} | "
        f"{test_lang} | "
        f"{prompt_name} | "
        f"{condition_label}"
    )

    # One directory per experimental condition, so this lines up with
    # evaluation.evaluate_all_predictions(), which expects
    # predictions_dir/<experiment>/<prompt>_<model>.tsv
    condition_dir = f"{test_lang}_{condition_label}"

    output_path = os.path.join(
        output_dir,
        condition_dir,
    )

    filename = f"{prompt_name}_{model_name}.tsv"
    filepath = os.path.join(output_path, filename)

    # Resume support: if this exact condition already produced a result
    # file (e.g. a previous run was interrupted partway through), skip
    # redoing it and just return what's already on disk.
    if os.path.exists(filepath):
        print(f"Already exists, skipping: {filepath}")
        return pd.read_csv(filepath, sep="\t")

    os.makedirs(output_path, exist_ok=True)

    query_df = load_query_set(test_lang)

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
            examples=examples,
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
            "shot_count": shot_count,
            "condition": condition_label,
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

    results_df.to_csv(
        filepath,
        sep="\t",
        index=False
    )

    print(f"Saved: {filepath}")

    return results_df