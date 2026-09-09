# src/inference.py
import os
import random

import pandas as pd
import torch

from .config import GENERATION_CONFIG, FEWSHOT_2SHOT_SEED, FEWSHOT_4SHOT_SEED
from .parser import parse_prediction
from .prompts import format_examples, build_prompt
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

#Added by Claude
def _draw_one(rng, pool):
    """
    Draws one row from `pool` (a DataFrame with a fresh 0..n-1 index)
    using `rng`. Returns (the row as a 1-row DataFrame, its position),
    so the caller can exclude that position from a later draw.
    """
    if len(pool) == 0:
        raise ValueError("Cannot draw from an empty pool.")

    position = rng.randrange(len(pool))
    return pool.iloc[[position]], position

#Modified with Claude and peronal reviews/additions
def build_fewshot_examples(pair_seed=FEWSHOT_2SHOT_SEED, four_shot_seed=FEWSHOT_4SHOT_SEED):
    """
    Selects, once and deterministically, the fixed cross-lingual few-shot
    examples used identically across every model/prompt/language.
    Label-stratified throughout, so every condition shows both classes
    rather than risking an all-one-label draw (verified possible with
    unconstrained random sampling on this project's actual data).

      - 2-shot pairs: the FIRST example presented is always
        metaphorical, the SECOND is always non-metaphorical, with
        language and label tied together:
          "2shot_DE_ES": [DE metaphorical, ES non-metaphorical]
          "2shot_ES_DE": [ES metaphorical, DE non-metaphorical]
        4 distinct sentences total (1 DE-pos, 1 DE-neg, 1 ES-pos, 1 ES-neg).
      - 4-shot: a separate, independent draw (its own seed), stratified
        the same way (1 more DE-pos, 1 DE-neg, 1 ES-pos, 1 ES-neg),
        excluding the four sentences already used in the pairs above.
      - 0-shot uses no examples.

    Returns a list of (shot_count, condition_label, examples_df) tuples,
    and prints exactly which sentences (and labels) were selected.
    """
    de_support = load_support_pool("DE")
    es_support = load_support_pool("ES")

    de_pos = de_support[de_support["isMetaphor"] == 1].reset_index(drop=True)
    de_neg = de_support[de_support["isMetaphor"] == 0].reset_index(drop=True)
    es_pos = es_support[es_support["isMetaphor"] == 1].reset_index(drop=True)
    es_neg = es_support[es_support["isMetaphor"] == 0].reset_index(drop=True)

    pair_rng = random.Random(pair_seed)

    de_pos_a, de_pos_a_pos = _draw_one(pair_rng, de_pos)
    de_neg_a, de_neg_a_pos = _draw_one(pair_rng, de_neg)
    es_pos_a, es_pos_a_pos = _draw_one(pair_rng, es_pos)
    es_neg_a, es_neg_a_pos = _draw_one(pair_rng, es_neg)

    two_shot_de_es = pd.concat([de_pos_a, es_neg_a], ignore_index=True)
    two_shot_es_de = pd.concat([es_pos_a, de_neg_a], ignore_index=True)

    # Independent 4-shot draw, same stratification, excluding the
    # sentences already used in the pairs above.
    four_shot_rng = random.Random(four_shot_seed)

    de_pos_remaining = de_pos.drop(index=de_pos_a_pos).reset_index(drop=True)
    de_neg_remaining = de_neg.drop(index=de_neg_a_pos).reset_index(drop=True)
    es_pos_remaining = es_pos.drop(index=es_pos_a_pos).reset_index(drop=True)
    es_neg_remaining = es_neg.drop(index=es_neg_a_pos).reset_index(drop=True)

    de_pos_b, _ = _draw_one(four_shot_rng, de_pos_remaining)
    de_neg_b, _ = _draw_one(four_shot_rng, de_neg_remaining)
    es_pos_b, _ = _draw_one(four_shot_rng, es_pos_remaining)
    es_neg_b, _ = _draw_one(four_shot_rng, es_neg_remaining)

    four_shot = pd.concat(
        [de_pos_b, de_neg_b, es_pos_b, es_neg_b],
        ignore_index=True
    )

    empty_examples = de_support.iloc[0:0]

    conditions = [
        (0, "0shot", empty_examples),
        (2, "2shot_DE_ES", two_shot_de_es),
        (2, "2shot_ES_DE", two_shot_es_de),
        (4, "4shot", four_shot),
    ]

    print(
        f"Fixed few-shot examples selected "
        f"(pair_seed={pair_seed}, four_shot_seed={four_shot_seed}):"
    )
    print(
        f"  2shot_DE_ES -> DE [metaphor]: \"{de_pos_a.iloc[0].statement}\" | "
        f"ES [non-metaphor]: \"{es_neg_a.iloc[0].statement}\""
    )
    print(
        f"  2shot_ES_DE -> ES [metaphor]: \"{es_pos_a.iloc[0].statement}\" | "
        f"DE [non-metaphor]: \"{de_neg_a.iloc[0].statement}\""
    )
    print(
        f"  4shot -> DE [metaphor]: \"{de_pos_b.iloc[0].statement}\", "
        f"DE [non-metaphor]: \"{de_neg_b.iloc[0].statement}\" | "
        f"ES [metaphor]: \"{es_pos_b.iloc[0].statement}\", "
        f"ES [non-metaphor]: \"{es_neg_b.iloc[0].statement}\""
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