# src/split.py
import os
import random

from .config import LANGUAGES, SPLIT_CONFIG
from .datasets import load_lang_dataset

OUTPUT_DIR = "data/processed/splits"


def split_dataset(dataset, split_seed):
    """
    Split a dataset into a support set (for drawing few-shot examples)
    and a query set (the held-out evaluation set), using a local RNG so
    this doesn't disturb global random state.
    """
    rng = random.Random(split_seed)

    support_size = int(len(dataset) * SPLIT_CONFIG["support_fraction"])
    support_indices = rng.sample(range(len(dataset)), support_size)

    support_dataset = dataset.iloc[support_indices].reset_index(drop=True)
    query_dataset = dataset.drop(index=support_indices).reset_index(drop=True)

    return support_dataset, query_dataset


def stratify(dataset, label_column):
    """Group rows of a dataset by their label value."""
    return {
        label: dataset[dataset[label_column] == label]
        for label in dataset[label_column].unique()
    }


def save_split(support_dataset, query_dataset, support_path, query_path):
    os.makedirs(os.path.dirname(support_path), exist_ok=True)
    os.makedirs(os.path.dirname(query_path), exist_ok=True)

    support_dataset.to_csv(support_path, sep="\t", index=False)
    query_dataset.to_csv(query_path, sep="\t", index=False)


def main():
    for language in LANGUAGES:

        dataset = load_lang_dataset(language)

        support_dataset, query_dataset = split_dataset(
            dataset,
            SPLIT_CONFIG["split_seed"]
        )

        support_path = os.path.join(OUTPUT_DIR, f"{language}_support.tsv")
        query_path = os.path.join(OUTPUT_DIR, f"{language}_query.tsv")

        save_split(
            support_dataset,
            query_dataset,
            support_path,
            query_path
        )

        print(
            f"{language}: {len(support_dataset)} support / "
            f"{len(query_dataset)} query rows "
            f"(saved to {support_path}, {query_path})"
        )


if __name__ == "__main__":
    main()