# src/datasets.py
import pandas as pd

from .config import DATASETS, LANGUAGES


def load_dataset(path, source, language):
    df = pd.read_csv(path, sep="\t")
    required_columns = {"statement", "isMetaphor"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f" The path {path} is missing required columns: {sorted(missing_columns)}")

    df = df[["statement", "isMetaphor"]].copy()
    df["source"] = source
    df["language"] = language
    df["row_id"] = [f"{source}:{index}" for index in df.index]

    return df


def combine_datasets(dataset_names):
    """
    Combine multiple named datasets (keys into DATASETS, e.g. "COMETA",
    "MIST") into a single DataFrame, tagging each row with its source
    dataset and language.
    """
    frames = []

    for name in dataset_names:
        if name not in DATASETS:
            raise ValueError(f"Dataset '{name}' is not defined in DATASETS.")

        language = next(
            (lang for lang, names in LANGUAGES.items() if name in names),
            None
        )
        if language is None:
            raise ValueError(f"Dataset '{name}' not found in LANGUAGES mapping.")

        df = load_dataset(DATASETS[name], source=name, language=language)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def load_lang_dataset(language):
    """
    Load (and combine, if there's more than one) all datasets belonging
    to one language, as defined in LANGUAGES.
    """
    if language not in LANGUAGES:
        raise ValueError(f"Language '{language}' is not supported. Supported languages: {list(LANGUAGES.keys())}")

    return combine_datasets(LANGUAGES[language])