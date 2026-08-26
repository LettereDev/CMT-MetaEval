"""
Preprocessing script for all the datasets in this repository.

Shuffles the rows of the csv for cometa and mist, that were made by extracting the sentences as statement and the metaphor labels were converted to a binary basis (0 and 1).
Also converts the csv files to tsv files.
Converts CoMeta token-level TSV files into one sentence-level dataset.

Usage (from the project root):
    .\\.venv\\Scripts\\python.exe src\\preprocess.py

Note: remove second escape character before running the command on Windows.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
import random


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH_COMETA_ES = PROJECT_ROOT / "data" / "processed" / "cometa_es.tsv"
SPLITS = {
    "train": RAW_DIR / "cometa_train.tsv",
    "test": RAW_DIR / "cometa_test.tsv",
}
CSVS = {
    "cometa": RAW_DIR / "cometa.csv",
    "mist": RAW_DIR / "mist.csv",
}

#Code generated with CODEX to format the token-level CoMeta dataset into a sentence-level dataset. Each sentence is labeled as 1 if it contains any non-O tags, and 0 otherwise. The token-level tags are also included for optional provenance.
#Source: https://chatgpt.com/s/cx_6a8853f043c881918a27d822c9a157ee
def detokenize(tokens: list[str]) -> str:
    """Rejoin CoNLL tokens into readable Spanish text."""
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%\)\]\}»])", r"\1", text)
    text = re.sub(r"([¿¡\(\[\{«])\s+", r"\1", text)
    text = text.replace(" n't", "n't")
    return text


def iter_sentences(path: Path):
    """Yield (tokens, tags) from a blank-line-separated CoNLL TSV file."""
    tokens: list[str] = []
    tags: list[str] = []

    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            line = line.rstrip("\r\n")
            if not line.strip():
                if tokens:
                    yield tokens, tags
                    tokens, tags = [], []
                continue

            try:
                token, tag = line.rsplit("\t", maxsplit=1)
            except ValueError as error:
                raise ValueError(f"Malformed row in {path.name}, line {line_number}") from error

            tokens.append(token)
            tags.append(tag)

    if tokens:
        yield tokens, tags

#Following functions were not generated in the previous conversation    
def shuffle_rows(rows: list[dict[str, str | int]]) -> list[dict[str, str | int]]:
    """Shuffle the rows of a dataset."""
    random.seed(42) #Added seed for reproducibility
    random.shuffle(rows)
    return rows

def write_tsv(rows: list[dict[str, str | int]], output_path: Path) -> None:
    """Write the rows to a TSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=rows[0].keys(),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def clean_cometa_es_rows(rows: list[dict[str, str | int]]) -> list[dict[str, str | int]]:
    """Remove empty placeholders and duplicate statement-label pairs."""
    cleaned_rows: list[dict[str, str | int]] = []
    seen: set[tuple[str, int]] = set()

    for row in rows:
        statement = str(row["statement"]).strip()
        is_metaphor = int(row["isMetaphor"])
        # CoMeta represents an empty sentence with a lone hyphen.
        if not statement or statement == "-":
            continue

        key = (statement, is_metaphor)
        if key in seen:
            continue

        seen.add(key)
        cleaned_rows.append({"statement": statement, "isMetaphor": is_metaphor})

    return cleaned_rows


def main() -> None:
    rows: list[dict[str, str | int]] = []

    for split, path in SPLITS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} input file: {path}")

        for tokens, tags in iter_sentences(path):
            rows.append(
                {
                    "statement": detokenize(tokens),
                    # A sentence is metaphorical when it has >= 1 metaphor-tagged token.
                    "isMetaphor": int(any(tag != "O" for tag in tags)),
                }
            )

    rows = clean_cometa_es_rows(rows)

    OUTPUT_PATH_COMETA_ES.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH_COMETA_ES.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=["statement", "isMetaphor"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    positives = sum(row["isMetaphor"] for row in rows) # type: ignore
    print(f"Wrote {len(rows)} sentences to {OUTPUT_PATH_COMETA_ES}")
    print(f"Metaphorical: {positives}; non-metaphorical: {len(rows) - positives}")
    
    #Preprocessing COMETA and MIST
    cometa = shuffle_rows(list(csv.DictReader(CSVS["cometa"].open(encoding="utf-8"), delimiter=",")))
    mist = shuffle_rows(list(csv.DictReader(CSVS["mist"].open(encoding="utf-8"), delimiter=",")))
    
    cometa_output_path = PROJECT_ROOT / "data" / "processed" / "cometa.tsv"
    mist_output_path = PROJECT_ROOT / "data" / "processed" / "mist.tsv"
    
    write_tsv(cometa, cometa_output_path)
    write_tsv(mist, mist_output_path)
    print(f"Wrote {len(cometa)} rows to {cometa_output_path}")
    print(f"Wrote {len(mist)} rows to {mist_output_path}")


if __name__ == "__main__":
    main()
