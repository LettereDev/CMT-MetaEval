# src/split.py
import pandas as pd
import random

from .datasets import combine_datasets
from .config import DATASETS, SPLIT_CONFIG

# Load DE and ES datasets
de_dataset = combine_datasets({DATASETS["COMETA"], DATASETS["MIST"]})
es_dataset = pd.read_csv(DATASETS["CoMeta"], sep="\t")

OUTPUT_DIR = "data/processed"

# Split datasets into support and query sets
def split_dataset(dataset):
    random.seed(SPLIT_CONFIG["split_seed"])
    support_size = int(len(dataset) * SPLIT_CONFIG["support_fraction"])
    support_indices = random.sample(range(len(dataset)), support_size)
    support_dataset = dataset.iloc[support_indices]
    query_dataset = dataset.drop(support_indices)
    return support_dataset, query_dataset

def stratify(dataset, label_column):
    stratified = {}
    for label in dataset[label_column].unique():
        stratified[label] = dataset[dataset[label_column] == label]
    return stratified

def save_split (support_dataset, query_dataset, support_path, query_path):
    support_dataset.to_csv(support_path, sep="\t", index=False)
    query_dataset.to_csv(query_path, sep="\t", index=False)