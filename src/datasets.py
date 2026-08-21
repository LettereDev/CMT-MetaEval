import pandas as pd


def load_dataset(path):
    df = pd.read_csv(path, sep="\t")

    return df[["statement", "label"]]


def combine_datasets(dataset_paths):
    frames = []

    for dataset_name, path in dataset_paths.items():
        df = load_dataset(path)
        df["dataset"] = dataset_name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)