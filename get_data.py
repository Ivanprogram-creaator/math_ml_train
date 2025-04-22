import os

import pandas as pd


def get_data():
    df = pd.read_excel('data.xlsx')
    df = df.drop(['Unnamed: 0'], axis=1)
    len_df = len(df.index)
    train_data = df.iloc[df.index <= len_df * 0.8]
    test_data = df.loc[df.index >= len_df * 0.8]
    sharded_train_paths = split_dataset(train_data, "train_data", 10)
    return train_data, test_data, sharded_train_paths


def split_dataset(
        dataset: pd.DataFrame, tmp_dir: str, num_shards: int
) -> list[str]:
    """Splits a csv file into multiple csv files."""

    os.makedirs(tmp_dir, exist_ok=True)
    num_row_per_shard = (dataset.shape[0] + num_shards - 1) // num_shards
    paths = []
    for shard_idx in range(num_shards):
        begin_idx = shard_idx * num_row_per_shard
        end_idx = (shard_idx + 1) * num_row_per_shard
        shard_dataset = dataset.iloc[begin_idx:end_idx]
        shard_path = os.path.join(tmp_dir, f"shard_{shard_idx}.csv")
        paths.append(shard_path)
        shard_dataset.to_csv(shard_path, index=False)
    return paths
