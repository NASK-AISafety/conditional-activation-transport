import json
from typing import Dict, List
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split


def create_contrastive_dataset(
    prompts: List[Dict[str, str]],
    start_index: int = 0,
    label: bool = False,
) -> List[str]:
    """
    Create a contrastive dataset from the given prompts.
    """
    dataset = []
    labels = []
    prompts0 = []
    prompts1 = []
    i = start_index

    while i < len(prompts):
        if prompts[i]["label"] == 0:
            prompts0.append(prompts[i]["prompt"])
        else:
            prompts1.append(prompts[i]["prompt"])
        i += 1
    for i in range(len(prompts0)):
        dataset.append(prompts0[i])
        labels.append(0)
        dataset.append(prompts1[i])
        labels.append(1)
    if label:
        return (dataset, labels)
    return dataset


def handle_i2p_dataset(df, test_size, seed=42):
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    bad_prompts = train[
        train["prompt_toxicity"] >= train["prompt_toxicity"].quantile(0.9)
    ]
    bad_prompts_test = test[
        test["prompt_toxicity"] >= test["prompt_toxicity"].quantile(0.9)
    ]
    train = train.sort_values(by=["prompt_toxicity"])
    good_prompts = train[: len(bad_prompts)]
    train_prompts = []
    for _, row in good_prompts.iterrows():
        train_prompts.append({"prompt": row["prompt"], "label": 0})
    for _, row in bad_prompts.iterrows():
        train_prompts.append({"prompt": row["prompt"], "label": 1})
    test_prompts = []
    for _, row in bad_prompts_test.iterrows():
        test_prompts.append(row["prompt"])
    return train_prompts, test_prompts


def handle_copro_dataset(json_dict, test_size, seed=42, max_size=1000):
    correct_keys = [
        "ID_train_data",
        "ID_test_data",
        "OOD_test_data",
        "ID_valid_data",
        "OOD_valid_data",
    ]
    all_data = []
    for key in correct_keys:
        for row in json_dict[key]:
            all_data.append(row)
    all_data = all_data[:max_size]
    train, test = train_test_split(all_data, test_size=test_size, random_state=seed)
    train_prompts, test_prompts = [], []
    for row in train:
        train_prompts.append({"prompt": row["safe_prompt"], "label": 0})
        train_prompts.append({"prompt": row["unsafe_prompt"], "label": 1})
    for row in test:
        test_prompts.append(row["unsafe_prompt"])
    return train_prompts, test_prompts


def handle_cats_dogs_dataset(json_dict, test_size):
    split = int(test_size * len(json_dict))
    split = split if split % 2 == 0 else split + 1
    test = json_dict[:split]
    train = json_dict[split:]
    test = create_contrastive_dataset(test, start_index=0, label=False)
    return train, test


def handle_parquet_dataset(df_train, df_test, config):
    test_prompts = []
    train = df_train[df_train["category"].isin(config["TRAIN_CATEGORIES"])]
    train_safe = pd.DataFrame(
        {
            "text": train["safe_prompt"].astype(str),
            "label": 0,
            "split": "train",
            "category": train["category"].astype(str),
            "source": "safe",
            "pair_id": train.index.astype(int),
        }
    )
    train_unsafe = pd.DataFrame(
        {
            "text": train["unsafe_prompt"].astype(str),
            "label": 1,
            "split": "train",
            "category": train["category"].astype(str),
            "source": "unsafe",
            "pair_id": train.index.astype(int),
        }
    )
    train_prompts = pd.concat([train_safe, train_unsafe], ignore_index=True)

    test = df_test[df_test["category"].isin(config["TEST_CATEGORIES"])]
    for _, row in test.iterrows():
        test_prompts.append(row["unsafe_prompt"])
    return train_prompts, test_prompts


def get_dataset(config, test_size=0.2):
    name = config["DATASET"]
    path = config["DATASET_PATH"]
    path_test = config["TEST_DATASET_PATH"]

    if name == "i2p":
        df = pd.read_csv(path)
        train, test_dataset = handle_i2p_dataset(df=df, test_size=test_size, seed=42)
        train_dataset = create_contrastive_dataset(train, start_index=0, label=True)

    elif name == "copro":
        with open(path, "r") as f:
            prompts = json.load(f)
        train, test_dataset = handle_copro_dataset(
            json_dict=prompts,
            test_size=test_size,
            seed=42,
            max_size=1000,
        )
        train_dataset = create_contrastive_dataset(train, start_index=0, label=True)

    elif name == "SafeSteer":
        df_train = pd.read_parquet(path)
        df_test = pd.read_parquet(path_test)
        train_dataset, test_dataset = handle_parquet_dataset(df_train, df_test, config)

    else:
        with open(path, "r") as f:
            prompts = json.load(f)
        train, test_dataset = handle_cats_dogs_dataset(prompts, test_size=test_size)
        train_dataset = create_contrastive_dataset(train, start_index=0, label=True)
    return train_dataset, test_dataset


def split_dataset_by_categories(config):
    path = config["DATASET_PATH"]
    df = pd.read_parquet(path)
    datasets = defaultdict(list)
    for category in pd.unique(df["category"]):
        for _, row in df[df["category"] == category].iterrows():
            datasets[category].append({"prompt": row["safe_prompt"], "label": 0})
            datasets[category].append({"prompt": row["unsafe_prompt"], "label": 1})
        datasets[category] = create_contrastive_dataset(
            datasets[category], start_index=0, label=True
        )
    datasets["All"] = create_contrastive_dataset(
        [{"prompt": row["safe_prompt"], "label": 0} for _, row in df.iterrows()]
        + [{"prompt": row["unsafe_prompt"], "label": 1} for _, row in df.iterrows()],
        start_index=0,
        label=True,
    )
    return datasets
