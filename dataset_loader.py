import json
import os
import pandas as pd
from datasets import load_dataset

INTENT2ID = {}
ID2INTENT = {}


def register_intents(intent_series):
    for intent in sorted(intent_series.unique()):
        intent_str = str(intent)
        if intent_str not in INTENT2ID:
            new_id = len(INTENT2ID)
            INTENT2ID[intent_str] = new_id
            ID2INTENT[new_id] = intent_str


def load_banking77():
    print("Loading Banking77...")
    ds = load_dataset("legacy-datasets/banking77")
    df = pd.concat([ds["train"].to_pandas(), ds["test"].to_pandas()])
    label_names = ds["train"].features["label"].names
    df["intent"] = df["label"].apply(lambda x: label_names[x])

    register_intents(df["intent"])
    df["label"] = df["intent"].map(INTENT2ID)
    return df[["text", "label"]]


def load_clinc150():
    print("Loading CLINC150...")
    ds = load_dataset("contemmcm/clinc150")
    df = ds["complete"].to_pandas()

    register_intents(df["intent"])
    df["label"] = df["intent"].map(INTENT2ID)
    return df[["text", "label"]]


def load_snips():
    print("Loading SNIPS...")
    # FIXED: Using legacy-datasets namespace
    ds = load_dataset("legacy-datasets/snips_built_in_intents")
    df = pd.concat([ds["train"].to_pandas(), ds["test"].to_pandas()])

    register_intents(df["intent"])
    df["label"] = df["intent"].map(INTENT2ID)
    return df[["text", "label"]]


def load_atis():
    print("Loading ATIS...")
    ds = load_dataset("tuetschek/atis")
    df = pd.concat([ds["train"].to_pandas(), ds["test"].to_pandas()])

    register_intents(df["intent"])
    df["label"] = df["intent"].map(INTENT2ID)
    return df[["text", "label"]]


def load_massive():
    print("Loading Amazon MASSIVE...")
    # FIXED: trust_remote_code=True allows script execution for legacy datasets
    # Alternatively, use "mteb/amazon_massive" if available
    try:
        ds = load_dataset(
            "AmazonScience/massive", "en-US", trust_remote_code=True
        )
    except Exception:
        ds = load_dataset("mteb/amazon_massive", "en-US")

    df = pd.concat(
        [
            ds["train"].to_pandas(),
            ds["validation"].to_pandas(),
            ds["test"].to_pandas(),
        ]
    )

    if "utt" in df.columns:
        df.rename(columns={"utt": "text"}, inplace=True)

    # Resolve label names
    if hasattr(ds["train"].features["intent"], "names"):
        intent_names = ds["train"].features["intent"].names
        df["intent"] = df["intent"].apply(
            lambda x: intent_names[x] if isinstance(x, int) else x
        )

    register_intents(df["intent"])
    df["label"] = df["intent"].map(INTENT2ID)
    return df[["text", "label"]]


def get_combined_dataset():
    loaders = [
        ("Banking77", load_banking77),
        ("CLINC150", load_clinc150),
        ("SNIPS", load_snips),
        ("ATIS", load_atis),
        ("Amazon MASSIVE", load_massive),
    ]

    loaded_dfs = []
    for name, loader in loaders:
        try:
            df = loader()
            loaded_dfs.append(df)
            print(f"✅ Loaded {name}")
        except Exception as e:
            print(f"⚠️ Failed to load {name}: {e}")

    if not loaded_dfs:
        raise RuntimeError("No datasets loaded successfully.")

    combined_df = pd.concat(loaded_dfs, ignore_index=True)

    # Add this right before saving 'intent_mapping.json' in get_combined_dataset()
    combined_df = combined_df.dropna(subset=["text", "label"])

    # Ensure labels are clean integers
    combined_df["label"] = combined_df["label"].astype(int)

    with open("intent_mapping.json", "w") as f:
        json.dump(ID2INTENT, f, indent=4)

    print(
        f"\nSuccessfully generated dataset with {len(ID2INTENT)} unique intent names."
    )
    return combined_df


if __name__ == "__main__":
    df = get_combined_dataset()