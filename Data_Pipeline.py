import os
import pandas as pd

DATA_ROOT = (
    "D:/ovgu_folder/advance_topics_of_kmd/Data"
    "/MuSe-Physio Multimodal Physiological-Arousal (MuSe2021)"
    "/c4_muse_physio/c4_muse_physio"
)

FEATURE_DIR    = os.path.join(DATA_ROOT, "feature_segments")
LABEL_DIR      = os.path.join(DATA_ROOT, "label_segments", "anno12_EDA")
PARTITION_FILE = os.path.join(DATA_ROOT, "metadata", "partition.csv")

OUTPUT_DIR = os.path.join(
    "D:/ovgu_folder/advance_topics_of_kmd/Data Engineering and EDA",
    "preprocessed_data"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = ["BPM", "ECG", "resp"]


def load_subject(subject_id: str) -> pd.DataFrame:
    merged = None
    for feat in FEATURES:
        path = os.path.join(FEATURE_DIR, feat, feat, f"{subject_id}.csv")
        df = pd.read_csv(path)[["timestamp", feat]]
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="timestamp", how="left")
    label_path = os.path.join(LABEL_DIR, f"{subject_id}.csv")
    if os.path.exists(label_path):
        label_df = pd.read_csv(label_path)[["timestamp", "value"]].rename(columns={"value": "label"})
        merged = merged.merge(label_df, on="timestamp", how="left")
    else:
        merged["label"] = float("nan")
    merged.insert(0, "subject_id", int(subject_id))
    return merged


def main():
    print(f"Features : {FEATURES}")
    print(f"Label    : anno12_EDA  (continuous regression target)")
    print(f"Output   : one row per 500 ms timestep — no windowing or aggregation")
    print()

    partition_df = pd.read_csv(PARTITION_FILE)
    partition_df.columns = ["subject_id", "partition"]
    partition_df["subject_id"] = partition_df["subject_id"].astype(str)

    all_data = {p: [] for p in ["train", "devel", "test"]}

    for _, row in partition_df.iterrows():
        sid       = row["subject_id"]
        partition = row["partition"]
        bpm_path  = os.path.join(FEATURE_DIR, "BPM", "BPM", f"{sid}.csv")
        if not os.path.exists(bpm_path):
            print(f"  [SKIP] subject {sid} — file not found")
            continue
        df = load_subject(sid)
        all_data[partition].append(df)
        print(f"  Subject {sid:>3} ({partition:>5}) | {len(df)} rows")

    print()

    COL_ORDER = ["subject_id", "timestamp"] + FEATURES + ["label"]

    for partition, dfs in all_data.items():
        out_df   = pd.concat(dfs, ignore_index=True)
        out_df   = out_df[COL_ORDER]
        out_path = os.path.join(OUTPUT_DIR, f"{partition}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Saved {partition:>5}: {len(out_df):>6} rows, {len(out_df.columns)} columns")

    print("\nDone.")


if __name__ == "__main__":
    main()
