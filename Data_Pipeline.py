import os
import pandas as pd
import numpy as np

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

SAMPLE_RATE_MS = 500
WINDOW_SEC     = 3
WINDOW_ROWS    = WINDOW_SEC * 1000 // SAMPLE_RATE_MS   # = 6
HOP_ROWS       = WINDOW_ROWS                           # non-overlapping

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
    label_df = pd.read_csv(label_path)[["timestamp", "value"]].rename(columns={"value": "label"})
    merged = merged.merge(label_df, on="timestamp", how="left")
    return merged


def safe_mode(series) -> float:
    m = series.mode()
    return round(float(m.iloc[0]), 6) if len(m) > 0 else float("nan")


def segment_subject(subject_id: str, df: pd.DataFrame) -> list[dict]:
    rows = []
    n = len(df)
    window_id = 0
    for start in range(0, n - WINDOW_ROWS + 1, HOP_ROWS):
        end = start + WINDOW_ROWS
        window = df.iloc[start:end]
        row = {
            "subject_id":      int(subject_id),
            "window_id":       window_id,
            "start_timestamp": int(window["timestamp"].iloc[0]),
            "end_timestamp":   int(window["timestamp"].iloc[-1]),
        }
        for feat in FEATURES:
            row[f"{feat}_mean"]   = round(window[feat].mean(),   6)
            row[f"{feat}_median"] = round(window[feat].median(), 6)
            row[f"{feat}_mode"]   = safe_mode(window[feat])
        row["label_mean"]   = round(window["label"].mean(),   6)
        row["label_median"] = round(window["label"].median(), 6)
        row["label_mode"]   = safe_mode(window["label"])
        rows.append(row)
        window_id += 1
    return rows


def main():
    print(f"Window size : {WINDOW_SEC} sec  ({WINDOW_ROWS} rows)")
    print(f"Hop size    : {HOP_ROWS} rows  ({'non-overlapping' if HOP_ROWS == WINDOW_ROWS else 'overlapping'})")
    print(f"Features    : {FEATURES}")
    print(f"Label       : anno12_EDA")
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
        df      = load_subject(sid)
        windows = segment_subject(sid, df)
        all_data[partition].extend(windows)
        print(f"  Subject {sid:>3} ({partition:>5}) | rows={len(df)} | windows={len(windows)}")

    print()

    FEATURE_COLS = [f"{feat}_{stat}"
                    for feat in FEATURES
                    for stat in ["mean", "median", "mode"]]

    train_df   = pd.DataFrame(all_data["train"])
    train_mean = train_df[FEATURE_COLS].mean()
    train_std  = train_df[FEATURE_COLS].std()

    stats_df = pd.DataFrame({"mean": train_mean, "std": train_std})
    stats_df.to_csv(os.path.join(OUTPUT_DIR, "norm_stats.csv"))
    print(f"Normalization stats saved -> {OUTPUT_DIR}/norm_stats.csv")
    print(stats_df.round(6).to_string())
    print()

    NORM_COLS  = [f"{col}_norm" for col in FEATURE_COLS]
    LABEL_COLS = ["label_mean", "label_median", "label_mode"]
    CLASS_COLS = ["class_mean", "class_median", "class_mode"]
    ID_COLS    = ["subject_id", "window_id", "start_timestamp", "end_timestamp"]

    # Tertile thresholds — Wiem & Lachiri (IJACSA 2017) + Sanchez-Reolid et al. (Sensors 2022)
    p33 = train_df["label_mean"].quantile(0.33)
    p66 = train_df["label_mean"].quantile(0.66)

    print(f"Arousal class thresholds (tertile split from train set):")
    print(f"  Low    : label  <  {p33:.4f}  (below 33rd percentile)")
    print(f"  Medium : {p33:.4f}  <=  label  <=  {p66:.4f}")
    print(f"  High   : label  >  {p66:.4f}  (above 66th percentile)")
    print()

    def assign_class(value: float) -> str:
        if pd.isna(value):
            return float("nan")
        if value < p33:
            return "Low"
        elif value <= p66:
            return "Medium"
        else:
            return "High"

    thresh_df = pd.DataFrame({
        "threshold": ["p33 (Low/Medium boundary)", "p66 (Medium/High boundary)"],
        "value":     [round(p33, 6), round(p66, 6)],
        "method":    ["33rd percentile of train label_mean",
                      "66th percentile of train label_mean"],
        "reference": ["Wiem & Lachiri, IJACSA 2017 | Sanchez-Reolid et al., Sensors 2022"] * 2
    })
    thresh_df.to_csv(os.path.join(OUTPUT_DIR, "class_thresholds.csv"), index=False)
    print(f"Class thresholds saved -> {OUTPUT_DIR}/class_thresholds.csv")
    print()

    for partition, records in all_data.items():
        out_df      = pd.DataFrame(records)
        norm_values = (out_df[FEATURE_COLS] - train_mean) / train_std
        norm_values.columns = NORM_COLS
        out_df["class_mean"]   = out_df["label_mean"].apply(assign_class)
        out_df["class_median"] = out_df["label_median"].apply(assign_class)
        out_df["class_mode"]   = out_df["label_mode"].apply(assign_class)
        out_df = pd.concat([out_df[ID_COLS], out_df[FEATURE_COLS],
                            norm_values, out_df[LABEL_COLS], out_df[CLASS_COLS]], axis=1)
        out_path = os.path.join(OUTPUT_DIR, f"{partition}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Saved {partition:>5}: {len(out_df):>4} windows, {len(out_df.columns)} columns")

    print("\nDone.")


if __name__ == "__main__":
    main()
