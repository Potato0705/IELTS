# scripts/preprocess_hf_dataset.py
from __future__ import annotations

import re
import random
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "data" / "raw" / "hf_dataset" / "train.csv"
OUT_DIR = BASE_DIR / "data" / "processed" / "hf_dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_PATH = OUT_DIR / "clean.csv"
TRAIN_PATH = OUT_DIR / "train_clean.csv"
EVAL_PATH = OUT_DIR / "eval_clean.csv"


# ========== 1) band robust parse ==========

def safe_parse_band(val) -> Optional[float]:
    """
    Robust parser for HF band field.
    Handles:
      "7.5", "5.0\\n\\n", "<4", "<4\\n\\n\\r\\r", "Band: 6.5"
    Rule:
      if startswith("<") -> treat as (x - 0.5).
    """
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None

    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if not nums:
        return None

    band = float(nums[0])
    if s.lstrip().startswith("<"):
        band -= 0.5

    # clamp + round to 0.5
    band = max(0.0, min(9.0, band))
    band = round(band * 2) / 2.0
    return band


# ========== 2) minimal cleaning ==========

def clean_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    stats = {
        "raw_rows": len(df),
        "drop_na": 0,
        "bad_band": 0,
        "bad_len": 0,
        "dedup": 0,
    }

    # keep key cols
    df = df.dropna(subset=["prompt", "essay", "band"]).reset_index(drop=True)
    stats["drop_na"] = stats["raw_rows"] - len(df)

    # parse band
    parsed = []
    for v in df["band"].tolist():
        parsed.append(safe_parse_band(v))
    df["band_clean"] = parsed

    bad_band_mask = df["band_clean"].isna()
    stats["bad_band"] = int(bad_band_mask.sum())
    df = df[~bad_band_mask].reset_index(drop=True)

    # word count filter (Stage-0)
    def wc(x: str) -> int:
        return len(str(x).split())

    df["word_count"] = df["essay"].apply(wc)
    bad_len_mask = (df["word_count"] < 50) | (df["word_count"] > 1200)
    stats["bad_len"] = int(bad_len_mask.sum())
    df = df[~bad_len_mask].reset_index(drop=True)

    # deduplicate by (prompt, essay)
    before = len(df)
    df = df.drop_duplicates(subset=["prompt", "essay"]).reset_index(drop=True)
    stats["dedup"] = before - len(df)

    # final rename
    # 先删掉原始 band，避免出现重复列名
    df = df.drop(columns=["band"], errors="ignore")
    df = df.rename(columns={"band_clean": "band"})

    df = df.drop(columns=["word_count"], errors="ignore")

    stats["clean_rows"] = len(df)
    return df, stats


# ========== 3) stratified downsample for devpack ==========

def stratified_devpack(
    df: pd.DataFrame,
    seed: int = 42,
    train_per_band: int = 100,
    eval_per_band: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)

    train_rows = []
    eval_rows = []

    for band, sub in df.groupby("band"):
        sub = sub.sample(frac=1, random_state=seed).reset_index(drop=True)

        # take eval first
        n_eval = min(eval_per_band, max(1, len(sub) // 5))
        eval_part = sub.iloc[:n_eval]
        rest = sub.iloc[n_eval:]

        n_train = min(train_per_band, len(rest))
        train_part = rest.iloc[:n_train]

        eval_rows.append(eval_part)
        train_rows.append(train_part)

    train_df = pd.concat(train_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    eval_df = pd.concat(eval_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train_df, eval_df


def main():
    print(f"Loading raw HF dataset from: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)

    clean, stats = clean_df(df)

    print("\n=== Cleaning Stats ===")
    for k, v in stats.items():
        print(f"{k:>10}: {v}")

    print("\n=== Band Distribution (clean) ===")
    print(clean["band"].value_counts().sort_index())

    clean.to_csv(CLEAN_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved clean full dataset to: {CLEAN_PATH}")

    train_df, eval_df = stratified_devpack(clean)
    train_df.to_csv(TRAIN_PATH, index=False, encoding="utf-8-sig")
    eval_df.to_csv(EVAL_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved devpack train to: {TRAIN_PATH} (rows={len(train_df)})")
    print(f"Saved devpack eval  to: {EVAL_PATH} (rows={len(eval_df)})")


if __name__ == "__main__":
    main()
