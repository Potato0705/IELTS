import pandas as pd
import re
import os

RAW_PATH = "data/raw/hf_dataset/train.csv"
OUT_PATH = "data/processed/train_clean.csv"

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv(RAW_PATH)

# 去除换行符、空白字符
df["band"] = df["band"].astype(str).str.extract(r"(\d+\.?\d*)")
df["band"] = df["band"].astype(float)

# 去除无效字符（如 \r \n 等）
df["essay"] = df["essay"].apply(lambda x: re.sub(r"\s+", " ", str(x)).strip())
df["prompt"] = df["prompt"].apply(lambda x: re.sub(r"\s+", " ", str(x)).strip())

print("清洗后数据:")
print(df.head())
print(f"有效样本: {len(df)}")

df.to_csv(OUT_PATH, index=False, encoding="utf-8")
print("已保存到:", OUT_PATH)
