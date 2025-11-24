from datasets import load_dataset
import pandas as pd

# 加载 HF 数据集
dataset = load_dataset("chillies/IELTS-writing-task-2-evaluation")

# 提取 train split
df = dataset["train"].to_pandas()

# 保存为本地 CSV
df.to_csv("data/raw/hf_dataset/train.csv", index=False, encoding="utf-8")
print("保存完成！共 {} 行数据".format(len(df)))
print(df.head())
