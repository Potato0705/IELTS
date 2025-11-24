# scorer/train_corrector.py
"""
训练 IELTS 自动评分校正器 (RandomForest)
- 使用 Kaggle IELTS 写作数据集
- 使用 Gemma 评分作为 LLM 初始预测
- 学习映射到人工 Overall 分数
"""

from __future__ import annotations
import os, re, time, json, random
from pathlib import Path
from typing import Tuple, List

import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt


# ================== 配置 ==================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "kaggle" / "ielts_writing_dataset.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "rf_corrector.pkl"
FEATURE_PATH = MODEL_DIR / "rf_features.json"

N_SAMPLES = 300  # 推荐免费 API 限制下使用

SCORING_INSTRUCTION = (
    "You are a certified IELTS examiner. Provide ONLY overall band score with high strictness. "
    "Follow the weakest criterion principle: overall score should not exceed the lowest major criterion. "
    "Do NOT give rubric explanations. If text is clearly under 250 words, lower score by at least 0.5. "
    "The output MUST be a number only (0 to 9, increments 0.5)."
)


# ================== 数字提取 ==================
def safe_parse(raw: str, default: float = 5.5) -> float:
    nums = re.findall(r"\d+(?:\.\d+)?", raw or "")
    if not nums:
        return default
    v = max(0.0, min(9.0, float(nums[0])))
    return round(v * 2) / 2


# ================== API调用 ==================
def call_gemma(prompt: str, max_retries: int = 4) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("❌ 环境变量 OPENROUTER_API_KEY 未设置！")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://ielts-evolve",
        "X-Title": "IELTS-Scorer",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "google/gemma-2-9b-it",
        "messages": [
            {"role": "system", "content": "You are an IELTS examiner."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 10,
    }

    for retry in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code in (401, 429):
                print(f"[限流] {resp.status_code} -> 等待 {3*(retry+1)}s")
                time.sleep(3*(retry+1))
                continue

            data = resp.json()

            # 兼容双格式（OpenAI vs OpenRouter）
            try:
                msg = data["choices"][0]["message"]["content"]
            except Exception:
                try:
                    msg = data["output"][0]["content"][0]["text"]
                except:
                    msg = ""

            if msg and any(ch.isdigit() for ch in msg):
                return msg.strip()

        except Exception as e:
            print(f"[Retry {retry+1}/{max_retries}] {type(e).__name__}: {e}")

        time.sleep(2*(retry+1))

    print("⚠ 多次失败 → 返回空文本")
    return ""


# ================== 特征 ==================
def extract_features(essay: str, llm_score: float, task_type: int):
    text = essay.strip()
    words = text.split()
    wc = len(words)
    return {
        "llm_score": llm_score,
        "word_count": wc,
        "char_len": len(text),
        "avg_word_len": (sum(len(w) for w in words) / wc) if wc else 0,
        "comma_count": text.count(","),
        "period_count": text.count("."),
        "sentence_count": max(1, text.count(".")+text.count("!")+text.count("?")),
        "task_type_1": 1 if task_type == 1 else 0,
        "task_type_2": 1 if task_type == 2 else 0,
    }


# ================== 构建数据 ==================
def build_dataset(n_samples=N_SAMPLES):
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Essay", "Overall"]).reset_index(drop=True)
    df = df.sample(n=min(n_samples, len(df)), random_state=42)

    X_rows, y_list = [], []

    for idx, row in df.reset_index(drop=True).iterrows():
        essay = str(row["Essay"])
        true = float(row["Overall"])
        task_type = int(row.get("Task_Type", 1))

        full_prompt = f"{SCORING_INSTRUCTION}\n\nEssay:\n{essay}"
        reply = call_gemma(full_prompt)
        band = safe_parse(reply, default=5.5)

        print(f"[{idx + 1}/{len(df)}] True={true:.1f} Raw={band:.1f} → Out={reply[:30]!r}")
        time.sleep(1.2)

        X_rows.append(extract_features(essay, band, task_type))
        y_list.append(true)

    return pd.DataFrame(X_rows), np.array(y_list)


# ================== 训练 ==================
def train_corrector():
    print("📌 开始构建训练数据…")
    X, y = build_dataset()

    print("📌 划分训练集/测试集…")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    print("📌 训练 RandomForest 校正器…")
    rf = RandomForestRegressor(
        n_estimators=300,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    joblib.dump(rf, MODEL_PATH)
    json.dump(list(X.columns), open(FEATURE_PATH, "w", encoding="utf8"), indent=2)

    print("\n🎯 训练完成！")
    print(f"✨ RF 校正器 Test RMSE: {rmse:.4f}")
    print(f"📁 模型: {MODEL_PATH}")
    print(f"📁 特征: {FEATURE_PATH}")

    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([0,9],[0,9], "k--")
    plt.title(f"RF Corrector - RMSE={rmse:.3f}")
    plt.xlabel("True Overall")
    plt.ylabel("Pred Corrected")
    plt.savefig(MODEL_DIR/"rf_true_vs_pred.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    train_corrector()
