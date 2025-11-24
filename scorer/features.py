# scorer/output/features.py
"""
特征抽取模块（B2 多特征版本）

输入：
    - 原始 LLM 打分 raw_band
    - 作文文本 essay

输出：
    - 一个固定长度的特征向量，用于校正模型（RandomForestRegressor）

说明：
    这里我们结合了：
        1）LLM 打分特征（raw_band, raw_band^2）
        2）篇幅特征（字数、词数、句子数）
        3）词汇多样性特征（TTR）
    后续如需扩展，可以在 build_feature_vector 里追加新特征，但一定要保持顺序一致。
"""

from __future__ import annotations

import re
from typing import List

import numpy as np


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


def _tokenize_words(text: str) -> List[str]:
    # 粗略分词：按非字母拆分
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    return tokens


def _split_sentences(text: str) -> List[str]:
    # 非严格句子切分，够用
    parts = re.split(r"[.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def extract_text_features(essay: str) -> dict:
    """
    从作文文本中抽取一些简单的统计特征。
    返回 dict，后面再按照固定顺序拼到向量里。
    """
    if not isinstance(essay, str):
        essay = str(essay or "")

    chars = len(essay)
    words = _tokenize_words(essay)
    num_words = len(words)
    num_unique_words = len(set(words))
    sentences = _split_sentences(essay)
    num_sents = len(sentences)

    avg_word_len = _safe_div(sum(len(w) for w in words), num_words)
    ttr = _safe_div(num_unique_words, num_words)  # Type-Token Ratio
    words_per_sent = _safe_div(num_words, num_sents)

    return {
        "num_chars": float(chars),
        "num_words": float(num_words),
        "num_sents": float(num_sents),
        "avg_word_len": float(avg_word_len),
        "ttr": float(ttr),
        "words_per_sent": float(words_per_sent),
    }


def build_feature_vector(raw_band: float, essay: str) -> np.ndarray:
    """
    B2: 多特征向量构建。
    特征顺序（不要改动顺序，保证训练 / 推理一致）：

        0: raw_band
        1: raw_band^2
        2: num_chars
        3: num_words
        4: num_sents
        5: avg_word_len
        6: ttr
        7: words_per_sent
    """
    if raw_band is None:
        raw_band = 5.0

    text_feats = extract_text_features(essay)

    vec = np.array(
        [
            float(raw_band),
            float(raw_band) ** 2,
            text_feats["num_chars"],
            text_feats["num_words"],
            text_feats["num_sents"],
            text_feats["avg_word_len"],
            text_feats["ttr"],
            text_feats["words_per_sent"],
        ],
        dtype=float,
    )

    return vec
