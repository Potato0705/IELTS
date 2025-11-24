# scorer/corrector.py
"""
Runtime RF corrector (optional teacher).
If model files missing, gracefully fall back to raw score.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import json
import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "rf_corrector.pkl"
FEATURE_PATH = MODEL_DIR / "rf_features.json"

_model_cache: Optional[object] = None
_feature_names: Optional[list] = None


def extract_features_runtime(essay: str, llm_score: float, task_type: int):
    text = (essay or "").strip()
    words = text.split()
    wc = len(words)
    sc = max(1, text.count('.') + text.count('!') + text.count('?'))
    avg_wl = (sum(len(w) for w in words) / wc) if wc else 0.0
    avg_sl = wc / sc

    return {
        "llm_score": llm_score,
        "word_count": wc,
        "char_len": len(text),
        "avg_word_len": avg_wl,
        "comma_count": text.count(","),
        "period_count": text.count("."),
        "exclam_count": text.count("!"),
        "question_count": text.count("?"),
        "task_type_1": 1 if task_type == 1 else 0,
        "task_type_2": 1 if task_type == 2 else 0,
        "avg_sent_len": avg_sl,
    }


def _load_model():
    global _model_cache, _feature_names
    if _model_cache is not None:
        return _model_cache, _feature_names

    if not MODEL_PATH.exists() or not FEATURE_PATH.exists():
        raise FileNotFoundError("RF model or features not found.")

    _model_cache = joblib.load(MODEL_PATH)
    with open(FEATURE_PATH, "r", encoding="utf-8") as f:
        _feature_names = json.load(f)
    return _model_cache, _feature_names


def correct_band(raw_band: float, essay: str, task_type: int = 1) -> float:
    try:
        model, feat_names = _load_model()
    except Exception:
        # fallback: no teacher
        return round(float(raw_band) * 2) / 2.0

    feat_dict = extract_features_runtime(essay, raw_band, task_type)

    X_vec = np.array(
        [feat_dict.get(name, 0.0) for name in feat_names],
        dtype=float
    ).reshape(1, -1)

    pred = float(model.predict(X_vec)[0])
    pred = max(0.0, min(9.0, pred))
    return round(pred * 2) / 2.0
