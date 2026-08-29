"""
Machine Learning Prediction & Simulation Caching System

Caches walk-forward model predictions and portfolio simulation outputs to disk
so subsequent runs, charts, and comparisons load instantly (<0.1s) without re-training models.
"""

import os
import sys
import pickle
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

CACHE_DIR = BASE_DIR / "Reports" / "Cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_key(model_type: str, rr_ratio: str, probability_threshold: float) -> str:
    safe_rr = rr_ratio.replace(":", "to")
    return f"pred_{model_type}_{safe_rr}_p{probability_threshold:.2f}.pkl"


def load_cached_predictions(model_type: str, rr_ratio: str, probability_threshold: float) -> pd.DataFrame:
    key_filename = get_cache_key(model_type, rr_ratio, probability_threshold)
    cache_file = CACHE_DIR / key_filename

    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                df = pickle.load(f)
            return df
        except Exception:
            return None
    return None


def save_predictions_to_cache(df: pd.DataFrame, model_type: str, rr_ratio: str, probability_threshold: float):
    key_filename = get_cache_key(model_type, rr_ratio, probability_threshold)
    cache_file = CACHE_DIR / key_filename
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(df, f)
    except Exception as e:
        print(f"Warning: Failed to cache predictions to {cache_file}: {e}")


def clear_prediction_cache():
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("pred_*.pkl"):
            try:
                os.remove(f)
            except Exception:
                pass
        print(f"Cleared prediction cache directory: {CACHE_DIR.resolve()}")

if __name__ == "__main__":
    print(f"Prediction Caching Directory ready at: {CACHE_DIR.resolve()}")
