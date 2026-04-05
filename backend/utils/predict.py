"""Load Keras model and run inference with label mapping."""

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

# Lazy import tensorflow to speed up non-ML imports
_model = None
_label_map: Dict[int, str] = {}


def _default_model_path() -> str:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "model", "fer_model.h5")


def _default_label_path() -> str:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "model", "label_map.json")


def load_label_map(path: str) -> Dict[int, str]:
    """Load emotion index -> name mapping from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_model_once(model_path: str, label_map_path: str) -> Tuple[Any, Dict[int, str]]:
    """
    Load Keras model and labels once (singleton-style for the process).

    Returns:
        (keras.Model, label_map dict)
    """
    global _model, _label_map
    if _model is not None and _label_map:
        return _model, _label_map

    from tensorflow import keras

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    _model = keras.models.load_model(model_path)
    _label_map = load_label_map(label_map_path)
    return _model, _label_map


def predict_emotion(
    tensor: np.ndarray,
    model_path: str,
    label_map_path: str,
) -> Tuple[str, float, Dict[str, float], List[str]]:
    """
    Run softmax prediction on a single preprocessed batch (1, 48, 48, 1).

    Returns:
        (top_emotion, confidence, all_probabilities dict ordered keys by label index, ordered emotion names)
    """
    model, label_map = load_model_once(model_path, label_map_path)
    probs = model.predict(tensor, verbose=0)[0]
    idx = int(np.argmax(probs))
    top = label_map[idx]
    conf = float(probs[idx])

    ordered_names = [label_map[i] for i in sorted(label_map.keys())]
    all_probs = {label_map[i]: float(probs[i]) for i in sorted(label_map.keys())}
    return top, conf, all_probs, ordered_names


def clear_loaded_model() -> None:
    """Release references (mainly for tests)."""
    global _model, _label_map
    _model = None
    _label_map = {}
