"""
Flask REST API for facial expression recognition (FER2013 7-class model).
"""

import logging
import os

from flask import Flask, jsonify, request
from flask_cors import CORS

from utils.preprocess import preprocess_for_model
from utils import predict as pred

logger = logging.getLogger(__name__)

app = Flask(__name__)

_cors_origins = os.environ.get("CORS_ORIGINS", "*")
if _cors_origins == "*":
    CORS(app)
else:
    CORS(app, origins=[o.strip() for o in _cors_origins.split(",") if o.strip()])

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODEL = os.path.join(_BASE_DIR, "model", "fer_model.h5")
_DEFAULT_LABELS = os.path.join(_BASE_DIR, "model", "label_map.json")

MODEL_PATH = os.environ.get("MODEL_PATH", _DEFAULT_MODEL)
LABEL_MAP_PATH = os.environ.get("LABEL_MAP_PATH", _DEFAULT_LABELS)

_model_loaded = False
_model_error = None

try:
    if os.path.isfile(MODEL_PATH):
        pred.load_model_once(MODEL_PATH, LABEL_MAP_PATH)
        _model_loaded = True
    else:
        _model_error = f"Model file not found at {MODEL_PATH}"
        logger.warning(_model_error)
except Exception as e:
    _model_error = str(e)
    logger.exception("Failed to load model at startup")


@app.route("/health", methods=["GET"])
def health():
    """Liveness probe: API and model availability."""
    return jsonify({"status": "ok", "model_loaded": _model_loaded})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept multipart field 'image', run face detection + model, return emotion JSON.
    """
    if not _model_loaded:
        return (
            jsonify(
                {
                    "error": "model_unavailable",
                    "message": _model_error or "Model not loaded",
                    "face_detected": False,
                    "emotion": None,
                }
            ),
            503,
        )

    if "image" not in request.files:
        return (
            jsonify(
                {
                    "error": "missing_image",
                    "message": 'Expected multipart field "image"',
                    "face_detected": False,
                    "emotion": None,
                }
            ),
            400,
        )

    file = request.files["image"]
    if file.filename == "":
        return (
            jsonify(
                {
                    "error": "empty_filename",
                    "message": "No file selected",
                    "face_detected": False,
                    "emotion": None,
                }
            ),
            400,
        )

    try:
        image_bytes = file.read()
        if not image_bytes:
            return (
                jsonify(
                    {
                        "error": "empty_body",
                        "message": "Empty image upload",
                        "face_detected": False,
                        "emotion": None,
                    }
                ),
                400,
            )

        tensor, ok, face_box = preprocess_for_model(image_bytes)
    except ValueError as e:
        return (
            jsonify(
                {
                    "error": "invalid_image",
                    "message": str(e),
                    "face_detected": False,
                    "emotion": None,
                }
            ),
            400,
        )
    except Exception:
        logger.exception("Preprocessing failed")
        return (
            jsonify(
                {
                    "error": "preprocess_failed",
                    "message": "Could not process image",
                    "face_detected": False,
                    "emotion": None,
                }
            ),
            400,
        )

    if not ok or tensor is None:
        return jsonify(
            {
                "face_detected": False,
                "emotion": None,
                "confidence": None,
                "all_probabilities": None,
                "face_box": None,
            }
        )

    try:
        emotion, confidence, all_probs, _ = pred.predict_emotion(
            tensor, MODEL_PATH, LABEL_MAP_PATH
        )
    except Exception:
        logger.exception("Inference failed")
        return (
            jsonify(
                {
                    "error": "inference_failed",
                    "message": "Model inference failed",
                    "face_detected": True,
                    "emotion": None,
                }
            ),
            500,
        )

    response = {
        "emotion": emotion,
        "confidence": confidence,
        "all_probabilities": all_probs,
        "face_detected": True,
    }
    if face_box is not None:
        x, y, w, h = face_box
        response["face_box"] = {"x": x, "y": y, "width": w, "height": h}
    else:
        response["face_box"] = None

    return jsonify(response)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
