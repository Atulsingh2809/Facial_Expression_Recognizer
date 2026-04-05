"""Image preprocessing for FER: face detection, crop, resize, and normalization."""

import io
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def _load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes to a BGR OpenCV image."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return img


def _load_image_from_numpy(arr: np.ndarray) -> np.ndarray:
    """Ensure ndarray is a valid BGR image."""
    if arr is None or arr.size == 0:
        raise ValueError("Empty image array")
    if len(arr.shape) == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if len(arr.shape) == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return arr


def detect_largest_face(
    gray: np.ndarray,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect faces with Haar cascade and return the largest bounding box (x, y, w, h).

    Returns None if no face is found.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError("Failed to load Haar cascade classifier")

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
    )
    if faces is None or len(faces) == 0:
        return None

    largest = max(faces, key=lambda r: r[2] * r[3])
    x, y, w, h = int(largest[0]), int(largest[1]), int(largest[2]), int(largest[3])
    return x, y, w, h


def preprocess_for_model(
    image_input: bytes,
) -> Tuple[Optional[np.ndarray], bool, Optional[Tuple[int, int, int, int]]]:
    """
    Decode image, detect largest face, crop, resize to 48x48 grayscale, normalize to [0,1].

    Args:
        image_input: Raw image file bytes (JPEG/PNG).

    Returns:
        Tuple of (model_input tensor shape (1,48,48,1) or None, success flag, face box in original image coords).
    """
    try:
        bgr = _load_image_from_bytes(image_input)
    except ValueError:
        return None, False, None

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    box = detect_largest_face(gray)
    if box is None:
        return None, False, None

    x, y, w, h = box
    h_img, w_img = gray.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)
    if w <= 0 or h <= 0:
        return None, False, None

    face_roi = gray[y : y + h, x : x + w]
    resized = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    tensor = np.expand_dims(np.expand_dims(normalized, axis=-1), axis=0)
    return tensor, True, (x, y, w, h)


def preprocess_from_array(
    rgb_or_bgr: np.ndarray,
) -> Tuple[Optional[np.ndarray], bool, Optional[Tuple[int, int, int, int]]]:
    """
    Same as preprocess_for_model but from an in-memory BGR/RGB ndarray (e.g. OpenCV frame).
    """
    try:
        bgr = _load_image_from_numpy(rgb_or_bgr)
    except ValueError:
        return None, False, None

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    box = detect_largest_face(gray)
    if box is None:
        return None, False, None

    x, y, w, h = box
    h_img, w_img = gray.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)
    if w <= 0 or h <= 0:
        return None, False, None

    face_roi = gray[y : y + h, x : x + w]
    resized = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    tensor = np.expand_dims(np.expand_dims(normalized, axis=-1), axis=0)
    return tensor, True, (x, y, w, h)


def image_bytes_from_pil(pil_image: Image.Image, fmt: str = "JPEG") -> bytes:
    """Encode a PIL image to bytes."""
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    return buf.getvalue()
