"""
Evaluate saved FER model: train/val/test accuracy, classification report, confusion matrix plot.
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras.utils import to_categorical


def load_fer2013(csv_path: str):
    """Load FER2013 splits (same logic as train.py)."""
    df = pd.read_csv(csv_path)
    required = {"emotion", "pixels", "Usage"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}")

    def rows_for_usage(usage: str):
        sub = df[df["Usage"] == usage]
        pixels = np.array(
            [np.asarray(str(p).split(), dtype=np.float32) for p in sub["pixels"]]
        )
        emotions = sub["emotion"].values.astype(np.int32)
        return pixels, emotions

    x_train, y_train = rows_for_usage("Training")
    x_val, y_val = rows_for_usage("PublicTest")
    x_test, y_test = rows_for_usage("PrivateTest")

    def reshape_norm(x):
        return x.reshape(-1, 48, 48, 1).astype(np.float32) / 255.0

    return (
        reshape_norm(x_train),
        y_train,
        reshape_norm(x_val),
        y_val,
        reshape_norm(x_test),
        y_test,
    )


def load_label_names(label_map_path: str):
    """Return ordered class names for indices from label_map.json (sorted by key)."""
    with open(label_map_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    indices = sorted(int(k) for k in raw.keys())
    return [raw[str(i)] for i in indices]


def main():
    """Load model and data, print metrics and save confusion_matrix.png."""
    parser = argparse.ArgumentParser(description="Evaluate FER Keras model on FER2013")
    parser.add_argument("--csv", type=str, default="fer2013.csv", help="Path to fer2013.csv")
    parser.add_argument(
        "--model",
        type=str,
        default="fer_model.h5",
        help="Path to saved Keras model",
    )
    parser.add_argument(
        "--label-map",
        type=str,
        default="label_map.json",
        help="Path to label_map.json",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="confusion_matrix.png",
        help="Output path for confusion matrix image",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = args.model if os.path.isabs(args.model) else os.path.join(base_dir, args.model)
    label_path = (
        args.label_map if os.path.isabs(args.label_map) else os.path.join(base_dir, args.label_map)
    )
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(os.getcwd(), args.csv)
    if not os.path.isfile(csv_path):
        csv_path = args.csv if os.path.isfile(args.csv) else os.path.join(base_dir, args.csv)

    if not os.path.isfile(csv_path):
        print(f"Error: CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(model_path):
        print(f"Error: Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    class_names = load_label_names(label_path)
    num_classes = len(class_names)

    x_train, y_train, x_val, y_val, x_test, y_test = load_fer2013(csv_path)

    model = keras.models.load_model(model_path)

    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    train_loss, train_acc = model.evaluate(x_train, y_train_cat, verbose=0)
    val_loss, val_acc = model.evaluate(x_val, y_val_cat, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)

    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred_classes,
            target_names=class_names,
            digits=4,
        )
    )

    cm = confusion_matrix(y_test, y_pred_classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="FER2013 Confusion Matrix (Private Test)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    out_path = args.out if os.path.isabs(args.out) else os.path.join(base_dir, args.out)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {out_path}")

    print(f"Test Accuracy: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
