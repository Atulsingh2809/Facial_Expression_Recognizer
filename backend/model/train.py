"""
Full FER2013 training pipeline: load CSV, augment, train CNN, save fer_model.h5 and metrics.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


class EpochProgressLogger(keras.callbacks.Callback):
    """Print elapsed and ETA after each epoch for clearer training progress."""

    def on_train_begin(self, logs=None):
        self._train_start = time.time()
        self._epoch_durations = []
        print("Training started...")

    def on_epoch_end(self, epoch, logs=None):
        now = time.time()
        if epoch == 0:
            duration = now - self._train_start
        else:
            duration = now - self._last_epoch_end
        self._last_epoch_end = now
        self._epoch_durations.append(duration)
        avg_epoch = sum(self._epoch_durations) / len(self._epoch_durations)
        total_epochs = int(self.params.get("epochs", 0))
        done_epochs = epoch + 1
        remain = max(0, total_epochs - done_epochs)
        eta_seconds = int(avg_epoch * remain)
        elapsed_seconds = int(now - self._train_start)
        print(
            f"[Progress] epoch {done_epochs}/{total_epochs} | "
            f"elapsed={elapsed_seconds}s | eta~{eta_seconds}s"
        )


def resolve_csv_path(csv_arg: str) -> str:
    """
    Resolve fer2013.csv path from common locations and fail with clear guidance.

    Search order:
      1) path exactly as provided
      2) relative to current working directory
      3) relative to this script directory
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        csv_arg,
        os.path.join(os.getcwd(), csv_arg),
        os.path.join(script_dir, csv_arg),
    ]
    seen = []
    for candidate in candidates:
        norm = os.path.abspath(candidate)
        if norm not in seen:
            seen.append(norm)
        if os.path.isfile(norm):
            return norm

    looked = "\n  - ".join(seen)
    raise FileNotFoundError(
        "FER2013 CSV not found.\n"
        f"Checked:\n  - {looked}\n\n"
        "Fix:\n"
        "  1) Download FER2013 CSV from Kaggle.\n"
        "  2) Run: python train.py --csv \"D:\\path\\to\\fer2013.csv\" --out-dir ."
    )


def load_fer2013(csv_path: str):
    """
    Load FER2013 from CSV with columns emotion, pixels, Usage.

    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test (numpy arrays, y as int 0-6).
    """
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
        x = x.reshape(-1, 48, 48, 1).astype(np.float32) / 255.0
        return x

    return (
        reshape_norm(x_train),
        y_train,
        reshape_norm(x_val),
        y_val,
        reshape_norm(x_test),
        y_test,
    )


def build_model(num_classes: int = 7) -> keras.Model:
    """Build the specified sequential CNN for FER (48x48 grayscale)."""
    model = keras.Sequential(
        [
            layers.Input(shape=(48, 48, 1)),
            # Block 1
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Block 2
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Block 3
            layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="fer_cnn",
    )
    return model


def main():
    """CLI entry: train on FER2013 and save model + training history metrics."""
    parser = argparse.ArgumentParser(description="Train FER CNN on FER2013 CSV")
    parser.add_argument(
        "--csv",
        type=str,
        default="fer2013.csv",
        help="Path to fer2013.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory to save fer_model.h5 and training_metrics.json",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Number of training epochs (default: 60)",
    )
    args = parser.parse_args()

    try:
        csv_path = resolve_csv_path(args.csv)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "fer_model.h5")

    print(f"Using FER2013 CSV: {csv_path}")
    x_train, y_train, x_val, y_val, x_test, y_test = load_fer2013(csv_path)

    num_classes = len(np.unique(y_train))
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)

    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )

    train_generator = train_datagen.flow(
        x_train, y_train_cat, batch_size=64, shuffle=True
    )

    model = build_model(num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1, min_lr=1e-7
        ),
        EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EpochProgressLogger(),
    ]

    steps_per_epoch = max(1, math.ceil(len(x_train) / 64))
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        validation_data=(x_val, y_val_cat),
        callbacks=callbacks,
        verbose=1,
    )

    # Persist final weights after best checkpoint (checkpoint may have best val)
    model.save(model_path)

    hist = history.history
    train_acc = float(hist["accuracy"][-1]) if hist.get("accuracy") else 0.0
    val_acc = float(hist["val_accuracy"][-1]) if hist.get("val_accuracy") else 0.0

    test_loss, test_acc = model.evaluate(x_test, to_categorical(y_test, num_classes), verbose=0)

    metrics = {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
    }
    metrics_path = os.path.join(args.out_dir, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Final test accuracy: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
