from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_FILE = Path("data/gesture_data.npz")
MODEL_PATH = Path("models/gesture_classifier.h5")
LABELS = ["Open Palm", "Fist", "Thumbs Up", "Peace", "Pointing"]


def build_model(input_shape, num_classes):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    if not DATA_FILE.exists():
        print(f"Dataset not found: {DATA_FILE}")
        print("Run collect_data.py first to record gesture examples.")
        return

    with np.load(DATA_FILE) as data:
        X = data["X"]
        y = data["y"]

    if len(X) == 0:
        print("No examples found in the dataset.")
        return

    model = build_model(input_shape=X.shape[1:], num_classes=len(LABELS))
    model.summary()

    model.fit(
        X,
        y,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        shuffle=True,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
