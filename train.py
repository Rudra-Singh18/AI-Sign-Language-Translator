"""
Train an ASL sign language classifier from MediaPipe hand landmarks.

Loads landmark .npy files from data/{LETTER}/ directories, trains a
neural network classifier, and saves the model + label mapping.

Usage:
    python train.py
    python train.py --epochs 100 --batch-size 64
    python train.py --no-augment
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF info/warnings

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "sign_language_model.keras"
LABEL_MAP_PATH = MODEL_DIR / "label_map.json"

MIN_SAMPLES_PER_CLASS = 10
FEATURE_DIM = 63  # 21 landmarks * 3 coords (x, y, z)


def load_data(data_dir, min_samples=MIN_SAMPLES_PER_CLASS):
    """Load all .npy landmark files from data/{LETTER}/ directories."""
    X = []
    y = []
    label_names = []
    skipped = []

    # Find all letter directories with data
    for letter_dir in sorted(data_dir.iterdir()):
        if not letter_dir.is_dir():
            continue
        letter = letter_dir.name
        if len(letter) != 1 or not letter.isalpha():
            continue
        letter = letter.upper()

        # Load all .npy files
        npy_files = sorted(letter_dir.glob("*.npy"))
        if len(npy_files) < min_samples:
            if len(npy_files) > 0:
                skipped.append((letter, len(npy_files)))
            continue

        label_idx = len(label_names)
        label_names.append(letter)

        loaded = 0
        for npy_file in npy_files:
            try:
                data = np.load(npy_file)
                if data.shape == (FEATURE_DIM,):
                    X.append(data.astype(np.float32))
                    y.append(label_idx)
                    loaded += 1
            except Exception:
                pass  # Skip corrupted files silently

        print(f"  {letter}: {loaded} samples loaded")

    if skipped:
        print(f"\n  Skipped (< {min_samples} samples):")
        for letter, count in skipped:
            print(f"    {letter}: {count} samples")

    return np.array(X), np.array(y), label_names


def augment_data(X, y, num_augmented=3, noise_scale=0.01):
    """Create augmented copies of the data by adding Gaussian noise."""
    X_aug = [X]
    y_aug = [y]

    for i in range(num_augmented):
        noise = np.random.normal(0, noise_scale, X.shape).astype(np.float32)
        X_noisy = X + noise
        # Clip to valid range (MediaPipe landmarks are 0-1 for x,y)
        X_noisy = np.clip(X_noisy, -1.0, 2.0)
        X_aug.append(X_noisy)
        y_aug.append(y)

    return np.concatenate(X_aug), np.concatenate(y_aug)


def build_model(num_classes):
    """Build the sign language classification model."""
    model = models.Sequential([
        layers.Input(shape=(FEATURE_DIM,)),

        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train ASL sign language classifier"
    )
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size (default: 32)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--augment-copies", type=int, default=3,
                        help="Number of augmented copies (default: 3)")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES_PER_CLASS,
                        help=f"Min samples per class (default: {MIN_SAMPLES_PER_CLASS})")
    args = parser.parse_args()

    print("=" * 60)
    print("  ASL Sign Language Model Trainer")
    print("=" * 60)
    print()

    # ── Load Data ───────────────────────────────────────────────
    print("Loading data from data/{LETTER}/*.npy ...")
    X, y, label_names = load_data(DATA_DIR, min_samples=args.min_samples)

    if len(X) == 0:
        print("\nERROR: No data found!")
        print("Run 'python collect_data.py' or 'python process_dataset.py' first.")
        sys.exit(1)

    num_classes = len(label_names)
    print(f"\nTotal: {len(X)} samples across {num_classes} classes")
    print(f"Classes: {', '.join(label_names)}")

    if num_classes < 2:
        print("\nERROR: Need at least 2 classes to train. Collect more data!")
        sys.exit(1)

    # ── Data Augmentation ───────────────────────────────────────
    if not args.no_augment:
        print(f"\nAugmenting data ({args.augment_copies}x noisy copies)...")
        X, y = augment_data(X, y, num_augmented=args.augment_copies)
        print(f"After augmentation: {len(X)} samples")

    # ── Train / Val / Test Split ────────────────────────────────
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\nSplit: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # ── Build Model ─────────────────────────────────────────────
    print(f"\nBuilding model ({num_classes} output classes)...")
    model = build_model(num_classes)
    model.summary()

    # ── Training Callbacks ──────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    training_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ── Train ───────────────────────────────────────────────────
    print(f"\nTraining for up to {args.epochs} epochs...")
    print("-" * 60)

    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        callbacks=training_callbacks,
        verbose=1,
    )

    # ── Evaluate on Test Set ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Evaluation Results")
    print("=" * 60)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc * 100:.1f}%)")

    # Classification report
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))

    # ── Save Model & Label Map ──────────────────────────────────
    model.save(MODEL_PATH)
    print(f"  Model saved to: {MODEL_PATH}")

    label_map = {str(i): name for i, name in enumerate(label_names)}
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"  Label map saved to: {LABEL_MAP_PATH}")

    # ── Training Summary ────────────────────────────────────────
    best_epoch = np.argmax(history.history["val_accuracy"]) + 1
    best_val_acc = max(history.history["val_accuracy"])
    print(f"\n  Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"  Total epochs trained: {len(history.history['loss'])}")

    print()
    print("  Done! Run 'python sign_detector.py' or 'python app.py' to test.")
    print("=" * 60)


if __name__ == "__main__":
    main()
