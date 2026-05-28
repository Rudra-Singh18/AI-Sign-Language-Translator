"""
Process ASL Alphabet images into MediaPipe hand landmark vectors.

Reads images from the Kaggle ASL Alphabet dataset, runs each through
MediaPipe Hands to extract 21 landmarks (63 features), and saves
the results as .npy files in data/{LETTER}/ folders.

Usage:
    python process_dataset.py
    python process_dataset.py --dataset-path /path/to/asl_alphabet_train
    python process_dataset.py --max-per-class 500
"""

import argparse
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np


def find_dataset_path():
    """Auto-detect the Kaggle dataset path from kagglehub cache."""
    # Common kagglehub cache locations
    home = os.path.expanduser("~")
    cache_base = os.path.join(home, ".cache", "kagglehub", "datasets",
                              "grassknoted", "asl-alphabet")

    if os.path.exists(cache_base):
        # Find the latest version
        versions = sorted(os.listdir(cache_base), reverse=True)
        for version in versions:
            version_path = os.path.join(cache_base, version)
            # Look for the training data directory
            for candidate in [
                os.path.join(version_path, "asl_alphabet_train", "asl_alphabet_train"),
                os.path.join(version_path, "asl_alphabet_train"),
                version_path,
            ]:
                if os.path.exists(candidate):
                    # Check if it has letter folders
                    items = os.listdir(candidate)
                    if "A" in items or "a" in items:
                        return candidate

    return None


def extract_landmarks(image_path, hands):
    """Extract 63-dim landmark vector from a single image."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    return np.array(landmarks, dtype=np.float32)


def process_letter(letter_dir, letter, output_dir, hands, max_per_class, start_index=0):
    """Process all images for a single letter and save landmarks."""
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(letter_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])

    if max_per_class:
        image_files = image_files[:max_per_class]

    saved = 0
    failed = 0
    idx = start_index

    for img_file in image_files:
        img_path = os.path.join(letter_dir, img_file)
        landmarks = extract_landmarks(img_path, hands)

        if landmarks is not None and landmarks.shape == (63,):
            np.save(os.path.join(output_dir, f"{idx}.npy"), landmarks)
            idx += 1
            saved += 1
        else:
            failed += 1

    return saved, failed


def main():
    parser = argparse.ArgumentParser(
        description="Process ASL images into MediaPipe landmarks"
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None,
        help="Path to the ASL alphabet training images directory. "
             "Auto-detected from kagglehub cache if not provided."
    )
    parser.add_argument(
        "--max-per-class", type=int, default=None,
        help="Maximum number of images to process per letter. "
             "Default: process all images."
    )
    parser.add_argument(
        "--keep-existing", action="store_true",
        help="Keep existing .npy files and append new ones. "
             "Default: overwrite existing data."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  ASL Dataset → MediaPipe Landmarks Processor")
    print("=" * 60)
    print()

    # Find dataset
    dataset_path = args.dataset_path
    if dataset_path is None:
        print("Auto-detecting Kaggle dataset path...")
        dataset_path = find_dataset_path()
        if dataset_path is None:
            print("ERROR: Could not find the ASL dataset.")
            print("Please run 'python download_dataset.py' first,")
            print("or specify the path with --dataset-path")
            sys.exit(1)

    print(f"Dataset path: {dataset_path}")
    print()

    # Find all letter directories (A-Z only)
    letters = []
    for item in sorted(os.listdir(dataset_path)):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path) and len(item) == 1 and item.upper().isalpha():
            letters.append((item.upper(), item_path))

    if not letters:
        print("ERROR: No letter directories (A-Z) found in the dataset path.")
        print(f"Contents of {dataset_path}:")
        for item in os.listdir(dataset_path):
            print(f"  {item}")
        sys.exit(1)

    print(f"Found {len(letters)} letter classes: {', '.join(l[0] for l in letters)}")
    if args.max_per_class:
        print(f"Processing up to {args.max_per_class} images per class")
    print()

    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3,
    )

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    total_saved = 0
    total_failed = 0
    start_time = time.time()

    results_summary = []

    for i, (letter, letter_path) in enumerate(letters, 1):
        output_dir = os.path.join(data_dir, letter)

        # Determine start index (keep existing or overwrite)
        start_index = 0
        if args.keep_existing and os.path.exists(output_dir):
            existing = [f for f in os.listdir(output_dir) if f.endswith(".npy")]
            start_index = len(existing)
            print(f"[{i:2d}/{len(letters)}] {letter}: Keeping {start_index} existing, ", end="")
        else:
            # Clear existing npy files if not keeping
            if os.path.exists(output_dir):
                for f in os.listdir(output_dir):
                    if f.endswith(".npy"):
                        os.remove(os.path.join(output_dir, f))
            print(f"[{i:2d}/{len(letters)}] {letter}: ", end="")

        num_images = len([
            f for f in os.listdir(letter_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])

        saved, failed = process_letter(
            letter_path, letter, output_dir, hands,
            args.max_per_class, start_index
        )
        total_saved += saved
        total_failed += failed

        elapsed = time.time() - start_time
        print(f"{saved} landmarks saved, {failed} failed "
              f"(of {num_images} images) [{elapsed:.0f}s elapsed]")

        results_summary.append((letter, saved + start_index, failed))

    hands.close()

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("  Processing Complete!")
    print("=" * 60)
    print(f"  Total landmarks saved: {total_saved}")
    print(f"  Total failed:          {total_failed}")
    print(f"  Time elapsed:          {elapsed:.1f}s")
    print()
    print("  Per-letter breakdown:")
    for letter, count, failed in results_summary:
        bar = "█" * min(count // 50, 40)
        print(f"    {letter}: {count:5d} samples {bar}")
    print()
    print("  Next step: Run 'python train.py' to train the model.")


if __name__ == "__main__":
    main()
