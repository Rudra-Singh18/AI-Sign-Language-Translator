"""
Download the ASL Alphabet Dataset from Kaggle.

Uses kagglehub to download the 'grassknoted/asl-alphabet' dataset,
which contains ~87,000 images of ASL hand signs (A-Z + space/delete/nothing).

No Kaggle API key is required for this public dataset.

Usage:
    python download_dataset.py
"""

import kagglehub
import os
import sys


def main():
    print("=" * 60)
    print("  ASL Alphabet Dataset Downloader")
    print("=" * 60)
    print()
    print("Downloading 'grassknoted/asl-alphabet' from Kaggle...")
    print("This dataset contains ~87,000 images of ASL hand signs.")
    print("Download size: ~1 GB. This may take a few minutes.")
    print()

    try:
        path = kagglehub.dataset_download("grassknoted/asl-alphabet")
        print()
        print(f"Download complete!")
        print(f"Dataset path: {path}")
        print()

        # List contents to verify
        if os.path.exists(path):
            contents = os.listdir(path)
            print(f"Contents ({len(contents)} items):")
            for item in sorted(contents):
                full_path = os.path.join(path, item)
                if os.path.isdir(full_path):
                    num_files = len(os.listdir(full_path))
                    print(f"  [DIR]  {item}/ ({num_files} files)")
                else:
                    size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    print(f"  [FILE] {item} ({size_mb:.1f} MB)")

            # Check for nested structure (common in Kaggle datasets)
            for subdir in ["asl_alphabet_train", "asl-alphabet", "asl_alphabet_train/asl_alphabet_train"]:
                nested = os.path.join(path, subdir)
                if os.path.exists(nested) and os.path.isdir(nested):
                    print(f"\nFound training data in: {nested}")
                    nested_contents = sorted(os.listdir(nested))
                    print(f"Classes found: {len(nested_contents)}")
                    for item in nested_contents:
                        item_path = os.path.join(nested, item)
                        if os.path.isdir(item_path):
                            count = len(os.listdir(item_path))
                            print(f"  {item}: {count} images")

        print()
        print("Next step: Run 'python process_dataset.py' to extract landmarks.")
        return path

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print()
        print("If this fails, you can manually download from:")
        print("  https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        print()
        print("Then extract to a folder and update the path in process_dataset.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
