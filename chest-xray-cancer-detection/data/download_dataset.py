"""
Download the Chest X-Ray dataset from Kaggle.

Requirements:
    - Kaggle account: https://www.kaggle.com
    - API credentials in .env file (KAGGLE_USERNAME, KAGGLE_KEY)
    - OR place kaggle.json in ~/.kaggle/

Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
"""

import os
import zipfile
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def setup_kaggle_credentials():
    """Set Kaggle credentials from .env if not already configured."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if not kaggle_json.exists():
        username = os.getenv("KAGGLE_USERNAME")
        key = os.getenv("KAGGLE_KEY")

        if not username or not key:
            raise ValueError(
                "Kaggle credentials not found!\n"
                "Option 1: Add KAGGLE_USERNAME and KAGGLE_KEY to your .env file\n"
                "Option 2: Download kaggle.json from https://www.kaggle.com/settings "
                "and place it at ~/.kaggle/kaggle.json"
            )

        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_json.write_text(f'{{"username":"{username}","key":"{key}"}}')
        kaggle_json.chmod(0o600)
        print("Kaggle credentials configured.")
    else:
        print("Kaggle credentials already exist.")


def download_dataset(output_dir: str = "data"):
    """Download and extract the chest X-ray dataset."""
    import kaggle 

    output_path = Path(output_dir)
    dataset_path = output_path / "chest_xray"

    if dataset_path.exists():
        print(f"Dataset already exists at {dataset_path}")
        return str(dataset_path)

    print("Downloading Chest X-Ray dataset from Kaggle...")
    output_path.mkdir(parents=True, exist_ok=True)

    kaggle.api.dataset_download_files(
        "paultimothymooney/chest-xray-pneumonia",
        path=str(output_path),
        unzip=True,
        quiet=False,
    )

    extracted = output_path / "chest_xray"
    if not extracted.exists():

        for item in output_path.iterdir():
            if item.is_dir() and "xray" in item.name.lower():
                item.rename(extracted)
                break

    print(f"Dataset downloaded and extracted to: {extracted}")
    print_dataset_stats(extracted)
    return str(extracted)


def print_dataset_stats(dataset_path: Path):
    """Print a summary of the dataset."""
    print("\nDataset Summary:")
    print("-" * 40)
    for split in ["train", "val", "test"]:
        split_path = dataset_path / split
        if split_path.exists():
            total = 0
            print(f"\n  {split.upper()}:")
            for cls in sorted(split_path.iterdir()):
                if cls.is_dir():
                    count = len(list(cls.glob("*.jpeg")) + list(cls.glob("*.jpg")) + list(cls.glob("*.png")))
                    print(f"    {cls.name}: {count} images")
                    total += count
            print(f"    Total: {total}")
    print("-" * 40)


if __name__ == "__main__":
    print("Chest X-Ray Cancer Detection — Dataset Downloader")
    print("=" * 50)
    setup_kaggle_credentials()
    dataset_path = download_dataset(output_dir="data")
    print(f"\nReady! Dataset located at: {dataset_path}")
    print("\nNext step: python src/train.py")
