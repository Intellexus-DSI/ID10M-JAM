"""
Download ID10M-JAM and id10m datasets from HuggingFace to local data/ directories.

Usage:
    python data/download.py                  # download all
    python data/download.py --dataset id10m_jam  # only ID10M-JAM
    python data/download.py --dataset id10m  # only id10m
"""

import argparse
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent


def download_id10m_jam():
    from datasets import load_dataset

    out_dir = DATA_DIR / "id10m_jam"
    out_dir.mkdir(exist_ok=True)

    for lang in ("english", "german"):
        print(f"Downloading ID10M-JAM ({lang})...")
        ds = load_dataset(
            "Intellexus/ID10M-JAM",
            data_files={lang: f"data/{lang}.json"},
            split=lang,
        )
        out_path = out_dir / f"{lang}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ds.to_list(), f, ensure_ascii=False, indent=1)
        print(f"  Saved {len(ds)} rows → {out_path}")


def download_id10m():
    print("id10m download: coming soon (not yet on HuggingFace).")
    print("Place files manually at data/raw_id10m_data/{language}/id10m_{language}_FINAL.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["id10m_jam", "id10m", "all"],
        default="all",
        help="Which dataset to download",
    )
    args = parser.parse_args()

    if args.dataset in ("id10m_jam", "all"):
        download_id10m_jam()
    if args.dataset in ("id10m", "all"):
        download_id10m()


if __name__ == "__main__":
    main()
