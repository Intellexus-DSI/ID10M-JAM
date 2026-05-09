"""
Download ID10M-JAM from HuggingFace to data_generation/id10m_jam/.

Dataset: https://huggingface.co/datasets/Intellexus/ID10M-JAM
Languages: english, german

Usage:
    python data_generation/download_id10m_jam.py
"""

import json
import shutil
from pathlib import Path

HF_REPO_ID = "Intellexus/ID10M-JAM"
LANGUAGES = ("english", "german")
OUT_DIR = Path(__file__).parent / "id10m_jam"


def download():
    from huggingface_hub import hf_hub_download

    OUT_DIR.mkdir(exist_ok=True)

    for lang in LANGUAGES:
        print(f"Downloading ID10M-JAM ({lang})...")
        cached = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"data/{lang}.json",
            repo_type="dataset",
        )
        out_path = OUT_DIR / f"{lang}.json"
        shutil.copy(cached, out_path)
        with open(out_path, encoding="utf-8") as f:
            n = len(json.load(f))
        print(f"  Saved {n} rows → {out_path}")


if __name__ == "__main__":
    download()
