import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from collections import defaultdict
from typing import List, Dict
from common import LANGS_TO_MODELS, LANGS, SEEDS, get_cur_run_dir, write_results_to_dir
import pandas as pd
from pathlib import Path
import os
import json
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent.parent


def aggregate_word_tag(tags: List[str]) -> str:
    """A word's tag equals the tag of its first sub-token."""
    return tags[0] if tags else "O"


def predict_bio_tags(sentences: List[str], model_path: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Loads a saved model, auto-repairs corrupted label mappings if necessary,
    runs inference, and returns a dict of {sentence: [{word, tag}]}.
    """
    print(f"Loading model and tokenizer from: {model_path}")

    # ==========================================
    # 1. Config Self-Healing
    # ==========================================
    config_file_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_file_path):
        with open(config_file_path, "r") as f:
            config_data = json.load(f)

        # Handle wrong id2label from earlier training runs (ids were sorted alphabetically).
        # Kept for backward compatibility — redundant for newly trained models.
        if "B-IDIOM" in config_data.get("id2label", {}):
            print(f"Repairing reversed labels in {config_file_path}...")
            correct_label_list = ["B-IDIOM", "I-IDIOM", "O"]
            config_data["id2label"] = {str(i): label for i, label in enumerate(correct_label_list)}
            config_data["label2id"] = {label: i for i, label in enumerate(correct_label_list)}
            with open(config_file_path, "w") as f:
                json.dump(config_data, f, indent=2)

    # ==========================================
    # 2. Load Model & Tokenizer
    # ==========================================
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    id2label = model.config.id2label

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train(False)  # inference mode

    final_results = {}

    # ==========================================
    # 3. Inference & Aggregation
    # ==========================================
    for sentence_str in sentences:
        words = sentence_str.split()

        inputs = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
        word_ids = inputs.word_ids()

        word_idx_to_tags = defaultdict(list)
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None:
                pred_id = predictions[token_idx]
                tag = id2label.get(pred_id) or id2label.get(str(pred_id))
                word_idx_to_tags[word_idx].append(tag)

        sentence_word_tags = []
        for word_idx, word in enumerate(words):
            raw_tags = word_idx_to_tags.get(word_idx, [])
            final_tag = aggregate_word_tag(raw_tags)
            sentence_word_tags.append({"word": word, "tag": final_tag})

        final_results[sentence_str] = sentence_word_tags

    return final_results


def get_checkpoint_dir(base_path: str | Path) -> Path:
    """
    Given a base path, finds the 'checkpoint*' subdirectory inside 'results/'.
    Assumes exactly one checkpoint directory exists.
    """
    results_dir = Path(base_path) / "results"
    try:
        return next(results_dir.glob("checkpoint*"))
    except StopIteration:
        print(f"No checkpoint directory found in {results_dir}")


def get_id10m_sents_for_lang(lang: str) -> List[str]:
    """Returns original id10m sentences from id10m_fixed."""
    fpath = REPO_ROOT / "data_generation" / "id10m_fixed" / f"{lang}.json"
    df = pd.read_json(fpath)
    return df["sentence"].dropna().unique().tolist()


def get_id10m_jam_sents_for_lang(lang: str) -> List[str]:
    """Returns id10m-jam variant sentences (download first via download_id10m_jam.py)."""
    fpath = REPO_ROOT / "data_generation" / "id10m_jam" / f"{lang}.json"
    if not fpath.exists():
        raise FileNotFoundError(
            f"id10m-jam data not found at {fpath}.\n"
            f"Run: python data_generation/download_id10m_jam.py"
        )
    df = pd.read_json(fpath)
    return df["variant_sentence"].dropna().unique().tolist()


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    abs_path = Path(__file__).parent  # encoders/ directory

    for lang in LANGS:
        id10m_sents = get_id10m_sents_for_lang(lang)
        id10m_jam_sents = get_id10m_jam_sents_for_lang(lang)
        all_sents = list(dict.fromkeys(id10m_sents + id10m_jam_sents))  # deduplicated

        for model_name in LANGS_TO_MODELS[lang]:
            for cur_seed in SEEDS:
                run_dir = abs_path / get_cur_run_dir(lang, model_name, cur_seed)
                model_dir = get_checkpoint_dir(run_dir)
                if model_dir is None:
                    continue

                print(f"{datetime.now()} Model dir: {model_dir}")
                results = predict_bio_tags(all_sents, model_dir)

                # Split into id10m and id10m_jam for downstream analysis
                id10m_set = set(id10m_sents)
                id10m_jam_set = set(id10m_jam_sents)
                results_id10m = {s: v for s, v in results.items() if s in id10m_set}
                results_id10m_jam = {s: v for s, v in results.items() if s in id10m_jam_set}

                out_dir = (
                    abs_path / "predictions" / lang
                    / get_cur_run_dir("", model_name, cur_seed).parts[-1]
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                write_results_to_dir(out_dir, "new_results_id10m.json", results_id10m)
                write_results_to_dir(out_dir, "new_results_id10m_jam.json", results_id10m_jam)
