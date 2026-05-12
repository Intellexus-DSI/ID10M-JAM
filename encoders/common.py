#This file hosts resources that are common across scripts in encoders directory
from pathlib import Path
from collections import defaultdict
import itertools
import re
from datetime import datetime
from typing import Any
import json



LANGS = ["english", "german"]
RESULTS_OUT_FILENAME = "aggregated_results.json"
PREDICTIONS_OUT_FILENAME = "sentences_predictions.json"
PREDICTIONS_OUT_FILENAME_PLUS_EXPLICIT_IDIOMS = "sentences_predictions_plus_explicit_idioms.json"
SEEDS = [5, 7, 42, 123, 1773]

DEFAULT_MODELS = ["google-bert/bert-base-multilingual-cased", "FacebookAI/xlm-roberta-base"]

LANGS_TO_MODELS = defaultdict(list)
for l, m in itertools.product(LANGS, DEFAULT_MODELS):
    LANGS_TO_MODELS[l].append(m)

LANGS_TO_MODELS["english"].extend(["google-bert/bert-base-uncased", "FacebookAI/roberta-base"])
LANGS_TO_MODELS["german"].extend(["deepset/gbert-base"])
'''
The following models are nice-to-have in the German case, but not a core part of the project:
"google-bert/bert-base-german-cased", "google-bert/bert-base-german-dbmdz-uncased",
"benjamin/roberta-base-wechsel-german"
'''

def get_cur_run_dir(cur_lang: str, cur_model_name: str, cur_seed: int) -> Path:
    model_name_for_print = re.sub(r'[-/]', '_', cur_model_name) 
    return Path(cur_lang, model_name_for_print, str(cur_seed))

def write_results_to_dir(dir: Path, fname: str, data_to_write: Any):
    full_path = Path(dir, fname)
    print(f"{datetime.now()} Writing to {full_path}")
    with open(full_path, "w") as write_file:
        json.dump(data_to_write, write_file, indent=4)