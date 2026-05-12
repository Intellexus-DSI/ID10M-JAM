# Encoders

BERT-based encoder experiments for idiom identification, fine-tuned on the id10m training set and evaluated on both id10m and id10m-jam.

## Models

| Language | Models |
|---|---|
| English | `google-bert/bert-base-uncased`, `google-bert/bert-base-multilingual-cased`, `FacebookAI/xlm-roberta-base`, `FacebookAI/roberta-base` |
| German | `deepset/gbert-base`, `google-bert/bert-base-multilingual-cased`, `FacebookAI/xlm-roberta-base` |

5 seeds per model: `[5, 7, 42, 123, 1773]`

## Structure

```
encoders/
  common.py                              — shared constants (models, seeds, helpers)
  tags_fine_tuning_and_inference.ipynb  — fine-tuning + evaluation notebook (GPU required)
  sents_inference.py                     — inference on saved checkpoints
  predictions/                           ← download from HuggingFace (gitignored)
    english/
      {model_name}/seed_{N}/
        new_results_id10m.json           — predictions on original id10m sentences
        new_results_id10m_jam.json       — predictions on id10m-jam variants
    german/
      ...
```

## Option A — Download pre-computed predictions (recommended)

Pre-computed predictions for all models and seeds are hosted on HuggingFace:

```bash
# Coming soon: HuggingFace download instructions
# Place predictions at encoders/predictions/
```

Then run the analysis:

```bash
python analysis/generate_encoder_comparison_tables.py
python analysis/generate_encoder_comparison_tables.py --language english
python analysis/generate_encoder_comparison_tables.py --language german
```

## Option B — Re-run encoder experiments from scratch

Requires a GPU and the id10m training set.

### 1. Get the id10m training set

Download the original ID10M dataset from [https://github.com/Babelscape/ID10M](https://github.com/Babelscape/ID10M) and place the training TSV files under:

```
data_generation/id10m_trainset/
  english/   ← TSV files from the id10m English train split
  german/    ← TSV files from the id10m German train split
```

### 2. Download id10m-jam data

```bash
python data_generation/download_id10m_jam.py
```

### 3. Fine-tune

Open and run `encoders/tags_fine_tuning_and_inference.ipynb` from the `encoders/` directory. This fine-tunes each model across all seeds and writes per-run results under `{lang}/{model_name}/{seed}/`.

### 4. Run inference on saved checkpoints

```bash
cd encoders
python sents_inference.py
```

Writes `new_results_id10m.json` and `new_results_id10m_jam.json` to `encoders/predictions/{lang}/{model}/seed_{N}/`.

### 5. Generate comparison tables

```bash
python analysis/generate_encoder_comparison_tables.py
```
