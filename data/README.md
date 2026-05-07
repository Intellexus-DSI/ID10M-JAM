# Data

## Downloading Datasets

Run the download script to fetch all datasets from HuggingFace:

```bash
python data/download.py
```

This places data at:
- `data/id10m_jam/` — ID10M-JAM benchmark (English + German)
- `data/raw_id10m_data/` — Original id10m dataset

## Data Generation Pipeline

The `generation/` folder contains the pipeline used to generate the ID10M-JAM
confusing-context variants using Gemini. Requires a Gemini API key in `.env`:

```bash
cd data/generation
python variants_generator.py
```
