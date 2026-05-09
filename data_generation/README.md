# Data Generation

This folder contains everything related to the datasets used in ID10M-JAM.

## id10m (fixed)

The `id10m_fixed/` folder contains the fixed id10m dataset — the standard idiom
identification benchmark used as our baseline. English (178 sentences) and German
(137 sentences), in both JSON and CSV format.

## id10m-jam

The full ID10M-JAM benchmark is hosted on HuggingFace:
[https://huggingface.co/datasets/Intellexus/ID10M-JAM](https://huggingface.co/datasets/Intellexus/ID10M-JAM)

Download it locally with:

```bash
python data_generation/download_id10m_jam.py
```

This saves `english.json` and `german.json` to `data_generation/id10m_jam/`
(gitignored — re-download as needed).

## Data Generation Pipeline

The `generation/` folder contains the Gemini-based pipeline used to generate the
confusing-context variants that form the id10m-jam benchmark. Requires a Gemini
API key in `.env`:

```bash
cd data_generation/generation
python variants_generator.py
```
