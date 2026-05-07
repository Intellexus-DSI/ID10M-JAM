# Encoders

This directory contains BERT-based encoder experiments for idiom identification.

## Structure

```
encoders/
  predictions/          ← download from HuggingFace (gitignored)
    english/
      {model_name}/seed_{N}/
        results_id10m.json
        results_id10m_jam.json
    german/
      ...
  generate_comparison_tables.py   ← analysis script (see analysis/)
```

## Running Encoder Analysis

1. Download encoder predictions from HuggingFace and place them at `encoders/predictions/`:

```bash
# Coming soon: HuggingFace download instructions
```

2. Run the comparison:

```bash
python analysis/generate_encoder_comparison_tables.py
python analysis/generate_encoder_comparison_tables.py --language english
```
