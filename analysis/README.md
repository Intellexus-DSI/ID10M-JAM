# Analysis

Scripts to reproduce all tables and plots from the paper. Run from the repo root after
experiments have completed and results are in `results/`.

## Scripts

### `compare_id10m_vs_id10m_jam.py`

Main comparison script. For each model/prompt/seed that has results in both `id10m`
and `id10m_jam`, computes S_M, ND_M, and the advanced context-confusion metrics
reported in the paper.

```bash
python analysis/compare_id10m_vs_id10m_jam.py                  # all languages
python analysis/compare_id10m_vs_id10m_jam.py --lang english
python analysis/compare_id10m_vs_id10m_jam.py --lang german
python analysis/compare_id10m_vs_id10m_jam.py --filter gpt-4o-mini
```

Outputs:
- `results/comparisons/{lang}/{model}/{prompt}/seed_{N}/metrics.json`
- `results/comparisons/{lang}/model_comparison_table.csv`

---

### `generate_comparison_table.py`

Builds the aggregated per-run comparison table (Table 3 in the paper) from the
per-run metrics written by `compare_id10m_vs_id10m_jam.py`.

```bash
python analysis/generate_comparison_table.py
```

Outputs:
- `results/comparisons/{lang}/model_comparison_table.csv`

---

### `plot_confusion_histogram.py`

Generates Figure 3 — histogram of how many models experienced negative drift per
hard variant, for each language.

```bash
python analysis/plot_confusion_histogram.py
python analysis/plot_confusion_histogram.py --prompt_type zero_shot --seed 42
python analysis/plot_confusion_histogram.py --prompt_type few_shot  --seed 42
```

Outputs:
- `results/plots/confusion_histogram_{lang}.png`

---

### `sentence_confusion_analysis.py`

For each original sentence, determines whether ALL, NONE, or a MIXED subset of its
hard variants confused the model (AC / NC / MX categories from the paper).

```bash
python analysis/sentence_confusion_analysis.py --language english
python analysis/sentence_confusion_analysis.py --language german
```

Outputs:
- `results/comparisons/{lang}/sentence_confusion_table.csv`

---

### `generate_encoder_comparison_tables.py`

Generates encoder comparison tables from prediction files. Requires encoder
predictions to be placed at `encoders/predictions/` (see `encoders/README.md`).

```bash
python analysis/generate_encoder_comparison_tables.py
python analysis/generate_encoder_comparison_tables.py --language english
```

Outputs:
- `results/encoders/{language}/encoder_comparison_table.csv`
- `results/encoders/{language}/encoder_sentence_confusion_table.csv`

---

## Suggested Order

```bash
# 1. Run experiments first (see root README)
python run_id10m.py
python run_id10m_jam.py

# 2. Compute comparisons
python analysis/compare_id10m_vs_id10m_jam.py

# 3. Generate tables and plots
python analysis/generate_comparison_table.py
python analysis/plot_confusion_histogram.py
python analysis/sentence_confusion_analysis.py --language english
python analysis/sentence_confusion_analysis.py --language german
```
