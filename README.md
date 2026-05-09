# ID10M-JAM: Stress-Testing Idiom Identification Under Challenging Context

**ACL 2026** · [[Paper]]() · [[Dataset (HuggingFace)](https://huggingface.co/datasets/Intellexus/ID10M-JAM)]

ID10M-JAM evaluates idiom identification under challenging confusing-context conditions. We benchmark 10 LLMs across English and German on two tasks: **id10m** — a standard idiom identification task — and **id10m-jam** — our new benchmark where each idiom-containing sentence is paired with LLM-generated confusing-context variants designed to mislead models into literal interpretations.

---

## Dataset

Both datasets are available on HuggingFace:

| Dataset | Description | Languages | Size |
|---|---|---|---|
| `id10m-jam` | Idiom identification with confusing-context variants (ours) | English, German | 534 EN / 411 DE |
| `id10m` | Standard idiom identification baseline | English, German | 178 EN |

```python
from datasets import load_dataset
ds = load_dataset("Intellexus/ID10M-JAM")
```

Download id10m-jam locally:

```bash
python data_generation/download_id10m_jam.py
```

> `id10m-jam` is also loaded automatically from HuggingFace by the experiment scripts (no manual download needed). The `id10m` dataset is committed at `data_generation/id10m_fixed/`.

---

## Repository Overview

| Path | Purpose |
|---|---|
| `experiments/src/` | Task utilities for `id10m_jam` and `id10m` (data loading, prompts, metrics) |
| `utils/` | Shared utilities: metrics, LLM wrappers, schemas |
| `data_generation/generation/` | Gemini-based pipeline that generated the confusing-context variants |
| `data_generation/id10m_fixed/` | id10m dataset (English + German, JSON + CSV) |
| `data_generation/download_id10m_jam.py` | Download id10m-jam from HuggingFace |
| `analysis/` | Scripts to reproduce all plots and comparison tables from the paper |
| `encoders/` | BERT encoder experiments (predictions downloaded from HuggingFace) |
| `assets/` | Paper figures |

---

## Quick Start

**Prerequisites:** Python 3.10+

```bash
pip install -r requirements.txt
```

Copy and fill in your API keys:

```bash
cp keys.yaml.example keys.yaml
# Edit keys.yaml with your OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY, TOGETHER_API_KEY
```

Run a zero-shot evaluation (GPT-4o-mini, id10m-jam, English):

```bash
python run_id10m_jam.py
```

---

## Running Experiments

### LLM Evaluation

Edit `config.yaml` to configure your task, model, and prompt type, then run:

```bash
# id10m task
python run_id10m.py

# id10m-jam task (supports self-consistency)
python run_id10m_jam.py

# Override config values via CLI
python run_id10m_jam.py --seed 43 --lang german --sc_runs 5
python run_id10m_jam.py --config_file my_config.yaml

# Re-evaluate existing responses without calling the API
python run_id10m_jam.py --responses_dir results/id10m_jam/english/<model>/<prompt_type>/seed_42
```

### Key Config Parameters (`config.yaml`)

| Parameter | Options | Description |
|---|---|---|
| `task` | `id10m_jam`, `id10m` | Task to run |
| `lang` | `english`, `german` | Language |
| `model` | see table below | LLM to evaluate |
| `prompt_type` | `zero_shot`, `few_shot_cot_best` | Prompting strategy |
| `shots` | `0`, `10` | Few-shot examples (must be even; 0 for zero-shot) |
| `sc_runs` | `1`, `5` | Self-consistency runs (1 = disabled) |
| `temperature` | `0.3`, `0.8` | Use 0.3 without SC, 0.8 with SC |
| `debug` | `true`, `false` | Run on 5 samples, skip W&B logging |

### Supported Models

| Provider | Model string |
|---|---|
| OpenAI | `gpt-4o`, `gpt-4o-mini`, `o3-mini` |
| Google | `gemini-2.5-pro`, `gemini-2.5-flash-lite` |
| Anthropic | `claude-sonnet-4-20250514`, `claude-3-haiku-20240307` |
| Together.ai | `meta-llama/Llama-4-Scout-17B-16E-Instruct`, `deepseek-ai/DeepSeek-R1`, `Qwen/Qwen2.5-72B-Instruct-Turbo` |

### Experiment Outputs

Each run writes to `results/{task}/{lang}/{model}/{prompt_type}/seed_{N}/`:

```
results/id10m_jam/english/<model>/<prompt_type>/seed_42/
    config.yaml               # experiment config snapshot
    responses.json            # raw LLM responses
    metrics.json              # precision / recall / F1
    results.tsv               # per-sample predictions
    conf_matrices_reports.txt
```

Aggregate results per task are written to `results/{task}/{lang}/full_results.csv`.

---

## Reproducing Paper Results

### Step 1 — Run experiments

```bash
# id10m baseline
python run_id10m.py

# id10m-jam benchmark
python run_id10m_jam.py
```

### Step 2 — Compute comparisons (id10m vs. id10m-jam)

```bash
python analysis/compare_id10m_vs_id10m_jam.py
# Outputs: results/comparisons/{lang}/model_comparison_table.csv
#          results/comparisons/{lang}/{model}/{prompt}/seed_{N}/metrics.json
```

### Step 3 — Generate analysis tables and plots

```bash
# Per-run comparison table
python analysis/generate_comparison_table.py

# Main LLM comparison table (id10m vs. id10m-jam)
python analysis/compare_id10m_vs_id10m_jam.py

# Confusion histograms (zero-shot and few-shot)
python analysis/plot_confusion_histogram.py

# Sentence-level confusion analysis
python analysis/sentence_confusion_analysis.py
```

---

## Encoder Experiments

1. Download BERT encoder prediction files from HuggingFace and place them at `encoders/predictions/{language}/{model_name}/`
2. Run the analysis:

```bash
python analysis/generate_encoder_comparison_tables.py
python analysis/generate_encoder_comparison_tables.py --language english
```

---

## Data Generation

The `data/generation/` folder contains the pipeline used to create the id10m-jam confusing-context variants using Gemini. Requires a Gemini API key in `.env`:

```bash
cd data/generation
python variants_generator.py
```

---

## Results

### Zero-Shot English (seed 42)

| Model | id10m F1 | id10m-jam F1 | Δ F1 |
|---|---|---|---|
| Gemini 2.5 Pro | .964 | .946 | −.018 |
| Gemini 2.5 Flash-Lite | .959 | .923 | −.035 |
| Claude Sonnet 4 | .949 | .938 | −.011 |
| GPT-4o | .945 | .937 | −.008 |
| Llama 4 Scout | .939 | .950 | +.012 |
| GPT-4o-mini | .942 | .942 | .000 |
| o3-mini | .937 | .946 | +.009 |
| Qwen 2.5 72B | .916 | .948 | +.032 |
| DeepSeek-R1 | .925 | .929 | +.004 |

> Full results (few-shot, German, self-consistency) are reported in the paper.

---

## Citation

If you use this work, please cite:

**APA:**
Hashiloni, K. G., Livyatan, L., Hefetz, O., Mannor, A., Cohen, B., & Bar, K. (2026). ID10M-JAM: Stress-Testing Idiom Identification Under Challenging Context. In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)*.

**BibTeX:**
```bibtex
@inproceedings{hashiloni2026idiojam,
  title     = {{ID10M-JAM}: Stress-Testing Idiom Identification Under Challenging Context},
  author    = {Hashiloni, Kai Golan and Livyatan, Lior and Hefetz, Ofri and Mannor, Alon and Cohen, Bar and Bar, Kfir},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)},
  year      = {2026},
}
```

---

## License

Code: Apache 2.0. See [LICENSE](LICENSE).
Data: See the HuggingFace dataset card for data licensing terms.

## Contact

For questions or contributions: [kai.golanhashiloni@post.runi.ac.il](mailto:kai.golanhashiloni@post.runi.ac.il?subject=ID10M-JAM)
