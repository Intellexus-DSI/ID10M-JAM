# ID10M-JAM: Stress-Testing Idiom Identification Under Challenging Context

**ACL 2026** · [[Paper]]() · [[Dataset (HuggingFace)]]()

ID10M-JAM evaluates idiom identification under challenging confusing-context conditions. We benchmark 10 LLMs across English and German on two tasks: **id10m** — a standard idiom identification task — and **hard_idioms** — our new benchmark where each idiom-containing sentence is paired with LLM-generated confusing-context variants designed to mislead models into literal interpretations.

---

## Dataset

Both datasets are available on HuggingFace:

| Dataset | Description | Languages | Size |
|---|---|---|---|
| `hard_idioms` | Idiom identification with confusing-context variants (ours) | English, German | 534 EN / 561 DE |
| `id10m` | Standard idiom identification baseline | English, German | 178 EN |

```python
from datasets import load_dataset
ds = load_dataset("PLACEHOLDER/idiojam-hard-idioms")
```

> After downloading, place data files at `data/hard_idioms_data/` and `data/raw_id10m_data/` as expected by the experiment scripts.

---

## Repository Overview

| Path | Purpose |
|---|---|
| `experiments/src/` | Task utilities for `hard_idioms` and `id10m` (data loading, prompts, metrics) |
| `data_generation/` | Gemini-based pipeline that generated the confusing-context variants |
| `analysis/` | Scripts to reproduce all plots and comparison tables from the paper |
| `encoders_experiment/` | Place downloaded BERT encoder predictions here before running encoder analysis |

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

Run a zero-shot evaluation (GPT-4o-mini, hard_idioms, English):

```bash
python run_llm_eval_hard.py
```

---

## Running Experiments

### LLM Evaluation

Edit `config.yaml` to configure your task, model, and prompt type, then run:

```bash
# id10m task
python run_llm_eval.py

# hard_idioms task (supports self-consistency)
python run_llm_eval_hard.py

# Override config values via CLI
python run_llm_eval_hard.py --seed 43 --lang german --sc_runs 5
python run_llm_eval_hard.py --config_file my_config.yaml

# Re-evaluate existing responses without calling the API
python run_llm_eval_hard.py --responses_dir experiments/logs/hard_idioms/english/<exp_name>/run_001
```

### Key Config Parameters (`config.yaml`)

| Parameter | Options | Description |
|---|---|---|
| `task` | `hard_idioms`, `id10m` | Task to run |
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

Outputs are written to `experiments/logs/{task}/{lang}/{exp_name}/`:

```
experiments/logs/hard_idioms/english/<exp_name>/run_001/
    config.yaml               # experiment config snapshot
    responses.json            # raw LLM responses
    metrics.json              # precision / recall / F1
    results.tsv               # per-sample predictions
    conf_matrices_reports.txt
```

---

## Encoder Experiments

1. Download BERT encoder prediction files from HuggingFace and place them at `encoders_experiment/{language}/{model_name}/`
2. Run the analysis:

```bash
python analysis/generate_encoder_comparison_tables.py
python analysis/generate_encoder_comparison_tables.py --language english
```

---

## Reproducing Paper Results

Run analysis scripts from the repo root after running experiments:

```bash
# Main LLM comparison table (id10m vs. hard_idioms)
python analysis/compare_id10m_vs_hard_idioms.py

# Legacy comparison table
python analysis/generate_legacy_comparison_table.py

# Confusion histograms (zero-shot and few-shot)
python analysis/plot_confusion_histogram.py

# Sentence-level confusion analysis
python analysis/sentence_confusion_analysis.py
```

---

## Data Generation

The `data_generation/` folder contains the pipeline used to create the hard_idioms confusing-context variants using Gemini. Requires a Gemini API key in `.env`:

```bash
cd data_generation
python variants_generator.py
```

---

## Results

### Zero-Shot English (seed 42)

| Model | id10m F1 | hard_idioms F1 | Δ F1 |
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

```bibtex
@inproceedings{idiojam2026,
  title     = {ID10M-JAM: Stress-Testing Idiom Identification Under Challenging Context},
  author    = {},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2026},
}
```

---

## License

Code: Apache 2.0. See [LICENSE](LICENSE).
Data: See the HuggingFace dataset card for data licensing terms.

## Contact

For questions, open a GitHub issue or contact the authors.
