# Camera-Ready Repo Design: IdioJam (ACL 2026)

## Context

Paper: **ID10M-JAM: Stress-Testing Idiom Identification Under Challenging Context** (ACL 2026)

Source working repo: `work_IdioJam` (private, messy research workspace)
Target public repo: `IdioJam` (this repo — camera-ready code release)

Data (hard_idioms + id10m datasets) will be released on HuggingFace, not in this repo.

---

## Goals

- Reproducible evaluation code for LLM experiments (hard_idioms + id10m tasks)
- Data generation pipeline for the hard idioms confusing variants
- Analysis scripts to reproduce plots and tables from the paper
- Clean README with Quick Start, Citation, dataset links
- No raw results, no secrets, no internal tooling

---

## Folder Structure

```
IdioJam/
├── README.md
├── LICENSE
├── requirements.txt
├── config.yaml
├── keys.yaml.example
├── .gitignore
│
├── run_llm_eval.py            # entry point for id10m (renamed from run_exp.py)
├── run_llm_eval_hard.py       # entry point for hard_idioms + SC (renamed from run_exp_hard.py)
│
├── experiments/
│   └── src/
│       ├── hard_idioms.py
│       ├── id10m_utils.py
│       ├── models.py
│       ├── utils.py
│       ├── pydantic_schemas.py
│       ├── typed_schemas.py
│       └── bmc_munkres/
│
├── data_generation/           # renamed from data/scripts/
│   ├── variants_generator.py
│   ├── extract_sentences.py
│   ├── combine_parsed_pie.py
│   └── system_prompts.py
│
└── analysis/
    ├── compare_id10m_vs_hard_idioms.py
    ├── generate_legacy_comparison_table.py
    ├── generate_encoder_comparison_tables.py  # moved from scripts/
    ├── plot_confusion_histogram.py
    └── sentence_confusion_analysis.py
```

---

## Key Decisions

- **No results in repo**: users reproduce by running scripts; encoder predictions downloaded from HuggingFace into `encoders_experiment/`
- **No encoder training code**: encoder experiments were run externally; only the analysis script is included
- **Only hard_idioms + id10m task utils**: coam, parseme, lcp, magpie utils removed
- **Top-level entry points**: two `run_*.py` files at root following DharmaBench pattern
- **keys.yaml.example**: template showing required API key names, never the actual values
- **README includes a results table**: key numbers from the paper for at-a-glance reference

---

## What Gets Excluded

- `venv/`, `wandb/`, `.DS_Store`, `__pycache__/`
- `keys.yaml`, `.env`
- `data/raw_id10m_data/`, `data/hard_idioms_data/`, `data/id10m/` (→ HuggingFace)
- `experiments/src/coam_*`, `lcp_*`, `magpie_*`, `parseme_*`
- `experiments/logs/`, `experiments/results/`, `results/`
- `encoders_experiment/` result folders (→ HuggingFace)
- `CLEANUP_PLAN.md`, `tasks.md`, `CLAUDE.md`
- `scripts/reeval_*.py`, `scripts/copy_and_split_*.py`
- `experiments/complete_english_few_shot_sc.py`
