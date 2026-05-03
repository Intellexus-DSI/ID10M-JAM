# Camera-Ready IdioJam Repo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate the empty public `IdioJam` repo with clean, reproducible code for the ACL 2026 paper "ID10M-JAM: Stress-Testing Idiom Identification Under Challenging Context."

**Architecture:** Copy relevant files from the private `work_IdioJam` workspace, strip unused task utilities (coam/parseme/lcp/magpie), add sys.path wiring so top-level runner scripts can reach `experiments/src/`, and write a paper-grade README.

**Tech Stack:** Python 3.13, LangChain, PyYAML, HuggingFace Transformers, wandb, agno (data generation)

**Source repo:** `/Users/liorlivyatan/LocalProjects/Thesis/work_IdioJam`
**Target repo:** `/Users/liorlivyatan/LocalProjects/Thesis/IdioJam`

---

## File Map

| Target path | Source | Action |
|---|---|---|
| `run_llm_eval.py` | `experiments/run_exp.py` | Copy + strip unused imports |
| `run_llm_eval_hard.py` | `experiments/run_exp_hard.py` | Copy + strip unused imports |
| `experiments/__init__.py` | — | Create (empty) |
| `experiments/src/__init__.py` | — | Create (empty) |
| `experiments/src/hard_idioms.py` | `experiments/src/hard_idioms.py` | Copy as-is |
| `experiments/src/id10m_utils.py` | `experiments/src/id10m_utils.py` | Copy as-is |
| `experiments/src/models.py` | `experiments/src/models.py` | Copy as-is |
| `experiments/src/utils.py` | `experiments/src/utils.py` | Copy as-is |
| `experiments/src/pydantic_schemas.py` | `experiments/src/pydantic_schemas.py` | Copy as-is |
| `experiments/src/typed_schemas.py` | `experiments/src/typed_schemas.py` | Copy as-is |
| `experiments/src/bmc_munkres/` | `experiments/src/bmc_munkres/` | Copy dir as-is |
| `data_generation/variants_generator.py` | `data/scripts/variants_generator.py` | Copy as-is |
| `data_generation/extract_sentences.py` | `data/scripts/extract_sentences.py` | Copy as-is |
| `data_generation/combine_parsed_pie.py` | `data/scripts/combine_parsed_pie.py` | Copy as-is |
| `data_generation/system_prompts.py` | `data/scripts/system_prompts.py` | Copy as-is |
| `analysis/compare_id10m_vs_hard_idioms.py` | `analysis/compare_id10m_vs_hard_idioms.py` | Copy as-is |
| `analysis/generate_legacy_comparison_table.py` | `analysis/generate_legacy_comparison_table.py` | Copy as-is |
| `analysis/generate_encoder_comparison_tables.py` | `scripts/generate_encoder_comparison_tables.py` | Copy as-is (ROOT path unaffected) |
| `analysis/plot_confusion_histogram.py` | `analysis/plot_confusion_histogram.py` | Copy as-is |
| `analysis/sentence_confusion_analysis.py` | `analysis/sentence_confusion_analysis.py` | Copy as-is |
| `config.yaml` | `config.yaml` | Copy as-is |
| `requirements.txt` | `requirements.txt` | Copy as-is |
| `keys.yaml.example` | — | Create new |
| `.gitignore` | — | Create new |
| `README.md` | `README.md` (stub) | Rewrite |

---

## Task 1: Create directory structure and copy boilerplate files

**Files:**
- Create: `experiments/__init__.py`
- Create: `experiments/src/__init__.py`
- Create: `data_generation/` (directory)
- Create: `analysis/` (directory)
- Copy: `config.yaml`, `requirements.txt`

- [ ] **Step 1: Create all directories and empty `__init__.py` files**

```bash
cd /Users/liorlivyatan/LocalProjects/Thesis/IdioJam
mkdir -p experiments/src data_generation analysis
touch experiments/__init__.py experiments/src/__init__.py
```

- [ ] **Step 2: Copy config and requirements**

```bash
cp /Users/liorlivyatan/LocalProjects/Thesis/work_IdioJam/config.yaml /Users/liorlivyatan/LocalProjects/Thesis/IdioJam/config.yaml
cp /Users/liorlivyatan/LocalProjects/Thesis/work_IdioJam/requirements.txt /Users/liorlivyatan/LocalProjects/Thesis/IdioJam/requirements.txt
```

- [ ] **Step 3: Create `.gitignore`**

Write to `/Users/liorlivyatan/LocalProjects/Thesis/IdioJam/.gitignore`:

```
# Secrets
keys.yaml
.env

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
venv/
.venv/
*.egg-info/

# Experiment outputs (reproduce locally)
experiments/logs/
experiments/results/
results/

# Encoder predictions (download from HuggingFace)
encoders_experiment/

# Data (on HuggingFace)
data/

# macOS
.DS_Store

# Weights & Biases
wandb/

# Editor
.idea/
.vscode/
```

- [ ] **Step 4: Create `keys.yaml.example`**

Write to `/Users/liorlivyatan/LocalProjects/Thesis/IdioJam/keys.yaml.example`:

```yaml
# Copy this file to keys.yaml and fill in your API keys.
# keys.yaml is gitignored — never commit your actual keys.
OPENAI_API_KEY: "sk-..."
GOOGLE_API_KEY: "..."
ANTHROPIC_API_KEY: "sk-ant-..."
TOGETHER_API_KEY: "..."
```

- [ ] **Step 5: Commit**

```bash
cd /Users/liorlivyatan/LocalProjects/Thesis/IdioJam
git add experiments/__init__.py experiments/src/__init__.py data_generation analysis config.yaml requirements.txt .gitignore keys.yaml.example
git commit -m "chore: scaffold directory structure, gitignore, keys template"
```

---

## Task 2: Copy experiments/src files

**Files:**
- Copy: all 6 src modules + bmc_munkres directory

- [ ] **Step 1: Copy the six source modules**

```bash
SRC=/Users/liorlivyatan/LocalProjects/Thesis/work_IdioJam/experiments/src
DST=/Users/liorlivyatan/LocalProjects/Thesis/IdioJam/experiments/src

cp "$SRC/hard_idioms.py" "$DST/hard_idioms.py"
cp "$SRC/id10m_utils.py" "$DST/id10m_utils.py"
cp "$SRC/models.py"      "$DST/models.py"
cp "$SRC/utils.py"       "$DST/utils.py"
cp "$SRC/pydantic_schemas.py" "$DST/pydantic_schemas.py"
cp "$SRC/typed_schemas.py"    "$DST/typed_schemas.py"
```

- [ ] **Step 2: Copy the vendored bmc_munkres algorithm**

```bash
cp -r /Users/liorlivyatan/LocalProjects/Thesis/work_IdioJam/experiments/src/bmc_munkres \
      /Users/liorlivyatan/LocalProjects/Thesis/IdioJam/experiments/src/bmc_munkres
```

- [ ] **Step 3: Verify all files landed correctly**

```bash
ls /Users/liorlivyatan/LocalProjects/Thesis/IdioJam/experiments/src/
```

Expected output (7 items + bmc_munkres dir):
```
__init__.py  bmc_munkres  hard_idioms.py  id10m_utils.py  models.py  pydantic_schemas.py  typed_schemas.py  utils.py
```

- [ ] **Step 4: Commit**

```bash
cd /Users/liorlivyatan/LocalProjects/Thesis/IdioJam
git add experiments/src/
git commit -m "feat: add experiments/src task utils for hard_idioms and id10m"
```

---

## Task 3: Create `run_llm_eval.py` (id10m runner)

**Files:**
- Create: `run_llm_eval.py` (based on `work_IdioJam/experiments/run_exp.py`)

The key changes from the original:
1. Add `sys.path` manipulation so `from src.utils import ...` resolves to `experiments/src/`
2. Remove imports for unused tasks: `lcp_utils`, `coam_utils`, `parseme_utils`, `parseme_vid_utils`, `magpie_utils`
3. Strip their branches from `get_task_utils()`

- [ ] **Step 1: Write `run_llm_eval.py`**

Write to `/Users/liorlivyatan/LocalProjects/Thesis/IdioJam/run_llm_eval.py`:

```python
"""
LLM evaluation runner for the id10m task.

Run from the repo root:
    python run_llm_eval.py
    python run_llm_eval.py --lang german
    python run_llm_eval.py --config_file my_config.yaml
    python run_llm_eval.py --responses_dir experiments/logs/id10m/english/some_exp/
"""

import sys
from pathlib import Path

# Allow `from src.X import ...` to resolve to experiments/src/
sys.path.insert(0, str(Path(__file__).parent / "experiments"))

import os
import yaml
import wandb
import json
import argparse
from argparse import Namespace
import pandas as pd
from transformers import set_seed
from datetime import datetime

from src.utils import (
    MERGE_COLUMNS,
    get_logger,
    set_keys,
    calc_metrics_cont,
    calc_metrics_classification,
    parse_response,
    calc_metrics_mwe,
    send_email
)
from src.models import get_model

from src.id10m_utils import ID10M_UTILS
from src.hard_idioms import HARD_IDIOMS_UTILS


# Define the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_file",
    type=str,
    default="config.yaml",
    help="Path to the config file",
)
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--lang", type=str, default=None, help="Language")
parser.add_argument(
    "--sc_runs",
    type=int,
    default=None,
    help="Number of self-consistency runs",
)
parser.add_argument(
    "--responses_dir",
    type=str,
    default=None,
    help="Directory with responses",
)

####################################################################################################
# Functions


def get_task_utils(task: str):
    if task == "id10m":
        utils = ID10M_UTILS
        calc_metrics = calc_metrics_classification
    elif task == "hard_idioms":
        utils = HARD_IDIOMS_UTILS
        calc_metrics = calc_metrics_classification
    else:
        raise ValueError(f"Task '{task}' is not supported. Choose from: id10m, hard_idioms")
    return Namespace(
        get_data=utils["get_data"],
        get_prompt_schema=utils["get_prompt_schema"],
        get_user_inputs=utils["get_user_inputs"],
        calc_metrics=calc_metrics,
        process_responses=utils["process_responses"],
    )


def main():
    # Get logger
    logger = get_logger(__name__)

    # Get CMD args
    cmd_args = parser.parse_args()
    logger.info(f"CMD args: {cmd_args}")

    # Load keys
    with open("keys.yaml", "r") as f:
        keys = yaml.safe_load(f)
    logger.info("Loaded API keys")
    # Set keys
    set_keys(keys)

    # Load config
    config_file = cmd_args.config_file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config: {config}")

    if "responses_dir" in cmd_args and cmd_args.responses_dir:
        config["responses_dir"] = cmd_args.responses_dir
        logger.info(f"Updated config with responses_dir: {config['responses_dir']}")

    # Check if responses were given
    if config["responses_dir"]:
        responses_dir = config["responses_dir"]
        # Get the original config
        with open(os.path.join(config["responses_dir"], "config.yaml"), "r") as f:
            orig_config = yaml.safe_load(f)
        # Update config with the original config
        config.update(orig_config)
        # Update responses_dir to the original one
        config["responses_dir"] = responses_dir
        logger.info(f"Updated config from {config['responses_dir']}")

    # Update config with CMD args
    for key, value in vars(cmd_args).items():
        if value is None:
            continue
        config[key] = value

    # Set seed
    set_seed(config["seed"])

    # Get experiment name
    model_name = config["model"].split("/")[-1]
    exp_name = f"{config['task']}_{model_name}_{config['prompt_type']}_shots_{config['shots']}_sc{config['sc_runs']}_tmp{config['temperature']}_seed{config['seed']}"

    # Get utils
    task_utils = get_task_utils(config["task"])

    # Add language to experiment name
    if config["lang"]:
        exp_name += f"_{config['lang']}"

    # Assert
    if "few" in config["prompt_type"]:
        assert config["shots"] > 0, "Shots must be greater than 0"
        assert config["shots"] % 2 == 0, "Shots must be even"
    if "zero" in config["prompt_type"]:
        assert config["shots"] == 0, "Shots must be 0 for zero-shot"
    if config["sc_runs"] > 1:
        assert (
            config["temperature"] == 0.8
        ), "Temperature must be 0.8 for self-consistency"

    # Load task results
    if config["lang"] and config["task"] in ["id10m", "hard_idioms"]:
        task_res_dir = os.path.join(
            config["results_dir"], config["task"], config["lang"]
        )
    else:
        task_res_dir = os.path.join(config["results_dir"], config["task"])
    os.makedirs(task_res_dir, exist_ok=True)

    task_res_file = os.path.join(task_res_dir, "full_results.csv")
    if os.path.exists(task_res_file):
        task_res = pd.read_csv(task_res_file)
    else:
        task_res = pd.DataFrame(columns=MERGE_COLUMNS)
        task_res.to_csv(task_res_file, index=False)

    # Create experiment directory
    if config["lang"] and config["task"] in ["id10m", "hard_idioms"]:
        exp_dir = os.path.join(config["logs_dir"], config["task"], config["lang"], exp_name)
    else:
        exp_dir = os.path.join(config["logs_dir"], config["task"], exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Add experiment metadata
    config["experiment_start_date"] = datetime.now().strftime("%Y-%m-%d")
    config["experiment_start_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config["model_checkpoint"] = "pending"

    # Write config to file
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # Initialize W&B
    if not config["debug"]:
        wandb.login()
        wandb.init(
            project=config["task"],
            config=config,
            name=exp_name,
            reinit=True,
        )

    # Load data
    train, test = task_utils.get_data(task=config["task"], lang=config["lang"])

    # Cut only respective language
    if config["lang"]:
        if "language" in test.columns:
            train = train[train["language"] == config["lang"]]
            test = test[test["language"] == config["lang"]]

    if train is not None:
        logger.info(f"Loaded train data: {train.shape}")
    logger.info(f"Loaded test data: {test.shape}")

    # Cut data for debugging
    if config["debug"]:
        if "debug_samples" in config and config.get("debug_samples") is not None:
            test = test.iloc[config["debug_samples"]]
        else:
            test = test[:config["num_samples"]]
        if train is not None:
            train = train.sample(min(len(train), 100), random_state=config["seed"])

    # Check if responses were given
    if config["responses_dir"]:
        responses_file = os.path.join(config["responses_dir"], "responses.json")
        with open(responses_file, "r", encoding="utf-8-sig") as f:
            responses = json.load(f)
        logger.info(f"Loaded responses: {len(responses)}")

    else:
        # Initialize model
        llm = get_model(config["model"], config["temperature"], config["use_rate_limiter"])

        # Get prompt template and schema
        prompt, schema = task_utils.get_prompt_schema(config=config, train=train)

        if schema:
            llm = llm.with_structured_output(schema, include_raw=True)
            structured = True
        else:
            structured = False

        # Build chain
        chain = prompt | llm

        # prepare data
        user_inputs = task_utils.get_user_inputs(test)

        # Aggregate results
        responses = []
        for _, row in test.iterrows():
            _data = {key: row[key] for key in test.columns}
            _data["responses"] = []
            responses.append(_data)

        # Run with multiple seeds/runs
        for run_index in range(config["sc_runs"]):
            if config["batched"]:
                logger.info(f"Running batch for run {run_index}")
                try:
                    raw_responses_run = chain.batch(user_inputs)
                    logger.info(
                        f"Generated {len(raw_responses_run)} raw responses for run {run_index}"
                    )
                except Exception as e:
                    logger.error(f"Error during batch run {run_index}: {e}")
            else:
                problematic_inputs_file = os.path.join(
                    exp_dir, f"problematic_inputs_run_{run_index}.txt"
                )
                with open(problematic_inputs_file, "w") as f:
                    f.write("Problematic inputs:\n")
                logger.info(f"Running individual invokes for run {run_index}")

                def invoke_row(i, row):
                    try:
                        return chain.invoke(row)
                    except Exception as e_individual:
                        logger.warning(
                            f"Error for input {i} in run {run_index}: {e_individual}"
                        )
                        with open(problematic_inputs_file, "a") as f:
                            f.write(f"Input {i}: {row}\n")
                            f.write(f"Error: {e_individual}\n\n")
                        return None

                raw_responses_run = list(
                    map(
                        lambda args: invoke_row(*args),
                        zip(range(len(user_inputs)), user_inputs),
                    )
                )

            # Capture model checkpoint from first valid response
            if config.get("model_checkpoint") == "pending" and raw_responses_run and run_index == 0:
                for resp in raw_responses_run:
                    if resp:
                        try:
                            if hasattr(resp, 'response_metadata'):
                                checkpoint = resp.response_metadata.get("model_name", config["model"])
                            elif isinstance(resp, dict) and "response_metadata" in resp:
                                checkpoint = resp["response_metadata"].get("model_name", config["model"])
                            else:
                                checkpoint = config["model"]
                            config["model_checkpoint"] = checkpoint
                            logger.info(f"Captured model checkpoint: {checkpoint}")
                            with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
                                yaml.dump(config, f)
                            break
                        except Exception as e:
                            logger.warning(f"Could not extract model checkpoint: {e}")
                            config["model_checkpoint"] = config["model"]

            # Save responses to results
            for i, resp in enumerate(raw_responses_run):
                try:
                    responses[i]["responses"].append(parse_response(resp, structured))
                except Exception as e:
                    logger.error(f"Error parsing response: {e}")
                    responses[i]["responses"].append({})

    # Save results
    with open(os.path.join(exp_dir, "responses.json"), "w", encoding="utf-8-sig") as f:
        json.dump(responses, f, indent=1, ensure_ascii=False)
    logger.info(f"Saved responses to {exp_dir}")

    # Calculate metrics
    try:
        metrics, test, run_res, log_cm_report = task_utils.process_responses(
            responses,
            test,
            task_utils.calc_metrics,
            lang=config["lang"],
            sc_runs=config["sc_runs"],
        )
    except Exception as e:
        raise RuntimeError(f"Error processing responses: {e}")

    logger.info(f"Metrics: {metrics}")

    if not config["debug"]:
        wandb.log({"metrics": metrics})

    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if log_cm_report:
        with open(os.path.join(exp_dir, "conf_matrices_reports.txt"), "w") as f:
            for lang in log_cm_report.keys():
                cm = log_cm_report[lang]["conf_matrix"]
                report = log_cm_report[lang]["report"]
                f.write(f"Language: {lang}\n")
                f.write(f"Confusion matrix:\n{cm}\n\n")
                f.write(f"Classification report:\n{report}\n")
                f.write("\n")

    test.to_csv(os.path.join(exp_dir, "results.tsv"), index=False, sep="\t")

    for col in MERGE_COLUMNS:
        run_res[col] = config[col]

    if not config["debug"]:
        task_res = task_res.set_index(MERGE_COLUMNS)
        run_res = run_res.set_index(MERGE_COLUMNS)
        task_res.update(run_res)
        task_res = task_res.combine_first(run_res)
        task_res.reset_index(inplace=True)
        task_res.to_csv(task_res_file, index=False)

    logger.info(f"Saved results to {exp_dir}")

    if not config["debug"]:
        wandb.finish()


####################################################################################################
if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
cd /Users/liorlivyatan/LocalProjects/Thesis/IdioJam
git add run_llm_eval.py
git commit -m "feat: add run_llm_eval.py entry point for id10m and hard_idioms tasks"
```

---

## Task 4: Create `run_llm_eval_hard.py` (hard idioms + SC runner)

**Files:**
- Create: `run_llm_eval_hard.py` (based on `work_IdioJam/experiments/run_exp_hard.py`)

Same changes as Task 3, applied to the hard idioms runner.

- [ ] **Step 1: Write `run_llm_eval_hard.py`**

Write to `/Users/liorlivyatan/LocalProjects/Thesis/IdioJam/run_llm_eval_hard.py`:

```python
"""
LLM evaluation runner for the hard_idioms task with self-consistency support.

Run from the repo root:
    python run_llm_eval_hard.py
    python run_llm_eval_hard.py --seed 43 --lang german --sc_runs 5
    python run_llm_eval_hard.py --config_file my_config.yaml
    python run_llm_eval_hard.py --responses_dir experiments/logs/hard_idioms/english/some_exp/run_001
"""

import sys
from pathlib import Path

# Allow `from src.X import ...` to resolve to experiments/src/
sys.path.insert(0, str(Path(__file__).parent / "experiments"))

import os
import yaml
import wandb
import json
import argparse
from argparse import Namespace
import pandas as pd
from transformers import set_seed
from datetime import datetime

from src.utils import (
    MERGE_COLUMNS,
    get_logger,
    set_keys,
    calc_metrics_classification,
    parse_response,
)
from src.models import get_model

from src.id10m_utils import ID10M_UTILS
from src.hard_idioms import HARD_IDIOMS_UTILS


# Define the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_file",
    type=str,
    default="config.yaml",
    help="Path to the config file",
)
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--lang", type=str, default=None, help="Language")
parser.add_argument(
    "--sc_runs",
    type=int,
    default=None,
    help="Number of self-consistency runs",
)
parser.add_argument(
    "--responses_dir",
    type=str,
    default=None,
    help="Directory with responses",
)
parser.add_argument(
    "--resume_dir",
    type=str,
    default=None,
    help="Resume a partial SC run from this directory (skips completed batches)",
)
parser.add_argument(
    "--data_path",
    type=str,
    default=None,
    help="Custom data file path (for correction runs)",
)

####################################################################################################
# Functions


def get_task_utils(task: str):
    if task == "id10m":
        utils = ID10M_UTILS
        calc_metrics = calc_metrics_classification
    elif task == "hard_idioms":
        utils = HARD_IDIOMS_UTILS
        calc_metrics = calc_metrics_classification
    else:
        raise ValueError(f"Task '{task}' is not supported. Choose from: id10m, hard_idioms")
    return Namespace(
        get_data=utils["get_data"],
        get_prompt_schema=utils["get_prompt_schema"],
        get_user_inputs=utils["get_user_inputs"],
        calc_metrics=calc_metrics,
        process_responses=utils["process_responses"],
    )


def create_run_directory(base_exp_dir: str) -> str:
    os.makedirs(base_exp_dir, exist_ok=True)
    existing_runs = []
    if os.path.exists(base_exp_dir):
        for item in os.listdir(base_exp_dir):
            item_path = os.path.join(base_exp_dir, item)
            if os.path.isdir(item_path) and item.startswith('run_'):
                try:
                    run_num = int(item.split('_')[1])
                    existing_runs.append(run_num)
                except (IndexError, ValueError):
                    continue
    next_run = max(existing_runs) + 1 if existing_runs else 1
    run_dir_name = f"run_{next_run:03d}"
    run_dir_path = os.path.join(base_exp_dir, run_dir_name)
    os.makedirs(run_dir_path, exist_ok=True)
    return run_dir_path


def main():
    logger = get_logger(__name__)
    cmd_args = parser.parse_args()
    logger.info(f"CMD args: {cmd_args}")

    with open("keys.yaml", "r") as f:
        keys = yaml.safe_load(f)
    logger.info("Loaded API keys")
    set_keys(keys)

    config_file = cmd_args.config_file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config: {config}")

    if "responses_dir" in cmd_args and cmd_args.responses_dir:
        config["responses_dir"] = cmd_args.responses_dir
        logger.info(f"Updated config with responses_dir: {config['responses_dir']}")

    if config["responses_dir"]:
        responses_dir = config["responses_dir"]
        with open(os.path.join(config["responses_dir"], "config.yaml"), "r") as f:
            orig_config = yaml.safe_load(f)
        config.update(orig_config)
        config["responses_dir"] = responses_dir
        logger.info(f"Updated config from {config['responses_dir']}")

    for key, value in vars(cmd_args).items():
        if value is None:
            continue
        config[key] = value

    set_seed(config["seed"])

    model_name = config["model"].split("/")[-1]
    exp_name = f"{config['task']}_{model_name}_{config['prompt_type']}_shots_{config['shots']}_sc{config['sc_runs']}_tmp{config['temperature']}_seed{config['seed']}"

    task_utils = get_task_utils(config["task"])

    if config["lang"]:
        exp_name += f"_{config['lang']}"

    if "few" in config["prompt_type"]:
        assert config["shots"] > 0, "Shots must be greater than 0"
        assert config["shots"] % 2 == 0, "Shots must be even"
    if "zero" in config["prompt_type"]:
        assert config["shots"] == 0, "Shots must be 0 for zero-shot"
    if config["sc_runs"] > 1:
        assert (
            config["temperature"] == 0.8
        ), "Temperature must be 0.8 for self-consistency"

    if config["lang"] and config["task"] in ["id10m", "hard_idioms"]:
        task_res_dir = os.path.join(
            config["results_dir"], config["task"], config["lang"]
        )
    else:
        task_res_dir = os.path.join(config["results_dir"], config["task"])
    os.makedirs(task_res_dir, exist_ok=True)

    task_res_file = os.path.join(task_res_dir, "full_results.csv")
    if os.path.exists(task_res_file):
        task_res = pd.read_csv(task_res_file)
    else:
        task_res = pd.DataFrame(columns=MERGE_COLUMNS)
        task_res.to_csv(task_res_file, index=False)

    resume_dir = cmd_args.resume_dir
    if resume_dir:
        exp_dir = os.path.abspath(resume_dir)
        if not os.path.isdir(exp_dir):
            raise ValueError(f"--resume_dir does not exist: {exp_dir}")
        run_number = os.path.basename(exp_dir)
        logger.info(f"Resuming run from {exp_dir}")
    else:
        if config["lang"] and config["task"] in ["id10m", "hard_idioms"]:
            mother_dir = os.path.join(config["logs_dir"], config["task"], config["lang"])
        else:
            mother_dir = os.path.join(config["logs_dir"], config["task"])
        base_exp_dir = os.path.join(mother_dir, exp_name)
        exp_dir = create_run_directory(base_exp_dir)
        logger.info(f"Created experiment directory: {exp_dir}")
        run_number = os.path.basename(exp_dir)

    config["experiment_start_date"] = datetime.now().strftime("%Y-%m-%d")
    config["experiment_start_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config["model_checkpoint"] = "pending"

    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    if not config["debug"]:
        wandb.login()
        wandb.init(
            project=config["task"],
            config=config,
            name=exp_name,
            reinit=True,
        )

    def convert_nan_to_none(obj):
        if isinstance(obj, dict):
            return {k: convert_nan_to_none(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_nan_to_none(item) for item in obj]
        elif isinstance(obj, float) and pd.isna(obj):
            return None
        return obj

    test = task_utils.get_data(
        task=config["task"],
        lang=config["lang"],
        data_path=config.get("data_path")
    )
    train = None

    if config["lang"]:
        if "language" in test.columns:
            test = test[test["language"] == config["lang"]]

    logger.info(f"Loaded test data: {test.shape}")

    if config["debug"]:
        if "debug_samples" in config and config.get("debug_samples") is not None:
            test = test.iloc[config["debug_samples"]]
        else:
            test = test[:config["num_samples"]]

    if config["responses_dir"]:
        responses_file = os.path.join(config["responses_dir"], "responses.json")
        with open(responses_file, "r", encoding="utf-8-sig") as f:
            responses = json.load(f)
        logger.info(f"Loaded responses: {len(responses)}")

    else:
        llm = get_model(config["model"], config["temperature"], config["use_rate_limiter"])
        prompt, schema = task_utils.get_prompt_schema(config=config, train=train)

        if schema:
            llm = llm.with_structured_output(schema, include_raw=True)
            structured = True
        else:
            structured = False

        chain = prompt | llm
        user_inputs = task_utils.get_user_inputs(test)

        partial_responses_file = os.path.join(exp_dir, "responses.json")
        if resume_dir and os.path.exists(partial_responses_file):
            with open(partial_responses_file, "r", encoding="utf-8-sig") as f:
                responses = json.load(f)
            start_run = min(len(r["responses"]) for r in responses)
            logger.info(f"Resuming from SC batch {start_run}")
        else:
            responses = []
            for _, row in test.iterrows():
                _data = {key: row[key] for key in test.columns}
                _data["responses"] = []
                responses.append(_data)
            start_run = 0

        for run_index in range(start_run, config["sc_runs"]):
            if config["batched"]:
                logger.info(f"Running batch for run {run_index}")
                try:
                    raw_responses_run = chain.batch(user_inputs)
                    logger.info(f"Generated {len(raw_responses_run)} raw responses for run {run_index}")
                except Exception as e:
                    logger.error(f"Error during batch run {run_index}: {e}")
                    raw_responses_run = []
                    continue
            else:
                problematic_inputs_file = os.path.join(
                    exp_dir, f"problematic_inputs_run_{run_index}.txt"
                )
                with open(problematic_inputs_file, "w") as f:
                    f.write("Problematic inputs:\n")
                logger.info(f"Running individual invokes for run {run_index}")

                def invoke_row(i, row):
                    try:
                        return chain.invoke(row)
                    except Exception as e_individual:
                        logger.warning(f"Error for input {i} in run {run_index}: {e_individual}")
                        with open(problematic_inputs_file, "a") as f:
                            f.write(f"Input {i}: {row}\n")
                            f.write(f"Error: {e_individual}\n\n")
                        return None

                raw_responses_run = list(
                    map(
                        lambda args: invoke_row(*args),
                        zip(range(len(user_inputs)), user_inputs),
                    )
                )

            if config.get("model_checkpoint") == "pending" and raw_responses_run and run_index == 0:
                for resp in raw_responses_run:
                    if resp:
                        try:
                            if hasattr(resp, 'response_metadata'):
                                checkpoint = resp.response_metadata.get("model_name", config["model"])
                            elif isinstance(resp, dict) and "response_metadata" in resp:
                                checkpoint = resp["response_metadata"].get("model_name", config["model"])
                            else:
                                checkpoint = config["model"]
                            config["model_checkpoint"] = checkpoint
                            logger.info(f"Captured model checkpoint: {checkpoint}")
                            with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
                                yaml.dump(config, f)
                            break
                        except Exception as e:
                            logger.warning(f"Could not extract model checkpoint: {e}")
                            config["model_checkpoint"] = config["model"]

            for i, resp in enumerate(raw_responses_run):
                try:
                    responses[i]["responses"].append(parse_response(resp, structured))
                except Exception as e:
                    logger.error(f"Error parsing response: {e}")
                    responses[i]["responses"].append({})

            # Checkpoint after every SC batch so runs can be resumed on failure
            _checkpoint_clean = convert_nan_to_none(responses)
            with open(os.path.join(exp_dir, "responses.json"), "w", encoding="utf-8-sig") as f:
                json.dump(_checkpoint_clean, f, indent=1, ensure_ascii=False)
            logger.info(f"Checkpointed responses after SC batch {run_index + 1}/{config['sc_runs']}")

    responses_clean = convert_nan_to_none(responses)
    with open(os.path.join(exp_dir, "responses.json"), "w", encoding="utf-8-sig") as f:
        json.dump(responses_clean, f, indent=1, ensure_ascii=False)
    logger.info(f"Saved responses to {exp_dir}")

    try:
        metrics, test, run_res, log_cm_report = task_utils.process_responses(
            responses,
            test,
            task_utils.calc_metrics,
            lang=config["lang"],
            sc_runs=config["sc_runs"],
        )
    except Exception as e:
        raise RuntimeError(f"Error processing responses: {e}")

    logger.info(f"Metrics: {metrics}")

    if not config["debug"]:
        wandb.log({"metrics": metrics})

    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if log_cm_report:
        with open(os.path.join(exp_dir, "conf_matrices_reports.txt"), "w") as f:
            for lang in log_cm_report.keys():
                cm = log_cm_report[lang]["conf_matrix"]
                report = log_cm_report[lang]["report"]
                f.write(f"Language: {lang}\n")
                f.write(f"Confusion matrix:\n{cm}\n\n")
                f.write(f"Classification report:\n{report}\n")
                f.write("\n")

    test.to_csv(os.path.join(exp_dir, "results.tsv"), index=False, sep="\t")

    for col in MERGE_COLUMNS:
        run_res[col] = config[col]

    if not config["debug"]:
        task_res = task_res.set_index(MERGE_COLUMNS)
        run_res = run_res.set_index(MERGE_COLUMNS)
        task_res.update(run_res)
        task_res = task_res.combine_first(run_res)
        task_res.reset_index(inplace=True)
        task_res.to_csv(task_res_file, index=False)

    logger.info(f"Saved results to {exp_dir}")

    if not config["debug"]:
        wandb.finish()


####################################################################################################
if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
cd /Users/liorlivyatan/LocalProjects/Thesis/IdioJam
git add run_llm_eval_hard.py
git commit -m "feat: add run_llm_eval_hard.py entry point for hard_idioms with SC support"
```

---

## Task 5: Copy data_generation scripts

**Files:**
- Copy: 4 scripts from `work_IdioJam/data/scripts/` → `data_generation/`

- [ ] **Step 1: Copy the scripts**

```bash
SRC=/Users/liorlivyatan/LocalProjects/Thesis/work_IdioJam/data/scripts
DST=/Users/liorlivyatan/LocalProjects/Thesis/IdioJam/data_generation

cp "$SRC/variants_generator.py"   "$DST/variants_generator.py"
cp "$SRC/extract_sentences.py"    "$DST/extract_sentences.py"
cp "$SRC/combine_parsed_pie.py"   "$DST/combine_parsed_pie.py"
cp "$SRC/system_prompts.py"       "$DST/system_prompts.py"
```

- [ ] **Step 2: Verify**

```bash
ls /Users/liorlivyatan/LocalProjects/Thesis/IdioJam/data_generation/
```

Expected: `variants_generator.py  extract_sentences.py  combine_parsed_pie.py  system_prompts.py`

- [ ] **Step 3: Commit**

```bash
cd /Users/liorlivyatan/LocalProjects/Thesis/IdioJam
git add data_generation/
git commit -m "feat: add data_generation pipeline scripts"
```

---

## Task 6: Copy analysis scripts

**Files:**
- Copy: 4 scripts from `work_IdioJam/analysis/` → `analysis/`
- Copy: 1 script from `work_IdioJam/scripts/` → `analysis/` (encoder table generator)

Note: `generate_encoder_comparison_tables.py` uses `ROOT = Path(__file__).resolve().parent.parent` to find the repo root. Moving it from `scripts/` to `analysis/` produces the same ROOT (one level up = repo root in both cases).

- [ ] **Step 1: Copy analysis scripts**

```bash
SRC=/Users/liorlivyatan/LocalProjects/Thesis/work_IdioJam/analysis
DST=/Users/liorlivyatan/LocalProjects/Thesis/IdioJam/analysis

cp "$SRC/compare_id10m_vs_hard_idioms.py"      "$DST/compare_id10m_vs_hard_idioms.py"
cp "$SRC/generate_legacy_comparison_table.py"  "$DST/generate_legacy_comparison_table.py"
cp "$SRC/plot_confusion_histogram.py"           "$DST/plot_confusion_histogram.py"
cp "$SRC/sentence_confusion_analysis.py"        "$DST/sentence_confusion_analysis.py"
```

- [ ] **Step 2: Copy encoder comparison script from scripts/**

```bash
cp /Users/liorlivyatan/LocalProjects/Thesis/work_IdioJam/scripts/generate_encoder_comparison_tables.py \
   /Users/liorlivyatan/LocalProjects/Thesis/IdioJam/analysis/generate_encoder_comparison_tables.py
```

- [ ] **Step 3: Verify**

```bash
ls /Users/liorlivyatan/LocalProjects/Thesis/IdioJam/analysis/
```

Expected (5 files):
```
compare_id10m_vs_hard_idioms.py  generate_encoder_comparison_tables.py
generate_legacy_comparison_table.py  plot_confusion_histogram.py  sentence_confusion_analysis.py
```

- [ ] **Step 4: Commit**

```bash
cd /Users/liorlivyatan/LocalProjects/Thesis/IdioJam
git add analysis/
git commit -m "feat: add analysis scripts for plots, tables, and encoder evaluation"
```

---

## Task 7: Write README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write the full README**

Write to `/Users/liorlivyatan/LocalProjects/Thesis/IdioJam/README.md`:

```markdown
# ID10M-JAM: Stress-Testing Idiom Identification Under Challenging Context

**ACL 2026** · [[Paper]]() · [[Dataset (HuggingFace)]]()

ID10M-JAM is a benchmark for evaluating idiom identification under challenging confusing-context conditions. We test 10 LLMs (GPT-4o, Gemini 2.5, Claude Sonnet, and others) across English and German on two tasks: a standard idiom identification task (id10m) and our new hard-idioms benchmark with LLM-generated confusing context variants.

---

## Dataset

Download the datasets from HuggingFace:

| Dataset | Languages | Size |
|---|---|---|
| `hard_idioms` (ours) | English, German | 534 (EN) / 561 (DE) |
| `id10m` (baseline) | English, German | 178 (EN) |

```bash
# Example: load via HuggingFace datasets
from datasets import load_dataset
ds = load_dataset("PLACEHOLDER/idiojam-hard-idioms")
```

> After downloading, place the data files at `data/hard_idioms_data/` and `data/raw_id10m_data/` as expected by the experiment scripts.

---

## Repository Overview

| Folder | Purpose |
|---|---|
| `experiments/src/` | Task utilities for `hard_idioms` and `id10m` (data loading, prompts, evaluation) |
| `data_generation/` | Pipeline that generated confusing-context variants using Gemini |
| `analysis/` | Scripts to reproduce all plots and tables from the paper |
| `encoders_experiment/` | Place downloaded encoder predictions here before running analysis |

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

Run your first experiment (zero-shot, GPT-4o-mini, hard_idioms, English):

```bash
python run_llm_eval_hard.py
```

---

## Running Experiments

### LLM Evaluation

Edit `config.yaml` to set your task, model, and prompt type, then run:

```bash
# id10m task
python run_llm_eval.py

# hard_idioms task (with self-consistency support)
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
| `model` | see below | LLM to evaluate |
| `prompt_type` | `zero_shot`, `few_shot_cot_best` | Prompting strategy |
| `shots` | `0`, `10` | Few-shot examples (must be even; 0 for zero-shot) |
| `sc_runs` | `1`, `5` | Self-consistency runs (1 = disabled) |
| `temperature` | `0.3`, `0.8` | Use 0.3 without SC, 0.8 with SC |
| `debug` | `true`, `false` | Run on 5 samples, skip W&B |

### Supported Models

| Provider | Model string |
|---|---|
| OpenAI | `gpt-4o`, `gpt-4o-mini`, `o3-mini` |
| Google | `gemini-2.5-pro`, `gemini-2.5-flash-lite` |
| Anthropic | `claude-sonnet-4-20250514`, `claude-3-haiku-20240307` |
| Together.ai | `meta-llama/Llama-4-Scout-17B-16E-Instruct`, `deepseek-ai/DeepSeek-R1`, `Qwen/Qwen2.5-72B-Instruct-Turbo` |

### Experiment Outputs

Results are written to `experiments/logs/{task}/{lang}/{exp_name}/`:

```
experiments/logs/hard_idioms/english/<exp_name>/run_001/
    config.yaml          # experiment config snapshot
    responses.json       # raw LLM responses
    metrics.json         # precision / recall / F1
    results.tsv          # per-sample predictions
    conf_matrices_reports.txt
```

---

## Encoder Experiments

1. Download encoder prediction files from HuggingFace and place them at `encoders_experiment/{language}/{model_name}/`
2. Run the analysis:

```bash
python analysis/generate_encoder_comparison_tables.py
python analysis/generate_encoder_comparison_tables.py --language english
```

---

## Reproducing Paper Results

Run analysis scripts from the repo root after running experiments:

```bash
# Main comparison table (id10m vs hard_idioms)
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

The `data_generation/` folder contains the pipeline used to create the hard-idioms confusing-context variants. Requires a Gemini API key in `.env`:

```bash
cd data_generation
python variants_generator.py
```

---

## Results

### Hard Idioms — Zero-Shot (English)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| GPT-4o | — | — | — | — |
| GPT-4o-mini | — | — | — | — |
| Gemini 2.5 Pro | — | — | — | — |
| Claude Sonnet 4 | — | — | — | — |
| DeepSeek-R1 | — | — | — | — |

> Fill in paper numbers before publishing.

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
Data: See the HuggingFace dataset card for data licensing details.

## Contact

For questions, open a GitHub issue or contact the authors.
```

- [ ] **Step 2: Commit**

```bash
cd /Users/liorlivyatan/LocalProjects/Thesis/IdioJam
git add README.md
git commit -m "docs: add camera-ready README with Quick Start, config reference, and citation"
```

---

## Self-Review

**Spec coverage check:**
- ✅ Top-level entry points (`run_llm_eval.py`, `run_llm_eval_hard.py`) — Tasks 3 & 4
- ✅ experiments/src for hard_idioms + id10m only — Task 2
- ✅ data_generation/ — Task 5
- ✅ analysis/ including encoder table generator — Task 6
- ✅ config.yaml, requirements.txt — Task 1
- ✅ keys.yaml.example — Task 1
- ✅ .gitignore — Task 1
- ✅ README with HuggingFace links, results table, citation — Task 7
- ✅ No results, no raw data, no secrets, no venv/wandb — enforced by .gitignore + file map

**Placeholder scan:**
- README results table has `—` placeholders with explicit instruction to fill before publishing — acceptable, user must fill with paper numbers
- HuggingFace URL has `PLACEHOLDER` — acceptable, URL not yet known
- Citation author field is empty — acceptable, user knows their author list

**Type consistency:** No new types introduced; all imports preserved from original scripts.
