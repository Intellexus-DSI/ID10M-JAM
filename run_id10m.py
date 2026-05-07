"""
LLM evaluation runner for the id10m task.

Run from the repo root:
    python run_id10m.py
    python run_id10m.py --lang german
    python run_id10m.py --config_file my_config.yaml
    python run_id10m.py --responses_dir results/id10m/english/<model>/<prompt_type>/seed_42
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

from utils.utils import (
    MERGE_COLUMNS,
    get_logger,
    set_keys,
    calc_metrics_cont,
    calc_metrics_classification,
    parse_response,
    calc_metrics_mwe,
    send_email
)
from utils.models import get_model

from src.id10m_utils import ID10M_UTILS
from src.id10m_jam import ID10M_JAM_UTILS


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
    elif task == "id10m_jam":
        utils = ID10M_JAM_UTILS
        calc_metrics = calc_metrics_classification
    else:
        raise ValueError(f"Task '{task}' is not supported. Choose from: id10m, id10m_jam")
    return Namespace(
        get_data=utils["get_data"],
        get_prompt_schema=utils["get_prompt_schema"],
        get_user_inputs=utils["get_user_inputs"],
        calc_metrics=calc_metrics,
        process_responses=utils["process_responses"],
    )


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

    if config["lang"] and config["task"] in ["id10m", "id10m_jam"]:
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

    exp_dir = os.path.join(
        config["results_dir"], config["task"], config["lang"],
        model_name, config["prompt_type"], f"seed_{config['seed']}"
    )
    os.makedirs(exp_dir, exist_ok=True)

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

    train, test = task_utils.get_data(task=config["task"], lang=config["lang"])

    if config["lang"]:
        if "language" in test.columns:
            train = train[train["language"] == config["lang"]]
            test = test[test["language"] == config["lang"]]

    if train is not None:
        logger.info(f"Loaded train data: {train.shape}")
    logger.info(f"Loaded test data: {test.shape}")

    if config["debug"]:
        if "debug_samples" in config and config.get("debug_samples") is not None:
            test = test.iloc[config["debug_samples"]]
        else:
            test = test[:config["num_samples"]]
        if train is not None:
            train = train.sample(min(len(train), 100), random_state=config["seed"])

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

        responses = []
        for _, row in test.iterrows():
            _data = {key: row[key] for key in test.columns}
            _data["responses"] = []
            responses.append(_data)

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

    with open(os.path.join(exp_dir, "responses.json"), "w", encoding="utf-8-sig") as f:
        json.dump(responses, f, indent=1, ensure_ascii=False)
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
