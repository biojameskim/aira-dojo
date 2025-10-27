#!/usr/bin/env python3
"""
Utility script to launch the three-task MCTS experiment locally without Slurm.

Runs each task in the `three_tasks` split with seeds [1, 2, 3] by calling
`python -m dojo.main_run` directly. Adjust the `TASKS` or `SEEDS` lists below
if you want to cover a different subset or number of seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Tasks pulled from src/dojo/configs/benchmark/mlebench/three_tasks.yaml
DEFAULT_TASKS = [
    "aptos2019-blindness-detection",
    "tabular-playground-series-may-2022",
    "mlsp-2013-birds",
]

DEFAULT_SEEDS = [1, 2, 3]
# Edit step_limit in /share/j_sun/jjk297/repos/aira-dojo/src/dojo/configs/solver/mlebench/mcts.yaml
# Edit time_limit_secs in /share/j_sun/jjk297/repos/aira-dojo/src/dojo/configs/solver/mcts.yaml


def _list_config_paths(base_dir: Path) -> set[Path]:
    if not base_dir.exists():
        return set()
    return {path for path in base_dir.glob("**/dojo_config.json")}


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _get_run_info(config_paths: set[Path], task: str, seed: int):
    matches: list[tuple[int, Path, dict, dict | None, bool]] = []
    for path in config_paths:
        data = _load_json(path)
        if not data:
            continue
        if data.get("task", {}).get("name") != task:
            continue
        if data.get("metadata", {}).get("seed") != seed:
            continue

        step_limit = data.get("solver", {}).get("step_limit")
        state_path = path.parent / "checkpoint/state.json"
        state_data = _load_json(state_path) if state_path.exists() else None
        has_summary = (path.parent / "best_node_summary.txt").exists()
        step_limit_val = step_limit if isinstance(step_limit, int) else -1
        matches.append((step_limit_val, path, data, state_data, has_summary))

    if not matches:
        return None

    matches.sort(key=lambda item: item[0], reverse=True)
    _, cfg_path, cfg_data, state_data, has_summary = matches[0]
    return cfg_path, cfg_data, state_data, has_summary


def _wait_for_run(base_dir: Path, known: set[Path], task: str, seed: int, timeout: float = 30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        current = _list_config_paths(base_dir)
        for path in current - known:
            data = _load_json(path)
            if not data:
                continue
            if data.get("task", {}).get("name") == task and data.get("metadata", {}).get("seed") == seed:
                known.add(path)
                return path, data
        time.sleep(1)
    return None, None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run three-task MCTS experiments locally.")
    parser.add_argument(
        "--exp-config",
        default="run_mlebench_aira_mcts_gdm",
        help="Hydra experiment config to launch (passed as +_exp=...).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help=(
            "List of task names to run (defaults to the three canonical MLE-Bench tasks). "
            "Example: --tasks aptos2019-blindness-detection tabular-playground-series-may-2022"
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="List of integer seeds to iterate over (e.g., --seeds 1 3 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent
    env_python = sys.executable
    logging_dir = os.environ.get("LOGGING_DIR")
    if not logging_dir:
        raise RuntimeError("LOGGING_DIR environment variable must be set to locate run outputs.")
    logs_base = Path(logging_dir) / "aira-dojo"
    config_paths = _list_config_paths(logs_base)
    known_configs = set(config_paths)

    base_cmd = [
        env_python,
        "-m",
        "dojo.main_run",
        f"+_exp={args.exp_config}",
        "logger.use_wandb=False",
    ]

    summary: list[tuple[str, int, str, Path, str]] = []

    for task in args.tasks:
        for seed in args.seeds:
            # Refresh the set of known configs to capture runs created outside this script
            config_paths |= _list_config_paths(logs_base)

            existing_info = _get_run_info(config_paths, task, seed)
            existing_run_id: str | None = None
            existing_path: Path | None = None
            existing_step_limit: int | None = None
            existing_current_step: int | None = None
            already_finished = False

            if existing_info:
                cfg_path, cfg_data, state_data, has_summary = existing_info
                existing_path = cfg_path
                existing_run_id = cfg_data.get("id", "<unknown>")
                step_limit = cfg_data.get("solver", {}).get("step_limit")
                if isinstance(step_limit, int):
                    existing_step_limit = step_limit
                if isinstance(state_data, dict):
                    step_val = state_data.get("current_step")
                    if isinstance(step_val, int):
                        existing_current_step = step_val

                already_finished = has_summary or (
                    existing_step_limit is not None
                    and existing_current_step is not None
                    and existing_current_step >= existing_step_limit
                )
                if already_finished:
                    print(
                        f"\n=== Skipping {task} (seed {seed}): already complete "
                        f"({existing_current_step}/{existing_step_limit}) ==="
                    )
                    summary.append(
                        (
                            task,
                            seed,
                            existing_run_id,
                            cfg_path.parent,
                            "already complete",
                        )
                    )
                    known_configs.add(cfg_path)
                    continue

            cmd = base_cmd + [
                f"task.name={task}",
                f"metadata.seed={seed}",
            ]
            print(f"\n=== Running {task} (seed {seed}) ===")
            result = subprocess.run(cmd, cwd=repo_root)
            if result.returncode != 0:
                print(f"Command failed with exit code {result.returncode}. Aborting remaining runs.")
                sys.exit(result.returncode)

            # Update the config cache and wait for the run metadata (only required for brand new runs)
            config_paths |= _list_config_paths(logs_base)
            if existing_info is None:
                config_path, _ = _wait_for_run(logs_base, known_configs, task, seed)
                if config_path:
                    config_paths.add(config_path)

            post_info = _get_run_info(config_paths, task, seed)
            if post_info:
                cfg_path, cfg_data, state_data, has_summary = post_info
                run_id = cfg_data.get("id", "<unknown>")
                run_dir = cfg_path.parent
                known_configs.add(cfg_path)
                status = "finished" if has_summary else "in progress"
                summary.append((task, seed, run_id, run_dir, status))
                print(f"Completed {task} (seed {seed}) -> {run_id} [{status}]")
            else:
                run_id = existing_run_id or "<unknown>"
                run_dir = existing_path.parent if existing_path else repo_root
                summary.append((task, seed, run_id, run_dir, "completed"))
                print(f"Completed {task} (seed {seed}) -> {run_id}")

    if summary:
        print("\nRun summary:")
        for task, seed, run_id, run_dir, status in summary:
            print(f"- {task} (seed {seed}) [{status}] -> {run_id} @ {run_dir}")


if __name__ == "__main__":
    main()
