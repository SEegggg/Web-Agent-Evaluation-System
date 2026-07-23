"""
WorkflowBenchmark — AgentLab-compatible Benchmark adapter for Agent A eval.

Wraps the .md task registry (ALL_TASK_IDS) into a `bgym.Benchmark` object
that AgentLab's `make_study()` / `Study.run()` can consume directly.

Usage:
    from browsergym.agent_a_eval.benchmark import create_workflow_benchmark

    benchmark = create_workflow_benchmark(
        task_filter=["data_analysis"],
        stability_seeds=[42, 123, 456],
        max_steps=200,
        headless=True,
    )

    from agentlab.experiments.study import make_study
    study = make_study(agent_args, benchmark)
    study.run(n_jobs=4, parallel_backend="joblib")
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import pandas as pd

from browsergym.experiments.benchmark.base import Benchmark, HighLevelActionSetArgs

# Use agentlab's EnvArgs as BASE CLASS (NOT browsergym.experiments.loop.EnvArgs).
# agentlab's _convert_env_args() keeps its own EnvArgs subclasses as-is
# (no conversion), avoiding TypeError from extra fields.
from agentlab.experiments.loop import EnvArgs as _AgentLabEnvArgs

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Custom EnvArgs subclass that guarantees gym environment registration
# in EVERY worker process (critical for Windows/joblib spawn mode).
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AgentAEvalEnvArgs(_AgentLabEnvArgs):
    """
    agentlab EnvArgs subclass that auto-registers agent_a_eval gym environments.

    On Windows, joblib uses the ``loky`` backend (spawn mode) — each worker
    is a fresh Python process that does NOT inherit gym registrations from the
    main process.  Overriding ``make_env()`` ensures ``import browsergym.agent_a_eval``
    runs in whichever process is about to call ``gym.make()``.
    """

    def make_env(self, action_mapping, exp_dir, exp_task_kwargs=None, use_raw_page_output=True):
        """Same as the parent, but first triggers gym environment registration."""
        # Ensure our tasks are registered in this process (idempotent import).
        # Must happen BEFORE gym.make() — otherwise NameNotFound on Windows.
        import browsergym.agent_a_eval  # noqa: F401

        return super().make_env(
            action_mapping=action_mapping,
            exp_dir=exp_dir,
            exp_task_kwargs=exp_task_kwargs or {},
            use_raw_page_output=use_raw_page_output,
        )

# Default action set for agent_a_eval tasks.
# Matches what DemoAgent + the driver prompt expects.
DEFAULT_ACTION_SET_ARGS = HighLevelActionSetArgs(
    subsets=("chat", "tab", "nav", "bid", "infeas"),
    multiaction=False,
    strict=False,
    retry_with_force=False,
    demo_mode="off",
)


def create_workflow_benchmark(
    task_filter: Optional[list[str]] = None,
    generalization_datasets: int = 3,
    stability_seeds: Optional[list[int]] = None,
    max_steps: int = 200,
    headless: bool = True,
    viewport: Optional[dict] = None,
    slow_mo: Optional[int] = None,
    storage_state: Optional[str] = None,
    user_data_dir: Optional[str] = None,
    skip_generalization: bool = False,
    skip_stability: bool = False,
    benchmark_name: str = "agent_a_eval",
) -> Benchmark:
    """
    Create an AgentLab-compatible Benchmark from registered agent_a_eval tasks.

    Each registered gym task ID (e.g. ``agent_a_eval.data_analysis.ds0``)
    becomes one or more EnvArgs entries depending on test mode:

    - **Generalization**: one entry per (task, dataset_index) with seed=42
    - **Stability**: one entry per (task, dataset_index=0, seed)

    When both are skipped, runs a single quick test: iris.csv + seed=42.

    Args:
        task_filter: optional list of task names (stem, without .md).
            ``[]`` or ``None`` = all discovered tasks.
        generalization_datasets: number of datasets per task for
            generalization testing.
        stability_seeds: seeds for stability testing (default: [42, 123, 456]).
        max_steps: max steps per experiment.
        headless: run browser headless.
        viewport: browser viewport dict (width, height).
        slow_mo: Playwright slow_mo in ms.
        storage_state: path to Playwright storage_state JSON.
        user_data_dir: path to persistent browser profile directory.
        skip_generalization: skip generalization tests.
        skip_stability: skip stability tests.
        benchmark_name: name for the returned Benchmark object.

    Returns:
        A ``bgym.Benchmark`` instance ready for AgentLab's ``make_study()``.
    """
    # Trigger task registration (idempotent — runs once on first import)
    import browsergym.agent_a_eval  # noqa: F401

    from browsergym.agent_a_eval import ALL_TASK_IDS
    from browsergym.agent_a_eval.utils import WorkflowConfig

    stability_seeds = stability_seeds or [42, 123, 456]

    # Build task name → [dataset_indices] mapping from ALL_TASK_IDS
    # Task IDs are like: agent_a_eval.<task_name>.ds<idx>
    task_datasets: dict[str, set[int]] = {}
    for tid in ALL_TASK_IDS:
        # Parse out task_name and dataset index
        # Format: agent_a_eval.{task_name}.ds{idx}
        parts = tid.split(".")
        if len(parts) >= 3 and parts[2].startswith("ds"):
            task_name = parts[1]
            try:
                ds_idx = int(parts[2][2:])  # after "ds"
            except ValueError:
                ds_idx = 0
        else:
            # Unconventional format — treat as single dataset
            task_name = parts[1] if len(parts) > 1 else tid
            ds_idx = 0

        task_datasets.setdefault(task_name, set()).add(ds_idx)

    # Apply task filter
    if task_filter:
        task_datasets = {k: v for k, v in task_datasets.items() if k in task_filter}
        if not task_datasets:
            logger.warning(
                f"No tasks matched filter: {task_filter}. "
                f"Available: {sorted(set().union(*[task_datasets.keys()]))}"
            )

    if not task_datasets:
        logger.warning("No tasks registered. Ensure tasks/*.md exist and are parseable.")
        # Return empty benchmark
        return Benchmark(
            name=benchmark_name,
            high_level_action_set_args=DEFAULT_ACTION_SET_ARGS,
            is_multi_tab=False,
            supports_parallel_seeds=True,
            backends=["agent_a_eval"],
            env_args_list=[],
        )

    # Build env_args_list
    env_args_list: list[AgentAEvalEnvArgs] = []
    all_task_names: set[str] = set()

    for task_name, ds_indices in sorted(task_datasets.items()):
        all_task_names.add(task_name)
        max_ds = max(ds_indices) if ds_indices else 0
        n_datasets = max_ds + 1  # dataset indices are 0-based

        if skip_generalization and skip_stability:
            # Quick test: single run with iris.csv (or first dataset) + seed=42
            # Find iris.csv index
            iris_idx = 0
            for i in sorted(ds_indices):
                tid = f"agent_a_eval.{task_name}.ds{i}"
                if "iris" in tid.lower():
                    iris_idx = i
                    break

            env_args_list.append(
                _make_env_args(
                    task_name=task_name,
                    dataset_index=iris_idx,
                    seed=42,
                    max_steps=max_steps,
                    headless=headless,
                    viewport=viewport,
                    slow_mo=slow_mo,
                    storage_state=storage_state,
                    user_data_dir=user_data_dir,
                )
            )
        else:
            # Generalization: N datasets, each with seed=42
            if not skip_generalization:
                n_gen = min(generalization_datasets, n_datasets)
                for ds_idx in range(n_gen):
                    env_args_list.append(
                        _make_env_args(
                            task_name=task_name,
                            dataset_index=ds_idx,
                            seed=42,
                            max_steps=max_steps,
                            headless=headless,
                            viewport=viewport,
                            slow_mo=slow_mo,
                            storage_state=storage_state,
                            user_data_dir=user_data_dir,
                        )
                    )

            # Stability: same dataset (index 0), varying seeds
            if not skip_stability:
                for seed in stability_seeds:
                    env_args_list.append(
                        _make_env_args(
                            task_name=task_name,
                            dataset_index=0,
                            seed=seed,
                            max_steps=max_steps,
                            headless=headless,
                            viewport=viewport,
                            slow_mo=slow_mo,
                            storage_state=storage_state,
                            user_data_dir=user_data_dir,
                        )
                    )

    # Build task_metadata DataFrame using full gym task IDs (not stems).
    # Benchmark.__post_init__ validates that every env_args.task_name
    # appears in task_metadata["task_name"].
    unique_env_task_names = sorted(
        set(e.task_name for e in env_args_list)
    )
    # Include the task stem as a separate column for grouping/filtering
    task_metadata = pd.DataFrame([
        {
            "task_name": tid,  # full gym ID (e.g. "agent_a_eval.data_analysis.ds0")
            "task_stem": tid.split(".")[1] if tid.count(".") >= 2 else tid,
        }
        for tid in unique_env_task_names
    ])

    logger.info(
        f"WorkflowBenchmark created: {len(all_task_names)} task(s), "
        f"{len(env_args_list)} experiment(s)"
    )

    return Benchmark(
        name=benchmark_name,
        high_level_action_set_args=DEFAULT_ACTION_SET_ARGS,
        is_multi_tab=False,
        supports_parallel_seeds=True,
        backends=["agent_a_eval"],
        env_args_list=env_args_list,
        task_metadata=task_metadata,
    )


def _make_env_args(
    task_name: str,
    dataset_index: int,
    seed: int,
    max_steps: int = 200,
    headless: bool = True,
    viewport: Optional[dict] = None,
    slow_mo: Optional[int] = None,
    storage_state: Optional[str] = None,
    user_data_dir: Optional[str] = None,
) -> AgentAEvalEnvArgs:
    """
    Build a single EnvArgs for a (task, dataset, seed) combination.

    The gym task ID format is: agent_a_eval.<task_name>.ds<dataset_index>
    """
    gym_id = f"agent_a_eval.{task_name}.ds{dataset_index}"

    # Resolve storage_state path
    ss = None
    if storage_state and Path(storage_state).exists():
        ss = storage_state

    # NOTE: user_data_dir is intentionally NOT passed to EnvArgs.
    # agentlab's _convert_env_args() does EnvArgs(**asdict(ea)), which would
    # fail because agentlab's EnvArgs doesn't accept user_data_dir.
    # Use storage_state for login persistence instead.
    return AgentAEvalEnvArgs(
        task_name=gym_id,
        task_seed=seed,
        max_steps=max_steps,
        headless=headless,
        viewport=viewport,
        slow_mo=slow_mo,
        storage_state=ss,
    )
