"""
BrowserGym benchmark for Agent A evaluation.

Registers workflow tasks defined in .md files as gym environments.
Each .md file in the tasks/ directory is registered as browsergym/agent_a_eval.<name>.

Usage:
    import browsergym.agent_a_eval  # triggers registration

    env = gym.make("browsergym/agent_a_eval.data_analysis.ds0")
"""

import logging
import os
from pathlib import Path

from browsergym.core.registration import register_task

from .task import WorkflowTask
from .utils import WorkflowConfig

logger = logging.getLogger(__name__)

# Default directories (relative to this package)
_PKG_ROOT = Path(__file__).resolve().parents[3]  # browsergym/agent_a_eval/
_DEFAULT_TASKS_DIR = _PKG_ROOT / "tasks"
_DEFAULT_DATASETS_DIR = _PKG_ROOT / "test_data" / "datasets"

# Allow override via environment variables
TASKS_DIR = Path(os.environ.get("AGENT_A_EVAL_TASKS_DIR", _DEFAULT_TASKS_DIR))
DATASETS_DIR = Path(os.environ.get("AGENT_A_EVAL_DATASETS_DIR", _DEFAULT_DATASETS_DIR))

# Ensure defaults exist so env var is set for task.py consumption
if "AGENT_A_EVAL_DATASETS_DIR" not in os.environ:
    os.environ["AGENT_A_EVAL_DATASETS_DIR"] = str(DATASETS_DIR.resolve())

ALL_TASK_IDS = []


def _register_workflows():
    """Auto-discover and register all .md task files."""
    if not TASKS_DIR.exists():
        logger.warning(
            f"Tasks directory not found: {TASKS_DIR}. "
            "No Agent A evaluation tasks registered. "
            "Set AGENT_A_EVAL_TASKS_DIR environment variable to specify tasks location."
        )
        return

    task_files = sorted(TASKS_DIR.glob("*.md"))
    if not task_files:
        logger.warning(f"No .md task files found in {TASKS_DIR}")
        return

    for task_file in task_files:
        task_name = task_file.stem

        # Parse the .md to know how many datasets are available
        config = WorkflowConfig.from_markdown(task_file)
        num_datasets = max(1, len(config.available_datasets))

        # Register one gym environment per dataset index
        # This allows running generalization tests as separate gym tasks.
        # The "ds<index>" suffix lets us run each dataset as a separate env.
        for ds_idx in range(num_datasets):
            gym_id = f"agent_a_eval.{task_name}.ds{ds_idx}"
            register_task(
                gym_id,
                WorkflowTask,
                task_kwargs={
                    "task_md_path": str(task_file),
                    "dataset_index": ds_idx,
                },
            )
            ALL_TASK_IDS.append(gym_id)

        logger.info(
            f"Registered: {task_file.name} → {num_datasets} dataset(s) "
            f"(agent_a_eval.{task_name}.ds0 .. ds{num_datasets - 1})"
        )

    logger.info(
        f"Registered {len(ALL_TASK_IDS)} Agent A evaluation tasks "
        f"from {len(task_files)} workflow definition(s)."
    )


# Auto-register on import
_register_workflows()
