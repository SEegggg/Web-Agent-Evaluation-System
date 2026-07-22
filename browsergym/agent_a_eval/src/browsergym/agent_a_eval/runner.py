"""
WorkflowRunner — orchestrates batch evaluation of Agent A.

Runs generalization tests (different datasets) and stability tests
(same dataset, different seeds) for each task .md file.

The runner operates in three phases per run:
1. BrowserGym session: Driver Agent executes the task
2. Artifact extraction: Collect screenshots, chat history, page state
3. EvaluatorAgent: Independent LLM scoring of artifacts
"""

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from browsergym.experiments import EnvArgs, ExpArgs, get_exp_result

from .evaluator import EvaluatorAgent
from .utils import WorkflowConfig

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result of a single workflow run."""

    task_name: str
    dataset_name: str
    seed: int
    status: str  # "completed" | "dead_loop" | "timeout" | "infeasible" | "error"
    overall_score: float = 0.0
    dimension_scores: dict = field(default_factory=dict)
    loop_detected: bool = False
    elapsed_seconds: float = 0
    error_message: Optional[str] = None
    artifacts: Optional[dict] = None

    # ── Benchmark-format fields ──
    format: str = "legacy"  # "legacy" | "benchmark"
    core_passed: Optional[bool] = None
    process_efficiency_scores: dict = field(default_factory=dict)
    resource_robustness_scores: dict = field(default_factory=dict)
    task_specific_scores: dict = field(default_factory=dict)


@dataclass
class TaskReport:
    """Aggregated report for a single task across all runs."""

    task_name: str
    format: str = "legacy"  # "legacy" | "benchmark"
    generalization: dict = field(default_factory=dict)  # mean, std, success_rate, scores
    stability: dict = field(default_factory=dict)  # mean, std, stability_score, scores
    runs: list[RunResult] = field(default_factory=list)


class WorkflowRunner:
    """
    Batch runner for Agent A evaluation workflows.

    Usage:
        from demo_agent.agent import DemoAgentArgs

        agent_args = DemoAgentArgs(model_name="gpt-4o", ...)
        runner = WorkflowRunner(
            agent_args=agent_args,
            tasks_dir="tasks/",
            exp_root="./results",
            eval_model="gpt-4o",
        )
        reports = runner.run_all()
        runner.save_reports(reports, "reports/")
    """

    def __init__(
        self,
        agent_args,  # AbstractAgentArgs
        tasks_dir: str = "tasks",
        exp_root: str = "./results",
        eval_model: str = "gpt-4o",
        eval_provider: Optional[str] = None,
        generalization_datasets: int = 3,
        stability_seeds: list[int] = None,
        max_steps: int = 200,
        headless: bool = True,
        task_filter: list[str] = None,
        task_datasets: dict[str, list[str]] = None,
        skip_generalization: bool = False,
        skip_stability: bool = False,
        storage_state: Optional[str] = None,
        user_data_dir: Optional[str] = None,
    ):
        """
        Args:
            agent_args: BrowserGym agent configuration (e.g., DemoAgentArgs)
            tasks_dir: directory containing task .md files
            exp_root: root directory for experiment outputs
            eval_model: LLM model for evaluation
            eval_provider: LLM provider for evaluation
            generalization_datasets: number of different datasets to test
            stability_seeds: seeds for stability testing (default: [42, 123, 456])
            max_steps: maximum steps per experiment
            headless: run browser in headless mode
            task_filter: optional list of task names to run ([] = all)
            task_datasets: optional per-task dataset override,
                e.g. {"data_analysis": ["custom.csv", "sales.csv"]}
            skip_generalization: skip generalization tests entirely
            skip_stability: skip stability tests entirely
            storage_state: path to Playwright storage_state JSON file
                for cookie/auth persistence (None = login every time)
            user_data_dir: path to persistent browser profile directory.
                When set, uses launch_persistent_context — behaves like regular
                Chrome: all cookies/localStorage persist automatically.
                This is the RECOMMENDED approach. Takes precedence over storage_state.
        """
        self.agent_args = agent_args
        self.tasks_dir = Path(tasks_dir)
        self.exp_root = Path(exp_root)
        self.eval_model = eval_model
        self.eval_provider = eval_provider
        self.generalization_datasets = generalization_datasets
        self.stability_seeds = stability_seeds or [42, 123, 456]
        self.max_steps = max_steps
        self.headless = headless
        self.task_filter = task_filter or []
        self.task_datasets = task_datasets or {}
        self.skip_generalization = skip_generalization
        self.skip_stability = skip_stability
        self.storage_state = storage_state
        self.user_data_dir = user_data_dir

        # The evaluator is created once and reused
        self.evaluator = EvaluatorAgent(
            model=eval_model,
            provider=eval_provider,
        )

    def run_all(self) -> list[TaskReport]:
        """
        Run all task .md files found in tasks_dir.

        For each task:
        - Generalization: N datasets, each with a fixed seed
        - Stability: same dataset, M different seeds

        Supports task_filter, task_datasets override, skip_generalization, skip_stability.

        Returns:
            list of TaskReport objects
        """
        task_files = sorted(self.tasks_dir.glob("*.md"))
        if not task_files:
            logger.warning(f"No .md task files found in {self.tasks_dir}")
            return []

        # Apply task filter
        if self.task_filter:
            task_files = [
                f for f in task_files
                if f.stem in self.task_filter
            ]
            if not task_files:
                logger.warning(
                    f"No task files matched filter: {self.task_filter}. "
                    f"Available: {[f.stem for f in sorted(self.tasks_dir.glob('*.md'))]}"
                )
                return []
            logger.info(
                f"Task filter applied: {self.task_filter} → {len(task_files)} task(s) selected"
            )

        logger.info(f"Found {len(task_files)} task(s) to evaluate")
        all_reports = []

        for task_file in task_files:
            logger.info(f"{'='*60}")
            logger.info(f"Processing task: {task_file.name}")
            logger.info(f"{'='*60}")

            config = WorkflowConfig.from_markdown(task_file)

            # Apply per-task dataset override
            task_name_key = task_file.stem
            if task_name_key in self.task_datasets:
                override_datasets = self.task_datasets[task_name_key]
                logger.info(
                    f"Dataset override for {task_name_key}: {override_datasets}"
                )
                config.available_datasets = override_datasets

            num_datasets = len(config.available_datasets)

            if num_datasets == 0:
                logger.warning(
                    f"Task {task_file.name} has no datasets defined. "
                    "Running single generalization run without dataset."
                )
                num_datasets = 1

            n_gen = min(self.generalization_datasets, num_datasets)
            n_stab = min(len(self.stability_seeds), 3)

            generalization_runs = []
            stability_runs = []

            # ---- Quick Test (both skipped → single run) ----
            if self.skip_generalization and self.skip_stability:
                # 找到 iris.csv，找不到就用第一个数据集
                iris_idx = 0
                for i, ds in enumerate(config.available_datasets):
                    if "iris" in ds.lower():
                        iris_idx = i
                        break
                logger.info(
                    f"--- Quick Test: iris.csv (dataset index {iris_idx}), seed=42 ---"
                )
                result = self._run_single(
                    task_file=task_file,
                    config=config,
                    dataset_index=iris_idx,
                    seed=42,
                )
                generalization_runs.append(result)
                logger.info(
                    f"  Quick Test dataset={result.dataset_name} "
                    f"score={result.overall_score:.1f} status={result.status}"
                )

            # ---- Generalization Tests ----
            elif self.skip_generalization:
                logger.info(f"--- Generalization: SKIPPED ---")
            else:
                logger.info(f"--- Generalization: {n_gen} dataset(s) ---")
                for ds_idx in range(n_gen):
                    result = self._run_single(
                        task_file=task_file,
                        config=config,
                        dataset_index=ds_idx,
                        seed=42,  # Fixed seed for generalization — variable is dataset
                    )
                    generalization_runs.append(result)
                    logger.info(
                        f"  Gen [{ds_idx}] dataset={result.dataset_name} "
                        f"score={result.overall_score:.1f} status={result.status}"
                    )

            # ---- Stability Tests ----
            if self.skip_stability:
                logger.info(f"--- Stability: SKIPPED ---")
            else:
                logger.info(f"--- Stability: {n_stab} seed(s) on dataset 0 ---")
                for seed in self.stability_seeds[:n_stab]:
                    result = self._run_single(
                        task_file=task_file,
                        config=config,
                        dataset_index=0,  # Fixed dataset for stability — variable is seed
                        seed=seed,
                    )
                    stability_runs.append(result)
                    logger.info(
                        f"  Stab [seed={seed}] score={result.overall_score:.1f} status={result.status}"
                    )

            # ---- Aggregate Results ----
            report = self._build_report(
                task_name=task_file.stem,
                generalization_runs=generalization_runs,
                stability_runs=stability_runs,
            )
            all_reports.append(report)

            # Log summary
            logger.info(f"Task {task_file.stem} summary:")
            if generalization_runs:
                if report.format == "benchmark":
                    logger.info(
                        f"  Generalization: pass_rate={report.generalization['pass_rate']:.1%}, "
                        f"passed={report.generalization.get('passed', 0)}, "
                        f"failed={report.generalization.get('failed', 0)}"
                    )
                else:
                    logger.info(
                        f"  Generalization: mean={report.generalization['mean']:.2f}, "
                        f"std={report.generalization['std']:.2f}, "
                        f"success_rate={report.generalization['success_rate']:.1%}"
                    )
            if stability_runs:
                logger.info(
                    f"  Stability: stability_score={report.stability['stability_score']:.2f}"
                )

        return all_reports

    def _run_single(
        self,
        task_file: Path,
        config: WorkflowConfig,
        dataset_index: int,
        seed: int,
    ) -> RunResult:
        """
        Run a single BrowserGym experiment and evaluate it.

        1. BrowserGym session (Driver Agent executes task)
        2. Extract artifacts from experiment result
        3. EvaluatorAgent scores the artifacts
        """
        task_name = f"agent_a_eval.{task_file.stem}.ds{dataset_index}"
        dataset_name = (
            config.available_datasets[dataset_index]
            if config.available_datasets
            and dataset_index < len(config.available_datasets)
            else "none"
        )

        start_time = time.time()

        try:
            # ---- Phase 1: BrowserGym session ----
            # Note: task_md_path and dataset_index are already baked into the
            # registered gym task (via register_task frozen kwargs).
            # Just use the registered task name; no need to override kwargs.
            if self.user_data_dir:
                logger.info(f"🔐 使用持久化浏览器 Profile: {self.user_data_dir}")
            elif self.storage_state and Path(self.storage_state).exists():
                logger.info(f"🔐 加载 cookie: {self.storage_state}")
            else:
                logger.info(
                    f"🔐 无持久化状态（每次都需要登录）"
                )

            _ss = (
                self.storage_state
                if self.storage_state and Path(self.storage_state).exists()
                else None
            )
            env_args = EnvArgs(
                task_name=task_name,
                task_seed=seed,
                max_steps=self.max_steps,
                headless=self.headless,
                storage_state=_ss,
                user_data_dir=self.user_data_dir,
            )

            exp_args = ExpArgs(
                agent_args=self.agent_args,
                env_args=env_args,
            )
            exp_args.prepare(self.exp_root)
            exp_args.run()

            exp_result = get_exp_result(exp_args.exp_dir)

            # ---- Phase 2: Extract artifacts ----
            artifacts = self._extract_artifacts(exp_result)
            elapsed = time.time() - start_time

            if artifacts is None:
                logger.warning(
                    f"No artifacts found in experiment {exp_args.exp_name}. "
                    "The task may have failed before completion."
                )
                self._save_run_log(
                    task_name=task_file.stem,
                    dataset_name=dataset_name,
                    seed=seed,
                    status="error",
                    artifacts={},
                    scores=None,
                    exp_result=exp_result,
                )
                return RunResult(
                    task_name=task_file.stem,
                    dataset_name=dataset_name,
                    seed=seed,
                    status="error",
                    error_message="No artifacts extracted",
                    elapsed_seconds=elapsed,
                )

            status = artifacts.get("completion_status", "unknown")

            # ---- Phase 3: EvaluatorAgent scores ----
            if config.evaluation_criteria:
                scores = self.evaluator.evaluate(
                    evaluation_criteria=config.evaluation_criteria,
                    artifacts=artifacts,
                    task_config=config,
                )
                fmt = scores.get("format", "legacy")

                if fmt == "benchmark":
                    core = scores.get("core", {})
                    overall_score = 10.0 if core.get("passed") else 2.0
                    dimension_scores = {}
                    loop_detected = False
                    core_passed = core.get("passed")
                    process_efficiency_scores = scores.get("process_efficiency", {})
                    resource_robustness_scores = scores.get("resource_robustness", {})
                    task_specific_scores = scores.get("task_specific", {})
                else:
                    overall_score = scores.get("overall_score", 0)
                    dimension_scores = scores.get("dimensions", {})
                    loop_detected = scores.get("loop_detected", False)
                    core_passed = None
                    process_efficiency_scores = {}
                    resource_robustness_scores = {}
                    task_specific_scores = {}
            else:
                # No evaluation criteria defined — use hard check only
                scores = None
                fmt = "legacy"
                overall_score = 10.0 if status == "completed" else 0.0
                dimension_scores = {}
                loop_detected = (status == "dead_loop")
                core_passed = None
                process_efficiency_scores = {}
                resource_robustness_scores = {}
                task_specific_scores = {}

            # ---- Phase 4: Save debug log ----
            self._save_run_log(
                task_name=task_file.stem,
                dataset_name=dataset_name,
                seed=seed,
                status=status,
                artifacts=artifacts,
                scores=scores if config.evaluation_criteria else None,
                exp_result=exp_result,
            )

            return RunResult(
                task_name=task_file.stem,
                dataset_name=dataset_name,
                seed=seed,
                status=status,
                overall_score=overall_score,
                dimension_scores=dimension_scores,
                loop_detected=loop_detected,
                elapsed_seconds=elapsed,
                artifacts=artifacts,
                format=fmt,
                core_passed=core_passed,
                process_efficiency_scores=process_efficiency_scores,
                resource_robustness_scores=resource_robustness_scores,
                task_specific_scores=task_specific_scores,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Run failed: {task_file.stem} ds={dataset_index} seed={seed}: {e}")
            # Try to save whatever we have
            try:
                self._save_run_log(
                    task_name=task_file.stem,
                    dataset_name=dataset_name,
                    seed=seed,
                    status="error",
                    artifacts=None,
                    scores=None,
                )
            except Exception:
                pass
            return RunResult(
                task_name=task_file.stem,
                dataset_name=dataset_name,
                seed=seed,
                status="error",
                error_message=str(e),
                elapsed_seconds=elapsed,
            )

    def _extract_artifacts(self, exp_result) -> Optional[dict]:
        """
        Extract artifacts collected by WorkflowTask from an ExpResult.

        Artifacts are stored in the last step's task_info dict.
        """
        try:
            # Get the last step info
            summary = exp_result.summary_info
            n_steps = summary.get("n_steps", 0)

            if n_steps < 0:
                return None

            last_step = exp_result.get_step_info(n_steps)
            if last_step is None or last_step.task_info is None:
                return None

            task_info = last_step.task_info
            artifacts = task_info.get("artifacts")

            # Also extract the goal from the first step for logging
            if artifacts is not None:
                try:
                    first_step = exp_result.get_step_info(0)
                    if first_step is not None:
                        artifacts["_goal"] = first_step.goal
                except Exception:
                    artifacts["_goal"] = None

            return artifacts

        except Exception as e:
            logger.error(f"Failed to extract artifacts: {e}")
            return None

    # =========================================================================
    # Per-Run Log Saving (for debugging)
    # =========================================================================

    def _save_run_log(
        self,
        task_name: str,
        dataset_name: str,
        seed: int,
        status: str,
        artifacts: Optional[dict],
        scores: Optional[dict],
        exp_result=None,  # ExpResult from BrowserGym, has all step data
    ) -> Path:
        """
        Save detailed per-run logs to a timestamped folder for debugging.

        Folder structure:
            logs/{task_name}/{dataset_name}_seed{seed}_{timestamp}/
              ├── goal.txt           — driver agent 收到的完整任务指令
              ├── step_by_step.txt   — ★ 每一步的详情（动作/思考/页面内容）
              ├── final_page.txt     — ★ 最后一步的完整页面 AXTree
              ├── chat_history.txt   — driver agent 与 Agent A 的全部对话
              ├── agent_a_output.txt — Agent A 的页面输出内容
              ├── result.json        — 状态、评分、校验结果
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = (
            Path(self.exp_root) / "logs" / task_name /
            f"{dataset_name}_seed{seed}_{timestamp}"
        )
        log_dir.mkdir(parents=True, exist_ok=True)

        if artifacts is None:
            artifacts = {}

        # ── 1. Goal ──
        goal = artifacts.get("_goal", "(未获取到 goal)")
        (log_dir / "goal.txt").write_text(str(goal), encoding="utf-8")

        # ── 2. Step-by-step detail ★ ──
        step_lines = []
        if exp_result is not None:
            try:
                n_steps = exp_result.summary_info.get("n_steps", 0)
                for i in range(n_steps + 1):  # step 0 is initial state
                    try:
                        si = exp_result.get_step_info(i)
                        if si is None:
                            continue
                    except Exception:
                        continue

                    step_lines.append(f"{'=' * 70}")
                    step_lines.append(f"=== Step {i} / {n_steps}")
                    step_lines.append(f"{'=' * 70}")

                    # 动作
                    action = si.action or "(无动作 — 初始状态)"
                    step_lines.append(f"\n  [动作 ACTION]")
                    step_lines.append(f"  {action}")

                    # Agent 的思考
                    agent_info = si.agent_info or {}
                    think = agent_info.get("think", "")
                    if think:
                        step_lines.append(f"\n  [思考 THINK]")
                        step_lines.append(f"  {think}")

                    # 页面内容（AXTree — Agent 看到的页面）
                    obs = si.obs or {}
                    if obs.get("axtree_object"):
                        try:
                            from browsergym.utils.obs import flatten_axtree_to_str
                            axtree_txt = flatten_axtree_to_str(obs["axtree_object"])
                            step_lines.append(f"\n  [页面 AXTree — Agent 看到的内容]")
                            step_lines.append(axtree_txt)
                        except Exception:
                            step_lines.append(f"\n  [页面 AXTree — 提取失败]")

                    # 对话消息
                    chat = obs.get("chat_messages", [])
                    if chat:
                        step_lines.append(f"\n  [对话消息]")
                        for msg in chat[-6:]:  # 最近 6 条
                            role = msg.get("role", "?")
                            text = str(msg.get("message", ""))[:500]
                            step_lines.append(f"  [{role}] {text}")

                    # task_info
                    ti = si.task_info or {}
                    if ti:
                        step_lines.append(f"\n  [任务状态]")
                        step_lines.append(f"  status={ti.get('status', '?')}")
                        loop_reason = ti.get("loop_reason", "")
                        if loop_reason:
                            step_lines.append(f"  loop_reason={loop_reason}")

                    step_lines.append("")

            except Exception as e:
                step_lines.append(f"\n(提取步骤详情时出错: {e})")

        (log_dir / "step_by_step.txt").write_text(
            "\n".join(step_lines) if step_lines else "(无步骤数据)",
            encoding="utf-8",
        )

        # ── 3. Final page AXTree ★ ──
        final_page = "(无页面数据)"
        if exp_result is not None:
            try:
                n_steps = exp_result.summary_info.get("n_steps", 0)
                last_step = exp_result.get_step_info(n_steps)
                obs = last_step.obs if last_step else {}
                if obs.get("axtree_object"):
                    from browsergym.utils.obs import flatten_axtree_to_str
                    final_page = flatten_axtree_to_str(obs["axtree_object"])
            except Exception:
                final_page = "(提取最终页面时出错)"
        (log_dir / "final_page.txt").write_text(final_page, encoding="utf-8")

        # ── 4. Chat history ──
        chat_lines = []
        for msg in artifacts.get("chat_history", []):
            role = msg.get("role", "?")
            message = msg.get("message", "")
            chat_lines.append(f"[{role}] {message}")
            chat_lines.append("-" * 60)
        (log_dir / "chat_history.txt").write_text(
            "\n".join(chat_lines) if chat_lines else "(无对话记录)",
            encoding="utf-8",
        )

        # ── 5. Agent A output ──
        agent_output_lines = []
        for resp in artifacts.get("agent_a_responses", []):
            agent_output_lines.append(f"=== 来源: {resp.get('selector', '?')} ===")
            agent_output_lines.append(resp.get("text", ""))
            agent_output_lines.append("")
        (log_dir / "agent_a_output.txt").write_text(
            "\n".join(agent_output_lines) if agent_output_lines else "(无 Agent A 输出)",
            encoding="utf-8",
        )

        # ── 6. Result summary ──
        if scores and scores.get("format") == "benchmark":
            result_data = {
                "task_name": task_name,
                "dataset_name": dataset_name,
                "seed": seed,
                "status": status,
                "format": "benchmark",
                "core": scores.get("core", {}),
                "process_efficiency": scores.get("process_efficiency", {}),
                "resource_robustness": scores.get("resource_robustness", {}),
                "task_specific": scores.get("task_specific", {}),
                "overall_comment": scores.get("overall_comment", ""),
                "page_url": artifacts.get("page_url", ""),
                "page_title": artifacts.get("page_title", ""),
                "elapsed_time_seconds": artifacts.get("elapsed_time_seconds", 0),
            }
        else:
            result_data = {
                "task_name": task_name,
                "dataset_name": dataset_name,
                "seed": seed,
                "status": status,
                "format": "legacy",
                "loop_detected": scores.get("loop_detected", False) if scores else False,
                "overall_score": scores.get("overall_score", 0) if scores else 0,
                "dimensions": scores.get("dimensions", {}) if scores else {},
                "overall_comment": scores.get("overall_comment", "") if scores else "",
                "page_url": artifacts.get("page_url", ""),
                "page_title": artifacts.get("page_title", ""),
                "elapsed_time_seconds": artifacts.get("elapsed_time_seconds", 0),
            }
        (log_dir / "result.json").write_text(
            json.dumps(result_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info(f"Run log saved to: {log_dir}")
        return log_dir

    def _build_report(
        self,
        task_name: str,
        generalization_runs: list[RunResult],
        stability_runs: list[RunResult],
    ) -> TaskReport:
        """Build aggregated TaskReport from individual runs."""

        def _compute_stats(runs: list[RunResult]) -> dict:
            if runs and all(r.format == "benchmark" for r in runs):
                # Benchmark format: compute pass rate
                passed = sum(1 for r in runs if r.core_passed)
                return {
                    "format": "benchmark",
                    "pass_rate": passed / len(runs) if runs else 0,
                    "total_runs": len(runs),
                    "passed": passed,
                    "failed": len(runs) - passed,
                    "statuses": [r.status for r in runs],
                }
            # Legacy format: compute mean/std of scores
            scores = [r.overall_score for r in runs]
            success_count = sum(1 for r in runs if r.status == "completed")
            return {
                "format": "legacy",
                "scores": scores,
                "mean": float(np.mean(scores)) if scores else 0.0,
                "std": float(np.std(scores)) if scores else 0.0,
                "success_rate": success_count / len(runs) if runs else 0.0,
                "num_runs": len(runs),
                "statuses": [r.status for r in runs],
            }

        gen_stats = _compute_stats(generalization_runs)
        stab_stats = _compute_stats(stability_runs)

        # Stability score: 1 - normalized std (higher is better)
        # For benchmark format, use pass_rate consistency instead
        if stab_stats.get("format") == "benchmark":
            stab_stats["stability_score"] = stab_stats.get("pass_rate", 0)
        elif stab_stats.get("std", 0) > 0:
            # Normalize: std of 0 → score 1.0, std of 5 → score 0.0
            stab_stats["stability_score"] = max(0.0, 1.0 - stab_stats["std"] / 5.0)
        else:
            stab_stats["stability_score"] = 1.0

        # Determine overall format
        report_format = "benchmark" if (
            generalization_runs and all(r.format == "benchmark" for r in generalization_runs)
        ) else "legacy"

        return TaskReport(
            task_name=task_name,
            format=report_format,
            generalization=gen_stats,
            stability=stab_stats,
            runs=generalization_runs + stability_runs,
        )

    # =========================================================================
    # Report Serialization
    # =========================================================================

    def save_reports(
        self,
        reports: list[TaskReport],
        output_dir: str = "reports",
    ):
        """
        Save evaluation reports as human-readable text, JSON, and CSV.

        Args:
            reports: list of TaskReport from run_all()
            output_dir: directory to save reports
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ── 文本报告（中文，人类可读） ──
        txt_path = output_dir / f"evaluation_report_{timestamp}.txt"
        txt_content = self._build_text_report(reports)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)
        logger.info(f"文本报告已保存至: {txt_path}")

        # ── JSON 详细数据 ──
        json_path = output_dir / f"evaluation_summary_{timestamp}.json"
        json_data = []
        for report in reports:
            json_runs = []
            for r in report.runs:
                run_entry = {
                    "dataset": r.dataset_name,
                    "seed": r.seed,
                    "status": r.status,
                    "elapsed_seconds": r.elapsed_seconds,
                    "error": r.error_message,
                    "format": r.format,
                }
                if r.format == "benchmark":
                    run_entry["core_passed"] = r.core_passed
                    run_entry["process_efficiency"] = {
                        k: v.get("score", v) if isinstance(v, dict) else v
                        for k, v in r.process_efficiency_scores.items()
                    }
                    run_entry["resource_robustness"] = {
                        k: v.get("score", v) if isinstance(v, dict) else v
                        for k, v in r.resource_robustness_scores.items()
                    }
                    run_entry["task_specific"] = {
                        k: v.get("score", v) if isinstance(v, dict) else v
                        for k, v in r.task_specific_scores.items()
                    }
                else:
                    run_entry["overall_score"] = r.overall_score
                    run_entry["dimension_scores"] = {
                        k: v.get("score", v) if isinstance(v, dict) else v
                        for k, v in r.dimension_scores.items()
                    }
                    run_entry["loop_detected"] = r.loop_detected
                json_runs.append(run_entry)

            json_data.append(
                {
                    "task_name": report.task_name,
                    "format": report.format,
                    "generalization": report.generalization,
                    "stability": report.stability,
                    "runs": json_runs,
                }
            )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON report saved to {json_path}")

        # ── CSV 表格 ──
        csv_path = output_dir / f"evaluation_runs_{timestamp}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "task_name",
                    "run_type",
                    "dataset",
                    "seed",
                    "status",
                    "format",
                    "overall_score",
                    "core_passed",
                    "loop_detected",
                    "elapsed_seconds",
                    "error",
                ]
            )
            for report in reports:
                gen_count = report.generalization.get("num_runs", len(report.generalization.get("scores", [])))
                for i, r in enumerate(report.runs):
                    run_type = (
                        "generalization"
                        if i < gen_count
                        else "stability"
                    )
                    writer.writerow(
                        [
                            report.task_name,
                            run_type,
                            r.dataset_name,
                            r.seed,
                            r.status,
                            r.format,
                            f"{r.overall_score:.2f}",
                            "PASS" if r.core_passed else ("FAIL" if r.core_passed is False else ""),
                            r.loop_detected,
                            f"{r.elapsed_seconds:.1f}",
                            r.error_message or "",
                        ]
                    )
        logger.info(f"CSV report saved to {csv_path}")

    # =========================================================================
    # Text Report Builder
    # =========================================================================

    STATUS_LABELS = {
        "completed": "✅ 完成",
        "dead_loop": "🔁 死循环",
        "timeout": "⏰ 超时",
        "infeasible": "❌ 无法执行",
        "error": "💥 错误",
    }

    def _build_text_report(self, reports: list[TaskReport]) -> str:
        """生成中文可读文本报告。"""
        lines = []
        lines.append("=" * 60)
        lines.append("  Agent A 自动评审报告")
        lines.append(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)

        for report in reports:
            lines.append("")
            lines.append("─" * 60)
            lines.append(f"  【任务】{report.task_name}")
            lines.append("─" * 60)

            # ── 泛化性测试 ──
            lines.extend(self._format_test_section(
                title="泛化性测试（固定 seed=42，变数据集）",
                stats=report.generalization,
                runs=report.runs[:report.generalization.get("num_runs", 0)],
            ))

            # ── 稳定性测试 ──
            gen_count = report.generalization.get("num_runs", 0)
            lines.extend(self._format_test_section(
                title="稳定性测试（固定数据集，变 seed）",
                stats=report.stability,
                runs=report.runs[gen_count:],
            ))

        lines.append("")
        lines.append("=" * 60)
        lines.append("  报告结束")
        lines.append("=" * 60)
        return "\n".join(lines)

    def _format_test_section(self, title: str, stats: dict, runs: list[RunResult]) -> list[str]:
        """格式化一个测试段（泛化性或稳定性）。"""
        lines = []
        lines.append(f"\n  ▸ {title}")

        if not runs:
            lines.append("    (未执行)")
            return lines

        is_benchmark = stats.get("format") == "benchmark"

        # 汇总
        if is_benchmark:
            pass_rate = stats.get("pass_rate", 0)
            passed = stats.get("passed", 0)
            failed = stats.get("failed", 0)
            lines.append(f"    通过率: {pass_rate:.0%}  |  通过: {passed}  |  失败: {failed}")
        else:
            mean_val = stats.get("mean", 0)
            std_val = stats.get("std", 0)
            success_rate = stats.get("success_rate", 0)
            lines.append(f"    平均分: {mean_val:.1f}/10  |  标准差: {std_val:.2f}  |  成功率: {success_rate:.0%}")

        # 稳定性额外指标
        if "stability_score" in stats:
            lines.append(f"    稳定性得分: {stats['stability_score']:.2f}（越接近 1 越稳定）")

        # 逐条运行结果
        for i, r in enumerate(runs):
            status_label = self.STATUS_LABELS.get(r.status, r.status)
            lines.append(f"\n    [{i + 1}] 数据集={r.dataset_name}  seed={r.seed}")

            if r.status == "completed":
                lines.append(f"        状态: {status_label}")

                if r.format == "benchmark":
                    # Benchmark format: show core pass/fail + sub-dimensions
                    core_label = "✅ PASS" if r.core_passed else "❌ FAIL"
                    lines.append(f"        核心判定: {core_label}")

                    # Process efficiency
                    if r.process_efficiency_scores:
                        lines.append(f"        过程与效率指标:")
                        for dim_name, dim_data in r.process_efficiency_scores.items():
                            if isinstance(dim_data, dict):
                                score = dim_data.get("score", "?")
                                reason = dim_data.get("justification", "")
                                lines.append(f"          - {dim_name}: {score}/10")
                                if reason:
                                    lines.append(f"            理由: {reason}")

                    # Resource robustness
                    if r.resource_robustness_scores:
                        lines.append(f"        资源与鲁棒性指标:")
                        for dim_name, dim_data in r.resource_robustness_scores.items():
                            if isinstance(dim_data, dict):
                                score = dim_data.get("score", "?")
                                reason = dim_data.get("justification", "")
                                lines.append(f"          - {dim_name}: {score}/10")
                                if reason:
                                    lines.append(f"            理由: {reason}")

                    # Task specific
                    if r.task_specific_scores:
                        lines.append(f"        任务专项指标:")
                        for dim_name, dim_data in r.task_specific_scores.items():
                            if isinstance(dim_data, dict):
                                score = dim_data.get("score", "?")
                                reason = dim_data.get("justification", "")
                                lines.append(f"          - {dim_name}: {score}/10")
                                if reason:
                                    lines.append(f"            理由: {reason}")
                else:
                    # Legacy format
                    lines.append(f"        综合得分: {r.overall_score:.1f}/10")
                    if r.loop_detected:
                        lines.append(f"        ⚠️ 检测到死循环")
                    if r.dimension_scores:
                        lines.append(f"        各维度评分:")
                        for dim_name, dim_data in r.dimension_scores.items():
                            if isinstance(dim_data, dict):
                                score = dim_data.get("score", "?")
                                reason = dim_data.get("justification", "")
                                lines.append(f"          - {dim_name}: {score}/10")
                                if reason:
                                    lines.append(f"            理由: {reason}")
                            else:
                                lines.append(f"          - {dim_name}: {dim_data}/10")

            else:
                # 失败：显示状态和原因
                lines.append(f"        状态: {status_label}")
                if r.error_message:
                    lines.append(f"        失败原因: {r.error_message}")
                if r.loop_detected:
                    lines.append(f"        ⚠️ 检测到死循环")

            lines.append(f"        耗时: {r.elapsed_seconds:.1f} 秒")

        return lines
