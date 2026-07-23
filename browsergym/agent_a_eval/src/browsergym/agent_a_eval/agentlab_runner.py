"""
AgentLabRunner — parallel evaluation runner using AgentLab's Study API.

Replaces WorkflowRunner's sequential for-loop with AgentLab's Ray/joblib
parallel execution. Produces identical TaskReport output format.

Architecture:
  1. Build WorkflowBenchmark from registered tasks
  2. Run all browser experiments in parallel via AgentLab Study
  3. Post-process: extract artifacts → EvaluatorAgent scores → TaskReport

Usage:
    from demo_agent.agent import DemoAgentArgs
    from browsergym.agent_a_eval.agentlab_runner import AgentLabRunner

    agent_args = DemoAgentArgs(model_name="gpt-4o", ...)
    runner = AgentLabRunner(
        agent_args=agent_args,
        tasks_dir="tasks/",
        exp_root="./results",
        eval_model="gpt-4o",
        n_jobs=4,
        parallel_backend="joblib",
    )
    reports = runner.run_all()
    runner.save_reports(reports, "reports/")
"""

import csv
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from browsergym.experiments.loop import EnvArgs, get_exp_result, yield_all_exp_results

from .benchmark import create_workflow_benchmark
from .evaluator import EvaluatorAgent
from .runner import RunResult, TaskReport, WorkflowRunner  # reuse data classes
from .utils import WorkflowConfig

logger = logging.getLogger(__name__)


class AgentLabRunner:
    """
    Parallel batch runner using AgentLab for experiment orchestration.

    Produces the same output format as ``WorkflowRunner`` (TaskReport, RunResult)
    but executes browser experiments in parallel via AgentLab's Study API.

    The evaluation phase (EvaluatorAgent) runs as a post-processing step after
    all browser experiments complete. This keeps the architecture clean:
    - Browser phase: parallel, I/O-bound (waiting on LLM + web server)
    - Evaluation phase: sequential, LLM-bound (API calls to evaluator model)

    Usage::

        runner = AgentLabRunner(
            agent_args=DemoAgentArgs(...),
            tasks_dir="tasks/",
            exp_root="./results",
            eval_model="gpt-4o",
            n_jobs=8,
            parallel_backend="ray",
        )
        reports = runner.run_all()
        runner.save_reports(reports, "reports/")
    """

    def __init__(
        self,
        agent_args,  # AbstractAgentArgs (e.g. DemoAgentArgs)
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
        # ── AgentLab-specific options ──
        n_jobs: int = 4,
        parallel_backend: str = "joblib",  # "ray", "joblib", "sequential"
        study_suffix: str = "agent_a_eval",
    ):
        """
        Args:
            agent_args: BrowserGym agent configuration (e.g., DemoAgentArgs).
            tasks_dir: directory containing task .md files.
            exp_root: root directory for experiment outputs
                (also used as AGENTLAB_EXP_ROOT for AgentXRay).
            eval_model: LLM model for evaluation.
            eval_provider: LLM provider for evaluation ("openai" or "anthropic").
            generalization_datasets: number of different datasets per task.
            stability_seeds: seeds for stability testing (default: [42, 123, 456]).
            max_steps: maximum steps per experiment.
            headless: run browser in headless mode.
            task_filter: optional list of task names to run ([] = all).
            task_datasets: optional per-task dataset override.
            skip_generalization: skip generalization tests entirely.
            skip_stability: skip stability tests entirely.
            storage_state: path to Playwright storage_state JSON file.
            user_data_dir: path to persistent browser profile directory.
            n_jobs: number of parallel jobs (AgentLab workers).
            parallel_backend: "ray", "joblib", or "sequential".
            study_suffix: suffix appended to AgentLab study name.
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
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.study_suffix = study_suffix

        # Set AGENTLAB_EXP_ROOT for AgentXRay compatibility
        os.environ.setdefault("AGENTLAB_EXP_ROOT", str(self.exp_root.resolve()))

        # Validate parallel_backend
        valid_backends = ("ray", "joblib", "sequential")
        if self.parallel_backend not in valid_backends:
            raise ValueError(
                f"Unknown parallel_backend '{self.parallel_backend}'. "
                f"Must be one of: {', '.join(valid_backends)}"
            )

        # Warn about user_data_dir in parallel mode
        if self.user_data_dir and self.n_jobs > 1 and self.parallel_backend != "sequential":
            logger.warning(
                "⚠️  user_data_dir is set with n_jobs > 1. "
                "Parallel browser instances may conflict on the same Chrome profile. "
                "Consider using storage_state instead, or set n_jobs=1."
            )

        # The evaluator is created once and reused
        self.evaluator = EvaluatorAgent(
            model=eval_model,
            provider=eval_provider,
        )

        # Will be populated after run()
        self._study = None
        self._study_dir = None

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    def run_all(self) -> list[TaskReport]:
        """
        Run all task .md files in parallel via AgentLab, then evaluate.

        Phases:
          1. Build WorkflowBenchmark → AgentLab-compatible Benchmark
          2. Run AgentLab Study (parallel browser experiments)
          3. Post-process: extract artifacts, evaluate, build reports

        Returns:
            list of TaskReport objects (same format as WorkflowRunner)
        """
        # ── Apply per-task dataset overrides ──
        # Parse task .md files first to apply overrides
        task_files = sorted(self.tasks_dir.glob("*.md"))
        if not task_files:
            logger.warning(f"No .md task files found in {self.tasks_dir}")
            return []

        # Apply task filter
        if self.task_filter:
            task_files = [
                f for f in task_files if f.stem in self.task_filter
            ]
            if not task_files:
                logger.warning(
                    f"No task files matched filter: {self.task_filter}."
                )
                return []

        # Apply dataset overrides to configs (same as WorkflowRunner)
        for tf in task_files:
            task_name = tf.stem
            if task_name in self.task_datasets:
                # We handle this by temporarily overriding environment
                # The WorkflowTask reads datasets from config at setup() time
                logger.info(
                    f"Per-task dataset override for '{task_name}': "
                    f"{self.task_datasets[task_name]}"
                )
                # Store for later use in the benchmark builder
                # (workflow task reads available_datasets from .md at setup time,
                #  so we need a different approach — see _apply_dataset_overrides)

        # ── Phase 1: Build Benchmark ──
        logger.info("Phase 1: Building WorkflowBenchmark...")
        benchmark = create_workflow_benchmark(
            task_filter=self.task_filter if self.task_filter else None,
            generalization_datasets=self.generalization_datasets,
            stability_seeds=self.stability_seeds,
            max_steps=self.max_steps,
            headless=self.headless,
            storage_state=self.storage_state,
            user_data_dir=self.user_data_dir,
            skip_generalization=self.skip_generalization,
            skip_stability=self.skip_stability,
        )

        if not benchmark.env_args_list:
            logger.warning("Benchmark has no experiments to run.")
            return []

        logger.info(
            f"Benchmark: {benchmark.name} — "
            f"{len(benchmark.env_args_list)} experiment(s) across "
            f"{len(set(e.task_name for e in benchmark.env_args_list))} task ID(s)"
        )

        # ── Phase 2: Run AgentLab Study ──
        logger.info(
            f"Phase 2: Running AgentLab Study "
            f"(n_jobs={self.n_jobs}, backend={self.parallel_backend})..."
        )
        study_start = time.time()

        try:
            from agentlab.experiments.study import make_study

            self._study = make_study(
                agent_args=self.agent_args,
                benchmark=benchmark,
                suffix=self.study_suffix,
            )
            self._study.run(
                n_jobs=self.n_jobs,
                parallel_backend=self.parallel_backend,
            )
            self._study_dir = self._study.dir

            study_elapsed = time.time() - study_start
            logger.info(
                f"AgentLab Study completed in {study_elapsed:.1f}s. "
                f"Results saved to: {self._study_dir}"
            )

        except ImportError as e:
            logger.error(
                f"Failed to import AgentLab: {e}. "
                "Install agentlab: pip install agentlab>=0.4.0"
            )
            raise
        except Exception as e:
            logger.error(f"AgentLab Study failed: {e}")
            raise

        # ── Phase 3: Post-process — extract artifacts, evaluate, build reports ──
        logger.info("Phase 3: Post-processing — extracting artifacts and evaluating...")
        reports = self._post_process_results()

        total_elapsed = time.time() - study_start
        logger.info(f"AgentLabRunner finished in {total_elapsed:.1f}s total.")

        return reports

    # =========================================================================
    # Post-processing — Evaluation Phase
    # =========================================================================

    def _post_process_results(self) -> list[TaskReport]:
        """
        Iterate over AgentLab experiment results, extract artifacts,
        run EvaluatorAgent on each, and build TaskReport objects.

        Groups runs by task_name (stem, without ds<N> suffix) and builds
        generalization + stability aggregation for each task.
        """
        if self._study_dir is None:
            logger.error("Study directory not set. Did run() succeed?")
            return []

        # Collect all experiment results from the study directory
        exp_results = list(yield_all_exp_results(
            self._study_dir, progress_fn=None, use_cache=False
        ))

        if not exp_results:
            logger.warning(f"No experiment results found in {self._study_dir}")
            return []

        logger.info(f"Found {len(exp_results)} experiment result(s) to evaluate.")

        # Parse task files for evaluation criteria
        task_configs: dict[str, WorkflowConfig] = {}
        task_files = sorted(self.tasks_dir.glob("*.md"))
        for tf in task_files:
            try:
                task_configs[tf.stem] = WorkflowConfig.from_markdown(tf)
            except Exception as e:
                logger.warning(f"Failed to parse task file {tf}: {e}")

        # Process each experiment result
        run_results: list[RunResult] = []
        for exp_result in exp_results:
            try:
                run_result = self._process_single_result(exp_result, task_configs)
                if run_result is not None:
                    run_results.append(run_result)
            except Exception as e:
                logger.error(
                    f"Failed to process experiment {exp_result.exp_dir}: {e}"
                )

        # Group by task name and build reports
        task_runs: dict[str, list[RunResult]] = {}
        for rr in run_results:
            task_runs.setdefault(rr.task_name, []).append(rr)

        reports = []
        for task_name, runs in sorted(task_runs.items()):
            is_quick = self.skip_generalization and self.skip_stability
            # Separate generalization and stability runs
            # Generalization: seed=42, varying datasets
            gen_runs = [r for r in runs if r.seed == 42]
            stab_runs = [r for r in runs if r.seed != 42]

            report = self._build_report(
                task_name=task_name,
                generalization_runs=gen_runs,
                stability_runs=stab_runs,
                is_quick_test=is_quick,
            )
            reports.append(report)

            # Log summary
            if gen_runs:
                label = "Quick Test" if is_quick else "Generalization"
                logger.info(
                    f"  {task_name} {label}: "
                    f"pass_rate={report.generalization['pass_rate']:.1%}, "
                    f"passed={report.generalization.get('passed', 0)}/"
                    f"{report.generalization.get('total_runs', 0)}"
                )
            if stab_runs:
                logger.info(
                    f"  {task_name} Stability: "
                    f"stability_score={report.stability['stability_score']:.2f}"
                )

        return reports

    def _process_single_result(
        self,
        exp_result,
        task_configs: dict[str, WorkflowConfig],
    ) -> Optional[RunResult]:
        """
        Process a single experiment result:
        1. Extract artifacts (same logic as WorkflowRunner._extract_artifacts)
        2. Run EvaluatorAgent
        3. Build RunResult

        Returns None if the experiment failed to produce usable data.
        """
        # Extract task name (stem) from the exp_args
        try:
            exp_task_name = exp_result.exp_args.env_args.task_name
            # Parse: agent_a_eval.<task_stem>.ds<idx>
            parts = exp_task_name.split(".")
            task_stem = parts[1] if len(parts) > 1 else exp_task_name
        except Exception:
            logger.warning(f"Cannot parse task name from {exp_result.exp_dir}")
            return None

        seed = exp_result.exp_args.env_args.task_seed
        if seed is None:
            seed = 42

        # Extract artifacts
        artifacts = self._extract_artifacts(exp_result)
        if artifacts is None:
            logger.warning(f"No artifacts in {exp_result.exp_dir} — skipping evaluation")
            return RunResult(
                task_name=task_stem,
                dataset_name="unknown",
                seed=seed,
                status="error",
                error_message="No artifacts extracted",
            )

        status = artifacts.get("completion_status", "unknown")
        dataset_name = artifacts.get("dataset_name", "unknown")

        # Get task config for evaluation criteria
        task_config = task_configs.get(task_stem)

        # Evaluate
        if task_config and task_config.evaluation_criteria:
            previous_result = self._load_previous_result(task_stem, dataset_name)
            scores = self.evaluator.evaluate(
                evaluation_criteria=task_config.evaluation_criteria,
                artifacts=artifacts,
                task_config=task_config,
                previous_result=previous_result,
            )

            core = scores.get("core", {})
            overall_score = 10.0 if core.get("passed") else 2.0
            core_passed = core.get("passed")
            process_efficiency_scores = scores.get("process_efficiency", {})
            resource_robustness_scores = scores.get("resource_robustness", {})
            task_specific_scores = scores.get("task_specific", {})
            comparison = scores.get("comparison", {})
            overall_comment = scores.get("overall_comment", "")
        else:
            scores = None
            overall_score = 10.0 if status == "completed" else 0.0
            core_passed = None
            process_efficiency_scores = {}
            resource_robustness_scores = {}
            task_specific_scores = {}
            comparison = None
            overall_comment = ""

        # Save per-run log (reuse same format as WorkflowRunner for compatibility)
        self._save_run_log(
            task_name=task_stem,
            dataset_name=dataset_name,
            seed=seed,
            status=status,
            artifacts=artifacts,
            scores=scores if task_config and task_config.evaluation_criteria else None,
            exp_result=exp_result,
        )

        elapsed = artifacts.get("elapsed_time_seconds", 0)

        return RunResult(
            task_name=task_stem,
            dataset_name=dataset_name,
            seed=seed,
            status=status,
            overall_score=overall_score,
            elapsed_seconds=elapsed,
            artifacts=artifacts,
            format="benchmark",
            core_passed=core_passed,
            process_efficiency_scores=process_efficiency_scores,
            resource_robustness_scores=resource_robustness_scores,
            task_specific_scores=task_specific_scores,
            comparison=comparison,
            overall_comment=overall_comment,
        )

    # =========================================================================
    # Artifact Extraction (mirrors WorkflowRunner._extract_artifacts)
    # =========================================================================

    def _extract_artifacts(self, exp_result) -> Optional[dict]:
        """
        Extract artifacts collected by WorkflowTask from an ExpResult.

        Artifacts are stored in the last step's task_info dict.
        """
        try:
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
    # Previous Result Loading (for iterative improvement comparison)
    # =========================================================================

    def _load_previous_result(
        self,
        task_name: str,
        dataset_name: str,
    ) -> Optional[dict]:
        """
        Load the most recent previous evaluation result for comparison.

        Same logic as WorkflowRunner._load_previous_result().
        """
        try:
            logs_dir = Path(self.exp_root) / "logs" / task_name
            if not logs_dir.exists():
                return None

            log_dirs = sorted(logs_dir.iterdir(), reverse=True)
            for log_dir in log_dirs:
                if not log_dir.is_dir():
                    continue
                if not log_dir.name.startswith(dataset_name):
                    continue
                result_file = log_dir / "result.json"
                if not result_file.exists():
                    continue
                try:
                    data = json.loads(result_file.read_text(encoding="utf-8"))
                    if data.get("core"):
                        logger.info(
                            f"Loaded previous result for {task_name}/{dataset_name}"
                        )
                        return {
                            "core": data.get("core", {}),
                            "process_efficiency": data.get("process_efficiency", {}),
                            "resource_robustness": data.get("resource_robustness", {}),
                            "task_specific": data.get("task_specific", {}),
                            "overall_comment": data.get("overall_comment", ""),
                        }
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse previous result: {e}")
                    continue

            return None
        except Exception as e:
            logger.warning(f"Error loading previous result: {e}")
            return None

    # =========================================================================
    # Per-Run Log Saving (same format as WorkflowRunner)
    # =========================================================================

    def _save_run_log(
        self,
        task_name: str,
        dataset_name: str,
        seed: int,
        status: str,
        artifacts: Optional[dict],
        scores: Optional[dict],
        exp_result=None,
    ) -> Path:
        """
        Save detailed per-run logs. Same format as WorkflowRunner._save_run_log().
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

        # ── 2. Step-by-step detail ──
        step_lines = []
        if exp_result is not None:
            try:
                n_steps = exp_result.summary_info.get("n_steps", 0)
                for i in range(n_steps + 1):
                    try:
                        si = exp_result.get_step_info(i)
                        if si is None:
                            continue
                    except Exception:
                        continue

                    step_lines.append(f"{'=' * 70}")
                    step_lines.append(f"=== Step {i} / {n_steps}")
                    step_lines.append(f"{'=' * 70}")

                    action = si.action or "(无动作 — 初始状态)"
                    step_lines.append(f"\n  [动作 ACTION]")
                    step_lines.append(f"  {action}")

                    agent_info = si.agent_info or {}
                    think = agent_info.get("think", "")
                    if think:
                        step_lines.append(f"\n  [思考 THINK]")
                        step_lines.append(f"  {think}")

                    obs = si.obs or {}
                    if obs.get("axtree_object"):
                        try:
                            from browsergym.utils.obs import flatten_axtree_to_str
                            axtree_txt = flatten_axtree_to_str(obs["axtree_object"])
                            step_lines.append(f"\n  [页面 AXTree]")
                            step_lines.append(axtree_txt)
                        except Exception:
                            pass

                    chat = obs.get("chat_messages", [])
                    if chat:
                        step_lines.append(f"\n  [对话消息]")
                        for msg in chat[-6:]:
                            role = msg.get("role", "?")
                            text = str(msg.get("message", ""))[:500]
                            step_lines.append(f"  [{role}] {text}")

                    ti = si.task_info or {}
                    if ti:
                        step_lines.append(f"\n  [任务状态]")
                        step_lines.append(f"  status={ti.get('status', '?')}")
                        if ti.get("loop_reason"):
                            step_lines.append(f"  loop_reason={ti['loop_reason']}")

                    step_lines.append("")
            except Exception as e:
                step_lines.append(f"\n(提取步骤详情时出错: {e})")

        (log_dir / "step_by_step.txt").write_text(
            "\n".join(step_lines) if step_lines else "(无步骤数据)",
            encoding="utf-8",
        )

        # ── 3. Final page AXTree ──
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
                pass
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
        agent_lines = []
        for resp in artifacts.get("agent_a_responses", []):
            agent_lines.append(f"=== 来源: {resp.get('selector', '?')} ===")
            agent_lines.append(resp.get("text", ""))
            agent_lines.append("")
        (log_dir / "agent_a_output.txt").write_text(
            "\n".join(agent_lines) if agent_lines else "(无 Agent A 输出)",
            encoding="utf-8",
        )

        # ── 6. Result summary ──
        if scores:
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
                "comparison": scores.get("comparison", {}),
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
                "format": "benchmark",
                "core": {"passed": None},
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

    # =========================================================================
    # Report Building (same as WorkflowRunner)
    # =========================================================================

    def _build_report(
        self,
        task_name: str,
        generalization_runs: list[RunResult],
        stability_runs: list[RunResult],
        is_quick_test: bool = False,
    ) -> TaskReport:
        """Build aggregated TaskReport (same logic as WorkflowRunner)."""

        def _compute_stats(runs):
            passed = sum(1 for r in runs if r.core_passed)
            return {
                "format": "benchmark",
                "pass_rate": passed / len(runs) if runs else 0,
                "total_runs": len(runs),
                "passed": passed,
                "failed": len(runs) - passed,
                "statuses": [r.status for r in runs],
            }

        gen_stats = _compute_stats(generalization_runs)
        stab_stats = _compute_stats(stability_runs)
        stab_stats["stability_score"] = stab_stats.get("pass_rate", 0)

        return TaskReport(
            task_name=task_name,
            format="benchmark",
            generalization=gen_stats,
            stability=stab_stats,
            runs=generalization_runs + stability_runs,
            is_quick_test=is_quick_test,
        )

    # =========================================================================
    # Report Serialization (delegates to WorkflowRunner's methods)
    # =========================================================================

    def save_reports(
        self,
        reports: list[TaskReport],
        output_dir: str = "reports",
    ):
        """
        Save evaluation reports using WorkflowRunner's save logic.

        Creates a temporary WorkflowRunner with a no-op config to reuse
        its save_reports(), _build_text_report(), etc.
        """
        # Create a minimal WorkflowRunner just for report saving
        temp_runner = _ReportSaver()
        temp_runner.save_reports(reports, output_dir)

    @property
    def study_dir(self) -> Optional[Path]:
        """Return the AgentLab Study directory (for AgentXRay)."""
        if self._study_dir is not None:
            return Path(self._study_dir)
        return None

    @property
    def xray_command(self) -> str:
        """Return the command to launch AgentXRay for this study."""
        exp_root = os.environ.get("AGENTLAB_EXP_ROOT", str(self.exp_root))
        return f"AGENTLAB_EXP_ROOT={exp_root} agentlab-xray"


class _ReportSaver(WorkflowRunner):
    """
    Minimal WorkflowRunner subclass that only exposes the report saving methods.

    We reuse WorkflowRunner.save_reports(), _build_text_report(), etc.
    without needing to set up a real runner.
    """

    def __init__(self):
        # Bypass WorkflowRunner.__init__ — we only need the save methods
        pass
