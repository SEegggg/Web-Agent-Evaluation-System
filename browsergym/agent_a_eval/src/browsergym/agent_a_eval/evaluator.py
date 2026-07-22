"""
EvaluatorAgent — independent LLM-based evaluator for Agent A outputs.

This module is COMPLETELY INDEPENDENT from BrowserGym.
It runs AFTER the BrowserGym session ends, receiving only artifacts
(screenshots, chat history, page state) collected by WorkflowTask.

Architecture principle:
    Driver Agent (BrowserGym) ≠ Evaluator Agent (this module)
    Separate sessions, separate responsibilities.

Evaluation framework (benchmark mode):
    - Core metrics (PASS/FAIL) — task-specific, from task .md
    - Process & efficiency (0-10) — framework-level, from system prompt
    - Resource & robustness (0-10) — framework-level, from system prompt
    - Task-specific metrics (0-10) — task-specific, from task .md
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from .utils import WorkflowConfig

logger = logging.getLogger(__name__)


class EvaluatorAgent:
    """
    Independent LLM evaluator for judging Agent A's task performance.

    Does NOT touch BrowserGym or any browser environment.
    Only consumes artifacts produced by WorkflowTask._collect_artifacts().

    All tasks are evaluated in benchmark format with four metric categories:
    1. Core (PASS/FAIL) — hard constraint, task-specific pass/fail conditions
    2. Process & efficiency (0-10) — framework-level, same dimensions for all tasks
    3. Resource & robustness (0-10) — framework-level, same dimensions for all tasks
    4. Task-specific (0-10) — task-specific, dimensions vary per task

    Usage:
        evaluator = EvaluatorAgent(model="gpt-4o")
        scores = evaluator.evaluate(
            evaluation_criteria=task_md_evaluation_section,
            artifacts=workflow_artifacts,
            task_config=config,
        )
    """

    # Supported models and their providers
    MODEL_PROVIDERS = {
        "gpt-4o": "openai",
        "gpt-4-turbo": "openai",
        "gpt-4o-mini": "openai",
        "gpt-4.1": "openai",
        "claude-sonnet-5": "anthropic",
        "claude-opus-4-8": "anthropic",
        "claude-haiku-4-5": "anthropic",
    }

    def __init__(
        self,
        model: str = "gpt-4o",
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Args:
            model: LLM model name for evaluation
            provider: "openai" or "anthropic" (auto-detected from model name if None)
            api_key: override API key (uses env var if None)
            base_url: override API base URL (uses default if None)
        """
        self.model = model

        # Auto-detect provider
        if provider is None:
            provider = self.MODEL_PROVIDERS.get(model, "openai")
        self.provider = provider

        self.api_key = api_key
        self.base_url = base_url
        self.client = self._init_client()
        logger.info(f"EvaluatorAgent initialized: model={model}, provider={provider}")

    def _init_client(self):
        """Initialize the LLM client based on provider."""
        if self.provider == "openai":
            import openai

            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            return openai.OpenAI(**kwargs)

        elif self.provider == "anthropic":
            import anthropic

            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            return anthropic.Anthropic(**kwargs)

        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def evaluate(
        self,
        evaluation_criteria: str,
        artifacts: dict,
        task_config: Optional[WorkflowConfig] = None,
        previous_result: Optional[dict] = None,
        max_retries: int = 2,
    ) -> dict:
        """
        Evaluate Agent A's performance based on collected artifacts.

        All evaluations use the benchmark format:
            core: {passed, justification}
            process_efficiency: {dim: {score, justification}}
            resource_robustness: {dim: {score, justification}}
            task_specific: {dim: {score, justification}}
            overall_comment: str
            comparison: {fixed_issues, new_issues, regressions, improvements, ...}

        Args:
            evaluation_criteria: raw markdown from the task .md's "## 评估标准" section
            artifacts: dict from WorkflowTask._collect_artifacts()
            task_config: optional WorkflowConfig with parsed core/task_specific criteria
            previous_result: optional dict of the previous run's evaluation result
                             (used for iterative improvement comparison)
            max_retries: number of retries if JSON parsing fails

        Returns:
            dict with benchmark format keys including comparison
        """
        # Build the evaluation prompt
        prompt = self._build_evaluation_prompt(
            evaluation_criteria, artifacts, task_config, previous_result,
        )

        # Build messages (text only — screenshots skipped to stay compatible
        # with non-multimodal models like DeepSeek)
        messages = self._build_messages(prompt, screenshot_base64=None)

        # Call LLM with retry for JSON parsing
        for attempt in range(max_retries + 1):
            try:
                raw_response = self._call_llm(messages)
                result = self._parse_response(raw_response)
                result["raw_llm_response"] = raw_response
                return result
            except json.JSONDecodeError as e:
                logger.warning(
                    f"JSON parsing failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
                if attempt < max_retries:
                    # Add a retry hint
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous response could not be parsed as JSON. "
                                "Please ensure your entire response is a valid JSON object "
                                "with the exact structure requested."
                            ),
                        }
                    )
                else:
                    # Final fallback
                    return {
                        "format": "benchmark",
                        "core": {"passed": False, "justification": "评估失败：LLM 响应无法解析"},
                        "process_efficiency": {},
                        "resource_robustness": {},
                        "task_specific": {},
                        "overall_comment": "",
                        "comparison": {
                            "fixed_issues": [],
                            "new_issues": [],
                            "regressions": [],
                            "improvements": [],
                            "unchanged_issues": [],
                            "overall_trend": "first_run",
                        },
                        "error": f"Failed to parse LLM response after {max_retries + 1} attempts",
                        "raw_llm_response": raw_response,
                    }

    # Path to the evaluator system prompt file
    _EVAL_PROMPT_PATH = (
        Path(__file__).resolve().parents[3] / "prompts" / "evaluator_agent_prompt.md"
    )

    @classmethod
    def _load_system_prompt(cls) -> str:
        """Load the evaluator system prompt from file."""
        try:
            path = cls._EVAL_PROMPT_PATH
            if path.exists():
                return path.read_text(encoding="utf-8").strip()
        except Exception:
            pass
        # Fallback (should not happen if file is present)
        return "你是一个严格但公正的数据科学 Agent 评审专家。"

    def _build_evaluation_prompt(
        self, criteria: str, artifacts: dict,
        task_config: Optional[WorkflowConfig] = None,
        previous_result: Optional[dict] = None,
    ) -> str:
        """Build the evaluation user prompt.

        Only injects task-specific content (core criteria from .md and
        task-specific metrics) plus artifacts.

        Framework-level dimensions (process_efficiency, resource_robustness)
        are defined in the system prompt (evaluator_agent_prompt.md) and
        do NOT need to be repeated here.

        When previous_result is provided, includes it with comparison
        instructions so the evaluator can track iterative improvement.
        """
        # Determine core criteria and task-specific metrics
        if task_config and task_config.is_benchmark_format:
            core_criteria = task_config.core_criteria or "（未提供核心判定条件，请根据 Agent A 输出和对话历史自行判定任务是否成功完成）"
            task_specific = task_config.task_specific_metrics or ""
        else:
            # Legacy .md without benchmark sub-sections:
            # treat the entire evaluation_criteria as both core hints and task-specific
            core_criteria = (
                f"（此任务使用旧版评估标准格式，请根据以下标准判定核心通过/失败）\n\n{criteria}"
            )
            task_specific = ""

        if not task_specific:
            task_specific = "本任务未定义专项评估指标，请输出空对象 {}。"

        # Format agent A responses
        agent_output = ""
        for resp in artifacts.get("agent_a_responses", []):
            agent_output += (
                f"\n### Output from {resp['selector']}\n"
                f"```\n{resp['text']}\n```\n"
            )
        if not agent_output:
            agent_output = "(Agent A 没有可见的输出内容)"

        # Format chat history summary
        chat_summary = ""
        for msg in artifacts.get("chat_history", [])[-20:]:
            role = msg.get("role", "?")
            message = msg.get("message", "")
            if len(message) > 500:
                message = message[:500] + "..."
            chat_summary += f"[{role}] {message}\n"

        prompt = f"""\
请根据以下信息对 Agent A 的表现进行基准测试（Benchmark）评分。

## 本任务的核心判定条件（PASS/FAIL）
{core_criteria}

请判定 Agent A 是否通过了核心指标，并给出判定理由。
重点说明失败原因是 Agent A 的问题还是外部因素（数据、指令、基础设施等）。

## 本任务的专项评估指标（0-10 评分）
{task_specific}

请根据上方列出的每个维度，逐一给出 0-10 评分和理由。
⚠️ 所有评分仅针对 Agent A（数据科学 Agent），不要评估驱动 Agent 的浏览器操作。
过程与效率、资源与鲁棒性这两类指标的维度已在系统 prompt 中定义，
请参考系统 prompt 中对 instruction_comprehension / task_execution_efficiency /
output_redundancy / computation_cost / task_execution_latency /
self_correction_rate 六个维度的说明进行评分。

## 任务元信息
- 任务名称: {artifacts.get('task_title', 'N/A')}
- 使用的数据集: {artifacts.get('dataset_name', 'N/A')}
- 执行耗时: {artifacts.get('elapsed_time_seconds', 0):.0f} 秒
- 完成状态: {artifacts.get('completion_status', 'unknown')}
  （completed=正常完成, dead_loop=检测到循环被终止, timeout=超时,
    infeasible=无法执行, error=出错）

## Agent A 的页面输出
{agent_output}

## 对话历史（驱动 Agent 与 Agent A 的交互）
{chat_summary}
{self._build_comparison_section(previous_result)}
## 输出格式
请仅输出系统 prompt 中定义的 JSON 格式（core + process_efficiency +
resource_robustness + task_specific + overall_comment + comparison）。
特别注意 comparison 字段必须根据本次与上次的对比填写。
不要添加任何解释性文字。"""
        return prompt

    @staticmethod
    def _build_comparison_section(previous_result: Optional[dict]) -> str:
        """Build the comparison section for the evaluation prompt.

        When previous_result is available, formats it for side-by-side comparison.
        When None (first run), returns empty string.
        """
        if not previous_result:
            return ""

        lines = [
            "",
            "## 历史评估结果（上次运行）",
            "",
            "⚠️ 重要：本次评估的核心目的是**追踪 Agent A 的迭代改进情况**。",
            "请仔细对比本次表现与上次评估结果，重点回答：",
            "1. 上次存在的问题本次是否已修复？（→ fixed_issues）",
            "2. 本次是否出现了新的错误？（→ new_issues）",
            "3. 有没有之前正常但本次退化的方面？（→ regressions）",
            "4. 有哪些方面比上次有明显提升？（→ improvements）",
            "5. 哪些上次的问题本次仍然存在？（→ unchanged_issues）",
            "",
            "以下是上次评估的完整结果：",
            "",
        ]

        # Format core result
        prev_core = previous_result.get("core", {})
        prev_passed = prev_core.get("passed")
        prev_passed_str = "PASS" if prev_passed else ("FAIL" if prev_passed is False else "N/A")
        lines.append(f"- 核心判定: {prev_passed_str}")
        if prev_core.get("justification"):
            lines.append(f"  理由: {prev_core['justification']}")

        # Format process_efficiency scores
        prev_pe = previous_result.get("process_efficiency", {})
        if prev_pe:
            lines.append("- 过程与效率指标:")
            for key, val in prev_pe.items():
                if isinstance(val, dict):
                    lines.append(f"  - {key}: {val.get('score', '?')}/10 — {val.get('justification', '')}")

        # Format resource_robustness scores
        prev_rr = previous_result.get("resource_robustness", {})
        if prev_rr:
            lines.append("- 资源与鲁棒性指标:")
            for key, val in prev_rr.items():
                if isinstance(val, dict):
                    lines.append(f"  - {key}: {val.get('score', '?')}/10 — {val.get('justification', '')}")

        # Format task_specific scores
        prev_ts = previous_result.get("task_specific", {})
        if prev_ts:
            lines.append("- 任务专项指标:")
            for key, val in prev_ts.items():
                if isinstance(val, dict):
                    lines.append(f"  - {key}: {val.get('score', '?')}/10 — {val.get('justification', '')}")

        # Format overall comment
        prev_comment = previous_result.get("overall_comment", "")
        if prev_comment:
            lines.append(f"- 上次整体评价: {prev_comment}")

        return "\n".join(lines)

    def _build_messages(
        self, prompt: str, screenshot_base64: Optional[str]
    ) -> list[dict]:
        """Build message list with system prompt from file."""
        content = [{"type": "text", "text": prompt}]

        if screenshot_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{screenshot_base64}",
                        "detail": "auto",
                    },
                }
            )

        system_msg = self._load_system_prompt()

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": content},
        ]

    def _call_llm(self, messages: list[dict]) -> str:
        """Call the LLM and return the text response."""
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=4000,  # Increased for benchmark + comparison output
                temperature=0.1,  # Low temperature for consistent scoring
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            # Convert OpenAI format to Anthropic format
            system = ""
            anthropic_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
                else:
                    anthropic_content = []
                    content = msg["content"]
                    if isinstance(content, str):
                        anthropic_content = [{"type": "text", "text": content}]
                    elif isinstance(content, list):
                        for block in content:
                            if block["type"] == "text":
                                anthropic_content.append(block)
                            elif block["type"] == "image_url":
                                # Extract base64 data
                                url = block["image_url"]["url"]
                                if url.startswith("data:image/png;base64,"):
                                    b64_data = url.split(",", 1)[1]
                                else:
                                    b64_data = url
                                anthropic_content.append(
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": b64_data,
                                        },
                                    }
                                )
                    anthropic_messages.append({"role": "user", "content": anthropic_content})

            kwargs = {"model": self.model, "max_tokens": 4000, "messages": anthropic_messages}
            if system:
                kwargs["system"] = system
            response = self.client.messages.create(**kwargs)
            return response.content[0].text

    def _parse_response(self, response: str) -> dict:
        """
        Parse the structured JSON from the LLM response.

        All evaluations use the benchmark format:
            core, process_efficiency, resource_robustness, task_specific, overall_comment
        """
        # Try to extract JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to find raw JSON object
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response

        # Fix common JSON issues
        json_str = self._sanitize_json(json_str)

        parsed = json.loads(json_str)

        # All results are benchmark format
        return {
            "format": "benchmark",
            "core": parsed.get("core", {}),
            "process_efficiency": parsed.get("process_efficiency", {}),
            "resource_robustness": parsed.get("resource_robustness", {}),
            "task_specific": parsed.get("task_specific", {}),
            "overall_comment": parsed.get("overall_comment", ""),
            "comparison": parsed.get("comparison", {
                "fixed_issues": [],
                "new_issues": [],
                "regressions": [],
                "improvements": [],
                "unchanged_issues": [],
                "overall_trend": "first_run",
            }),
            "raw_llm_response": response,
        }

    @staticmethod
    def _sanitize_json(json_str: str) -> str:
        """Fix common JSON formatting issues from LLM outputs."""
        # Remove trailing commas before closing brackets
        json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)
        # Remove comments
        json_str = re.sub(r"//.*?\n", "\n", json_str)
        # Fix single quotes
        # (be careful not to break apostrophes within strings)
        return json_str
