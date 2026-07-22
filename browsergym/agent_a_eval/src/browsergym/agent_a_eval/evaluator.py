"""
EvaluatorAgent — independent LLM-based evaluator for Agent A outputs.

This module is COMPLETELY INDEPENDENT from BrowserGym.
It runs AFTER the BrowserGym session ends, receiving only artifacts
(screenshots, chat history, page state) collected by WorkflowTask.

Architecture principle:
    Driver Agent (BrowserGym) ≠ Evaluator Agent (this module)
    Separate sessions, separate responsibilities.
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

    Usage:
        evaluator = EvaluatorAgent(model="gpt-4o")
        scores = evaluator.evaluate(
            evaluation_criteria=task_md_evaluation_section,
            artifacts=workflow_artifacts,
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
        max_retries: int = 2,
    ) -> dict:
        """
        Evaluate Agent A's performance based on collected artifacts.

        Args:
            evaluation_criteria: raw markdown from the task .md's "## 评估标准" section
            artifacts: dict from WorkflowTask._collect_artifacts()
            task_config: optional WorkflowConfig — used to detect benchmark vs legacy format
            max_retries: number of retries if JSON parsing fails

        Returns:
            dict (legacy format):
                dimensions: {dim_name: {score, justification, max_score}}
                overall_score: float (0-10)
                loop_detected: bool
                raw_llm_response: str
            dict (benchmark format):
                format: "benchmark"
                core: {passed, justification}
                process_efficiency: {dim: {score, justification}}
                resource_robustness: {dim: {score, justification}}
                task_specific: {dim: {score, justification}}
                overall_comment: str
        """
        # Build the evaluation prompt
        prompt = self._build_evaluation_prompt(evaluation_criteria, artifacts, task_config)

        # Build messages (text only — screenshots skipped to stay compatible
        # with non-multimodal models like DeepSeek)
        messages = self._build_messages(prompt, screenshot_base64=None)

        # Call LLM with retry for JSON parsing
        for attempt in range(max_retries + 1):
            try:
                raw_response = self._call_llm(messages)
                result = self._parse_response(raw_response, evaluation_criteria)
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
                        "dimensions": {},
                        "overall_score": 0,
                        "loop_detected": False,
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
    ) -> str:
        """Build the evaluation prompt.

        Branches on task_config.is_benchmark_format:
        - True → benchmark-style prompt with four metric categories
        - False/None → legacy weighted-scoring prompt
        """
        if task_config and task_config.is_benchmark_format:
            return self._build_benchmark_evaluation_prompt(task_config, artifacts)
        else:
            return self._build_legacy_evaluation_prompt(criteria, artifacts)

    def _build_legacy_evaluation_prompt(self, criteria: str, artifacts: dict) -> str:
        """Build the legacy weighted-scoring evaluation prompt (backward compat)."""

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
请根据以下信息对 Agent A 的表现进行客观评分。

## 评估标准
{criteria}

## 任务元信息
- 任务名称: {artifacts.get('task_title', 'N/A')}
- 使用的数据集: {artifacts.get('dataset_name', 'N/A')}
- 执行耗时: {artifacts.get('elapsed_time_seconds', 0):.0f} 秒
- 完成状态: {artifacts.get('completion_status', 'unknown')}

## Agent A 的页面输出
{agent_output}

## 对话历史（驱动 Agent 与 Agent A 的交互）
{chat_summary}

## 完成状态说明
- 任务状态: {artifacts.get('completion_status', 'unknown')}
  （completed=正常完成, dead_loop=检测到循环被终止, timeout=超时, infeasible=无法执行, error=出错）
- 如果状态不是 completed，说明任务在 BrowserGym 层面已终止，请结合对话历史和 Agent A 输出判断：
  - Agent A 是否已经正确完成了任务？（如果是，失败责任在驱动 Agent 或基础设施）
  - 还是 Agent A 确实没有产出正确的结果？（如果是，失败责任在 Agent A）
"""
        return prompt

    def _build_benchmark_evaluation_prompt(
        self, config: "WorkflowConfig", artifacts: dict
    ) -> str:
        """Build the benchmark-style evaluation prompt with four metric categories."""

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
请以基准测试（Benchmark）方式对 Agent A 的表现进行客观评分。

## ⚠️ 评分原则：严格区分失败归因
核心指标只衡量"Agent A 是否按要求完成了该做的事"，不衡量"最终结果有多好"。
- 如果 Agent A 正确完成了所有流程，但最终数值不好（如模型 RMSE 高），
  而这是由人类指定的数据集/模型类型导致的 → 核心指标判定为 **PASS**
- 如果 Agent A 遗漏了步骤、使用错误方法、输出不完整 → 核心指标判定为 **FAIL**
- 当无法确定失败原因时，优先检查 Agent A 的输出中是否有逻辑错误或遗漏步骤

## 一、核心指标: 任务成功率（PASS/FAIL 二元判定）
{config.core_criteria}

请判定: 任务是否成功？输出 PASS 或 FAIL，并给出判定理由。
重点说明失败原因是 Agent A 的问题还是外部因素（数据、指令、基础设施等）。

## 二、过程与效率指标（辅助参考，0-10 评分）
{config.process_efficiency_criteria or '评估驱动 Agent 的操作效率：'}

对以下维度给出 0-10 评分：
- 工具调用准确率 (Tool Call Accuracy): 每次工具调用是否正确达成了预期效果
- 轨迹步骤效率 (Trajectory Efficiency): 完成任务所用的步数是否合理，有无绕路
- 冗余/无效操作率: 是否存在重复操作、无效点击、无意义的页面浏览

## 三、资源与鲁棒性指标（辅助参考，0-10 评分）
{config.resource_robustness_criteria or '评估资源消耗和异常处理：'}

对以下维度给出 0-10 评分：
- Token 消耗成本: 对话历史和工具调用产生的 Token 消耗是否合理
- 任务执行时延 (Latency): 总执行时间是否在合理范围内
- 自我纠错与异常恢复率: 遇到错误时能否自主发现并纠正

## 四、任务专项指标（0-10 评分）
{config.task_specific_metrics or '无特定任务专项指标。'}

请根据上方任务专项指标中列出的每个维度，逐一给出 0-10 评分和理由。
如果未列出具体维度，则输出空对象 {{}}。

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

## 输出格式
请仅输出以下 JSON（不要添加任何解释性文字）：
```json
{{
  "core": {{
    "passed": true,
    "justification": "判定通过/失败的具体理由，说明是否是 Agent A 的问题"
  }},
  "process_efficiency": {{
    "tool_call_accuracy": {{"score": 8, "justification": "..."}},
    "trajectory_efficiency": {{"score": 7, "justification": "..."}},
    "redundant_operation_rate": {{"score": 6, "justification": "..."}}
  }},
  "resource_robustness": {{
    "token_consumption_cost": {{"score": 7, "justification": "..."}},
    "task_execution_latency": {{"score": 8, "justification": "..."}},
    "self_correction_rate": {{"score": 6, "justification": "..."}}
  }},
  "task_specific": {{
    "示例指标名": {{"score": 7, "justification": "..."}}
  }},
  "overall_comment": "一句话总结整体表现"
}}
```"""
        return prompt

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
            # Convert content array to string for models that don't support content arrays
            # (gpt-4o does support it though)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2000,
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

            kwargs = {"model": self.model, "max_tokens": 2000, "messages": anthropic_messages}
            if system:
                kwargs["system"] = system
            response = self.client.messages.create(**kwargs)
            return response.content[0].text

    def _parse_response(self, response: str, criteria: str) -> dict:
        """
        Parse the structured JSON from the LLM response.

        Handles both legacy format (dimensions, overall_score, loop_detected)
        and benchmark format (core, process_efficiency, resource_robustness, task_specific).
        Auto-detects which format by checking for the "core" key.
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

        # ── Detect format: benchmark format has "core" key ──
        if "core" in parsed:
            return {
                "format": "benchmark",
                "core": parsed.get("core", {}),
                "process_efficiency": parsed.get("process_efficiency", {}),
                "resource_robustness": parsed.get("resource_robustness", {}),
                "task_specific": parsed.get("task_specific", {}),
                "overall_comment": parsed.get("overall_comment", ""),
                "raw_llm_response": response,
            }

        # ── Legacy format ──
        result = {
            "format": "legacy",
            "dimensions": parsed.get("dimensions", {}),
            "loop_detected": parsed.get("loop_detected", False),
            "overall_comment": parsed.get("overall_comment", ""),
            "raw_llm_response": response,
        }

        # Compute overall score (weighted average if no explicit overall)
        if result["dimensions"]:
            scores = [
                dim.get("score", 0) for dim in result["dimensions"].values()
            ]
            result["overall_score"] = sum(scores) / len(scores) if scores else 0
        else:
            result["overall_score"] = 0

        return result

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
