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
        max_retries: int = 2,
    ) -> dict:
        """
        Evaluate Agent A's performance based on collected artifacts.

        Args:
            evaluation_criteria: raw markdown from the task .md's "## 评估标准" section
            artifacts: dict from WorkflowTask._collect_artifacts()
            max_retries: number of retries if JSON parsing fails

        Returns:
            dict:
                dimensions: {dim_name: {score, justification, max_score}}
                overall_score: float (0-10)
                loop_detected: bool
                raw_llm_response: str
        """
        # Build the evaluation prompt
        prompt = self._build_evaluation_prompt(evaluation_criteria, artifacts)

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
        Path(__file__).parent.parent.parent.parent / "prompts" / "evaluator_agent_prompt.md"
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

    def _build_evaluation_prompt(self, criteria: str, artifacts: dict) -> str:
        """Build the evaluation prompt with task-specific context."""

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

        Handles cases where the JSON is wrapped in markdown code blocks,
        contains trailing commas, or other minor formatting issues.
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

        # Validate structure
        result = {
            "dimensions": parsed.get("dimensions", {}),
            "loop_detected": parsed.get("loop_detected", False),
            "overall_comment": parsed.get("overall_comment", ""),
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
