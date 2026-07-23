#!/usr/bin/env python
"""
Task Generator — LLM-powered .md task definition generator for Agent A evaluation.

Reads LLM configuration from config.yaml (evaluator or task_generator section),
calls the LLM to auto-generate a benchmark-format task .md file, validates the
output, and saves it to the tasks/ directory.

Usage:
    cd browsergym/agent_a_eval
    # Command-line mode
    python scripts/task_generator.py -d model_training -n "时间序列预测" \\
        --datasets "weather.csv:天气数据,stock.csv:股价数据" \\
        -r "需要评估模型的时序预测能力"

    # Interactive mode
    python scripts/task_generator.py --interactive

    # Custom config
    python scripts/task_generator.py --config my_config.yaml -d custom -n "my_task"
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# LLM Client (supports OpenAI and Anthropic, mirrors evaluator.py pattern)
# =============================================================================


def _init_llm_client(provider: str, api_key: Optional[str], base_url: Optional[str]):
    """Initialize an LLM client based on provider."""
    if provider == "openai":
        import openai

        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return openai.OpenAI(**kwargs)

    elif provider == "anthropic":
        import anthropic

        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return anthropic.Anthropic(**kwargs)

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def _call_llm(
    client,
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> str:
    """Call the LLM and return the text response."""
    if provider == "openai":
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    elif provider == "anthropic":
        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content[0].text


# =============================================================================
# TaskGenerator
# =============================================================================


class TaskGenerator:
    """Generates BrowserGym Agent A evaluation task .md files via LLM."""

    # Path to example .md files (used as reference in the generation prompt)
    _TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"

    SYSTEM_PROMPT = """\
你是一个 BrowserGym Agent 评测任务定义专家。你的职责是根据用户提供的参数，
生成一份符合规范的 Agent A 评测任务 .md 文件。

Agent A 是一个运行在 Web 页面上的数据科学 Agent，用户通过浏览器与它交互。
你的任务是定义评测标准来评估 Agent A 在特定领域的能力。

生成的 .md 文件将用于自动化评测流水线：驱动 Agent 操作浏览器执行任务步骤，
然后独立的评估 Agent 根据你定义的评估标准对 Agent A 的表现进行评分。"""

    def __init__(self, llm_config: dict):
        """
        Args:
            llm_config: dict with keys: provider, model, api_key, base_url,
                        temperature, max_tokens
        """
        self.provider = llm_config.get("provider", "openai")
        self.model = llm_config.get("model", "gpt-4o")
        self.api_key = llm_config.get("api_key")
        self.base_url = llm_config.get("base_url")
        self.temperature = llm_config.get("temperature", 0.3)
        self.max_tokens = llm_config.get("max_tokens", 4096)
        self.client = _init_llm_client(self.provider, self.api_key, self.base_url)
        logger.info(
            f"TaskGenerator initialized: provider={self.provider}, model={self.model}"
        )

    def generate(self, params: dict) -> str:
        """
        Generate a .md task definition.

        Args:
            params: dict with keys:
                - domain: task domain (data_analysis/data_cleaning/model_training/custom)
                - task_name: display name for the task
                - datasets: list of (filename, description) tuples
                - requirements: optional extra instructions (free text)

        Returns:
            Raw markdown string for the .md file.
        """
        user_prompt = self._build_generation_prompt(params)

        logger.info("Calling LLM to generate task .md ...")
        raw_response = _call_llm(
            client=self.client,
            provider=self.provider,
            model=self.model,
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Strip markdown code fences if the LLM wrapped the output
        md_content = self._unwrap_code_fence(raw_response)
        return md_content

    def generate_with_retry(self, params: dict, max_retries: int = 2) -> str:
        """
        Generate with validation + automatic retry on failure.

        Returns the validated markdown string, or raises ValueError after retries.
        """
        for attempt in range(max_retries + 1):
            md_content = self.generate(params)
            errors = self.validate(md_content)

            if not errors:
                logger.info("Generated .md passed validation.")
                return md_content

            logger.warning(
                f"Validation failed (attempt {attempt + 1}/{max_retries + 1}): "
                f"{len(errors)} error(s)"
            )
            for e in errors:
                logger.warning(f"  - {e}")

            if attempt < max_retries:
                # Tell the LLM to fix the issues
                logger.info("Requesting LLM to fix validation errors...")
                fix_prompt = self._build_fix_prompt(params, md_content, errors)
                raw_response = _call_llm(
                    client=self.client,
                    provider=self.provider,
                    model=self.model,
                    system_prompt=self.SYSTEM_PROMPT,
                    user_prompt=fix_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                md_content = self._unwrap_code_fence(raw_response)
            else:
                raise ValueError(
                    f"Failed to generate valid .md after {max_retries + 1} attempts. "
                    f"Errors: {'; '.join(errors)}"
                )

        return md_content

    def validate(self, md_content: str) -> list[str]:
        """
        Validate generated .md has all required sections and correct formatting.

        Returns:
            List of error messages (empty = valid).
        """
        errors = []

        # 1. Must have a top-level title
        if not re.search(r"^#\s+", md_content, re.MULTILINE):
            errors.append("缺少一级标题 (# 标题)")

        # 2. Must have 可用数据集 section
        if "## 可用数据集" not in md_content:
            errors.append("缺少 '## 可用数据集' 章节")

        # 3. Must have 步骤 section
        if "## 步骤" not in md_content:
            errors.append("缺少 '## 步骤' 章节")

        # 4. Must have 评估标准 section
        if "## 评估标准" not in md_content:
            errors.append("缺少 '## 评估标准' 章节")

        # 5. 评估标准 should have at least one ### sub-section (核心指标 is required)
        eval_match = re.search(r"## 评估标准\n+(.*?)(?=\n## |\Z)", md_content, re.DOTALL)
        if eval_match:
            eval_body = eval_match.group(1)
            if not re.search(r"^###\s+", eval_body, re.MULTILINE):
                errors.append("'## 评估标准' 中缺少 '### ' 子章节（至少需要核心指标）")
            # Check that core metrics section exists
            if not re.search(r"###\s*核心指标", eval_body):
                errors.append("'## 评估标准' 中缺少 '### 核心指标' 子章节")

        # 6. Dataset entries should match expected pattern
        dataset_section = re.search(
            r"## 可用数据集\n+(.*?)(?=\n## |\Z)", md_content, re.DOTALL
        )
        if dataset_section:
            dataset_lines = dataset_section.group(1).strip().split("\n")
            for line in dataset_lines:
                line = line.strip()
                if line and not re.match(r"^[-*]\s+[\w\-\.]+\.(csv|xlsx?|json|parquet)", line):
                    if line and not line.startswith("#"):
                        errors.append(f"数据集行格式不正确: '{line[:60]}'")

        # 7. Steps should be numbered
        steps_section = re.search(
            r"## 步骤\n+(.*?)(?=\n## |\Z)", md_content, re.DOTALL
        )
        if steps_section:
            steps_body = steps_section.group(1).strip()
            numbered_steps = re.findall(r"^\d+\.", steps_body, re.MULTILINE)
            if not numbered_steps:
                errors.append("'## 步骤' 中的步骤应使用编号列表 (1. 2. 3. ...)")

        return errors

    def save(self, md_content: str, output_dir: Path, task_name: str) -> Path:
        """
        Save the generated .md to the tasks directory.

        Args:
            md_content: the markdown content
            output_dir: directory to save to
            task_name: used as filename stem (sanitized)

        Returns:
            Path to the saved file.
        """
        # Sanitize filename
        safe_name = re.sub(r"[^\w\-一-鿿]", "_", task_name).strip("_")
        output_path = Path(output_dir) / f"{safe_name}.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md_content, encoding="utf-8")
        logger.info(f"Task .md saved to: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_generation_prompt(self, params: dict) -> str:
        """Build the main generation prompt."""
        domain = params.get("domain", "custom")
        task_name = params.get("task_name", "未命名任务")
        datasets = params.get("datasets", [])
        requirements = params.get("requirements", "")

        # Format datasets
        dataset_lines = []
        for fname, desc in datasets:
            if desc:
                dataset_lines.append(f"- {fname}: {desc}")
            else:
                dataset_lines.append(f"- {fname}")
        dataset_str = "\n".join(dataset_lines) if dataset_lines else "- (请根据领域自行拟定合适的数据集)"

        # Load reference examples
        examples = self._load_reference_examples(domain)

        prompt = f"""\
请生成一份 Agent A 评测任务定义文件 (.md)，具体参数如下：

## 任务信息
- 任务领域: {domain}
- 任务名称: {task_name}
- 可用数据集:
{dataset_str}
- 特殊要求: {requirements if requirements else "无"}

---

## ⚠️ 十分重要 — 请先阅读以下两条原则

### 第一条：严格区分失败归因
核心指标只衡量"Agent A 是否按要求完成了该做的事"，不衡量"最终结果有多好"。

例如：
- 模型训练的 RMSE 很高，如果是因为人类指定的数据集噪声大或模型简单，
  而 Agent A 正确完成了训练→评估→保存→注册全流程，则核心指标判定为**通过**。
- 反之，如果 Agent A 遗漏了数据预处理步骤、未做训练/测试划分、或错误使用评估指标，
  即使最终 RMSE 数值"看起来还行"，核心指标也判定为**失败**。

### 第二条：自主分析任务类型
你需要**自主分析任务领域特点**，判断该任务的核心成功条件和最重要的专项评估维度。
在模板基础上，可以添加额外的任务专项指标，但只添加**重要且必要的**，
严禁盲目堆砌与任务无关的指标。每个添加的指标都必须有明确的理由。

参考示例：
- 数据清洗核心指标 → 缺失值是否已按指定方法填充、重复行是否已删除、清洗后有无异常零值
- 模型训练核心指标 → 是否完成训练全流程、模型是否正确保存注册、评估指标是否按要求输出
- 数据分析核心指标 → 报告是否覆盖所有要求维度、结论是否有数据支撑

---

## 输出格式要求

⚠️ 重要说明：评估框架中有两类指标是**所有任务共用的框架级指标**，已在评估 Agent 的系统 Prompt 中统一定义，**不需要**在 .md 中生成：
- **过程与效率指标**（工具调用准确率、轨迹步骤效率、冗余/无效操作率）
- **资源与鲁棒性指标**（Token 消耗成本、任务执行时延、自我纠错与异常恢复率）

你只需要生成**任务特化**的内容。

生成的 .md 文件必须严格包含以下章节：

```
# {task_name}

## 任务描述（可选）
简要说明这个任务评估 Agent A 的什么能力，1-2段。

## 可用数据集
- filename1.csv: 数据描述
- filename2.csv: 数据描述

## 步骤
1. 具体可操作步骤（如"点击上传按钮，选择文件上传"）
2. ...

## 评估标准

### 核心指标: 任务成功率（一票通过/否决）
判定任务是否成功完成的硬性条件。必须包含：
- 明确的通过条件（Agent A 必须做到什么）
- 明确的失败条件（什么样算失败）
- 归因说明（哪些失败是 Agent A 的问题，哪些不是）

### 任务专项指标（灵活，根据任务类型自主添加）
评估 Agent A 在该任务领域的专业表现质量。关注"做得好不好"。
只添加重要且必要的指标，每个指标需有评分标准。
```

## 格式规则
- 数据集名称使用真实文件名（含扩展名，如 .csv）
- 步骤描述具体可操作（"点击上传按钮，选择 sales_2024.csv 上传"而非"上传数据"）
- 评估标准的每个指标都要写出**具体的判定标准**，而非仅列指标名称
- 严禁用 ``` 代码块包裹整个输出，直接输出 markdown
- **禁止**在评估标准中生成"过程与效率指标"或"资源与鲁棒性指标"章节（这两类是框架级指标）

## 参考示例
以下是一个完整的数据清洗任务的 .md 文件，供参考格式：
{examples}

请直接输出生成的 .md 内容（只输出 markdown，不要添加任何解释性文字）。
"""
        return prompt

    def _build_fix_prompt(
        self, params: dict, previous_output: str, errors: list[str]
    ) -> str:
        """Build a prompt requesting the LLM to fix validation errors."""
        error_list = "\n".join(f"- {e}" for e in errors)
        return f"""\
你之前生成的 .md 文件有以下格式问题，请修复后重新输出完整的 .md：

## 验证失败的问题
{error_list}

## 之前生成的 .md
{previous_output}

请根据上述问题修复 .md 文件，补充缺失的章节或修正格式错误。
只输出修复后的完整 markdown，不要加任何解释。"""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_reference_examples(self, domain: str) -> str:
        """Load 1-2 existing .md files as reference examples.

        Picks examples relevant to the requested domain when possible.
        """
        domain_map = {
            "data_cleaning": ["data_cleaning.md"],
            "data_analysis": ["data_analysis.md"],
            "model_training": ["model_training.md"],
            "custom": ["data_analysis.md", "model_training.md"],
        }
        example_files = domain_map.get(domain, ["data_analysis.md"])
        examples = []
        for fname in example_files:
            path = self._TASKS_DIR / fname
            if path.exists():
                content = path.read_text(encoding="utf-8")
                examples.append(f"### 示例: {fname}\n{content}")
        if not examples:
            # Load any available .md
            for path in sorted(self._TASKS_DIR.glob("*.md"))[:2]:
                content = path.read_text(encoding="utf-8")
                examples.append(f"### 示例: {path.name}\n{content}")
        return "\n\n---\n\n".join(examples) if examples else "(无可用示例)"

    @staticmethod
    def _unwrap_code_fence(raw: str) -> str:
        """Remove markdown code fences if the LLM wrapped the output."""
        raw = raw.strip()
        # Remove leading ```markdown / ```md / ```
        if raw.startswith("```"):
            first_newline = raw.find("\n")
            if first_newline != -1:
                raw = raw[first_newline + 1:]
        # Remove trailing ```
        if raw.endswith("```"):
            raw = raw[:-3].rstrip()
        return raw.strip()


# =============================================================================
# CLI
# =============================================================================


def load_config(config_path: str) -> dict:
    """Load LLM config from config.yaml."""
    config_path = Path(config_path)

    # 加载 .env 文件（与 run_eval.py / run_agentlab.py 保持一致）
    # 确保 .env 中的 OPENAI_API_KEY / ANTHROPIC_API_KEY 等环境变量已注入
    try:
        from run_eval import load_dotenv
    except ImportError:
        # 如果导入失败，尝试在当前目录找到 run_eval.py
        _scripts_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(_scripts_dir))
        from run_eval import load_dotenv

    dotenv_path = config_path.parent / ".env"
    load_dotenv(dotenv_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Prefer task_generator section, fall back to evaluator section
    llm_section = config.get("task_generator") or config.get("evaluator")
    if not llm_section:
        raise ValueError(
            "Config file must have either 'task_generator' or 'evaluator' section"
        )

    provider = llm_section.get("provider", "openai")

    # Resolve api_key from env if null
    api_key = llm_section.get("api_key")
    if api_key is None:
        import os
        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
        elif provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Resolve base_url from env if null
    base_url = llm_section.get("base_url")
    if base_url is None:
        import os
        if provider == "openai":
            base_url = os.environ.get("OPENAI_BASE_URL")
        elif provider == "anthropic":
            base_url = os.environ.get("ANTHROPIC_BASE_URL")

    return {
        "provider": provider,
        "model": llm_section.get("model", "gpt-4o"),
        "api_key": api_key,
        "base_url": base_url,
        "temperature": llm_section.get("temperature", 0.3),
        "max_tokens": llm_section.get("max_tokens", 4096),
    }


def parse_datasets_arg(datasets_str: str) -> list[tuple[str, str]]:
    """
    Parse --datasets argument.

    Format: "file1.csv:desc,file2.csv:desc" or "file1.csv,file2.csv"
    Returns list of (filename, description) tuples.
    """
    if not datasets_str:
        return []
    result = []
    for part in datasets_str.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            fname, desc = part.split(":", 1)
            result.append((fname.strip(), desc.strip()))
        else:
            result.append((part, ""))
    return result


def interactive_mode() -> dict:
    """Run interactive prompts to collect task parameters."""
    print("\n" + "=" * 50)
    print("  Agent A 评测任务生成器 — 交互模式")
    print("=" * 50 + "\n")

    # Domain
    print("任务领域:")
    print("  1. data_analysis  (数据分析)")
    print("  2. data_cleaning  (数据清洗)")
    print("  3. model_training (模型训练)")
    print("  4. custom         (自定义)")
    domain_choice = input("请选择 [1-4]: ").strip()
    domain_map = {
        "1": "data_analysis",
        "2": "data_cleaning",
        "3": "model_training",
        "4": "custom",
    }
    domain = domain_map.get(domain_choice, "custom")
    print(f"  → 已选择: {domain}\n")

    # Task name
    task_name = input("任务名称（中文，将用作文件名）: ").strip()
    if not task_name:
        task_name = "未命名任务"

    # Datasets
    print("\n数据集（逗号分隔，格式: file.csv:描述, 直接回车跳过）")
    datasets_str = input("> ").strip()
    datasets = parse_datasets_arg(datasets_str)
    if not datasets:
        print("  → 未指定数据集，LLM 将自行拟定")

    # Requirements
    print("\n特殊要求或备注（直接回车跳过）:")
    requirements = input("> ").strip()

    # Output directory
    output_dir = input("\n输出目录 [默认: tasks]: ").strip()
    if not output_dir:
        output_dir = "tasks"

    return {
        "domain": domain,
        "task_name": task_name,
        "datasets": datasets,
        "requirements": requirements,
        "output_dir": output_dir,
    }


def main():
    parser = argparse.ArgumentParser(
        description="LLM-powered task .md generator for Agent A evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/task_generator.py -d model_training -n "时间序列预测" \\
      --datasets "weather.csv:天气数据,stock.csv:股价数据"
  python scripts/task_generator.py --interactive
  python scripts/task_generator.py --config my_config.yaml -d custom -n "my_task"
        """,
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to config.yaml (default: ./config.yaml)",
    )
    parser.add_argument(
        "--domain", "-d",
        choices=["data_analysis", "data_cleaning", "model_training", "custom"],
        help="Task domain",
    )
    parser.add_argument(
        "--task-name", "-n",
        help="Task display name (used as filename stem)",
    )
    parser.add_argument(
        "--datasets",
        default="",
        help='Comma-separated dataset files. Format: "file.csv:desc,file2.csv:desc"',
    )
    parser.add_argument(
        "--requirements", "-r",
        default="",
        help="Additional requirements or special instructions (free text)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="tasks",
        help="Output directory (default: tasks/)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode: step-by-step prompts",
    )
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="Disable automatic retry on validation failure",
    )
    args = parser.parse_args()

    # Load config
    try:
        llm_config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Collect parameters
    if args.interactive:
        params = interactive_mode()
        # Resolve config path for interactive mode
        config_path = args.config
    else:
        if not args.domain:
            parser.error("--domain/-d is required (or use --interactive)")
        if not args.task_name:
            parser.error("--task-name/-n is required (or use --interactive)")
        datasets = parse_datasets_arg(args.datasets)
        params = {
            "domain": args.domain,
            "task_name": args.task_name,
            "datasets": datasets,
            "requirements": args.requirements,
            "output_dir": args.output_dir,
        }

    output_dir = Path(params.pop("output_dir", "tasks"))

    # Generate
    generator = TaskGenerator(llm_config)

    try:
        if args.no_retry:
            md_content = generator.generate(params)
            errors = generator.validate(md_content)
            if errors:
                logger.warning(f"Validation found {len(errors)} issue(s):")
                for e in errors:
                    logger.warning(f"  - {e}")
        else:
            md_content = generator.generate_with_retry(params)

        # Preview
        print("\n" + "=" * 60)
        print("  生成的 .md 内容预览")
        print("=" * 60)
        print(md_content[:2000])
        if len(md_content) > 2000:
            print(f"\n... (共 {len(md_content)} 字符，已截断预览)")

        # Confirm
        if args.interactive:
            confirm = input("\n是否保存？[Y/n]: ").strip().lower()
            if confirm and confirm != "y":
                print("已取消保存。")
                return
        else:
            print()  # spacing

        # Save
        saved_path = generator.save(md_content, output_dir, params["task_name"])
        print(f"\n✅ 任务文件已保存至: {saved_path}")

        # Quick parse test
        print("🔍 验证可解析性...")
        try:
            from browsergym.agent_a_eval.utils import WorkflowConfig

            config = WorkflowConfig.from_markdown(saved_path)
            print(f"   ✅ 解析成功: title={config.title}, "
                  f"datasets={config.available_datasets}, "
                  f"benchmark={config.is_benchmark_format}")
        except ImportError:
            # Fallback: package not installed, add src/ to path
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
            try:
                from browsergym.agent_a_eval.utils import WorkflowConfig

                config = WorkflowConfig.from_markdown(saved_path)
                print(f"   ✅ 解析成功: title={config.title}, "
                      f"datasets={config.available_datasets}, "
                      f"benchmark={config.is_benchmark_format}")
            except Exception as e:
                print(f"   ⚠️ 无法验证解析（可能需要 pip install -e .）: {e}")
        except Exception as e:
            print(f"   ⚠️ 无法验证解析: {e}")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
