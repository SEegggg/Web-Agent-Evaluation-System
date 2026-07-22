"""
Utility functions and classes for the Agent A evaluation benchmark.

Includes:
- LoopDetector: multi-signal dead-loop detection
- Markdown parsing: extract sections from task .md files
- Screenshot collection helpers
"""

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# =============================================================================
# Markdown Section Parser
# =============================================================================


@dataclass
class WorkflowConfig:
    """Parsed representation of a task .md file.

    Only contains content from the .md file that is natural language.
    Technical parameters (URL, timeouts, hard checks) come from config.yaml and task code.
    """

    title: str = ""
    max_execution_time_minutes: int = 15  # default, can be overridden per task
    available_datasets: list[str] = field(default_factory=list)
    steps: str = ""  # natural language task steps
    evaluation_criteria: str = ""  # raw evaluation criteria string (backward compat)

    # ── Benchmark-format fields ──
    task_description: str = ""  # parsed from ## 任务描述
    core_criteria: str = ""  # parsed from ### 核心指标（task-specific hard constraints）
    task_specific_metrics: str = ""  # parsed from ### 任务专项指标（domain-specific quality metrics）
    is_benchmark_format: bool = False  # True when ### 核心指标 sub-section detected
    # NOTE: process_efficiency and resource_robustness dimensions are framework-level,
    # defined in evaluator_agent_prompt.md — NOT stored in per-task config.

    @classmethod
    def from_markdown(cls, path: str | Path) -> "WorkflowConfig":
        """
        Parse a task .md file into a WorkflowConfig.

        Expected sections (delimited by ## headers):
        - # Title (top-level)
        - ## 可用数据集
        - ## 步骤
        - ## 评估标准
        """
        content = Path(path).read_text(encoding="utf-8")
        config = cls()

        # Extract top-level title (line starting with "# ")
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if title_match:
            config.title = title_match.group(1).strip()

        # Extract each ## section by splitting on "## "
        sections = cls._split_sections(content)

        for header, body in sections.items():
            header_lower = header.lower().strip()

            if "可用数据集" in header_lower:
                config._parse_datasets(body)
            elif "任务描述" in header_lower:
                config.task_description = body.strip()
            elif "步骤" in header_lower:
                config.steps = body.strip()
            elif "评估标准" in header_lower:
                config.evaluation_criteria = body.strip()

                # Check for benchmark-format sub-sections (### headers)
                sub_sections = cls._split_h3_sections(body)
                for sub_header, sub_body in sub_sections.items():
                    header_lower_h3 = sub_header.lower().strip()
                    if "核心指标" in header_lower_h3 or "成功率" in header_lower_h3:
                        config.core_criteria = sub_body.strip()
                        config.is_benchmark_format = True
                    elif "专项" in header_lower_h3 or "任务特定" in header_lower_h3:
                        config.task_specific_metrics = sub_body.strip()
                # process_efficiency and resource_robustness are framework-level
                # and defined in evaluator_agent_prompt.md — not parsed here

        return config

    @staticmethod
    def _split_h3_sections(body: str) -> dict[str, str]:
        """Split a section body into sub-sections by ### headers.

        Mirrors _split_sections() but operates on ### (H3) level.
        Used to parse benchmark-format evaluation criteria blocks.
        """
        sub_sections = {}
        # Split on "### " at start of line
        parts = re.split(r"\n(?=###\s)", body)

        for part in parts:
            part = part.strip()
            if not part:
                continue
            header_match = re.match(r"^###\s+(.+)$", part, re.MULTILINE)
            if header_match:
                header = header_match.group(1).strip()
                body_lines = part.split("\n", 1)
                sub_body = body_lines[1].strip() if len(body_lines) > 1 else ""
                sub_sections[header] = sub_body

        return sub_sections

    @staticmethod
    def _split_sections(content: str) -> dict[str, str]:
        """Split markdown content into a dict of {header: body} for ## sections."""
        # Remove the top-level # title line for cleaner parsing
        content = re.sub(r"^#\s+.+$", "", content, count=1, flags=re.MULTILINE)

        sections = {}
        # Split on "## " at start of line
        parts = re.split(r"\n(?=##\s)", content)

        for part in parts:
            part = part.strip()
            if not part:
                continue
            header_match = re.match(r"^##\s+(.+)$", part, re.MULTILINE)
            if header_match:
                header = header_match.group(1).strip()
                # Body is everything after the header line
                body_lines = part.split("\n", 1)
                body = body_lines[1].strip() if len(body_lines) > 1 else ""
                sections[header] = body

        return sections

    def _parse_datasets(self, body: str):
        """Parse the 可用数据集 section into a list of filenames."""
        # Match lines like: "- sales_2024.csv: description"
        # or "- sales_2024.csv"
        dataset_pattern = re.findall(
            r"^[-*]\s+(?:`?)([\w\-\.]+\.(?:csv|xlsx?|json|parquet))", body, re.MULTILINE
        )
        self.available_datasets = dataset_pattern


# =============================================================================
# Loop Detector — Multi-Signal Dead-Loop Detection
# =============================================================================


@dataclass
class LoopDetector:
    """
    Multi-signal dead-loop detector.

    Three independent signals, each can trigger an alarm individually.
    Two or more simultaneous alarms → confirmed dead loop → terminate.

    Signal 1: Action repetition — same action sequence repeats
    Signal 2: DOM stagnation — page structure hasn't changed
    Signal 3: State reversion — page returns to previously seen states
    """

    # Signal 1: Action repetition
    action_window: int = 5
    action_repeat_threshold: int = 4  # 需要更多次重复才算（避免误判 waiting）
    action_history: list = field(default_factory=list)
    # 这些是被动/通信类动作，不参与重复检测
    _passive_actions = {"noop", "send_msg_to_user", "report_infeasible"}

    # Signal 2: DOM stagnation
    dom_hash_history: list = field(default_factory=list)
    dom_window: int = 12  # 更大窗口，容忍长时间等待
    dom_similarity_threshold: float = 0.85  # 85% 相同才算停滞

    # Signal 3: State reversion
    state_snapshot_history: dict = field(default_factory=dict)  # {hash: [step_indices]}
    state_reversion_threshold: int = 4  # 需要更多回流次数
    step_counter: int = 0

    def update(self, action: str, dom_object: Optional[dict], url: str):
        """
        Call once per step to feed new observations into the detector.

        Args:
            action: the action string the driver agent took
            dom_object: the BrowserGym dom_object observation (or None if unavailable)
            url: current page URL
        """
        self.step_counter += 1

        # Signal 1: record action (skip passive/wait actions)
        action_key = action.strip().split("(")[0] if action else ""
        if action_key not in self._passive_actions:
            self.action_history.append(action)
        # Keep last 2*window actions for sliding window comparison
        if len(self.action_history) > self.action_window * 3:
            self.action_history = self.action_history[-self.action_window * 3 :]

        # Signal 2 & 3: record DOM fingerprint
        dom_hash = self._compute_dom_fingerprint(dom_object, url)
        self.dom_hash_history.append(dom_hash)
        if len(self.dom_hash_history) > self.dom_window:
            self.dom_hash_history = self.dom_hash_history[-self.dom_window :]

        # Signal 3: track state reversion
        if dom_hash not in self.state_snapshot_history:
            self.state_snapshot_history[dom_hash] = []
        self.state_snapshot_history[dom_hash].append(self.step_counter)

    def is_loop(self) -> tuple[bool, str]:
        """
        Check if the agent appears to be in a dead loop.

        Returns:
            (is_loop: bool, reason: str)
            - is_loop=True: confirmed dead loop, should terminate
            - is_loop=False, reason non-empty: warning, log but continue
            - is_loop=False, reason empty: all clear
        """
        alarms = []

        if self._detect_action_loop():
            alarms.append("action_loop")
        if self._detect_dom_stagnation():
            alarms.append("dom_stagnation")
        if self._detect_state_reversion():
            alarms.append("state_reversion")

        if len(alarms) >= 2:
            return True, f"dead_loop_confirmed: {', '.join(alarms)}"
        elif len(alarms) == 1:
            return False, f"loop_warning: {alarms[0]}"
        else:
            return False, ""

    def _detect_action_loop(self) -> bool:
        """Check if recent actions form a repeating pattern."""
        if len(self.action_history) < self.action_window * 2:
            return False

        recent = tuple(self.action_history[-self.action_window :])
        repeat_count = 0
        for i in range(len(self.action_history) - self.action_window):
            window = tuple(self.action_history[i : i + self.action_window])
            if window == recent:
                repeat_count += 1
        return repeat_count >= self.action_repeat_threshold

    def _detect_dom_stagnation(self) -> bool:
        """Check if DOM structure is unchanged over recent steps."""
        if len(self.dom_hash_history) < self.dom_window:
            return False

        recent = self.dom_hash_history[-self.dom_window :]
        # Find the most common hash
        hash_counts = {}
        for h in recent:
            hash_counts[h] = hash_counts.get(h, 0) + 1
        most_common_count = max(hash_counts.values())
        return most_common_count / len(recent) >= self.dom_similarity_threshold

    def _detect_state_reversion(self) -> bool:
        """Check if the page keeps returning to the same state."""
        for dom_hash, indices in list(self.state_snapshot_history.items()):
            if len(indices) >= self.state_reversion_threshold:
                # Check if revisits are roughly evenly spaced
                # (even spacing suggests a loop; uneven spacing may be normal navigation)
                if len(indices) >= 3:
                    intervals = [
                        indices[i + 1] - indices[i] for i in range(len(indices) - 1)
                    ]
                    if max(intervals) > 0 and min(intervals) / max(intervals) < 0.3:
                        # Highly uneven intervals — might be normal operation
                        continue
                return True
        return False

    @staticmethod
    def _compute_dom_fingerprint(
        dom_object: Optional[dict], url: str
    ) -> str:
        """
        Compute a lightweight structural fingerprint of the DOM.

        Focuses on structural features (node count, text content hash, URL path)
        and ignores dynamic content like timestamps.
        """
        try:
            # URL path (ignore query strings which may contain nonces/timestamps)
            url_path = urlparse(url).path

            if dom_object is None or not isinstance(dom_object, dict):
                return f"{url_path}|no_dom"

            documents = dom_object.get("documents", [])
            if not documents:
                return f"{url_path}|no_docs"

            doc = documents[0] if isinstance(documents, list) else documents
            nodes = doc.get("nodes", {})
            strings = dom_object.get("strings", [])

            # Structural feature 1: node count
            parent_indices = nodes.get("parentIndex", [])
            node_count = len(parent_indices) if isinstance(parent_indices, list) else 0

            # Structural feature 2: visible text hash (first 500 chars)
            node_values = nodes.get("nodeValue", [])
            text_parts = []
            for idx in (node_values if isinstance(node_values, list) else [])[:50]:
                if isinstance(idx, int) and idx < len(strings):
                    text_parts.append(str(strings[idx]))
            text_sample = "".join(text_parts)[:500]
            text_hash = hashlib.md5(text_sample.encode("utf-8", errors="replace")).hexdigest()[:8]

            # Structural feature 3: node type counts
            node_types = nodes.get("nodeType", [])
            type_counts = {}
            for t in (node_types if isinstance(node_types, list) else []):
                type_counts[t] = type_counts.get(t, 0) + 1
            type_sig = ",".join(f"{k}:{v}" for k, v in sorted(type_counts.items())[:10])

            fingerprint = f"{url_path}|nodes={node_count}|text={text_hash}|types={type_sig}"
            return fingerprint

        except Exception as e:
            logger.debug(f"DOM fingerprint computation failed: {e}")
            return f"fallback:{hash(str(dom_object)[:200])}"


# =============================================================================
# Screenshot Collection
# =============================================================================


def collect_screenshots(
    page, step_indices: list[int], screenshot_dir: Path
) -> list[dict]:
    """
    Collect screenshots at specific step indices for evaluation.

    Args:
        page: Playwright page object
        step_indices: which steps to capture
        screenshot_dir: directory to save screenshots

    Returns:
        list of dicts with {step, path, base64_data}
    """
    import base64
    import io

    from PIL import Image

    screenshots = []
    screenshot_dir = Path(screenshot_dir)
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    for step_idx in step_indices:
        try:
            path = screenshot_dir / f"eval_screenshot_step_{step_idx}.png"
            page.screenshot(path=str(path))
            with open(path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode()
            screenshots.append(
                {
                    "step": step_idx,
                    "path": str(path),
                    "base64": b64_data,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to capture screenshot at step {step_idx}: {e}")

    return screenshots


# =============================================================================
# DOM-based Page State Checks
# =============================================================================


class PageStateChecker:
    """
    Executes hard page state checks against the current page.

    Checks are configured from the task .md file's "页面硬校验规则" section.

    Rule format (natural language, each numbered line is one rule group):
      1. 存在 .selector-a 或 .selector-b 元素
         → CSS OR group: any ONE of these selectors must exist
      2. 不存在 .loading 或 .spinner 元素
         → CSS AND group: ALL of these selectors must NOT exist
      3. 文本包含 "关键词A" 或 "关键词B"
         → Text OR group: any ONE of these texts must appear in page body
      4. 文本包含 "关键词A"
         → Text single: this text must appear in page body

    "或" = OR logic (any one passes → group passes)
    Multiple items without "或" = items that all get checked individually

    Each numbered rule is a separate group. ALL groups must pass.
    Within each group:
      - "或"-separated items: any ONE passing → group passes
      - Comma-separated items: EACH must pass
    """

    def __init__(self, hard_check_rules: str):
        self.rules_text = hard_check_rules
        self.rule_groups = self._parse_rules()

    def _parse_rules(self) -> list[dict]:
        """
        Parse hard check rules into structured rule groups.

        Each group is a dict:
          {
            "type": "css_exist" | "css_not_exist" | "text_contain",
            "items": [["item1", "item2"], ...],  # inner lists = OR groups
            "raw_line": str,
          }

        The outer list = AND (all groups must pass).
        Inner lists within items = OR (any one passing → sub-check passes).
        """
        groups = []

        for line in self.rules_text.split("\n"):
            line = line.strip()
            if not line or not re.search(r'\d', line[:3] if len(line) >= 3 else line):
                # Skip lines that don't look like rules (no leading number)
                if line and not line.startswith(("#", "-", "*", ">")):
                    continue

            # Extract CSS selectors (.class or #id)
            css_matches = re.findall(r"[\.\#][\w\-\.]+", line)

            # Extract quoted text: "xxx" or 「xxx」
            text_matches = re.findall(r"[\"「]([^\"」]+)[\"」]", line)

            # Check for negation
            is_negation = bool(re.search(
                r"不存在|不应存在|不能有|禁止出现|不应出现|没有",
                line,
            ))

            # Check if line uses "或" (OR logic within group)
            has_or = "或" in line

            if text_matches:
                # Text-based check
                if has_or:
                    # OR group: any one text must exist
                    groups.append({
                        "type": "text_contain_any",
                        "items": text_matches,  # flat list, any one must match
                        "raw_line": line,
                    })
                else:
                    # Each text individually required
                    for t in text_matches:
                        groups.append({
                            "type": "text_contain",
                            "items": [t],
                            "raw_line": line,
                        })

            elif css_matches:
                if is_negation:
                    # CSS must-not-exist: each individually
                    for css in css_matches:
                        groups.append({
                            "type": "css_not_exist",
                            "items": [css],
                            "raw_line": line,
                        })
                elif has_or:
                    # CSS OR group: any one must exist
                    groups.append({
                        "type": "css_exist_any",
                        "items": css_matches,
                        "raw_line": line,
                    })
                else:
                    # CSS must-exist: each individually
                    for css in css_matches:
                        groups.append({
                            "type": "css_exist",
                            "items": [css],
                            "raw_line": line,
                        })

        return groups

    def check_all(self, page) -> dict:
        """
        Run all hard checks against the current page.

        Returns:
            dict with individual check results and all_passed flag.
            Keys: "all_passed", "groups_passed", "group_details"
        """
        if not self.rule_groups:
            return {
                "all_passed": True,
                "groups_passed": 0,
                "group_details": [],
                "_note": "No structured rules; skipping hard checks",
            }

        # Pre-cache body text for text checks
        body_text = None
        try:
            body_text = page.locator("body").inner_text()
        except Exception:
            body_text = ""

        group_results = []
        all_passed = True

        for i, group in enumerate(self.rule_groups):
            gtype = group["type"]
            items = group["items"]
            group_label = f"规则{i+1}: {group['raw_line'][:80]}"

            if gtype == "css_exist":
                # EACH CSS selector must exist
                details = {}
                passed = True
                for css in items:
                    try:
                        exists = page.locator(css).count() > 0
                        details[f"css_exists:{css}"] = exists
                        if not exists:
                            passed = False
                    except Exception:
                        details[f"css_exists:{css}"] = False
                        passed = False

            elif gtype == "css_exist_any":
                # ANY ONE CSS selector must exist
                details = {}
                passed = False
                for css in items:
                    try:
                        exists = page.locator(css).count() > 0
                        details[f"css_exists:{css}"] = exists
                        if exists:
                            passed = True
                    except Exception:
                        details[f"css_exists:{css}"] = False

            elif gtype == "css_not_exist":
                # EACH CSS selector must NOT exist
                details = {}
                passed = True
                for css in items:
                    try:
                        exists = page.locator(css).count() > 0
                        details[f"css_not_exists:{css}"] = not exists
                        if exists:
                            passed = False
                    except Exception:
                        details[f"css_not_exists:{css}"] = True  # invalid = doesn't exist

            elif gtype == "text_contain":
                # EACH text must appear
                details = {}
                passed = True
                for t in items:
                    found = t in body_text
                    details[f"text:{t[:40]}"] = found
                    if not found:
                        passed = False

            elif gtype == "text_contain_any":
                # ANY ONE text must appear
                details = {}
                passed = False
                for t in items:
                    found = t in body_text
                    details[f"text:{t[:40]}"] = found
                    if found:
                        passed = True

            else:
                details = {"unknown_type": gtype}
                passed = True

            group_results.append({
                "label": group_label,
                "type": gtype,
                "passed": passed,
                "details": details,
            })

            if not passed:
                all_passed = False

        return {
            "all_passed": all_passed,
            "groups_passed": sum(1 for g in group_results if g["passed"]),
            "total_groups": len(group_results),
            "group_details": group_results,
        }
