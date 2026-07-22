"""
WorkflowTask — core BrowserGym task for Agent A evaluation.

Drives a BrowserGym agent through a multi-step workflow defined in a .md file,
then collects artifacts for independent LLM evaluation.
"""

import logging
import os
import time
from pathlib import Path
from typing import Tuple

import playwright.sync_api

from browsergym.core.task import AbstractBrowserTask

from .utils import LoopDetector, PageStateChecker, WorkflowConfig

logger = logging.getLogger(__name__)


class WorkflowTask(AbstractBrowserTask):
    """
    A BrowserGym task that executes a multi-step evaluation workflow against Agent A.

    Lifecycle:
    1. setup(): navigate to Agent A, present the task steps as the goal
    2. step() loop: driver agent operates the browser
       - Each validate() call runs loop detection
       - When agent sends WORKFLOW_DONE, page hard checks are executed
       - If hard checks pass, artifacts are collected and episode terminates
    3. After BrowserGym session ends, EvaluatorAgent (external) scores the artifacts

    Task info dict (accessible via StepInfo.task_info):
        - status: "completed" | "dead_loop" | "timeout" | "infeasible" | "max_steps"
        - artifacts: dict with screenshots, chat_history, hard_check_results, etc.
        - loop_reason: str (if dead loop detected)
        - error: str (if any)
    """

    @classmethod
    def get_task_id(cls):
        raise NotImplementedError(
            "WorkflowTask is registered with specific task IDs via register_task()"
        )

    # Per-task hard check rules (text-based, defined in code, not in .md)
    _HARD_CHECK_RULES = {
        "data_analysis": (
            "1. 文本包含 \"分析报告\" 或 \"核心发现\" 或 \"相关性\" 或 \"analysis_report\"\n"
            "2. 文本包含 \"✅ 分析完成\" 或 \"全部完成\" 或 \"交付产物\"\n"
            "3. 不存在 .loading 或 .spinner 元素"
        ),
        "data_cleaning": (
            "1. 文本包含 \"cleaned\" 或 \"清洗\" 或 \"处理后\" 或 \"缺失值\" 或 \"重复\"\n"
            "2. 文本包含 \"✅\" 或 \"完成\" 或 \"成功\"\n"
            "3. 不存在 .loading 或 .spinner 元素"
        ),
        "model_training": (
            "1. 文本包含 \"准确率\" 或 \"精确率\" 或 \"RMSE\" 或 \"R²\" 或 \"评估\" 或 \"模型\"\n"
            "2. 文本包含 \"✅\" 或 \"完成\" 或 \"训练完成\" 或 \"注册\"\n"
            "3. 不存在 .loading 或 .spinner 元素"
        ),
    }

    def _get_hard_check_rules(self) -> str:
        """Get hard check rules for this task (from code, not .md)."""
        return self._HARD_CHECK_RULES.get(self.task_name, "")

    # Path to the driver agent prompt file (relative to agent_a_eval package)
    _DRIVER_PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "driver_agent_prompt.md"

    @classmethod
    def _load_driver_prompt(cls) -> str:
        """Load the driver agent technical operations manual."""
        try:
            path = cls._DRIVER_PROMPT_PATH
            if path.exists():
                return path.read_text(encoding="utf-8").strip()
        except Exception:
            pass
        return ""

    def __init__(
        self,
        seed: int,
        task_md_path: str,
        dataset_index: int = 0,
    ) -> None:
        """
        Args:
            seed: random seed for reproducibility
            task_md_path: path to the .md task definition file
            dataset_index: which dataset to use from available_datasets (0-based)
        """
        super().__init__(seed)

        # Browser configuration
        # Allow viewport override via env vars (e.g. screen too small)
        import os
        vp_w = int(os.environ.get("AGENT_A_EVAL_VIEWPORT_WIDTH", "1280"))
        vp_h = int(os.environ.get("AGENT_A_EVAL_VIEWPORT_HEIGHT", "720"))
        self.viewport = {"width": vp_w, "height": vp_h}
        self.slow_mo = 500  # ms — slightly slower for visibility
        self.timeout = 10000  # ms per Playwright action

        # Task configuration
        self.config = WorkflowConfig.from_markdown(task_md_path)
        self.task_md_path = Path(task_md_path)
        self.task_name = self.task_md_path.stem
        self.dataset_index = dataset_index

        # Hard check rules — defined per task in code (not in .md)
        self._hard_check_rules = self._get_hard_check_rules()

        # State tracking
        self.loop_detector = LoopDetector()
        self.page_checker = PageStateChecker(self._hard_check_rules)
        self._hard_check_retries = 0
        self._max_hard_check_retries = 3
        self._storage_state_saved = False
        self.start_time: float = None
        self.artifacts: dict = None
        self.episode_screenshots: list = []  # paths to saved screenshots
        self.done_status: str = None

        # Cache for loop detector (dom_object is extracted in _get_obs but not in validate)
        self._last_dom_object: dict = None

        logger.info(
            f"WorkflowTask initialized: {self.config.title} "
            f"(dataset_index={dataset_index}, seed={seed})"
        )

    def setup(
        self, page: playwright.sync_api.Page
    ) -> tuple[str, dict]:
        """
        Navigate to Agent A's web interface and construct the goal.

        The goal contains the full task steps from the .md file, plus the
        specific dataset to use for this run.
        """
        self.start_time = time.time()

        # Navigate to Agent A
        agent_a_url = os.environ.get(
            "AGENT_A_EVAL_LOGIN_URL", "http://180.76.145.245:8080/chat"
        )
        logger.info(f"Navigating to Agent A: {agent_a_url}")
        page.goto(agent_a_url, timeout=30000)

        # Resolve datasets directory
        datasets_dir = os.environ.get(
            "AGENT_A_EVAL_DATASETS_DIR",
            str(Path(__file__).resolve().parents[3] / "test_data" / "datasets"),
        )
        datasets_dir = str(Path(datasets_dir).resolve())

        # Select the dataset for this run
        if self.config.available_datasets:
            dataset_idx = self.dataset_index % len(self.config.available_datasets)
            dataset = self.config.available_datasets[dataset_idx]
            dataset_full_path = str(Path(datasets_dir) / dataset)
        else:
            dataset = "(无预定义数据集, 请使用页面上已有的数据)"
            dataset_full_path = "(无本地文件)"

        # Cache DOM for loop detector
        try:
            from browsergym.core.observation import extract_dom_snapshot

            self._last_dom_object = extract_dom_snapshot(page)
        except Exception:
            self._last_dom_object = None

        # ── Programmatic login (bypass driver agent) ──
        login_username = os.environ.get("AGENT_A_EVAL_LOGIN_USERNAME", "")
        login_password = os.environ.get("AGENT_A_EVAL_LOGIN_PASSWORD", "")
        login_url = os.environ.get("AGENT_A_EVAL_LOGIN_URL", "")

        login_section = ""
        new_session_section = ""
        if login_username and login_password:
            logged_in = self._programmatic_login(page, login_username, login_password)
            if logged_in:
                logger.info("✅ 程序化登录成功, goal 中跳过登录步骤")
                new_session_section = """\
### 新建会话(首先执行)
1. 找到并点击「新建会话」/「新建对话」/「New Chat」/「+」按钮, 创建一个全新的会话
2. 确认新会话已创建(页面显示空白对话区域, 没有历史消息)

"""
            else:
                logger.info("⚠️ 程序化登录失败, 回退到驱动 Agent 登录")
                login_section = self._build_login_section(login_url, login_username, login_password)
        elif login_username or login_password:
            login_section = self._build_login_section(login_url, login_username, login_password)

        # Build dataset file list (with paths — needed for upload_file)
        dataset_list_lines = []
        for d in self.config.available_datasets:
            full_p = str(Path(datasets_dir) / d)
            dataset_list_lines.append(f"- {d}(路径: {full_p})")
        dataset_list = "\n".join(dataset_list_lines) if dataset_list_lines else "(无)"

        # ── Load driver technical operations manual ──
        driver_prompt = self._load_driver_prompt()

        # ── Assemble the goal ──
        goal = f"""\
{driver_prompt}

---

## 评测任务: {self.config.title}

### 本轮使用的数据集
文件名: {dataset}
文件路径: {dataset_full_path}

### 所有可用数据集(含完整文件路径)
{dataset_list}

{login_section}{new_session_section}### 执行步骤
{self.config.steps}

---
请严格按以上步骤顺序执行。所有步骤完成后, 调用 send_msg_to_user("WORKFLOW_DONE") 以触发完成校验。
"""

        task_info = {
            "task_title": self.config.title,
            "dataset": dataset,
            "dataset_index": self.dataset_index,
            "seed": self.random.get_state()[1][0],  # capture seed for reproducibility
        }

        logger.info(f"Task setup complete. Goal built with dataset: {dataset}")
        return goal, task_info

    def teardown(self) -> None:
        """Clean up. Most resources are handled by BrowserEnv."""
        self.artifacts = None
        self.episode_screenshots = []
        self._last_dom_object = None
        logger.info(
            f"WorkflowTask teardown: {self.config.title} "
            f"(status={self.done_status})"
        )

    def validate(
        self,
        page: playwright.sync_api.Page,
        chat_messages: list[dict],
    ) -> Tuple[float, bool, str, dict]:
        """
        Called after every step. Implements:

        1. Loop detection (every step)
        2. Timeout check (every step)
        3. WORKFLOW_DONE signal -> hard page check -> artifacts collection
        4. Infeasible report -> terminate with status

        Returns:
            (reward, done, user_message, task_info)
        """
        # ---- Update cached DOM for loop detector ----
        try:
            from browsergym.core.observation import extract_dom_snapshot

            self._last_dom_object = extract_dom_snapshot(page)
        except Exception:
            pass

        # ---- 0. Auto-save cookies after login (first time only) ----
        self._maybe_save_storage_state(page)

        # ---- 1. Extract last action for loop detection ----
        last_action = ""
        if chat_messages:
            # The last "info" message usually contains the action
            for msg in reversed(chat_messages):
                if msg.get("role") == "info" and "action:" in msg.get("message", ""):
                    last_action = msg["message"]
                    break

        # ---- 2. Loop detection ----
        self.loop_detector.update(
            action=last_action,
            dom_object=self._last_dom_object,
            url=page.url,
        )
        is_loop, loop_reason = self.loop_detector.is_loop()

        if is_loop:
            logger.warning(f"Dead loop detected: {loop_reason}")
            self.done_status = "dead_loop"
            self.artifacts = self._collect_artifacts(page, chat_messages)
            return (
                0,
                True,
                "",
                {
                    "status": "dead_loop",
                    "loop_reason": loop_reason,
                    "artifacts": self.artifacts,
                },
            )

        if loop_reason:
            # Single signal warning — log but don't stop
            logger.info(f"Loop warning (not terminal): {loop_reason}")

        # ---- 3. Timeout check ----
        elapsed_minutes = (time.time() - self.start_time) / 60
        if elapsed_minutes > self.config.max_execution_time_minutes:
            logger.warning(
                f"Task timeout: {elapsed_minutes:.1f} min > "
                f"{self.config.max_execution_time_minutes} min"
            )
            self.done_status = "timeout"
            self.artifacts = self._collect_artifacts(page, chat_messages)
            return (
                0,
                True,
                "",
                {
                    "status": "timeout",
                    "elapsed_minutes": elapsed_minutes,
                    "artifacts": self.artifacts,
                },
            )

        # ---- 4. Check for infeasible report ----
        if chat_messages and chat_messages[-1].get("role") == "infeasible":
            reason = chat_messages[-1].get("message", "No reason provided")
            logger.info(f"Agent reported infeasible: {reason}")
            self.done_status = "infeasible"
            self.artifacts = self._collect_artifacts(page, chat_messages)
            return (
                0,
                True,
                "",
                {
                    "status": "infeasible",
                    "reason": reason,
                    "artifacts": self.artifacts,
                },
            )

        # ---- 5. Check for WORKFLOW_DONE signal ----
        driver_done = (
            chat_messages
            and chat_messages[-1].get("role") == "assistant"
            and "WORKFLOW_DONE" in chat_messages[-1].get("message", "")
        )

        if not driver_done:
            # Still working — continue
            return 0, False, "", {"status": "in_progress"}

        # ---- 6. WORKFLOW_DONE received: execute hard page checks ----
        logger.info("WORKFLOW_DONE received. Running hard page checks...")
        check_results = self.page_checker.check_all(page)

        if not check_results.get("all_passed", False):
            self._hard_check_retries += 1

            # Build human-readable feedback from the new group-based results
            failed_groups = [
                g for g in check_results.get("group_details", [])
                if not g["passed"]
            ]
            failed_lines = []
            for g in failed_groups:
                # Show which items in the group failed
                item_details = ", ".join(
                    f"{k}={v}" for k, v in g.get("details", {}).items()
                )
                failed_lines.append(f"- {g['label']} [{item_details}]")

            if self._hard_check_retries >= self._max_hard_check_retries:
                # 重试次数用尽 — 不再让 driver agent 继续, 直接收工交给 Evaluator
                logger.warning(
                    f"Hard check retries exhausted ({self._hard_check_retries}/{self._max_hard_check_retries}). "
                    f"Terminating and delegating to EvaluatorAgent."
                )
                self.done_status = "completed"
                self.artifacts = self._collect_artifacts(page, chat_messages)
                return (
                    1.0, True, "",
                    {"status": "completed", "hard_check": check_results,
                     "note": "hard_check_retries_exhausted", "artifacts": self.artifacts},
                )

            # Still have retries — tell the agent what's missing and continue
            feedback = (
                f"页面硬校验未通过(第{self._hard_check_retries}/{self._max_hard_check_retries}次, "
                f"通过 {check_results.get('groups_passed', 0)}/{check_results.get('total_groups', 0)} 组)。"
                "以下条件未满足, 请继续操作: \n"
                + "\n".join(failed_lines)
            )
            logger.info(
                f"Hard check failed ({self._hard_check_retries}/{self._max_hard_check_retries}): "
                f"{len(failed_groups)} group(s) failed"
            )
            return 0, False, feedback, {"status": "hard_check_failed", "hard_check": check_results}

        # ---- 7. All checks passed: collect artifacts and terminate ----
        logger.info("All hard checks passed. Collecting artifacts...")
        self.done_status = "completed"
        self.artifacts = self._collect_artifacts(page, chat_messages)

        return (
            1.0,  # Reward for successful completion
            True,  # Episode done
            "",
            {
                "status": "completed",
                "hard_check": check_results,
                "artifacts": self.artifacts,
            },
        )

    # =========================================================================
    # Artifact Collection
    # =========================================================================

    @staticmethod
    def _build_login_section(login_url: str, username: str, password: str) -> str:
        """Build the login instruction section for the goal (fallback)."""
        nav1 = f"导航到 {login_url}" if login_url else "在当前页面上找到登录表单"
        nav2 = f"导航到 Agent A 工作区: {login_url}" if login_url else "在当前页面继续"
        return f"""\
### 登录(必须先完成)
1. 定位登录页面: {nav1}
2. 在用户名/邮箱输入框中输入: {username}
3. 在密码输入框中输入: {password}
4. 点击登录/登入/Sign In 按钮
5. 确认登录成功(页面跳转到工作区, 不再显示登录表单)

### 新建会话(登录后立即执行)
6. 登录成功后, {nav2}
7. 找到并点击「新建会话」/「新建对话」/「New Chat」/「+」按钮, 创建一个全新的会话
8. 确认新会话已创建(页面显示空白对话区域, 没有历史消息)

"""

    def _programmatic_login(
        self, page: playwright.sync_api.Page, username: str, password: str
    ) -> bool:
        """
        Perform login directly via Playwright (no driver agent involvement).

        Strategy (matches how a human's browser works):
        1. Navigate to the page — storage_state may have restored
           localStorage token from a previous session.
        2. WAIT for the SPA to auto-authenticate from that token.
           A human opens Chrome → sees a brief loading state →
           gets auto-logged-in. We must give the SPA the same chance.
        3. Only if the login form is STILL visible after waiting,
           fill in credentials programmatically.
        4. After successful login (auto or manual), save storage_state
           for next time.

        Returns True if login succeeded, False if fallback to agent-driven login is needed.
        """
        try:
            # ── Step 1: Wait for SPA to auto-authenticate from storage_state ──
            # storage_state restores localStorage (including auth tokens).
            # The SPA needs time to: read the token → send validation API call →
            # receive response → hide login form / show main UI.
            # This is exactly what happens when you reopen Chrome manually.
            logger.info("⏳ 等待 SPA 自动认证（从 storage_state 恢复登录状态）...")
            try:
                page.wait_for_selector(
                    'input[type="password"]',
                    state="hidden",
                    timeout=8000,  # 8 seconds max for auto-auth
                )
                # Password field disappeared → SPA auto-authenticated!
                logger.info("✅ storage_state 恢复登录成功！无需重新输入密码。")
                # Refresh storage_state (tokens may have been renewed by the server)
                self._save_storage_state(page)
                return True
            except Exception:
                # Password field still visible after waiting
                # → auto-auth failed (no valid token, or token expired)
                logger.info("⏳ SPA 自动认证超时，需要程序化填写登录凭据")

            # ── Step 2: Verify login form is present ──
            pw_input = page.locator('input[type="password"]')
            if pw_input.count() == 0 or not pw_input.first.is_visible():
                # No password field at all → already logged in (unlikely after timeout)
                logger.info("未检测到登录表单，已处于登录状态")
                self._save_storage_state(page)
                return True

            logger.info("🔑 开始程序化登录...")

            # Find username field (try common selectors)
            username_field = (
                page.locator('input[name="username"]').first
                if page.locator('input[name="username"]').count() > 0
                else page.locator('input[type="text"]').first
            )
            if username_field.count() == 0:
                logger.warning("找不到用户名输入框")
                return False

            # Find login button (try common selectors)
            login_btn = page.locator('button[type="submit"]').first
            if login_btn.count() == 0:
                login_btn = page.locator('button:has-text("登录")').first
            if login_btn.count() == 0:
                login_btn = page.locator('button:has-text("登入")').first
            if login_btn.count() == 0:
                login_btn = page.locator('button:has-text("Sign")').first
            if login_btn.count() == 0:
                logger.warning("找不到登录按钮")
                return False

            # Fill credentials
            username_field.fill(username)
            page.wait_for_timeout(300)
            pw_input.first.fill(password)
            page.wait_for_timeout(300)

            # Click login
            login_btn.click()

            # Wait for navigation and auth token to be saved to localStorage
            page.wait_for_load_state("networkidle", timeout=15000)
            # Extra wait: the SPA may make additional API calls after login
            # to fetch user profile, projects, etc. Wait for those to settle.
            page.wait_for_timeout(3000)

            # ── Check if login succeeded ──
            current_url = page.url
            still_on_login = "/login" in current_url.lower() or (
                page.locator('input[type="password"]').count() > 0
                and page.locator('input[type="password"]').first.is_visible()
            )

            if still_on_login:
                logger.warning(f"登录可能失败: url={current_url}")
                return False

            logger.info(f"✅ 程序化登录完成: url={current_url}, title={page.title()}")

            # ── Save storage_state for next run ──
            # This ensures cookies + localStorage (including the fresh auth token)
            # are persisted immediately, not waiting for the next validate() call.
            self._save_storage_state(page)

            return True

        except Exception as e:
            logger.warning(f"程序化登录异常: {e}")
            return False

    def _save_storage_state(self, page: playwright.sync_api.Page):
        """
        Save browser storage state (cookies + localStorage) to disk immediately.

        Called right after successful login (auto or manual), NOT waiting for
        the next validate() cycle. This ensures the auth token is captured
        while it's fresh.
        """
        if self._storage_state_saved:
            return

        storage_state_path = os.environ.get("AGENT_A_EVAL_STORAGE_STATE_PATH", "")
        if not storage_state_path:
            return

        try:
            pw_inputs = page.locator('input[type="password"]')
            if pw_inputs.count() > 0 and pw_inputs.first.is_visible():
                return  # Still on login page, don't save yet

            # Small extra wait: ensure all async storage writes have completed
            page.wait_for_timeout(500)

            page.context.storage_state(path=storage_state_path)
            self._storage_state_saved = True
            logger.info(f"💾 Cookie/localStorage 已保存到: {storage_state_path}")
        except Exception as e:
            logger.warning(f"保存 storage state 失败: {e}")

    def _maybe_save_storage_state(self, page: playwright.sync_api.Page):
        """
        Periodic saving from validate() — kept as a safety net.

        If _programmatic_login already saved, this is a no-op.
        If there was no programmatic login (e.g. agent-driven login),
        this provides a fallback save path.
        """
        self._save_storage_state(page)

    def _collect_artifacts(
        self,
        page: playwright.sync_api.Page,
        chat_messages: list[dict],
    ) -> dict:
        """
        Collect all artifacts needed for LLM evaluation.

        This runs AFTER the BrowserGym session ends (during the final validate call).
        The EvaluatorAgent will consume these artifacts independently.
        """
        # Capture final screenshot
        import base64
        import io

        try:
            screenshot_bytes = page.screenshot(type="png")
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
        except Exception as e:
            logger.error(f"Failed to capture final screenshot: {e}")
            screenshot_b64 = None

        # Extract Agent A's chat output from the page (the actual responses,
        # not the driver agent's actions)
        agent_a_responses = self._extract_agent_a_responses(page)

        # Filter chat_messages to relevant entries
        filtered_chat = [
            {
                "role": msg.get("role", "unknown"),
                "message": msg.get("message", "")[:1000],  # Truncate long messages
                "timestamp": msg.get("timestamp", 0),
            }
            for msg in chat_messages
            if msg.get("role") in ("user", "assistant", "infeasible", "info")
        ]

        artifacts = {
            "screenshot_base64": screenshot_b64,
            "chat_history": filtered_chat,
            "agent_a_responses": agent_a_responses,
            "page_url": page.url,
            "page_title": page.title() if not page.is_closed() else "",
            "completion_status": self.done_status,
            "elapsed_time_seconds": time.time() - self.start_time if self.start_time else 0,
            "task_title": self.config.title,
            "dataset_index": self.dataset_index,
            "dataset_name": (
                self.config.available_datasets[self.dataset_index]
                if self.config.available_datasets
                and self.dataset_index < len(self.config.available_datasets)
                else "unknown"
            ),
        }

        return artifacts

    def _extract_agent_a_responses(self, page: playwright.sync_api.Page) -> list[dict]:
        """
        Extract Agent A's visible responses from the page DOM.

        Strategy:
        1. Try known selectors first (specific to common chat UI patterns)
        2. Fallback: use JS to find ALL visible elements with substantial text,
           recording their tag/class/id so we can identify the right selectors.
        3. Ultimate fallback: grab the entire body visible text
        """
        responses = []

        # ── Strategy 1: Known selectors ──
        # These cover common Agent/chat UI patterns:
        # - Generic: .agent-response, .chat-message, .output-area
        # - Markdown reports: .markdown-body, .report-content
        # - Code blocks: pre, .code-output
        # - Tailwind/shadcn: prose (typography), article
        selectors_to_try = [
            # Semantic chat classes
            ".agent-response", ".chat-message", ".message",
            ".output-area", ".report-content", ".analysis-result",
            # Markdown / rendered content
            ".markdown-body", ".prose",
            # Raw code/output blocks
            "pre", ".code-output", ".code-block",
            # Fallback: any article-like content
            "article",
            # shadcn/ui card content (common in data science platforms)
            '[class*="prose"]', '[class*="markdown"]', '[class*="report"]',
        ]

        for selector in selectors_to_try:
            try:
                elements = page.locator(selector)
                count = elements.count()
                for i in range(min(count, 10)):
                    try:
                        elem = elements.nth(i)
                        if elem.is_visible():
                            text = elem.inner_text()
                            if text and len(text.strip()) > 10:
                                responses.append({
                                    "selector": selector,
                                    "index": i,
                                    "text": text[:3000],
                                })
                    except Exception:
                        pass
            except Exception:
                pass

        # ── Strategy 2: JS-powered scan for any text-heavy visible elements ──
        # This finds Agent A output regardless of what CSS framework is used.
        try:
            js_results = page.evaluate("""() => {
                const results = [];
                const allElements = document.querySelectorAll('*');
                const seen = new Set();

                for (const el of allElements) {
                    // Skip hidden elements
                    const style = window.getComputedStyle(el);
                    if (style.display === 'none' || style.visibility === 'hidden') continue;

                    // Only leaf or near-leaf elements (text is here)
                    const childCount = el.children.length;
                    const text = (el.innerText || '').trim();

                    // Target: elements with meaningful text, especially longer content
                    if (text.length > 80) {
                        // Prefer elements that are containers (some children, but not the whole body)
                        const isInteresting =
                            text.length > 150 ||                        // substantial text
                            el.tagName === 'PRE' ||                     // code blocks
                            el.tagName === 'CODE' ||                    // inline code
                            /markdown|prose|report|output|result|response|chat|message/i.test(
                                el.className || ''
                            );  // likely a content container

                        if (isInteresting && childCount <= 20 && !seen.has(text.substring(0, 50))) {
                            seen.add(text.substring(0, 50));
                            results.push({
                                tag: el.tagName,
                                id: el.id || '',
                                className: (el.className || '').substring(0, 200),
                                text: text.substring(0, 3000),
                                childCount: childCount,
                            });
                        }
                    }

                    if (results.length >= 20) break;  // Enough samples
                }
                return results;
            }""")

            for item in js_results:
                # Avoid duplicates with strategy 1
                text_start = item["text"][:50] if item["text"] else ""
                is_dup = any(r["text"][:50] == text_start for r in responses)
                if not is_dup and len(item["text"].strip()) > 10:
                    responses.append({
                        "selector": f"js_fallback:{item['tag']}.{item['className'][:50]}",
                        "index": 0,
                        "text": item["text"][:3000],
                        "element_info": f"<{item['tag']}> class='{item['className'][:100]}' id='{item['id']}' children={item['childCount']}",
                    })

        except Exception as e:
            logger.warning(f"JS-based response extraction failed: {e}")

        # ── Strategy 3: Ultimate fallback — body text ──
        if not responses:
            try:
                body_text = page.locator("body").inner_text()
                if body_text and len(body_text.strip()) > 10:
                    responses.append({
                        "selector": "fallback:body",
                        "index": 0,
                        "text": body_text[:5000],
                        "element_info": "FULL BODY TEXT — no structured selectors matched",
                    })
            except Exception:
                pass

        # Deduplicate by text content
        seen_texts = set()
        unique_responses = []
        for r in responses:
            key = r["text"][:100]
            if key not in seen_texts:
                seen_texts.add(key)
                unique_responses.append(r)

        return unique_responses
