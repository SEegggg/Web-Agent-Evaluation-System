"""
Agent A 自动评审系统 — 测试入口程序
====================================

用法:
    python run_eval.py                        # 使用 config.yaml 默认配置运行
    python run_eval.py --config my_conf.yaml  # 使用自定义配置文件
    python run_eval.py --task data_analysis   # 只运行指定任务
    python run_eval.py --skip-stability       # 跳过稳定性测试
    python run_eval.py --no-headless          # 显示浏览器窗口（调试用）

快速上手 —— 编辑下方 ★ 标记区域即可：
    1. 选择要测试的任务
    2. 调整稳定性测试的种子
    3. 指定数据集位置
    4. 选择是否跳过某类测试
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# 把项目根目录加到 sys.path，使 demo_agent 模块可被导入
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

# ============================================================
# ★ EDIT HERE —— 快速配置覆盖（优先级高于 config.yaml）
# ============================================================

# 要测试的任务列表，留空 = 自动扫描 config.yaml 中 paths.tasks_dir 下的全部 .md
# 示例: TASKS = ["data_analysis", "data_cleaning"]
TASKS = ["data_analysis"]

# 稳定性测试的随机种子，None = 使用 config.yaml 中的 stability_seeds
# 示例: SEEDS = [42, 123, 456, 789]
SEEDS = None

# 泛化性测试使用的数据集数量，None = 使用 config.yaml 中的值
# 示例: GENERALIZATION_DATASETS = 5
GENERALIZATION_DATASETS = None

# 数据集文件所在目录的绝对路径，None = 使用 config.yaml 中的值
# 示例: DATASETS_DIR = r"D:\my_work\eval_datasets"
DATASETS_DIR = None

# 跳过某类测试: True = 跳过
SKIP_STABILITY = True
SKIP_GENERALIZATION = True

# 浏览器是否无头模式，None = 使用 config.yaml 中的 browser.headless
# 调试时设为 False 可以看到浏览器操作过程
HEADLESS = None

# 每个实验最大步数，None = 使用 config.yaml 中的值
MAX_STEPS = None

# ============================================================
# 以下为程序逻辑，通常不需要修改
# ============================================================


def parse_args():
    """解析命令行参数。命令行参数优先级高于 config.yaml 和上方快速配置。"""
    p = argparse.ArgumentParser(
        description="Agent A 自动评审系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="配置文件路径 (默认: 与 run_eval.py 同目录的 config.yaml)",
    )
    p.add_argument(
        "--task", type=str, action="append", dest="tasks",
        help="指定要运行的任务（可多次使用），不指定则运行全部",
    )
    p.add_argument(
        "--seed", type=int, action="append", dest="seeds",
        help="稳定性测试的种子值（可多次使用）",
    )
    p.add_argument(
        "--gen-datasets", type=int, default=None,
        help="泛化性测试使用的数据集数量",
    )
    p.add_argument(
        "--datasets-dir", type=str, default=None,
        help="数据集文件所在目录的绝对路径",
    )
    p.add_argument(
        "--skip-stability", action="store_true",
        help="跳过稳定性测试",
    )
    p.add_argument(
        "--skip-generalization", action="store_true",
        help="跳过泛化性测试",
    )
    p.add_argument(
        "--no-headless", action="store_true",
        help="显示浏览器窗口（调试用）",
    )
    p.add_argument(
        "--max-steps", type=int, default=None,
        help="每个实验的最大步数",
    )
    p.add_argument(
        "--exp-root", type=str, default=None,
        help="实验结果输出目录",
    )
    p.add_argument(
        "--reports-dir", type=str, default=None,
        help="评分报告输出目录",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="显示详细日志",
    )
    return p.parse_args()


def _resolve_path(base: Path, value: str) -> Path:
    """如果 value 是相对路径，则相对于 base 所在目录解析。"""
    p = Path(value)
    if p.is_absolute():
        return p
    return (base.parent / p).resolve()


def load_config(config_path: Path) -> dict:
    """加载 YAML 配置并解析相对路径。"""
    if not config_path.exists():
        print(f"[ERROR] 配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 解析相对路径 → 绝对路径
    paths = config.setdefault("paths", {})
    for key in ["tasks_dir", "datasets_dir", "exp_root", "reports_dir"]:
        if key in paths and paths[key]:
            paths[key] = str(_resolve_path(config_path, paths[key]))

    return config


def merge_overrides(config: dict, args) -> dict:
    """将命令行参数和快速配置区域的值合并到 config 中。"""
    cfg = config

    # ---- 快速配置区域 ----
    if TASKS:
        cfg["evaluation"]["tasks"] = TASKS
    if SEEDS is not None:
        cfg["evaluation"]["stability_seeds"] = SEEDS
    if GENERALIZATION_DATASETS is not None:
        cfg["evaluation"]["generalization_datasets"] = GENERALIZATION_DATASETS
    if HEADLESS is not None:
        cfg["browser"]["headless"] = HEADLESS
    if MAX_STEPS is not None:
        cfg["evaluation"]["max_steps"] = MAX_STEPS

    # ---- 命令行参数（优先级最高） ----
    if args.tasks:
        cfg["evaluation"]["tasks"] = args.tasks
    if args.seeds:
        cfg["evaluation"]["stability_seeds"] = args.seeds
    if args.gen_datasets is not None:
        cfg["evaluation"]["generalization_datasets"] = args.gen_datasets
    if args.no_headless:
        cfg["browser"]["headless"] = False
    if args.max_steps is not None:
        cfg["evaluation"]["max_steps"] = args.max_steps
    if args.exp_root:
        cfg["paths"]["exp_root"] = args.exp_root
    if args.reports_dir:
        cfg["paths"]["reports_dir"] = args.reports_dir

    # 跳过标志
    if SKIP_STABILITY or args.skip_stability:
        cfg["evaluation"]["_skip_stability"] = True
    if SKIP_GENERALIZATION or args.skip_generalization:
        cfg["evaluation"]["_skip_generalization"] = True

    # 数据集目录（快速配置）
    if DATASETS_DIR:
        cfg["paths"]["datasets_dir"] = DATASETS_DIR
    if args.datasets_dir:
        cfg["paths"]["datasets_dir"] = args.datasets_dir

    return cfg


def setup_env(cfg: dict, config_path: Path):
    """根据配置设置环境变量，供 __init__.py 和 task.py 读取。"""
    paths = cfg.get("paths", {})

    # 数据集目录
    ds_dir = paths.get("datasets_dir", "")
    if ds_dir:
        os.environ["AGENT_A_EVAL_DATASETS_DIR"] = str(ds_dir)
        print(f"[CONFIG] 数据集目录: {ds_dir}")

    # 任务目录
    tasks_dir = paths.get("tasks_dir", "")
    if tasks_dir:
        os.environ["AGENT_A_EVAL_TASKS_DIR"] = str(tasks_dir)
        print(f"[CONFIG] 任务目录: {tasks_dir}")

    # 登录凭据
    login = cfg.get("login", {})
    if login.get("username") and login.get("password"):
        os.environ["AGENT_A_EVAL_LOGIN_URL"] = login.get("url", "")
        os.environ["AGENT_A_EVAL_LOGIN_USERNAME"] = login["username"]
        os.environ["AGENT_A_EVAL_LOGIN_PASSWORD"] = login["password"]
        print(f"[CONFIG] 登录配置: 用户名={login['username']}")

    # 持久化浏览器 profile 目录（优先使用 user_data_dir）
    user_data_dir = login.get("user_data_dir", "")
    if user_data_dir:
        os.environ["AGENT_A_EVAL_USER_DATA_DIR"] = str(
            _resolve_path(config_path, user_data_dir)
        )
        print(f"[CONFIG] 浏览器 Profile: {os.environ['AGENT_A_EVAL_USER_DATA_DIR']}")

    # Cookie 持久化路径（向后兼容，user_data_dir 优先时不必配置）
    storage_path = login.get("storage_state_path", "")
    if storage_path:
        os.environ["AGENT_A_EVAL_STORAGE_STATE_PATH"] = str(
            _resolve_path(config_path, storage_path)
        )
        print(f"[CONFIG] Cookie 持久化: {os.environ['AGENT_A_EVAL_STORAGE_STATE_PATH']}")


def create_agent_args(cfg: dict):
    """
    根据配置创建 DemoAgentArgs。

    如果你使用的是自定义 Agent，请在此处修改。
    """
    from demo_agent.agent import DemoAgentArgs

    driver_cfg = cfg.get("driver_agent", {})

    # 设置 API key 环境变量
    _setup_api_env(driver_cfg)

    return DemoAgentArgs(
        model_name=driver_cfg.get("model", "gpt-4o"),
        chat_mode=driver_cfg.get("chat_mode", False),
        demo_mode=driver_cfg.get("demo_mode", "off"),
        use_html=driver_cfg.get("use_html", False),
        use_axtree=driver_cfg.get("use_axtree", True),
        use_screenshot=driver_cfg.get("use_screenshot", False),
    )


def _setup_api_env(llm_cfg: dict):
    """设置 LLM API 相关的环境变量。"""
    provider = llm_cfg.get("provider", "openai")
    api_key = llm_cfg.get("api_key")
    base_url = llm_cfg.get("base_url")

    if api_key:
        if provider == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
        elif provider == "anthropic":
            os.environ["ANTHROPIC_API_KEY"] = api_key

    if base_url:
        if provider == "openai":
            os.environ["OPENAI_BASE_URL"] = base_url
        elif provider == "anthropic":
            os.environ["ANTHROPIC_BASE_URL"] = base_url


def print_banner(cfg: dict):
    """打印运行配置摘要。"""
    driver = cfg.get("driver_agent", {})
    evaluator = cfg.get("evaluator", {})
    browser = cfg.get("browser", {})
    eval_cfg = cfg.get("evaluation", {})
    paths = cfg.get("paths", {})

    print("=" * 60)
    print("  Agent A 自动评审系统")
    print("=" * 60)
    print(f"  驱动 Agent: {driver.get('model')} ({driver.get('provider')})")
    print(f"  评审 Agent: {evaluator.get('model')} ({evaluator.get('provider')})")
    print(f"  浏览器模式: {'headless' if browser.get('headless') else 'visible'}")
    print(f"  任务目录:   {paths.get('tasks_dir', 'N/A')}")
    print(f"  数据集目录: {paths.get('datasets_dir', 'N/A')}")
    print(f"  结果目录:   {paths.get('exp_root', 'N/A')}")
    print(f"  报告目录:   {paths.get('reports_dir', 'N/A')}")

    tasks = eval_cfg.get("tasks", [])
    if tasks:
        print(f"  指定任务:   {', '.join(tasks)}")
    else:
        print(f"  任务范围:   全部 .md 文件")

    gen_info = "跳过" if eval_cfg.get("_skip_generalization") else f"{eval_cfg.get('generalization_datasets', 3)} 个数据集"
    stab_info = "跳过" if eval_cfg.get("_skip_stability") else f"种子: {eval_cfg.get('stability_seeds', [42, 123, 456])}"
    print(f"  泛化性测试: {gen_info}")
    print(f"  稳定性测试: {stab_info}")
    print(f"  最大步数:   {eval_cfg.get('max_steps', 200)}")
    print("-" * 60)


def main():
    args = parse_args()

    # 确定配置文件路径
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).parent / "config.yaml"

    # 加载配置
    config = load_config(config_path)

    # 合并覆盖
    config = merge_overrides(config, args)

    # 设置环境变量
    setup_env(config, config_path)

    # 设置日志级别
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # 打印配置摘要
    print_banner(config)

    # ---- 导入并创建 runner ----
    print("[INIT] 正在注册任务...")
    import browsergym.agent_a_eval  # noqa: F401 — triggers task registration

    from browsergym.agent_a_eval.runner import WorkflowRunner

    agent_args = create_agent_args(config)

    eval_cfg = config.get("evaluation", {})
    paths = config.get("paths", {})
    browser_cfg = config.get("browser", {})
    evaluator_cfg = config.get("evaluator", {})
    login_cfg = config.get("login", {})

    # 解析持久化路径（相对于 config 文件所在目录）
    user_data_dir = login_cfg.get("user_data_dir", "")
    if user_data_dir:
        user_data_dir = str(_resolve_path(config_path, user_data_dir))

    storage_state_path = login_cfg.get("storage_state_path", "")
    if storage_state_path:
        storage_state_path = str(_resolve_path(config_path, storage_state_path))

    runner = WorkflowRunner(
        agent_args=agent_args,
        tasks_dir=paths.get("tasks_dir", "tasks"),
        exp_root=paths.get("exp_root", "./results"),
        eval_model=evaluator_cfg.get("model", "gpt-4o"),
        eval_provider=evaluator_cfg.get("provider", "openai"),
        generalization_datasets=eval_cfg.get("generalization_datasets", 3),
        stability_seeds=eval_cfg.get("stability_seeds", [42, 123, 456]),
        max_steps=eval_cfg.get("max_steps", 200),
        headless=browser_cfg.get("headless", True),
        storage_state=storage_state_path or None,
        user_data_dir=user_data_dir or None,
        # 新增选项
        task_filter=eval_cfg.get("tasks", []),
        task_datasets=eval_cfg.get("task_datasets", {}),
        skip_generalization=eval_cfg.get("_skip_generalization", False),
        skip_stability=eval_cfg.get("_skip_stability", False),
    )

    # ---- 执行评测 ----
    print("\n[RUN] 开始评测...\n")
    reports = runner.run_all()

    # ---- 保存报告 ----
    reports_dir = paths.get("reports_dir", "./reports")
    runner.save_reports(reports, reports_dir)

    print(f"\n[DONE] 评测完成！共 {len(reports)} 个任务。")
    for r in reports:
        g = r.generalization
        s = r.stability
        print(f"  {r.task_name}:")
        print(f"    泛化性 → mean={g['mean']:.2f}, std={g['std']:.2f}, success={g['success_rate']:.1%}")
        print(f"    稳定性 → mean={s['mean']:.2f}, std={s['std']:.2f}, score={s['stability_score']:.2f}")


if __name__ == "__main__":
    main()
