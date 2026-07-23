"""
Agent A 自动评审系统 — AgentLab 并行评测入口
=============================================

基于 AgentLab 的并行评测运行器。与 run_eval.py 功能等价，
但使用 Ray/joblib 并行执行浏览器实验。

用法:
    cd browsergym/agent_a_eval

    # 默认并行（joblib，4 workers）
    python scripts/run_agentlab.py

    # 指定 worker 数量和后端
    python scripts/run_agentlab.py --n-jobs 8 --backend joblib

    # 仅评测单个任务
    python scripts/run_agentlab.py --task data_analysis

    # 使用 Ray 后端（Linux/Mac，需要 ray[default]）
    python scripts/run_agentlab.py --n-jobs 16 --backend ray

    # 串行模式（等同于 run_eval.py）
    python scripts/run_agentlab.py --n-jobs 1 --backend sequential

    # 完成后打印 AgentXRay 启动命令
    python scripts/run_agentlab.py --xray

环境变量:
    AGENTLAB_EXP_ROOT   — AgentXRay 读取的实验根目录（默认 = exp_root）
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# 把项目根目录加到 sys.path，使 demo_agent 模块可被导入
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # BrowserGym workspace root
_SCRIPTS_DIR = Path(__file__).resolve().parent  # scripts/ directory

# Add all browsergym src/ directories to sys.path.
# This ensures namespace packages (browsergym.*) and bgym are importable.
_SRC_PATHS = [
    str(_PROJECT_ROOT),
    str(_SCRIPTS_DIR),
    str(_PROJECT_ROOT / "browsergym" / "core" / "src"),
    str(_PROJECT_ROOT / "browsergym" / "experiments" / "src"),
    str(_PROJECT_ROOT / "browsergym" / "agent_a_eval" / "src"),
]

# Insert local src paths at BEGINNING so they shadow any stock PyPI browsergym
for _p in reversed(_SRC_PATHS):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
for _p in _SRC_PATHS:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── agentlab compatibility stubs ──
# agentlab unconditionally imports webarena/visualwebarena modules.
# Provide lightweight stubs so imports succeed without full installations.
_STUB_DIR = Path(__file__).resolve().parent / "_agentlab_stubs"
_STUB_DIR.mkdir(exist_ok=True)

_webarena_dir = _STUB_DIR / "browsergym" / "webarena"
_webarena_dir.mkdir(parents=True, exist_ok=True)
(_webarena_dir / "__init__.py").touch(exist_ok=True)
(_webarena_dir / "instance.py").write_text('''
# Stub: agentlab imports this but agent_a_eval doesn\'t use it.
class WebArenaInstance:
    def __init__(self, **kwargs): pass
    def full_reset(self): pass
    def check_status(self): pass
''')

_vwa_dir = _STUB_DIR / "browsergym" / "visualwebarena"
_vwa_dir.mkdir(parents=True, exist_ok=True)
(_vwa_dir / "__init__.py").touch(exist_ok=True)
(_vwa_dir / "instance.py").write_text('''
# Stub: agentlab imports this but agent_a_eval doesn\'t use it.
class VisualWebArenaInstance:
    def __init__(self, **kwargs): pass
    def full_reset(self): pass
''')

if str(_STUB_DIR) not in sys.path:
    sys.path.insert(0, str(_STUB_DIR))


# ============================================================
# ★ EDIT HERE —— 快速配置覆盖（优先级高于 config.yaml）
#   独立于 run_eval.py，互不影响
# ============================================================

# 要测试的任务列表，留空 = 自动扫描全部 .md 任务
# 示例: TASKS = ["data_analysis", "data_cleaning"]
TASKS = []

# 稳定性测试的随机种子，None = 使用 config.yaml 中的值
SEEDS = None

# 泛化性测试使用的数据集数量，None = 使用 config.yaml 中的值
GENERALIZATION_DATASETS = None

# 数据集文件所在目录的绝对路径，None = 使用 config.yaml 中的值
DATASETS_DIR = None

# 跳过某类测试: True = 跳过
SKIP_STABILITY = True
SKIP_GENERALIZATION = True

# 浏览器是否无头模式，None = 使用 config.yaml 中的值
HEADLESS = None

# 每个实验最大步数，None = 使用 config.yaml 中的值
MAX_STEPS = None

# ============================================================


def parse_args():
    """解析命令行参数。"""
    p = argparse.ArgumentParser(
        description="Agent A 自动评审系统 — AgentLab 并行版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="配置文件路径 (默认: config.yaml)",
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
    # ── AgentLab 特有参数 ──
    p.add_argument(
        "--n-jobs", type=int, default=4,
        help="并行 worker 数量 (默认: 4)",
    )
    p.add_argument(
        "--backend", type=str, default="joblib",
        choices=["ray", "joblib", "sequential"],
        help="并行后端: ray | joblib | sequential (默认: joblib)",
    )
    p.add_argument(
        "--xray", action="store_true",
        help="完成后打印 AgentXRay 启动命令",
    )
    return p.parse_args()


def _resolve_path(base: Path, value: str) -> Path:
    """如果 value 是相对路径，则相对于 base 所在目录解析。"""
    p = Path(value)
    if p.is_absolute():
        return p
    return (base.parent / p).resolve()


def _merge_overrides(config: dict, args) -> dict:
    """将命令行参数和本脚本的快速配置合并到 config 中（独立于 run_eval.py）。"""
    cfg = config

    # ---- 本脚本的快速配置 ----
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

    # 数据集目录
    if DATASETS_DIR:
        cfg["paths"]["datasets_dir"] = DATASETS_DIR
    if args.datasets_dir:
        cfg["paths"]["datasets_dir"] = args.datasets_dir

    return cfg


def main():
    args = parse_args()

    # 确定配置文件路径
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"

    # 加载 .env 文件
    from run_eval import load_dotenv
    dotenv_path = config_path.parent / ".env"
    load_dotenv(dotenv_path)

    # 加载配置
    from run_eval import load_config, setup_env, \
        create_agent_args, _setup_api_env

    config = load_config(config_path)
    config = _merge_overrides(config, args)
    setup_env(config, config_path)

    # 设置日志级别
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # ── 打印配置摘要 ──
    driver = config.get("driver_agent", {})
    evaluator = config.get("evaluator", {})
    browser_cfg = config.get("browser", {})
    eval_cfg = config.get("evaluation", {})
    paths = config.get("paths", {})

    print("=" * 60)
    print("  Agent A 自动评审系统 (AgentLab 并行版)")
    print("=" * 60)
    print(f"  驱动 Agent: {driver.get('model')} ({driver.get('provider')})")
    print(f"  评审 Agent: {evaluator.get('model')} ({evaluator.get('provider')})")
    print(f"  浏览器模式: {'headless' if browser_cfg.get('headless') else 'visible'}")
    print(f"  并行后端:   {args.backend}")
    print(f"  Worker 数:  {args.n_jobs}")
    print(f"  任务目录:   {paths.get('tasks_dir', 'N/A')}")
    print(f"  数据集目录: {paths.get('datasets_dir', 'N/A')}")
    print(f"  结果目录:   {paths.get('exp_root', 'N/A')}")
    print(f"  报告目录:   {paths.get('reports_dir', 'N/A')}")

    tasks = eval_cfg.get("tasks", [])
    if tasks:
        print(f"  指定任务:   {', '.join(tasks)}")

    gen_info = "跳过" if eval_cfg.get("_skip_generalization") else \
        f"{eval_cfg.get('generalization_datasets', 3)} 个数据集"
    stab_info = "跳过" if eval_cfg.get("_skip_stability") else \
        f"种子: {eval_cfg.get('stability_seeds', [42, 123, 456])}"
    print(f"  泛化性测试: {gen_info}")
    print(f"  稳定性测试: {stab_info}")
    print(f"  最大步数:   {eval_cfg.get('max_steps', 200)}")
    print("-" * 60)

    # ── 注册任务 ──
    print("[INIT] 正在注册任务...")
    import browsergym.agent_a_eval  # noqa: F401 — triggers task registration

    from browsergym.agent_a_eval.agentlab_runner import AgentLabRunner

    # 创建 agent_args
    agent_args = create_agent_args(config)

    # 设置评审 Agent 的 API key
    evaluator_cfg = config.get("evaluator", {})
    _setup_api_env(evaluator_cfg)

    # 解析路径
    login_cfg = config.get("login", {})
    user_data_dir = login_cfg.get("user_data_dir", "")
    if user_data_dir:
        user_data_dir = str(_resolve_path(config_path, user_data_dir))

    storage_state_path = login_cfg.get("storage_state_path", "")
    if storage_state_path:
        storage_state_path = str(_resolve_path(config_path, storage_state_path))

    # ── 创建 AgentLabRunner ──
    runner = AgentLabRunner(
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
        task_filter=eval_cfg.get("tasks", []),
        task_datasets=eval_cfg.get("task_datasets", {}),
        skip_generalization=eval_cfg.get("_skip_generalization", False),
        skip_stability=eval_cfg.get("_skip_stability", False),
        # AgentLab 特有
        n_jobs=args.n_jobs,
        parallel_backend=args.backend,
        study_suffix="agent_a_eval",
    )

    # ── 执行评测 ──
    print(f"\n[RUN] 开始并行评测 (n_jobs={args.n_jobs}, backend={args.backend})...\n")
    reports = runner.run_all()

    # ── 保存报告 ──
    reports_dir = paths.get("reports_dir", "./reports")
    runner.save_reports(reports, reports_dir)

    # ── 打印摘要 ──
    print(f"\n[DONE] 评测完成！共 {len(reports)} 个任务。")
    for r in reports:
        g = r.generalization
        s = r.stability
        print(f"  {r.task_name}:")
        label = "快速测试" if r.is_quick_test else "泛化性"
        print(f"    {label} → pass_rate={g['pass_rate']:.1%}, "
              f"passed={g['passed']}/{g['total_runs']}")
        print(f"    稳定性 → score={s['stability_score']:.2f}")

    # ── AgentXRay 提示 ──
    study_dir = runner.study_dir
    if study_dir:
        print(f"\n[AgentLab] Study 目录: {study_dir}")
    if args.xray:
        print(f"\n[AgentXRay] 启动可视化界面:")
        print(f"  {runner.xray_command}")
        print(f"  启动后在浏览器中打开 Gradio 界面，选择实验 → Agent → 任务 → seed 即可查看 trace。")


if __name__ == "__main__":
    main()
