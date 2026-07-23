# Agent A 自动评审系统

基于 [BrowserGym](https://github.com/ServiceNow/BrowserGym) 构建的网页端数据科学 Agent 能力评审系统。

支持 **串行** 和 **joblib 多进程并行** 两种执行模式，一键并行评测全部任务。

## 目录结构

```
agent_a_eval/
├── config.yaml                  # ★ 统一配置文件（LLM、浏览器、路径）
├── .env.example                 # 环境变量模板
├── scripts/
│   ├── run_eval.py              # 串行评测入口
│   ├── run_agentlab.py          # ★ 并行评测入口（AgentLab + joblib）
│   ├── task_generator.py        # ★ LLM 驱动的任务 .md 生成器
│   └── _agentlab_stubs/         # agentlab 兼容性桩（webarena/visualwebarena）
├── README.md
├── pyproject.toml
├── requirements.txt
├── prompts/
│   ├── driver_agent_prompt.md       # 驱动 Agent 系统 prompt（含 Skill 使用指南）
│   └── evaluator_agent_prompt.md    # 评审 Agent 系统 prompt
├── tasks/                       # ★ 任务定义 .md 文件
│   ├── data_analysis.md
│   ├── data_cleaning.md
│   ├── model_training.md
│   └── 数据注册.md               # LLM 生成的示例任务
├── test_data/datasets/          # ★ 测试数据集
│   ├── sales_2024.csv           # 干净销售数据（25行）
│   ├── customers.csv            # 干净客户数据
│   ├── iris.csv                 # 干净鸢尾花数据
│   ├── dirty_sales.csv          # 脏销售数据（含缺失值/重复行/异常值）
│   ├── messy_customers.csv      # 混乱客户数据（含日期格式不一致/拼写错误）
│   └── raw_iris.csv             # 原始鸢尾花数据（含缺失值/极端异常值）
└── src/browsergym/agent_a_eval/ # 核心源码
    ├── __init__.py              # 自动注册任务 + get_benchmark()
    ├── task.py                  # WorkflowTask（BrowserGym 任务核心 + Skill API）
    ├── evaluator.py             # EvaluatorAgent（独立 LLM 评审）
    ├── runner.py                # WorkflowRunner（串行批量+泛化性+稳定性）
    ├── agentlab_runner.py       # AgentLabRunner（并行批量+泛化性+稳定性）
    ├── benchmark.py             # AgentLab Benchmark 适配层
    ├── network_interceptor.py   # 网络拦截器（SSE + JSON 捕获）
    └── utils.py                 # WorkflowConfig / LoopDetector / PageStateChecker
```

---

## 一、环境配置

### 1. 创建 conda 环境

```bash
conda create -n agent-a-eval python=3.11 -y
conda activate agent-a-eval
```

### 2. 安装依赖

以下命令均在项目根目录 `BrowserGym/` 下执行：

```bash
# browsergym 核心模块
pip install -e ./browsergym/core
pip install -e ./browsergym/experiments

# agent_a_eval 模块（含 agentlab + joblib）
pip install -e ./browsergym/agent_a_eval

# demo_agent 依赖（驱动 Agent 使用的 DemoAgent）
pip install -r ./demo_agent/requirements.txt

# Playwright 浏览器（Chromium）
playwright install chromium
```

> **注意**：`agentlab` 依赖 `ray[default]`，Windows 下 ray 可能无法安装。`
> run_agentlab.py` 默认使用 joblib 后端，不依赖 ray。

### 3. 验证安装

```bash
cd browsergym/agent_a_eval
python -c "import browsergym.agent_a_eval; print('OK:', len(browsergym.agent_a_eval.ALL_TASK_IDS), 'tasks registered')"
```

---

## 二、配置说明

### 🔐 敏感信息管理（.env）

API Key、登录密码等敏感信息通过 `.env` 文件管理：

```bash
cp .env.example .env
# 编辑 .env，填入真实密钥
```

**配置优先级**（高 → 低）：系统环境变量 > `.env` 文件 > `config.yaml`

### 环境变量对照表

| 用途 | 环境变量 | 说明 |
|------|---------|------|
| OpenAI API Key | `OPENAI_API_KEY` | 驱动/评审 Agent 共用 |
| OpenAI API 地址 | `OPENAI_BASE_URL` | 使用 DeepSeek 等兼容接口时设置 |
| Anthropic API Key | `ANTHROPIC_API_KEY` | 驱动/评审 Agent 共用 |
| Anthropic API 地址 | `ANTHROPIC_BASE_URL` | 自定义 Anthropic 端点 |
| 登录 URL | `AGENT_A_EVAL_LOGIN_URL` | Agent A 的登录页面地址 |
| 登录用户名 | `AGENT_A_EVAL_LOGIN_USERNAME` | Agent A 登录用户名 |
| 登录密码 | `AGENT_A_EVAL_LOGIN_PASSWORD` | Agent A 登录密码 |
| Skill API | `AGENT_A_API_BASE_URL` | Agent A 后端 API 地址（Skill 查询） |
| Skill API Token | `AGENT_A_API_TOKEN` | Skill API 认证 Token |

---

## 三、运行测试

### 并行评测（推荐）

```bash
cd browsergym/agent_a_eval

# 默认并行（joblib，4 workers）
python scripts/run_agentlab.py

# 指定 worker 数量
python scripts/run_agentlab.py --n-jobs 8

# 仅评测单个任务
python scripts/run_agentlab.py --task data_analysis

# 串行模式（等同于 run_eval.py）
python scripts/run_agentlab.py --n-jobs 1 --backend sequential
```

### 串行评测（传统模式）

```bash
cd browsergym/agent_a_eval
python scripts/run_eval.py
```

### 常用命令

```bash
# 只运行指定任务
python scripts/run_eval.py --task data_analysis

# 跳过稳定性测试
python scripts/run_eval.py --skip-stability

# 跳过泛化性测试
python scripts/run_eval.py --skip-generalization

# 显示浏览器窗口（调试用）
python scripts/run_eval.py --no-headless -v

# 指定数据集目录
python scripts/run_eval.py --datasets-dir D:\my_data\eval_datasets
```

### 快速配置

编辑 `scripts/run_agentlab.py` 顶部 `★ EDIT HERE` 区域：

```python
TASKS = ["data_analysis"]          # 只跑指定任务
SKIP_STABILITY = True              # 跳过稳定性测试
SKIP_GENERALIZATION = True         # 跳过泛化性测试
HEADLESS = False                   # 显示浏览器窗口
```

配置优先级：**命令行参数 > run_agentlab.py ★ 区域 > config.yaml**

---

## 四、添加新任务

### 方式一：手写 .md

1. 在 `tasks/` 目录下新建 `.md` 文件，参考现有任务格式：

```markdown
# 任务标题

## 可用数据集
- dataset1.csv: 描述
- dataset2.csv: 描述

## 步骤
1. 具体可操作步骤
2. ...

## 评估标准

### 核心指标: 任务成功率（一票通过/否决）
- 通过条件: ...
- 失败条件: ...
- 归因说明: ...

### 任务专项指标
- 具体评分维度及标准
```

2. 将数据集文件放入 `test_data/datasets/`

3. 无需修改代码，再次运行会自动发现新任务

### 方式二：LLM 生成

```bash
# 命令行模式
python scripts/task_generator.py \
  -d custom \
  -n "任务名称" \
  --datasets "file1.csv:描述,file2.csv:描述" \
  -r "额外的特殊要求"

# 交互模式
python scripts/task_generator.py --interactive
```

生成器会：
1. 从 `.env` 和 `config.yaml` 读取 LLM 配置
2. 调用 LLM 生成符合规范的 .md 文件
3. 自动校验格式（必需章节、数据集格式、编号步骤等）
4. 校验失败自动让 LLM 修复（最多 2 次重试）
5. 保存前验证 `WorkflowConfig` 可正常解析

---

## 五、输出报告

评测结束后，报告保存在 `reports/` 目录：

- `evaluation_report_YYYYMMDD_HHMMSS.txt` — **中文文字报告**
- `evaluation_summary_YYYYMMDD_HHMMSS.json` — 结构化 JSON
- `evaluation_runs_YYYYMMDD_HHMMSS.csv` — 每次运行摘要

AgentLab 模式还会在 `results/` 下生成 AgentXRay 兼容的实验目录。

---

## 六、架构说明

```
┌─ 浏览器阶段（并行）────────────────────────────────────┐
│  Driver Agent (BrowserGym + AgentLab)                  │
│  ├─ 读取 tasks/*.md → 构建 goal                        │
│  ├─ 通过 Skill API 获取 Agent A 可用 Skill 列表         │
│  ├─ 操作浏览器执行任务步骤                              │
│  ├─ NetworkInterceptor 捕获 Agent A SSE/JSON 响应       │
│  ├─ 死循环检测 / 硬校验 / 超时控制                      │
│  └─ 收集 artifacts → 写入日志目录                       │
├────────────────────────────────────────────────────────┤
│  评审阶段（串行）                                       │
│  Evaluator Agent (独立 LLM)                            │
│  ├─ 读取 artifacts (agent_a_output.txt, chat_history)   │
│  ├─ 根据任务 .md 中的评估标准评分                        │
│  └─ 输出 TaskReport (pass/fail + 各维度评分 + 理由)     │
└────────────────────────────────────────────────────────┘

AgentLabRunner
├─ 快速测试: 固定 seed=42, 单数据集
├─ 泛化性测试: 固定 seed=42, 变数据集
└─ 稳定性测试: 固定数据集, 变 seed
```

**驱动 Agent 和评审 Agent 完全分离** — 不同 session，不同 LLM，不同职责。

### 关键设计

- **NetworkInterceptor**：在 Playwright 网络层拦截 Agent A 后端 API 响应（SSE 流 + JSON），比 DOM 抓取更可靠
- **Skill API**：每次任务 setup 时从 Agent A 后端获取可用 Skill 列表，注入到 Driver Agent 的 goal 中
- **程序化登录**：先尝试从 `storage_state` 恢复登录态，失败则自动填写登录表单
- **死循环检测**：多信号（动作重复 + DOM 状态回退）防止 Driver Agent 卡死
