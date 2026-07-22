# Agent A 自动评审系统

基于 [BrowserGym](https://github.com/ServiceNow/BrowserGym) 构建的网页端数据科学 Agent 能力评审系统。

## 目录结构

```
agent_a_eval/
├── config.yaml              # ★ 统一配置文件（LLM、浏览器、路径）
├── run_eval.py              # ★ 测试入口脚本
├── README.md
├── pyproject.toml
├── requirements.txt
├── prompts/
│   ├── driver_agent_prompt.md    # 驱动 Agent 系统 prompt
│   └── evaluator_agent_prompt.md # 评审 Agent 系统 prompt
├── tasks/                   # ★ 任务定义 .md 文件（真人维护）
│   ├── data_analysis.md
│   ├── data_cleaning.md
│   └── model_training.md
├── test_data/datasets/      # ★ 测试数据集
│   ├── sales_2024.csv
│   ├── customers.csv
│   └── iris.csv
└── src/browsergym/agent_a_eval/
    ├── __init__.py           # 自动注册任务
    ├── task.py               # WorkflowTask（BrowserGym 任务核心）
    ├── evaluator.py          # EvaluatorAgent（独立 LLM 评审）
    ├── runner.py             # WorkflowRunner（批量+泛化性+稳定性）
    └── utils.py              # WorkflowConfig / LoopDetector / PageStateChecker
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

# agent_a_eval 模块
pip install -e ./browsergym/agent_a_eval

# demo_agent 依赖（驱动 Agent 使用的 DemoAgent）
pip install -r ./demo_agent/requirements.txt

# Playwright 浏览器（Chromium）
playwright install chromium
```

> **说明**：
> - `-e` = editable 可编辑安装，改源码后立即生效无需重装
> - 必须装 `browsergym/core` 和 `browsergym/experiments`，agent_a_eval 依赖它们
> - Playwright 的 Chromium 是定制版浏览器，与你本机的 Chrome 相互独立
> - 如果 `pip install -e ./browsergym/agent_a_eval` 报 `README.md does not exist`，先 `echo "" > browsergym/agent_a_eval/README.md`

### 3. 验证安装

```bash
cd browsergym/agent_a_eval
python -c "import browsergym.agent_a_eval; print('OK:', len(browsergym.agent_a_eval.ALL_TASK_IDS), 'tasks registered')"
```

预期输出：
```
OK: 9 tasks registered
```

---

## 二、配置说明

### 🔐 敏感信息管理（.env）

API Key、登录密码等敏感信息**不要**写在 [config.yaml](config.yaml) 中，而应通过 `.env` 文件管理：

```bash
# 1. 复制模板
cp .env.example .env

# 2. 编辑 .env，填入真实密钥
# .env 已加入 .gitignore，不会被提交到 Git
```

**配置优先级**（高 → 低）：
1. 系统环境变量（`export` / `set` 设置的）
2. 同目录下的 `.env` 文件
3. `config.yaml` 中的值

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

### config.yaml 结构

所有配置集中在 [config.yaml](config.yaml) 中，分为 6 个部分：

### driver_agent — 驱动 Agent LLM

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `provider` | `openai` 或 `anthropic` | `openai` |
| `model` | 模型名称 | `gpt-4o` |
| `api_key` | API key，`null` 时从环境变量读取 | `null` |
| `base_url` | API 地址，`null` 使用默认 | `null` |
| `temperature` | 生成温度 (0-2) | `0.1` |
| `max_tokens` | 最大输出 token | `4096` |
| `chat_mode` | 对话模式 | `false` |
| `demo_mode` | 视觉特效 (`"default"` / `"off"`) | `"off"` |

> **注意**：DemoAgent 当前硬编码了 `openai.OpenAI()`，只支持 OpenAI 兼容接口。如果用 DeepSeek 等其他厂商，需使用 OpenAI 兼容端点（如 `base_url: https://api.deepseek.com/v1`）。

### evaluator — 评审 Agent LLM

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `provider` | 同上 | `openai` |
| `model` | 建议使用强模型保证评分质量 | `gpt-4o` |
| `api_key` | 同上 | `null` |
| `base_url` | 同上 | `null` |
| `temperature` | 低温度保证评分一致性 | `0.1` |
| `max_tokens` | | `2000` |

### login — 登录配置

Agent A 访问前需要登录时填写。配置后所有任务的 setup 阶段会自动插入登录步骤。

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `url` | 登录页面 URL，为空则跳过登录 | `""` |
| `username` | 登录用户名 | `""` |
| `password` | 登录密码 | `""` |

> **说明**：
> - 如果登录表单在 Agent A 主页面上，`url` 填 Agent A 的地址
> - 如果登录是独立的页面，`url` 填登录页地址，登录成功后再跳转到 Agent A
> - 用户名和密码都为空时，自动跳过登录步骤

### browser — 浏览器配置

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `headless` | `true` = 后台运行，`false` = 显示窗口 | `true` |
| `viewport_width` | 窗口宽度 | `1280` |
| `viewport_height` | 窗口高度 | `900` |
| `slow_mo` | 操作间隔 ms | `500` |
| `timeout` | 单次操作超时 ms | `10000` |

### paths — 路径配置

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `tasks_dir` | .md 任务文件目录 | `tasks` |
| `datasets_dir` | 测试数据集目录 | `test_data/datasets` |
| `exp_root` | 实验结果输出目录 | `./results` |
| `reports_dir` | 评分报告输出目录 | `./reports` |

相对路径均相对于 config.yaml 所在目录解析。

### evaluation — 评测设置

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `tasks` | 要跑的任务列表，`[]` = 全部 | `[]` |
| `generalization_datasets` | 泛化性测试用几个数据集 | `3` |
| `stability_seeds` | 稳定性测试的种子 | `[42, 123, 456]` |
| `max_steps` | 每实验最大步数 | `200` |
| `task_datasets` | 按任务覆盖数据集（可选） | `{}` |

---

## 三、运行测试

```bash
cd browsergym/agent_a_eval
python run_eval.py
```

### 常用命令

```bash
# 查看所有参数
python run_eval.py --help

# 只运行指定任务
python run_eval.py --task data_analysis

# 跳过稳定性测试（只跑泛化性）
python run_eval.py --skip-stability

# 跳过泛化性测试（只跑稳定性）
python run_eval.py --skip-generalization

# 显示浏览器窗口（调试用）
python run_eval.py --no-headless -v

# 指定数据集目录
python run_eval.py --datasets-dir D:\my_data\eval_datasets

# 用自定义配置文件
python run_eval.py --config my_config.yaml
```

### 快速配置（编辑 run_eval.py 顶部 ★ 区域）

```python
TASKS = ["data_analysis"]      # 只跑 data_analysis
SEEDS = [42, 123, 456, 789]    # 稳定性测试用 4 个种子
GENERALIZATION_DATASETS = 5    # 泛化性测试 5 个数据集
DATASETS_DIR = r"D:\work\data" # 数据集路径
SKIP_STABILITY = True          # 跳过稳定性测试
HEADLESS = False               # 显示浏览器窗口
```

配置优先级：**命令行参数 > run_eval.py ★ 区域 > config.yaml**

---

## 四、添加新任务

1. 在 `tasks/` 目录下新建 `.md` 文件，参考 [data_analysis.md](tasks/data_analysis.md) 的格式：

```markdown
# 任务标题

## 基本信息
- Agent A 地址: http://localhost:8080/workspace
- 最大执行时间: 10 分钟
- 最大操作步数: 200
- 需要登录: 是（登录凭据在 config.yaml 中配置，执行时自动处理）

## 可用数据集
- dataset1.csv: 描述
- dataset2.csv: 描述

## 页面硬校验规则
1. 存在 .report-content 元素
2. 不存在 .loading 元素

## 步骤
1. ...

## 评估标准
### 维度1 (权重 ...)
...
```

2. 将数据集文件放入 `test_data/datasets/`

3. 无需修改代码，再次运行 `run_eval.py` 会自动发现新任务

---

## 五、输出报告

评测结束后，报告保存在 `reports/` 目录：

- `evaluation_report_YYYYMMDD_HHMMSS.txt` — **★ 中文文字报告**（可直接阅读，含各维度评分和失败原因）
- `evaluation_summary_YYYYMMDD_HHMMSS.json` — 结构化 JSON 数据
- `evaluation_runs_YYYYMMDD_HHMMSS.csv` — 每次运行摘要表格

---

## 六、架构说明

```
Driver Agent (BrowserGym 内)          Evaluator Agent (独立)
├─ 读取 tasks/*.md                    ├─ 读取 artifacts
├─ 操作浏览器                          ├─ 调用 LLM 评分
├─ 发送 WORKFLOW_DONE 信号            └─ 输出结构化 JSON
└─ 硬校验通过 → 收集 artifacts

WorkflowRunner
├─ 泛化性测试: 固定 seed=42, 变数据集
└─ 稳定性测试: 固定数据集, 变 seed
```

**驱动 Agent 和评审 Agent 完全分离** — 不同 session，不同 LLM，不同职责。
