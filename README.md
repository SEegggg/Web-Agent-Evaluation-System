# Web Agent Test System

基于 [BrowserGym](https://github.com/ServiceNow/BrowserGym) 构建的**网页端 AI Agent 自动化评测系统**。

> 本项目 fork 自 [ServiceNow/BrowserGym](https://github.com/ServiceNow/BrowserGym)，在此基础上新增了 Agent A 自动评审模块。

---

## 核心功能：Agent A 自动评审系统

针对数据科学类网页 Agent（Agent A）的自动化能力评测，支持：

- **驱动 Agent（Driver Agent）**：基于 BrowserGym 的 LLM Agent，自动操作浏览器、上传数据集、与 Agent A 对话、触发任务执行
- **评审 Agent（Evaluator Agent）**：独立的 LLM 评审器，不依赖浏览器，仅消费 Driver Agent 收集的 artifacts（页面输出、对话历史），对 Agent A 的多维度表现打分
- **泛化性测试**：固定 workflow，变换数据集，评估 Agent A 对不同数据的适应能力
- **稳定性测试**：固定数据集，变换随机种子，评估 Agent A 输出的确定性/稳定性

📖 详细文档：[browsergym/agent_a_eval/README.md](browsergym/agent_a_eval/README.md)

### 快速开始

```bash
# 1. 安装依赖
pip install -e ./browsergym/core
pip install -e ./browsergym/experiments
pip install -e ./browsergym/agent_a_eval
pip install -r ./demo_agent/requirements.txt
playwright install chromium

# 2. 配置 config.yaml（LLM、登录、浏览器等）

# 3. 运行评测
cd browsergym/agent_a_eval
python run_eval.py
```

### 项目结构

```
browsergym/agent_a_eval/
├── config.yaml                    # 统一配置文件
├── run_eval.py                    # 测试入口
├── prompts/
│   ├── driver_agent_prompt.md     # 驱动 Agent 操作手册
│   └── evaluator_agent_prompt.md  # 评审 Agent 系统提示
├── tasks/                         # 任务定义（.md 文件）
│   ├── data_analysis.md
│   ├── data_cleaning.md
│   └── model_training.md
├── test_data/datasets/            # 测试数据集
└── src/browsergym/agent_a_eval/   # 源码
    ├── __init__.py                # Gym 环境注册
    ├── task.py                    # WorkflowTask（核心）
    ├── evaluator.py               # EvaluatorAgent（LLM 评审）
    ├── runner.py                  # WorkflowRunner（批量编排）
    └── utils.py                   # LoopDetector / PageStateChecker
```

### 架构

```
Driver Agent (BrowserGym)              Evaluator Agent (独立)
├─ 读取 tasks/*.md 任务定义            ├─ 读取 artifacts
├─ 操作浏览器与 Agent A 交互            ├─ 调用 LLM 评分
├─ 触发 WORKFLOW_DONE 信号              └─ 输出结构化 JSON 评分
└─ 硬校验 → 收集 artifacts

WorkflowRunner
├─ 泛化性测试: 固定 seed=42, 变数据集
└─ 稳定性测试: 固定数据集, 变 seed
```

---

## BrowserGym 生态

本项目底层使用 BrowserGym 提供的标准 Gymnasium 环境和实验工具：

- **BrowserGym Core**：Playwright 驱动的浏览器环境，提供标准化的观察/动作空间
- **BrowserGym Experiments**：实验基础设施（`ExpArgs`, `EnvArgs`, `Agent` 基类）
- **[AgentLab](https://github.com/ServiceNow/AgentLab)**：大规模实验编排框架（Ray 并行、结果分析、AgentXRay 可视化）

BrowserGym 内置以下基准测试：

- [MiniWoB](https://miniwob.farama.org/)
- [WebArena](https://webarena.dev/)
- [WebArenaVerified](https://github.com/ServiceNow/webarena-verified)
- [VisualWebArena](https://jykoh.com/vwa)
- [WorkArena](https://github.com/ServiceNow/WorkArena)
- [AssistantBench](https://github.com/oriyor/assistantbench)
- [WebLINX](https://github.com/McGill-NLP/weblinx)
- [OpenApps](https://facebookresearch.github.io/OpenApps/)
- [TimeWarp](https://timewarp-web.github.io)

---

## Citation

```tex
@article{
    chezelles2025browsergym,
    title={The BrowserGym Ecosystem for Web Agent Research},
    author={Thibault Le Sellier de Chezelles and Maxime Gasse and Alexandre Lacoste and
            Massimo Caccia and Alexandre Drouin and L{\'e}o Boisvert and Megh Thakkar and
            Tom Marty and Rim Assouel and Sahar Omidi Shayegan and Lawrence Keunho Jang and
            Xing Han L{\`u} and Ori Yoran and Dehan Kong and Frank F. Xu and Siva Reddy and
            Graham Neubig and Quentin Cappart and Russ Salakhutdinov and Nicolas Chapados},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2025},
    url={https://openreview.net/forum?id=5298fKGmv3},
    note={Expert Certification}
}
```
