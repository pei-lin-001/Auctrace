<div align="center">
  <h1>
    <b>Auctrace: Trustworthy Automated Scientific Research</b>
  </h1>
  <p><i>可信、可审计、有韧性的自动化科研系统</i></p>
</div>

<p align="center">
  🏗️ 基于 <a href="https://github.com/SakanaAI/AI-Scientist_v2">AI Scientist-v2</a> |
  📚 <a href="https://pub.sakana.ai/ai-scientist-v2/paper">[原论文]</a>
</p>

---

**Auctrace** 是基于 [AI Scientist-v2](https://github.com/SakanaAI/AI-Scientist_v2) 的增强版本，专注于解决自动化科研系统中的**可信性问题**。

当前由 LLM 驱动的自动化科研系统虽然能够完成 `idea → experiment → writeup → review` 的闭环并生成论文草稿，但论文内容的真实性仍存在根本缺陷：

- 幻觉内容可能进入论文文本
- 数字、图表和结论可能脱节
- 研究主张往往缺乏可追溯的证据绑定
- 失败实验和不稳定结果容易被错误解释

**Auctrace 的目标**：让自动化科研系统产出的论文内容更真实、更可核验、更可信——从"能生成"转向"能被相信"。

> **注意！**
> 此代码库将执行由大语言模型（LLM）编写的代码。这种自主性存在各种风险和挑战，包括可能使用危险软件包、不受控制的网络访问以及产生意外进程的可能性。请确保在受控的沙盒环境（如 Docker 容器）中运行。请自行斟酌使用。

## 目录

1.  [核心特性](#核心特性)
2.  [环境要求](#环境要求)
    *   [安装](#安装)
    *   [支持的模型与 API 密钥](#支持的模型与-api-密钥)
3.  [生成研究想法](#生成研究想法)
4.  [运行实验](#运行实验)
5.  [执行后端](#执行后端)
    *   [本地 GPU](#本地-gpu)
    *   [WSL SSH](#wsl-ssh)
6.  [可靠性模块](#可靠性模块)
7.  [项目结构](#项目结构)
8.  [致谢](#致谢)
9.  [许可证](#许可证)

## 核心特性

Auctrace 在 AI Scientist-v2 基础上新增以下可信性增强模块：

### 符号事实锚定 (Symbolic Fact Anchoring)

- **FactStore**：结构化事实变量库，统一管理实验结果数字
- **Symbolic LaTeX**：论文写作使用 `\fact{KEY}` 占位符，渲染器确定性填充
- **Claim Ledger**：研究主张与证据的双向追溯链路

### 分级门禁机制 (Budgeted Gates)

- **Gate A-E**：从实验产物入库到旁观者审计的五层检查
- **确定性校验优先**：schema、引用键、变量解析等低成本检查
- **旁观者审计**：薄上下文输入，避免被错误叙事链路污染

### 评测协议 (MAC Framework)

- **Numeric Integrity**：NCS（数值主张健全性）、NIA（数值一致性）
- **Claim Traceability**：CEB（主张证据绑定）、PTC（来源可追溯性）
- **Auditability**：ICS（内部一致性）、GPC（门禁覆盖率）

---

## 环境要求

此代码设计在 Linux 环境下运行，需要 NVIDIA GPU（CUDA）和 PyTorch。也支持通过 WSL SSH 进行远程执行。

### 安装

```bash
# 创建新的 conda 环境
conda create -n ai_scientist python=3.11
conda activate ai_scientist

# 安装支持 CUDA 的 PyTorch（根据你的配置调整 pytorch-cuda 版本）
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# 安装 PDF 和 LaTeX 工具
conda install anaconda::poppler
conda install conda-forge::chktex

# 安装 Python 依赖包
pip install -r requirements.txt
```

安装通常不超过一小时。

### 支持的模型与 API 密钥

#### OpenAI 模型

默认情况下，系统使用 `OPENAI_API_KEY` 环境变量来访问 OpenAI 模型。

#### OpenAI 兼容模型

本仓库支持在 v2 实验路径、想法生成、绘图、论文写作、LLM 审稿和 VLM 审稿中使用**任意 OpenAI 兼容的模型名称**。

设置方式一：

```bash
export OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
export OPENAI_API_KEY="YOUR_COMPATIBLE_KEY"
```

`OPENAI_API_BASE` 也可作为 `OPENAI_BASE_URL` 的别名使用。

设置方式二（显式别名）：

```bash
export OPENAI_COMPATIBLE_BASE_URL="https://your-compatible-endpoint/v1"
export OPENAI_COMPATIBLE_API_KEY="YOUR_COMPATIBLE_KEY"
```

本仓库启动时也会读取本地 `.env` 文件。如需将配置的 OpenAI 兼容模型的 token 预算统一为 128k，可设置：

```bash
AI_SCIENTIST_CONTEXT_TOKENS=131072
```

本仓库请求侧的生成上限单独控制：

```bash
AI_SCIENTIST_MAX_OUTPUT_TOKENS=8192
```

当配置了兼容的 base URL 时，普通模型名称如 `Qwen/Qwen3-32B` 或 `meta-llama/llama-3.1-70b-instruct` 将通过该 OpenAI 兼容端点发送。本仓库不再维护特定提供商的 SDK 路由；如需使用非 OpenAI 模型，请通过你的 OpenAI 兼容网关暴露。

对于 v2 树搜索路径，每个阶段配置现在可以声明一个显式的 `fallback_model`。当主 OpenAI 兼容模型在可重试的传输/提供商错误上耗尽 SDK 重试预算时，请求将在配置的备用模型上重试一次，并显式记录切换，而不是在原模型上无限循环。

示例：

```yaml
agent:
  code:
    model: MiniMax-M2.5
    fallback_model: nvidia/nemotron-3-super-120b-a12b
```

#### Semantic Scholar API（文献搜索）

我们的代码可选择使用 Semantic Scholar API 密钥（`S2_API_KEY`）以在文献搜索期间获得更高吞吐量，[如果你有的话](https://www.semanticscholar.org/product/api)。这在想法生成和论文写作阶段都会使用。没有它系统也能工作，但可能会遇到速率限制或想法生成期间的新颖性检查受限。如果遇到 Semantic Scholar 问题，可以跳过论文生成期间的引用阶段。

#### 设置 API 密钥

确保为你打算使用的模型提供必要的 API 密钥作为环境变量。例如：
```bash
export OPENAI_API_KEY="YOUR_OPENAI_KEY_HERE"
export OPENAI_BASE_URL="https://your-compatible-endpoint/v1"  # 可选，用于 OpenAI 兼容 API
export S2_API_KEY="YOUR_S2_KEY_HERE"
```

## 生成研究想法

在运行完整的 AI Scientist-v2 实验管线之前，首先使用 `ai_scientist/perform_ideation_temp_free.py` 脚本生成潜在的研究想法。该脚本使用 LLM 根据你提供的高层级主题描述进行头脑风暴和细化想法，并与 Semantic Scholar 等工具交互以检查新颖性。

1.  **准备主题描述：** 创建一个 Markdown 文件（如 `my_research_topic.md`），描述你想让 AI 探索的研究领域或主题。该文件应包含 `Title`、`Keywords`、`TL;DR` 和 `Abstract` 等部分来定义研究范围。参考示例文件 `ai_scientist/ideas/i_cant_believe_its_not_better.md` 了解预期的结构和内容格式。将文件放在脚本可访问的位置（如 `ai_scientist/ideas/` 目录）。

2.  **运行想法生成脚本：** 从主项目目录执行脚本，指向你的主题描述文件并指定所需的 LLM。

    ```bash
    python ai_scientist/perform_ideation_temp_free.py \
     --workshop-file "ai_scientist/ideas/my_research_topic.md" \
     --model gpt-4o-2024-05-13 \
     --max-num-generations 20 \
     --num-reflections 5
    ```
    *   `--workshop-file`：主题描述 Markdown 文件的路径。
    *   `--model`：用于生成想法的 LLM。现在可以是任何支持的本地模型，或在配置 `OPENAI_BASE_URL` 时的任何 OpenAI 兼容模型名称。
    *   `--max-num-generations`：尝试生成多少个不同的研究想法。
    *   `--num-reflections`：LLM 对每个想法执行多少次细化步骤。

3.  **输出：** 脚本将生成一个以你的输入 Markdown 文件命名的 JSON 文件（如 `ai_scientist/ideas/my_research_topic.json`）。该文件将包含结构化研究想法列表，包括假设、提议的实验和相关工作分析。

4.  **继续实验：** 生成包含研究想法的 JSON 文件后，可以继续下一节运行实验。

此想法生成步骤引导 AI Scientist 关注特定兴趣领域，并产生可在主实验管线中测试的具体研究方向。

## 运行实验

使用生成的 idea JSON 文件启动完整的实验管线：

```bash
python launch_scientist_bfts.py \
  --load_ideas "ai_scientist/ideas/my_research_topic.json" \
  --idea_idx 0
```

### 关键命令行参数

| 参数 | 说明 |
|------|------|
| `--load_ideas` | idea JSON 文件路径 |
| `--idea_idx` | 选择第几个 idea（从 0 开始） |
| `--load_code` | 使用初始代码模板启动实验 |
| `--add_dataset_ref` | 添加数据集引用 |
| `--writeup-symbolic-facts` | 启用符号事实模式（推荐） |
| `--resume-checkpoint` | 从检查点恢复运行 |

### 输出目录结构

```
experiments/<timestamp>_<idea>_attempt_<id>/
├── logs/0-run/
│   ├── fact_store.json          # 事实变量库
│   ├── claim_ledger.json        # 主张追溯账本
│   ├── outsider_audit.json      # 旁观者审计报告
│   └── unified_tree_viz.html    # 搜索树可视化
├── latex/
│   ├── template.tex             # 符号 LaTeX 稿
│   ├── template.rendered.tex    # 渲染后的数值稿
│   └── used_facts.json         # 使用的事实键列表
└── <timestamp>_<idea>.pdf       # 最终论文
```

### 从检查点恢复

如果之前的运行在某个阶段中断，可以从检查点继续：

```bash
python launch_scientist_bfts.py \
  --resume-checkpoint "experiments/<previous_run>/logs/0-run/stage_2_baseline_tuning_1_first_attempt/checkpoint.pkl"
```

## 执行后端

Auctrace 支持多种实验执行后端，通过 `bfts_config.yaml` 中的 `exec.backend` 配置：

### 本地 GPU

默认模式，在本地 CUDA GPU 上执行实验。

```yaml
exec:
  backend: local
```

### WSL SSH

通过 SSH 连接到 WSL 远程主机执行实验。适用于 Windows + WSL 环境。

```yaml
exec:
  backend: wsl_ssh
```

环境变量配置：

```bash
AI_SCIENTIST_WSL_SSH_HOST=your-wsl-host
AI_SCIENTIST_WSL_SSH_USER=username
AI_SCIENTIST_WSL_SSH_PASSWORD=password
AI_SCIENTIST_WSL_REMOTE_ROOT=/home/user/auctrace
AI_SCIENTIST_WSL_VENV_PATH=/home/user/auctrace/.venv
```

## 可靠性模块

Auctrace 的可靠性增强模块位于 `ai_scientist/reliable/` 目录：

| 模块 | 文件 | 功能 |
|------|------|------|
| **FactStore** | `facts.py` | 结构化事实变量库 |
| **Fact Extraction** | `fact_extraction.py`, `extract_facts.py` | 从实验产物提取事实 |
| **Claim Ledger** | `claim_ledger.py` | 主张-证据追溯账本 |
| **Symbolic Postprocess** | `symbolic_postprocess.py` | 符号稿渲染与修复 |
| **Gates** | `gates.py` | 分级门禁检查 |
| **Numeric Lint** | `numeric_lint.py` | 裸数字检测 |
| **Outsider Audit** | `outsider_audit.py` | 旁观者审计 |
| **Remediation** | `remediation.py` | 写稿失败修复循环 |
| **Context Pack** | `context_pack.py` | 分层上下文打包 |

### 环境变量开关

| 变量 | 默认值 | 作用 |
|------|--------|------|
| `AI_SCIENTIST_WRITEUP_SYMBOLIC_FACTS` | `1` | 启用符号事实模式 |
| `AI_SCIENTIST_SKIP_NUMERIC_LINT` | `0` | 跳过数字 lint |
| `AI_SCIENTIST_SKIP_CLAIM_LEDGER` | `0` | 跳过 Claim Ledger |
| `AI_SCIENTIST_OUTSIDER_AUDIT` | `1` | 启用旁观者审计 |
| `AI_SCIENTIST_NUMERIC_BACKANCHOR` | `0` | 数值反向锚定 |

---

## 项目结构

```
auctrace/
├── launch_scientist_bfts.py    # 主入口脚本
├── bfts_config.yaml            # BFTS 配置
├── requirements.txt            # Python 依赖
├── .env                        # 环境变量配置
│
├── ai_scientist/
│   ├── llm/                    # LLM 调用层
│   │   ├── client.py           # 客户端创建与路由选择
│   │   ├── completion.py       # 请求执行与重试
│   │   ├── models.py           # 模型能力声明
│   │   └── ...
│   │
│   ├── treesearch/             # BFTS 树搜索核心
│   │   ├── agent_manager.py    # 四阶段实验管理
│   │   ├── node.py             # 搜索节点
│   │   └── ...
│   │
│   ├── reliable/               # 可靠性增强模块
│   │   ├── facts.py            # FactStore
│   │   ├── claim_ledger.py     # Claim Ledger
│   │   ├── gates.py            # 分级门禁
│   │   └── ...
│   │
│   ├── perform_ideation_temp_free.py  # Idea 生成
│   ├── perform_writeup.py      # 论文写作
│   ├── perform_llm_review.py   # LLM 审稿
│   └── perform_vlm_review.py   # VLM 审稿
│
├── build/                      # 项目规划文档
│   ├── 项目目标.md
│   ├── 四大研究问题.md
│   └── ...
│
└── experiments/                # 实验输出目录
```

---

## 致谢

本项目基于以下开源项目构建：

- **[AI Scientist-v2](https://github.com/SakanaAI/AI-Scientist_v2)** - Sakana AI 的自动化科研系统
- **[AIDE](https://github.com/WecoAI/aideml)** - 树搜索实验框架

如果使用本项目，请同时引用 AI Scientist-v2：

```bibtex
@article{aiscientist_v2,
  title={The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search},
  author={Yamada, Yutaro and Lange, Robert Tjarko and Lu, Cong and Hu, Shengran and Lu, Chris and Foerster, Jakob and Clune, Jeff and Ha, David},
  journal={arXiv preprint arXiv:2504.08066},
  year={2025}
}
```

---

## 许可证

本项目继承 AI Scientist-v2 的许可证：**The AI Scientist Source Code License**（基于 Responsible AI License）。

**强制披露**：使用本代码生成的任何科学论文或手稿，必须明确披露 AI 的使用。建议在论文的摘要或方法部分添加：

> "本手稿使用 [Auctrace](https://github.com/pei-lin-001/Auctrace) 生成，基于 [The AI Scientist](https://github.com/SakanaAI/AI-Scientist)。"
