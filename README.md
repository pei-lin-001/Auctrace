<div align="center">
  <h1>
    <b>Auctrace: Trustworthy Automated Scientific Research</b>
  </h1>
  <p><i>可信、可审计、有韧性的自动化科研系统</i></p>
</div>

<p align="center">
  🏗️ Based on <a href="https://github.com/SakanaAI/AI-Scientist_v2">AI Scientist-v2</a> |
  � <a href="https://pub.sakana.ai/ai-scientist-v2/paper">[Original Paper]</a>
</p>

---

**Auctrace** 是基于 [AI Scientist-v2](https://github.com/SakanaAI/AI-Scientist_v2) 的增强版本，专注于解决自动化科研系统中的**可信性问题**。

当前由 LLM 驱动的自动化科研系统虽然能够完成 `idea → experiment → writeup → review` 的闭环并生成论文草稿，但论文内容的真实性仍存在根本缺陷：

- 幻觉内容可能进入论文文本
- 数字、图表和结论可能脱节
- 研究主张往往缺乏可追溯的证据绑定
- 失败实验和不稳定结果容易被错误解释

**Auctrace 的目标**：让自动化科研系统产出的论文内容更真实、更可核验、更可信——从"能生成"转向"能被相信"。

> **Caution!**
> 此代码库将执行由大语言模型（LLM）编写的代码。这种自主性存在各种风险和挑战，包括可能使用危险软件包、不受控制的网络访问以及产生意外进程的可能性。请确保在受控的沙盒环境（如 Docker 容器）中运行。请自行斟酌使用。

## Table of Contents

1.  [核心特性](#核心特性)
2.  [Requirements](#requirements)
    *   [Installation](#installation)
    *   [Supported Models and API Keys](#supported-models-and-api-keys)
3.  [Generate Research Ideas](#generate-research-ideas)
4.  [Run Experiments](#run-experiments)
5.  [Execution Backends](#execution-backends)
    *   [Local GPU](#local-gpu)
    *   [WSL SSH](#wsl-ssh)
6.  [Reliability Modules](#reliability-modules)
7.  [Project Structure](#project-structure)
8.  [Acknowledgement](#acknowledgement)
9.  [License](#license)

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

## Requirements

此代码设计在 Linux 环境下运行，需要 NVIDIA GPU（CUDA）和 PyTorch。也支持通过 WSL SSH 进行远程执行。

### Installation

```bash
# Create a new conda environment
conda create -n ai_scientist python=3.11
conda activate ai_scientist

# Install PyTorch with CUDA support (adjust pytorch-cuda version for your setup)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install PDF and LaTeX tools
conda install anaconda::poppler
conda install conda-forge::chktex

# Install Python package requirements
pip install -r requirements.txt
```

Installation usually takes no more than one hour.

### Supported Models and API Keys

#### OpenAI Models

By default, the system uses the `OPENAI_API_KEY` environment variable for OpenAI models.

#### OpenAI-compatible Models

The repository now supports **arbitrary OpenAI-compatible model names** across the v2 experiment path, ideation, plotting, writeup, LLM review, and VLM review.

Set either:

```bash
export OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
export OPENAI_API_KEY="YOUR_COMPATIBLE_KEY"
```

`OPENAI_API_BASE` is also accepted as an alias for `OPENAI_BASE_URL`.

or the explicit aliases:

```bash
export OPENAI_COMPATIBLE_BASE_URL="https://your-compatible-endpoint/v1"
export OPENAI_COMPATIBLE_API_KEY="YOUR_COMPATIBLE_KEY"
```

The repository also reads a local `.env` file at startup if present. To unify the token budget for the configured OpenAI-compatible models to 128k, set:

```bash
AI_SCIENTIST_CONTEXT_TOKENS=131072
```

This repository's request-side generation cap is controlled separately:

```bash
AI_SCIENTIST_MAX_OUTPUT_TOKENS=8192
```

When a compatible base URL is configured, plain model names such as `Qwen/Qwen3-32B` or `meta-llama/llama-3.1-70b-instruct` are sent through that OpenAI-compatible endpoint. This repository no longer maintains provider-specific SDK routes; if you want to use non-OpenAI models, expose them via your OpenAI-compatible gateway.

For the v2 tree-search path, each stage config can now declare an explicit `fallback_model`. When the primary OpenAI-compatible model exhausts the SDK's retry budget on retryable transport/provider errors, the request is retried once on the configured fallback model and the switch is logged explicitly instead of looping forever on the original model.

Example:

```yaml
agent:
  code:
    model: MiniMax-M2.5
    fallback_model: nvidia/nemotron-3-super-120b-a12b
```

#### Semantic Scholar API (Literature Search)

Our code can optionally use a Semantic Scholar API Key (`S2_API_KEY`) for higher throughput during literature search [if you have one](https://www.semanticscholar.org/product/api). This is used during both the ideation and paper writing stages. The system should work without it, though you might encounter rate limits or reduced novelty checking during ideation. If you experience issues with Semantic Scholar, you can skip the citation phase during paper generation.

#### Setting API Keys

Ensure you provide the necessary API keys as environment variables for the models you intend to use. For example:
```bash
export OPENAI_API_KEY="YOUR_OPENAI_KEY_HERE"
export OPENAI_BASE_URL="https://your-compatible-endpoint/v1"  # optional, for OpenAI-compatible APIs
export S2_API_KEY="YOUR_S2_KEY_HERE"
```

## Generate Research Ideas

Before running the full AI Scientist-v2 experiment pipeline, you first use the `ai_scientist/perform_ideation_temp_free.py` script to generate potential research ideas. This script uses an LLM to brainstorm and refine ideas based on a high-level topic description you provide, interacting with tools like Semantic Scholar to check for novelty.

1.  **Prepare a Topic Description:** Create a Markdown file (e.g., `my_research_topic.md`) describing the research area or theme you want the AI to explore. This file should contain sections like `Title`, `Keywords`, `TL;DR`, and `Abstract` to define the scope of the research. Refer to the example file `ai_scientist/ideas/i_cant_believe_its_not_better.md` for the expected structure and content format. Place your file in a location accessible by the script (e.g., the `ai_scientist/ideas/` directory).

2.  **Run the Ideation Script:** Execute the script from the main project directory, pointing it to your topic description file and specifying the desired LLM.

    ```bash
    python ai_scientist/perform_ideation_temp_free.py \
     --workshop-file "ai_scientist/ideas/my_research_topic.md" \
     --model gpt-4o-2024-05-13 \
     --max-num-generations 20 \
     --num-reflections 5
    ```
    *   `--workshop-file`: Path to your topic description Markdown file.
    *   `--model`: The LLM to use for generating ideas. This can now be any supported native model or any OpenAI-compatible model name when `OPENAI_BASE_URL` is configured.
    *   `--max-num-generations`: How many distinct research ideas to attempt generating.
    *   `--num-reflections`: How many refinement steps the LLM should perform for each idea.

3.  **Output:** The script will generate a JSON file named after your input Markdown file (e.g., `ai_scientist/ideas/my_research_topic.json`). This file will contain a list of structured research ideas, including hypotheses, proposed experiments, and related work analysis.

4.  **Proceed to Experiments:** Once you have the generated JSON file containing research ideas, you can proceed to the next section to run the experiments.

This ideation step guides the AI Scientist towards specific areas of interest and produces concrete research directions to be tested in the main experimental pipeline.

## Run Experiments

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

## Execution Backends

Auctrace 支持多种实验执行后端，通过 `bfts_config.yaml` 中的 `exec.backend` 配置：

### Local GPU

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

## Reliability Modules

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

## Project Structure

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

## Acknowledgement

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

## License

本项目继承 AI Scientist-v2 的许可证：**The AI Scientist Source Code License**（基于 Responsible AI License）。

**强制披露**：使用本代码生成的任何科学论文或手稿，必须明确披露 AI 的使用。建议在论文的 Abstract 或 Methods 部分添加：

> "This manuscript was generated using [Auctrace](https://github.com/your-repo/auctrace), based on [The AI Scientist](https://github.com/SakanaAI/AI-Scientist)."
