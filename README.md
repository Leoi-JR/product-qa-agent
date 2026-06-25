# Product QA Agent

https://github.com/user-attachments/assets/339c1929-cadf-4bd4-be8d-6334e4ccb8a3

> 电商商品智能问答 Agent —— 基于 BGE-M3 + ChromaDB 的混合检索（语义相似度 + 结构化条件过滤），LLM 驱动查询解析，FastAPI 流式 Web 界面。
>
> AI shopping agent with hybrid retrieval: multi-field fusion scoring combining BGE-M3 semantic search and structured metadata filtering, LLM-powered query parsing, FastAPI streaming UI.

---

## 项目特点

- **数据治理与特征工程**：基于 11.7 万条 Amazon 商品记录构建清洗管线。解析 categories 层级逻辑填补 2.4 万条主要类目缺失；提取并格式化嵌套字典中的颜色、材质等离散属性；针对 30% 价格缺失与异常零值进行标识统一；将多字段重组拼接为平均长度约 1800 字符的语义文本，为向量模型提供输入。
- **混合检索策略**：结合语义相似度（BGE-M3 Dense）与结构化字段过滤（品牌 / 颜色 / 分类），实现精准且具备长尾泛化能力的商品召回。
- **高性能向量预计算**：通过预构建字段级唯一值向量表，显著降低实时计算开销与存储成本。
- **多维度排序融合**：结合价格区间精确过滤与多字段特征软加权，平衡推荐的准确性与多样性。
- **流式 Web 界面体验**：采用 FastAPI + SSE，实现打字机效果的流式输出，并实时渲染骨架屏动效与检索卡片。
- **高效资源利用**：服务启动时一次性加载 BGE-M3 与 ChromaDB 实例，全链路请求复用，确保低延迟响应。
- **LLM 全链路可观测**：集成 Langfuse，对 query 解析、混合检索、答案生成全链路进行 trace/span 追踪，支持云端与自托管两种接入方式。

---

## 架构总览

```text
浏览器 (index.html)
    │  POST /api/chat {query, top_k}
    ↓
FastAPI (server.py)
    │  启动时初始化一次 ProductAgent（模型常驻）
    │
    │  每次请求分阶段推送 SSE 事件：
    │
    │  1. status:  "正在解析查询..."
    │  2. parsed:  {semantic_query, extracted}    ← 展示 Agent 的意图识别
    │  3. status:  "正在检索商品..."
    │  4. retrieved: [Top-K 商品卡片]              ← 实时渲染检索结果
    │  5. status:  "正在生成回答..."
    │  6. token:   "针" / "对" / ...               ← 流式打字机输出
    │  7. done
    ↓
浏览器逐事件渲染响应
```

---

## 工作流程

```text
用户自然语言查询 (支持中/英双语)
    │
    ↓
[1] LLM 查询解析 (DeepSeek-V3.2 via 千帆)
    │   输出: semantic_query（英文核心语义）+ extracted（price/brand/color/category）
    ↓
[2] 混合检索引擎 (HybridRetriever)
    │   ├─ ChromaDB 执行语义 Top-K 初筛
    │   ├─ 结构化字段条件校验与多维度加权打分
    │   ├─ 价格区间硬过滤
    │   └─ 返回精选 Top-K 商品集合
    ↓
[3] LLM 答案生成 (DeepSeek-V3.2 via 千帆, stream=True)
    │   基于检索到的事实型商品数据，生成带推荐理由的中文回答，逐 token 流式输出
    ↓
前端动态渲染（骨架屏过渡 + 渐进式卡片 + Markdown 流式解析）
```

---

## 目录结构

```text
product-qa-agent/
├── src/                            # 核心业务逻辑模块
│   ├── agent.py                    # 问答 Agent 主调度逻辑
│   ├── query_parser.py             # 用户意图与查询参数解析
│   ├── hybrid_retriever.py         # 混合检索核心引擎
│   └── answer_generator.py         # 推荐理由生成与格式化
├── scripts/                        # 数据工程与准备脚本
│   ├── data_governance.py          # 原始数据清洗与特征工程
│   ├── build_vector_db.py          # 构建并持久化 ChromaDB 向量库
│   ├── build_field_embeddings.py   # 构建字段级特征向量表
│   ├── download_model.py           # 本地化部署 BGE-M3 模型
│   ├── build_benchmark_dataset.py  # 构建检索评估 benchmark（750 条）
│   └── run_retrieval_eval.py       # 多策略检索指标评估（Recall / MRR）
├── web/                            # Web 服务层
│   ├── server.py                   # FastAPI SSE 接口层
│   └── static/
│       └── index.html              # 纯前端交互界面
├── infra/
│   └── langfuse/                   # Langfuse 自托管部署（Docker Compose）
├── data/                           # 数据集与数据库目录（需通过脚本生成）
│   ├── chroma_db/                  # ChromaDB 向量库
│   ├── field_embeddings.npz        # 字段级唯一值向量表
│   ├── product_field_indices.parquet
│   ├── cleaned_products.parquet
│   └── eval_results/               # 检索评估结果
├── models/                         # 本地模型文件存储目录
├── run_agent.py                    # 命令行 CLI 交互入口
├── requirements.txt                # 依赖清单
└── .env.example                    # 环境变量配置模板
```

---

## 环境要求

- **Python 3.12**（推荐使用 conda 虚拟环境）
- **硬件支持**：推荐配备 GPU 以加速 BGE-M3 编码过程
- **显存与磁盘**：数据构建阶段建议 8GB+ 显存；整体项目需预留约 4GB 磁盘空间（模型与向量库）

---

## 快速开始

### 1. 安装依赖

```bash
conda create -n py312 python=3.12 -y
conda activate py312
pip install -r requirements.txt
```

### 2. 配置 API Key

本项目使用**百度千帆**（DeepSeek-V3.2）驱动 LLM 节点，复制环境变量模板并填入配置（注册地址：https://console.bce.baidu.com/qianfan/）：

```bash
cp .env.example .env
# 编辑 .env，填入以下必填项：
# QIANFAN_API_KEY  —— 千帆 API Key
# QIANFAN_BASE_URL —— https://qianfan.baidubce.com/v2
# QIANFAN_MODEL    —— deepseek-v3.2（默认）
```

> **Langfuse 可观测（可选）**：如需追踪 LLM 调用链路，在 `.env` 中额外填入 `LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_HOST`。未配置时自动跳过，不影响功能。

### 3. 下载模型与数据准备

原始数据集来自 [Amazon Product Reviews（Hugging Face）](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)，下载 `train-00000-of-00001.parquet` 后放置于 `data/` 目录，再依次执行数据管线：

```bash
# 下载本地 BGE-M3 模型
python scripts/download_model.py

# 数据清洗 → 构建向量库 → 构建字段向量表
python scripts/data_governance.py
python scripts/build_vector_db.py
python scripts/build_field_embeddings.py
```

### 4. 启动服务

**Web 模式**（推荐体验）：

```bash
conda run -n py312 uvicorn web.server:app --host 0.0.0.0 --port 8000
```

启动后访问 `http://localhost:8000` 即可开始智能电商对话。

**CLI 模式**（调试用）：

```bash
python run_agent.py
```

---

## 评估体系

### Benchmark 构建

通过「反向构造 + 多路召回 + LLM Judge 判定」方法构建 750 条检索评估测试集：

```text
[阶段1] 分层采样 150 个 anchor 商品（5 种 query 类型 × 各 30 个）
[阶段2] LLM 反向构造 query（每商品 5 条，共 750 条）
[阶段3] 三路召回建候选池（dense + BM25 + M3-sparse，每路 top-20）
[阶段4] LLM Judge 并发判定相关性（~3 万次调用）
[阶段5] 输出 benchmark_dataset.jsonl（750 行，含 relevant_ids 金标准）
```

```bash
# 构建 benchmark（支持 --stage N 断点续跑）
python scripts/build_benchmark_dataset.py

# 运行多策略对比评估
python scripts/run_retrieval_eval.py
```

### 检索指标对比

| 指标 | 策略A（字段加权生产链路）| 策略B（dense+sparse RRF）| 策略C（纯 dense 对照）|
|------|:-:|:-:|:-:|
| Recall@5 | 0.2635 | 0.2414 | 0.3535 |
| MRR@5 | **0.7623** | 0.7162 | 0.9159 |
| Recall@20 | 0.4939 | 0.4617 | 0.7590 |
| MRR@20 | **0.7700** | 0.7255 | 0.9179 |

> **注**：策略C 的 Recall 虚高源于候选池构建时的同路重叠偏差，并非真实生产效果。策略A 的 MRR@5 = 0.76 说明相关结果确实排在前列，字段加权在精确约束查询（如「Nike 运动鞋预算 100 美元」）场景下优于纯语义检索。

---

## LLM 可观测性（Langfuse）

项目在 `query_parser`、`hybrid_retriever`、`answer_generator` 全链路挂载了 `@observe` 装饰器，所有 LLM 调用通过 `langfuse.openai.OpenAI` 上报 trace。

### 接入方式

**方式一：Langfuse Cloud**（零部署，直接用）

在 [cloud.langfuse.com](https://cloud.langfuse.com) 注册项目，将密钥填入 `.env`：

```bash
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

**方式二：自托管**（数据不出本地）

使用 `infra/langfuse/` 提供的 Docker Compose 配置，在宿主机一键启动完整 Langfuse v3 服务（6 个容器）：

```bash
cd infra/langfuse
cp .env.example .env   # 填入各服务密钥
./setup-data-dirs.sh   # 初始化数据目录
docker compose up -d
```

启动后访问 `http://<宿主机IP>:3001`，注册账号并创建项目，将拿到的密钥回填到项目根 `.env`。详细部署说明见 [`infra/langfuse/README.md`](infra/langfuse/README.md)。
