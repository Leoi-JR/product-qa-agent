# Product QA Agent

https://github.com/user-attachments/assets/339c1929-cadf-4bd4-be8d-6334e4ccb8a3

> 电商商品智能问答 Agent —— 基于 BGE-M3 + ChromaDB 的混合检索（语义相似度 + 结构化条件过滤），LLM 驱动查询解析，FastAPI 流式 Web 界面。
>
> AI shopping agent with hybrid retrieval: multi-field fusion scoring combining BGE-M3 semantic search and structured metadata filtering, LLM-powered query parsing, FastAPI streaming UI.

---

## 项目特点

- **混合检索策略**：结合语义相似度（BGE-M3 Dense）与结构化字段过滤（品牌 / 颜色 / 分类），实现精准且具备长尾泛化能力的商品召回。
- **高性能向量预计算**：通过预构建字段级唯一值向量表，显著降低实时计算开销与存储成本。
- **多维度排序融合**：结合价格区间精确过滤与多字段特征软加权，平衡推荐的准确性与多样性。
- **流式 Web 界面体验**：采用 FastAPI + SSE，实现打字机效果的流式输出，并实时渲染骨架屏动效与检索卡片。
- **高效资源利用**：服务启动时一次性加载 BGE-M3 与 ChromaDB 实例，全链路请求复用，确保低延迟响应。

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
[1] LLM 查询解析 (GLM-4-Flash)
    │   输出: semantic_query（英文核心语义）+ extracted（price/brand/color/category）
    ↓
[2] 混合检索引擎 (HybridRetriever)
    │   ├─ ChromaDB 执行语义 Top-K 初筛
    │   ├─ 结构化字段条件校验与多维度加权打分
    │   ├─ 价格区间硬过滤
    │   └─ 返回精选 Top-K 商品集合
    ↓
[3] LLM 答案生成 (GLM-4-Flash, stream=True)
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
│   └── download_model.py           # 本地化部署 BGE-M3 模型
├── web/                            # Web 服务层
│   ├── server.py                   # FastAPI SSE 接口层
│   └── static/
│       └── index.html              # 纯前端交互界面
├── data/                           # 数据集与数据库目录（需通过脚本生成）
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

本项目默认使用智谱 AI 驱动 LLM 节点，复制环境变量模板并填入你的 Key（注册地址：https://open.bigmodel.cn/）：

```bash
cp .env.example .env
# 编辑 .env，填入 ZHIPU_API_KEY
```

### 3. 下载模型与数据准备

```bash
# 下载本地 BGE-M3 模型
python scripts/download_model.py

# 将原始商品数据放置于 data/train-00000-of-00001.parquet 后依次执行数据管线：
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
