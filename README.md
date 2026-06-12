# Product QA Agent

> 电商商品智能问答 Agent —— 基于 BGE-M3 + ChromaDB 的多字段加权融合混合检索（语义相似度 + 结构化条件过滤），LLM 驱动查询解析，FastAPI 流式 Web 界面。
>
> AI shopping agent with hybrid retrieval: multi-field weighted fusion scoring combining BGE-M3 semantic search and structured metadata filtering, LLM-powered query parsing, FastAPI streaming UI.

---

## 项目特点

- **混合检索策略**：语义相似度（BGE-M3 Dense）+ 结构化字段加权（brand / color / category），可控性优于纯 LLM 调用
- **字段向量表预计算**：为每个字段的唯一值预先计算 embedding，查询时只对用户输入做一次编码 → 余弦相似度查表，存储与计算成本从 O(商品数) 降到 O(唯一值数)
- **价格硬过滤 + 字段软加权**：价格区间做精确过滤，其它字段通过归一化加权融合，平衡严格性与容错性
- **流式 Web 界面**：FastAPI + SSE，打字机效果输出答案，并实时展示 Agent 的解析过程和检索结果
- **模型常驻内存**：服务启动时一次性加载 BGE-M3 + ChromaDB + 字段向量表，所有请求复用，无重复加载开销

---

## 架构总览

```
浏览器 (index.html)
    │  POST /api/chat {query, top_k}
    ↓
FastAPI (server.py)
    │  启动时初始化一次 ProductAgent（模型常驻）
    │
    │  每次请求分阶段推送 SSE 事件：
    │
    │  1. status:  "正在解析查询..."
    │  2. parsed:  {semantic_query, extracted}    ← 展示 Agent 的"理解"
    │  3. status:  "正在检索商品..."
    │  4. retrieved: [Top-K 商品卡片]              ← 展示检索结果
    │  5. status:  "正在生成回答..."
    │  6. token:   "针" / "对" / ...               ← 流式打字机
    │  7. done
    ↓
浏览器逐事件渲染
```

### 检索打分公式

```
最终分数 = w_field × 字段分 + w_text × 语义分
```

- **字段分**：对 LLM 提取到的每个字段（brand / color / category），计算查询值向量与该字段唯一值向量的余弦相似度，再通过商品→字段值索引广播到每条商品；缺失字段给保守低分 0.2；多字段取平均。
- **语义分**：`semantic_query` 与商品 `search_text` 的余弦相似度（ChromaDB Top-N 召回）。
- 两项分别 min-max 归一化到 `[0,1]` 再加权（默认 `w_field = w_text = 0.5`）。
- `price` 作为硬过滤，不参与分数计算。

---

## 工作流程

```
用户查询 (中/英文)
    │
    ↓
[1] LLM 查询解析 (GLM-4-Flash)
    │   输出: semantic_query（英文完整句）+ extracted（price/brand/color/category）
    ↓
[2] 混合检索 (HybridRetriever)
    │   ├─ ChromaDB 取语义 Top-200 候选
    │   ├─ 候选集上计算字段分（向量查表 + 索引广播）
    │   ├─ 价格硬过滤
    │   ├─ min-max 归一化 → 加权融合
    │   └─ 返回 Top-K (默认 5)
    ↓
[3] LLM 答案生成 (GLM-4-Flash, stream=True)
    │   基于 Top-K 商品生成带推荐理由的中文回答，逐 token 流式输出
    ↓
浏览器渲染（解析结果 + 商品卡片 + 打字机答案）
```

---

## 目录结构

```
product-qa-agent/
├── src/                            # 运行时模块
│   ├── agent.py                    # 主 Agent 类（CLI + chat_stream）
│   ├── query_parser.py             # LLM 查询解析
│   ├── hybrid_retriever.py         # 多字段加权融合检索
│   └── answer_generator.py         # LLM 答案生成（同步 + 流式）
├── scripts/                        # 一次性数据准备脚本
│   ├── explore_data.py             # 原始数据探索
│   ├── data_governance.py          # 数据清洗与治理
│   ├── build_vector_db.py          # 生成 search_text 向量并存入 ChromaDB
│   ├── build_field_embeddings.py   # 字段唯一值向量表预计算
│   └── download_model.py           # BGE-M3 模型下载
├── web/                            # FastAPI Web 服务
│   ├── server.py
│   └── static/
│       └── index.html              # 单页前端
├── data/                           # 数据目录（gitignore）
│   ├── train-00000-of-00001.parquet    # 原始 Amazon 数据
│   ├── cleaned_products.parquet        # 治理后数据
│   ├── field_embeddings.npz            # 字段唯一值向量
│   ├── product_field_indices.parquet   # 商品→字段值索引
│   └── chroma_db/                      # ChromaDB 持久化
├── models/                         # 模型目录（gitignore）
│   └── bge-m3/
├── run_agent.py                    # CLI 入口
├── requirements.txt
├── .env.example                    # 环境变量模板
└── README.md
```

---

## 环境要求

- **Python 3.12**（推荐使用 conda 虚拟环境）
- **GPU**：BGE-M3 encode 需要 CUDA（已用 RTX A6000 测试；CPU 可运行但数据准备阶段会很慢）
- **显存**：≥ 8GB（数据构建批次大小 64，FP16）
- **磁盘空间**：
  - 原始数据 ~300MB
  - BGE-M3 模型 ~2GB
  - ChromaDB ~1GB
  - 字段向量表 ~120MB

---

## 快速开始

### 1. 安装依赖

```bash
conda create -n py312 python=3.12 -y
conda activate py312
pip install -r requirements.txt
```

### 2. 配置 API Key

复制环境变量模板并填入你的智谱 AI Key（注册地址：https://open.bigmodel.cn/）：

```bash
cp .env.example .env
# 编辑 .env，填入 ZHIPU_API_KEY
```

`.env` 内容：

```dotenv
ZHIPU_API_KEY=your_key_here
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
ZHIPU_MODEL=glm-4-flash
```

### 3. 下载 BGE-M3 模型

```bash
# 直连
python scripts/download_model.py

# 通过代理
python scripts/download_model.py --proxy http://127.0.0.1:7890
```

模型将保存至 `models/bge-m3/`。

### 4. 准备数据

将原始 Amazon 商品数据放到 `data/train-00000-of-00001.parquet`，然后依次执行：

```bash
# (可选) 探索原始数据结构
python scripts/explore_data.py

# Step 1: 数据治理 — 清洗、字段提取、拼接 search_text
python scripts/data_governance.py

# Step 2: 生成 search_text 向量，写入 ChromaDB（GPU）
python scripts/build_vector_db.py

# Step 3: 字段唯一值向量表预计算（brand / color / category）
python scripts/build_field_embeddings.py
```

完成后 `data/` 目录下应包含：

- `cleaned_products.parquet`
- `chroma_db/`
- `field_embeddings.npz`
- `product_field_indices.parquet`

### 5. 启动服务

**Web 模式**（推荐，便于演示）：

```bash
conda run -n py312 uvicorn web.server:app --host 0.0.0.0 --port 8000
```

或：

```bash
python -m web.server
```

访问 `http://localhost:8000` 即可开始对话。

**CLI 模式**（调试用）：

```bash
python run_agent.py
```

---

## API 接口

### `POST /api/chat`

SSE 流式接口，请求体：

```json
{ "query": "推荐一款 Nike 的运动鞋，预算100美元", "top_k": 5 }
```

响应 `text/event-stream`，事件类型：

| event      | data                                          | 说明                       |
| ---------- | --------------------------------------------- | -------------------------- |
| `status`   | 进度文本                                       | "正在解析查询..." 等        |
| `parsed`   | `{semantic_query, extracted}` JSON            | Agent 的解析结果           |
| `retrieved`| `[商品卡片 JSON]`                              | Top-K 检索结果             |
| `token`    | 答案文本片段                                   | 逐 token 推送（打字机效果） |
| `done`     | 空                                            | 完成                       |
| `error`    | 错误信息                                       | 异常                       |

### `GET /api/health`

健康检查，返回 `{"status": "ok"}`。

---

## 配置说明

| 环境变量           | 必填 | 默认值                                          | 说明                |
| ------------------ | ---- | ----------------------------------------------- | ------------------- |
| `ZHIPU_API_KEY`    | 是   | -                                               | 智谱 AI API Key     |
| `ZHIPU_BASE_URL`   | 否   | `https://open.bigmodel.cn/api/paas/v4/`         | LLM 服务地址        |
| `ZHIPU_MODEL`      | 否   | `glm-4-flash`                                   | 模型名              |

检索器内部参数（在 `src/hybrid_retriever.py` 顶部可调）：

- `DEFAULT_W_FIELD` / `DEFAULT_W_TEXT`：字段分 / 语义分权重（默认 0.5 / 0.5）
- `DEFAULT_TOP_K`：返回商品数（默认 5）
- `n_candidates`：ChromaDB 召回候选数（默认 200）
- `MISSING_FIELD_SCORE`：商品字段缺失时的保守低分（默认 0.2）

---

## 技术栈

| 类别        | 选型                                       |
| ----------- | ------------------------------------------ |
| Embedding   | BGE-M3（仅用 Dense，1024 维）              |
| 向量库      | ChromaDB（cosine，HNSW）                   |
| LLM         | 智谱 GLM-4-Flash（通过 OpenAI SDK 调用）   |
| Web 框架    | FastAPI + SSE                              |
| 数据处理    | pandas + numpy                             |
| 深度学习    | PyTorch（CUDA）                            |

---

## 已知限制

- **无 rerank 阶段**：当前流程是召回 + 融合打分，未引入 cross-encoder 精排。某些场景（如 "徒步背包"）会因语义干扰影响排序质量。
- **`color` 字段噪声较多**：数据中 `detail_color` 有 13K+ 唯一值且写法不规范，依赖 embedding 的自然聚类效果；缺失字段统一给 0.2 保守分。
- **价格缺失商品保留**：原始数据 price 缺失率约 30%，这部分商品不会被价格过滤排除，但会标注"价格待定"。

---

## 示例查询

```
200美元以内适合徒步的红色背包
推荐一款蓝牙耳机
Nike 的运动鞋，预算100美元
waterproof jacket under 100 dollars
适合户外露营的防水帐篷
```
