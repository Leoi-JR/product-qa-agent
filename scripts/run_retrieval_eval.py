"""
召回策略评估脚本

对 benchmark_dataset.jsonl 中的 750 条 query，分别用三种召回策略跑检索，
计算 Recall@K 和 MRR@K 指标，输出策略对比报告。

三种策略：
  A: 生产链路（LLM 解析 query → HybridRetriever 字段加权融合）
  B: dense + M3-sparse 双路 → RRF 融合（全量 sparse 倒排索引）
  C: 纯 dense（ChromaDB 向量检索，对照组）

用法：
  # 跑全部策略，K=5,10,20
  conda run -n py312 python scripts/run_retrieval_eval.py

  # 只跑 B 和 C，支持断点续跑
  conda run -n py312 python scripts/run_retrieval_eval.py --strategy B,C --resume

  # 冒烟测试（只跑前 10 条 query）
  conda run -n py312 python scripts/run_retrieval_eval.py --smoke 10

输出到 data/eval_results/：
  eval_results_raw.jsonl    各策略原始检索结果（断点续跑用）
  eval_metrics_summary.md   指标汇总表（策略 × K × query_type）
  eval_metrics_detail.csv   逐条 query 指标明细
"""

import os
import sys
import json
import time
import math
import pickle
import logging
import argparse
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

# ── 路径定位 ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EVAL_DIR = os.path.join(DATA_DIR, "eval_results")
BENCHMARK_PATH = os.path.join(DATA_DIR, "benchmark_dataset.jsonl")
# Strategy B 的 sparse 倒排索引缓存（直接复用 build_benchmark_dataset.py 的路径）
SPARSE_INDEX_PATH = os.path.join(DATA_DIR, "benchmark_work", "sparse_index.pkl")
RAW_RESULTS_PATH = os.path.join(EVAL_DIR, "eval_results_raw.jsonl")

sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── 配置 ───────────────────────────────────────────────────────────
QIANFAN_BASE_URL = os.environ.get("QIANFAN_BASE_URL", "https://qianfan.baidubce.com/v2")
QIANFAN_API_KEY  = os.environ.get("QIANFAN_API_KEY", "")
QIANFAN_MODEL    = os.environ.get("QIANFAN_MODEL", "deepseek-v3.2")
LLM_RPM          = int(os.environ.get("LLM_RPM", "2000"))
LLM_CONCURRENCY  = int(os.environ.get("LLM_CONCURRENCY", "32"))
LLM_THINKING     = os.environ.get("LLM_THINKING", "false").lower() in ("1", "true", "yes")
LLM_MAX_RETRIES  = int(os.environ.get("LLM_MAX_RETRIES", "4"))

DEFAULT_K_VALUES  = [5, 10, 20]
DEFAULT_N_CAND    = 200   # Strategy A HybridRetriever 候选数
RRF_K             = 60    # RRF 参数（常用值）


# ════════════════════════════════════════════════════════════════════
# 限速器（复用 build_benchmark_dataset.py 的设计）
# ════════════════════════════════════════════════════════════════════

class _RateLimiter:
    def __init__(self, rpm: int):
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._next_allowed = 0.0
        self._lock = threading.Lock()

    def acquire(self):
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.time()
            wait = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self.min_interval
        if wait > 0:
            time.sleep(wait)


_RATE_LIMITER = _RateLimiter(LLM_RPM)


def _retry_after_seconds(exc):
    if getattr(exc, "status_code", None) != 429:
        return None
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    for key in ("Retry-After", "retry-after", "x-ratelimit-reset"):
        val = headers.get(key)
        if val:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return 30.0


def _llm_chat(client, *, model, messages, timeout=120, **kwargs):
    extra_body = dict(kwargs.pop("extra_body", {}))
    extra_body.setdefault("enable_thinking", LLM_THINKING)
    for attempt in range(LLM_MAX_RETRIES + 1):
        _RATE_LIMITER.acquire()
        try:
            return client.chat.completions.create(
                model=model, messages=messages,
                timeout=timeout, extra_body=extra_body, **kwargs,
            )
        except Exception as e:
            if attempt >= LLM_MAX_RETRIES:
                raise
            ra = _retry_after_seconds(e)
            time.sleep((ra + 2.0) if ra is not None else 2.0 * (attempt + 1))


# ════════════════════════════════════════════════════════════════════
# 指标计算
# ════════════════════════════════════════════════════════════════════

def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Recall@K：Top-K 中覆盖了多少 relevant（占 relevant 总数的比例）。"""
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / len(relevant)


def mrr_at_k(retrieved: list, relevant: set, k: int) -> float:
    """MRR@K：Top-K 中第一个 relevant 的倒数排名。"""
    for rank, r in enumerate(retrieved[:k], start=1):
        if r in relevant:
            return 1.0 / rank
    return 0.0


def compute_metrics(retrieved: list, relevant_ids: list, k_values: list) -> dict:
    """计算所有 K 值的 Recall 和 MRR。"""
    relevant = set(relevant_ids)
    metrics = {}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        metrics[f"mrr@{k}"]    = mrr_at_k(retrieved, relevant, k)
    return metrics


# ════════════════════════════════════════════════════════════════════
# Strategy C：纯 dense（ChromaDB）
# ════════════════════════════════════════════════════════════════════

class StrategyC:
    """纯 dense 检索：直接用原始 query 文本编码后查 ChromaDB。"""

    def __init__(self, model, collection):
        self.model = model
        self.collection = collection

    def retrieve(self, query: str, top_k: int) -> list:
        try:
            q_vec = self.model.encode([query], max_length=8192)["dense_vecs"][0].astype("float32")
            result = self.collection.query(
                query_embeddings=q_vec.tolist(),
                n_results=min(top_k, self.collection.count()),
                include=["distances"],
            )
            return result["ids"][0]
        except Exception as e:
            logger.warning("Strategy C 失败 (query=%s...): %s", query[:30], e)
            return []


# ════════════════════════════════════════════════════════════════════
# Strategy B：dense + M3-sparse 双路 → RRF
# ════════════════════════════════════════════════════════════════════

class SparseRetriever:
    """
    从倒排索引中检索：加载已有的 sparse_index.pkl，
    查询时对 query sparse 向量做倒排索引点积打分。
    """

    def __init__(self, model, df: pd.DataFrame, cache_path: str, rebuild: bool = False):
        self.model = model
        self.ids = [f"prod_{i}" for i in df.index]
        self.cache_path = cache_path

        if not rebuild and os.path.exists(cache_path):
            logger.info("Sparse: 加载倒排索引缓存 %s", cache_path)
            with open(cache_path, "rb") as f:
                self.inverted_index, cached_ids = pickle.load(f)
            if cached_ids == self.ids:
                logger.info("Sparse 缓存命中（%d 个 token）", len(self.inverted_index))
                return
            logger.warning("Sparse 缓存 id 不匹配，重新编码")

        logger.info("Sparse: 对全库编码 lexical_weights（%d 条）…", len(df))
        texts = df["search_text"].fillna("").astype(str).tolist()
        batch_size, max_length = 128, 1024
        all_sparse = []
        t0 = time.time()
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            out = model.encode(
                batch, return_dense=False, return_sparse=True,
                return_colbert_vecs=False, max_length=max_length,
            )
            for lw in out["lexical_weights"]:
                all_sparse.append({str(k): float(v) for k, v in lw.items()})
            if (i // batch_size + 1) % 50 == 0:
                done = i + len(batch)
                elapsed = time.time() - t0
                logger.info(
                    "  Sparse 编码进度 %d/%d (%.1fs, %.1f 条/秒)",
                    done, len(texts), elapsed, done / elapsed,
                )

        logger.info("Sparse: 构建倒排索引…")
        inverted_index: dict = defaultdict(list)
        for doc_idx, doc_lw in enumerate(all_sparse):
            for token_id, weight in doc_lw.items():
                inverted_index[token_id].append((doc_idx, weight))
        self.inverted_index = dict(inverted_index)
        del all_sparse

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump((self.inverted_index, self.ids), f)
        logger.info(
            "Sparse 编码完成并缓存（耗时 %.1fs，%d 个 token）",
            time.time() - t0, len(self.inverted_index),
        )

    def retrieve(self, query: str, top_k: int) -> list:
        out = self.model.encode(
            [query], return_dense=False, return_sparse=True,
            return_colbert_vecs=False, max_length=256,
        )
        q_lw = out["lexical_weights"][0]
        q_sparse = {str(k): float(v) for k, v in q_lw.items()}
        if not q_sparse:
            return []
        scores: dict = defaultdict(float)
        for token_id, q_weight in q_sparse.items():
            postings = self.inverted_index.get(token_id)
            if not postings:
                continue
            for doc_idx, doc_weight in postings:
                scores[doc_idx] += q_weight * doc_weight
        if not scores:
            return []
        sorted_docs = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [self.ids[idx] for idx, _ in sorted_docs]


def rrf_merge(ranked_lists: list[list], k: int = RRF_K) -> list:
    """
    倒数排名融合（Reciprocal Rank Fusion）。
    ranked_lists: 每个元素是一个有序 id 列表（已按各自策略排好序）。
    返回融合后的有序 id 列表。
    """
    scores: dict = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: -x[1])]


class StrategyB:
    """dense + M3-sparse 双路 → RRF 融合。"""

    def __init__(self, model, collection, sparse_retriever: SparseRetriever, n_candidates: int):
        self.model = model
        self.collection = collection
        self.sparse = sparse_retriever
        self.n_candidates = n_candidates

    def retrieve(self, query: str, top_k: int) -> list:
        try:
            # Dense 路
            q_vec = self.model.encode([query], max_length=8192)["dense_vecs"][0].astype("float32")
            dense_result = self.collection.query(
                query_embeddings=q_vec.tolist(),
                n_results=min(self.n_candidates, self.collection.count()),
                include=["distances"],
            )
            dense_ids = dense_result["ids"][0]
        except Exception as e:
            logger.warning("Strategy B dense 失败 (query=%s...): %s", query[:30], e)
            dense_ids = []

        # Sparse 路
        sparse_ids = self.sparse.retrieve(query, top_k=self.n_candidates)

        # RRF 融合，取 Top-K
        merged = rrf_merge([dense_ids, sparse_ids])
        return merged[:top_k]


# ════════════════════════════════════════════════════════════════════
# Strategy A：HybridRetriever（生产链路）
# ════════════════════════════════════════════════════════════════════

def _parse_query_llm(client, query: str, max_retries: int = 3):
    """用 LLM 解析 query → {semantic_query, extracted}，带限速和重试。"""
    from src.query_parser import PARSER_SYSTEM_PROMPT, PARSER_USER_TEMPLATE
    messages = [
        {"role": "system", "content": PARSER_SYSTEM_PROMPT},
        {"role": "user", "content": PARSER_USER_TEMPLATE.format(query=query)},
    ]
    for attempt in range(max_retries + 1):
        try:
            resp = _llm_chat(
                client, model=QIANFAN_MODEL, messages=messages,
                temperature=0.1, response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            if not content:
                raise ValueError("空 content")
            data = json.loads(content)
            extracted = data.get("extracted", {})
            for key in ["price_min", "price_max", "brand", "color", "category"]:
                extracted.setdefault(key, None)
            return {
                "semantic_query": data.get("semantic_query", query),
                "extracted": extracted,
            }
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
            else:
                logger.warning("parse_query 失败 (query=%s...): %s", query[:30], e)
                return None
    return None


class StrategyA:
    """生产链路：LLM 解析 → HybridRetriever 多字段加权融合。"""

    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm_client = llm_client
        self._cache: dict = {}   # query -> parsed dict

    def preparse(self, queries: list, concurrency: int = LLM_CONCURRENCY):
        """并发预解析所有 query，减少评估时的串行等待。"""
        unique_queries = list({q["query"] for q in queries})
        logger.info("Strategy A: 并发预解析 %d 条 query（并发 %d）", len(unique_queries), concurrency)
        t0 = time.time()
        completed = 0
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(_parse_query_llm, self.llm_client, q): q
                for q in unique_queries
            }
            for future in as_completed(futures):
                q = futures[future]
                try:
                    self._cache[q] = future.result()
                except Exception as e:
                    logger.warning("preparse future 异常: %s", e)
                    self._cache[q] = None
                completed += 1
                if completed % 50 == 0 or completed == len(unique_queries):
                    logger.info(
                        "  preparse 进度 %d/%d (%.1fs)",
                        completed, len(unique_queries), time.time() - t0,
                    )
        ok = sum(1 for v in self._cache.values() if v)
        logger.info("Strategy A 预解析完成: %d/%d 成功", ok, len(unique_queries))

    def retrieve(self, query: str, top_k: int) -> list:
        parsed = self._cache.get(query)
        if parsed is None:
            parsed = _parse_query_llm(self.llm_client, query)
        if not parsed:
            return []
        try:
            results = self.retriever.retrieve(
                semantic_query=parsed["semantic_query"],
                extracted=parsed["extracted"],
                top_k=top_k,
                n_candidates=DEFAULT_N_CAND,
            )
            return [r["id"] for r in results]
        except Exception as e:
            logger.warning("Strategy A retrieve 失败 (query=%s...): %s", query[:30], e)
            return []


# ════════════════════════════════════════════════════════════════════
# 主评估循环
# ════════════════════════════════════════════════════════════════════

def load_benchmark(path: str, smoke: int = 0) -> list:
    """加载 benchmark_dataset.jsonl，返回样本列表。"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if smoke > 0:
        samples = samples[:smoke]
        logger.info("冒烟模式：只取前 %d 条 query", smoke)
    logger.info("加载 benchmark：共 %d 条 query", len(samples))
    return samples


def load_existing_results(path: str) -> dict:
    """
    加载已有的 eval_results_raw.jsonl，返回
    {strategy -> {query -> retrieved_ids_list}}
    用于断点续跑时跳过已完成的条目。
    """
    existing: dict = defaultdict(dict)
    if not os.path.exists(path):
        return existing
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                strategy = record["strategy"]
                query = record["query"]
                existing[strategy][query] = record["retrieved_ids"]
            except Exception:
                pass
    total = sum(len(v) for v in existing.values())
    logger.info("断点续跑：已载入 %d 条已完成结果", total)
    return existing


def run_strategy(
    strategy_name: str,
    strategy_obj,
    samples: list,
    k_values: list,
    existing: dict,
    raw_out_f,
    max_k: int,
    preparse_fn=None,
) -> dict:
    """
    对单个策略跑全量 benchmark，收集 retrieved_ids，计算指标。
    返回 {query -> metrics_dict}。

    raw_out_f: 打开的文件对象，用于写入 eval_results_raw.jsonl（支持断点续跑）。
    """
    done_queries = existing.get(strategy_name, {})
    todo = [s for s in samples if s["query"] not in done_queries]
    logger.info(
        "[策略 %s] 共 %d 条，已完成 %d，待执行 %d",
        strategy_name, len(samples), len(done_queries), len(todo),
    )

    # Strategy A 需要先并发预解析
    if preparse_fn is not None and todo:
        preparse_fn(todo)

    t0 = time.time()
    for i, sample in enumerate(todo):
        query = sample["query"]
        retrieved = strategy_obj.retrieve(query, top_k=max_k)
        done_queries[query] = retrieved

        # 写入 raw 文件（逐行 append，宕机也不丢）
        record = {
            "strategy": strategy_name,
            "query": query,
            "query_type": sample["query_type"],
            "retrieved_ids": retrieved,
            "relevant_ids": sample["relevant_ids"],
        }
        raw_out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        raw_out_f.flush()

        if (i + 1) % 50 == 0 or i == len(todo) - 1:
            logger.info(
                "  [策略 %s] 进度 %d/%d (%.1fs)",
                strategy_name, i + 1, len(todo), time.time() - t0,
            )

    # 计算指标
    all_metrics: dict = {}
    for sample in samples:
        query = sample["query"]
        retrieved = done_queries.get(query, [])
        metrics = compute_metrics(retrieved, sample["relevant_ids"], k_values)
        metrics["query_type"] = sample["query_type"]
        metrics["anchor_id"]  = sample["anchor_id"]
        metrics["retrieved_count"] = len(retrieved)
        all_metrics[query] = metrics

    logger.info("[策略 %s] 检索完成，耗时 %.1fs", strategy_name, time.time() - t0)
    return all_metrics


# ════════════════════════════════════════════════════════════════════
# 报告生成
# ════════════════════════════════════════════════════════════════════

def aggregate_metrics(
    all_strategy_metrics: dict,   # {strategy -> {query -> metrics_dict}}
    k_values: list,
    strategies: list,
) -> dict:
    """
    聚合指标：整体 + 按 query_type 分类。
    返回 {strategy -> {"overall": {...}, "by_type": {type -> {...}}}}
    """
    query_types = ["brand_model", "natural_desc", "cross_lang", "constrained", "vague"]
    agg = {}
    for strategy in strategies:
        metrics_map = all_strategy_metrics[strategy]
        agg[strategy] = {"overall": {}, "by_type": {}}
        all_values: dict = defaultdict(list)
        by_type_values: dict = defaultdict(lambda: defaultdict(list))

        for query, m in metrics_map.items():
            qt = m["query_type"]
            for k in k_values:
                for metric in [f"recall@{k}", f"mrr@{k}"]:
                    all_values[metric].append(m[metric])
                    by_type_values[qt][metric].append(m[metric])

        for metric, vals in all_values.items():
            agg[strategy]["overall"][metric] = float(np.mean(vals))
        for qt, type_vals in by_type_values.items():
            agg[strategy]["by_type"][qt] = {
                metric: float(np.mean(vals)) for metric, vals in type_vals.items()
            }
    return agg


def write_summary_markdown(agg: dict, k_values: list, strategies: list, output_path: str):
    """生成 eval_metrics_summary.md。"""
    lines = [
        "# 召回策略评估报告",
        "",
        f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 策略说明",
        "",
        "| 策略 | 描述 |",
        "|------|------|",
        "| **A** | 生产链路：LLM 解析 query → HybridRetriever 多字段加权融合（dense Top-200 候选 + brand/color/category 字段 embedding 加权）|",
        "| **B** | dense + M3-sparse 双路 → RRF 融合（全量 sparse 倒排索引，RRF K=60）|",
        "| **C** | 纯 dense 对照组：原始 query 直接 ChromaDB 向量检索 |",
        "",
    ]

    for k in k_values:
        lines += [
            f"## Recall@{k} 整体对比",
            "",
            "| Query类型 | 策略A | 策略B | 策略C |",
            "|-----------|-------|-------|-------|",
        ]
        query_types_order = [
            "brand_model", "natural_desc", "cross_lang", "constrained", "vague", "**整体**"
        ]
        for qt_label in query_types_order:
            qt = qt_label.strip("*")
            if qt == "整体":
                row = [qt_label]
                for s in strategies:
                    val = agg[s]["overall"].get(f"recall@{k}", 0.0)
                    row.append(f"**{val:.4f}**")
            else:
                row = [qt_label]
                for s in strategies:
                    val = agg[s]["by_type"].get(qt, {}).get(f"recall@{k}", 0.0)
                    row.append(f"{val:.4f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        lines += [
            f"## MRR@{k} 整体对比",
            "",
            "| Query类型 | 策略A | 策略B | 策略C |",
            "|-----------|-------|-------|-------|",
        ]
        for qt_label in query_types_order:
            qt = qt_label.strip("*")
            if qt == "整体":
                row = [qt_label]
                for s in strategies:
                    val = agg[s]["overall"].get(f"mrr@{k}", 0.0)
                    row.append(f"**{val:.4f}**")
            else:
                row = [qt_label]
                for s in strategies:
                    val = agg[s]["by_type"].get(qt, {}).get(f"mrr@{k}", 0.0)
                    row.append(f"{val:.4f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines += [
        "## 完整指标汇总（整体平均）",
        "",
        "| 指标 | 策略A | 策略B | 策略C |",
        "|------|-------|-------|-------|",
    ]
    for k in k_values:
        for metric_name in [f"recall@{k}", f"mrr@{k}"]:
            row = [metric_name]
            for s in strategies:
                val = agg[s]["overall"].get(metric_name, 0.0)
                row.append(f"{val:.4f}")
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("指标汇总报告已写入: %s", output_path)


def write_detail_csv(
    all_strategy_metrics: dict,
    k_values: list,
    strategies: list,
    samples: list,
    output_path: str,
):
    """生成逐条 query 明细 CSV。"""
    rows = []
    query_to_sample = {s["query"]: s for s in samples}
    for query, sample in query_to_sample.items():
        row = {
            "query": query,
            "query_type": sample["query_type"],
            "anchor_id": sample["anchor_id"],
            "relevant_count": len(sample["relevant_ids"]),
        }
        for strategy in strategies:
            m = all_strategy_metrics[strategy].get(query, {})
            row[f"strategy_{strategy}_retrieved_count"] = m.get("retrieved_count", 0)
            for k in k_values:
                row[f"strategy_{strategy}_recall@{k}"] = round(m.get(f"recall@{k}", 0.0), 4)
                row[f"strategy_{strategy}_mrr@{k}"]    = round(m.get(f"mrr@{k}", 0.0), 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("逐条明细 CSV 已写入: %s", output_path)


# ════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="召回策略评估")
    parser.add_argument(
        "--strategy", default="A,B,C",
        help="要评估的策略，逗号分隔，例如 A,B,C 或 B,C（默认: A,B,C）",
    )
    parser.add_argument(
        "--top-k", default="5,10,20",
        help="评估 K 值，逗号分隔（默认: 5,10,20）",
    )
    parser.add_argument(
        "--n-candidates", type=int, default=DEFAULT_N_CAND,
        help="Strategy A/B 的候选集大小（默认: 200）",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="断点续跑：跳过已完成的 query，追加写入 eval_results_raw.jsonl",
    )
    parser.add_argument(
        "--rebuild-sparse", action="store_true",
        help="强制重新编码全库 sparse 向量（忽略缓存）",
    )
    parser.add_argument(
        "--smoke", type=int, default=0,
        help="冒烟测试：只跑前 N 条 query（默认: 0 = 跑全部）",
    )
    parser.add_argument(
        "--concurrency", type=int, default=LLM_CONCURRENCY,
        help="Strategy A LLM 解析并发数（默认读 .env LLM_CONCURRENCY）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    strategies = [s.strip().upper() for s in args.strategy.split(",")]
    k_values   = [int(k) for k in args.top_k.split(",")]
    max_k      = max(k_values)

    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "benchmark_work"), exist_ok=True)

    logger.info("=" * 60)
    logger.info("召回策略评估启动")
    logger.info("策略: %s | K 值: %s | 断点续跑: %s | 冒烟: %s",
                strategies, k_values, args.resume, args.smoke or "否")
    logger.info("=" * 60)

    # ── 加载 benchmark ─────────────────────────────────────────────
    samples = load_benchmark(BENCHMARK_PATH, smoke=args.smoke)

    # ── 断点续跑：加载已有结果 ─────────────────────────────────────
    existing: dict = {}
    if args.resume:
        existing = load_existing_results(RAW_RESULTS_PATH)

    # ── 加载公共资源 ───────────────────────────────────────────────
    logger.info("加载 BGE-M3 模型…")
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel(
        os.path.join(BASE_DIR, "models", "bge-m3"),
        use_fp16=True, devices=["cuda:3"],
    )

    logger.info("加载 ChromaDB…")
    import chromadb
    chroma_client = chromadb.PersistentClient(path=os.path.join(DATA_DIR, "chroma_db"))
    collection = chroma_client.get_collection("products")

    # ── 按需加载额外资源 ──────────────────────────────────────────
    hybrid_retriever = None
    llm_client = None
    sparse_retriever = None
    df = None

    if "A" in strategies:
        logger.info("Strategy A: 加载 HybridRetriever…")
        from src.hybrid_retriever import HybridRetriever
        hybrid_retriever = HybridRetriever()
        from openai import OpenAI
        llm_client = OpenAI(api_key=QIANFAN_API_KEY, base_url=QIANFAN_BASE_URL)

    if "B" in strategies:
        logger.info("Strategy B: 加载商品数据（sparse 编码用）…")
        df = pd.read_parquet(os.path.join(DATA_DIR, "cleaned_products.parquet"))
        sparse_retriever = SparseRetriever(
            model=model, df=df,
            cache_path=SPARSE_INDEX_PATH,
            rebuild=args.rebuild_sparse,
        )

    # ── 构建策略对象 ──────────────────────────────────────────────
    strategy_objs = {}
    if "A" in strategies:
        strategy_objs["A"] = StrategyA(hybrid_retriever, llm_client)
    if "B" in strategies:
        strategy_objs["B"] = StrategyB(model, collection, sparse_retriever, args.n_candidates)
    if "C" in strategies:
        strategy_objs["C"] = StrategyC(model, collection)

    # ── 打开 raw 输出文件（resume 模式追加，否则覆盖）────────────
    raw_mode = "a" if args.resume else "w"
    all_strategy_metrics: dict = {}

    with open(RAW_RESULTS_PATH, raw_mode, encoding="utf-8") as raw_out_f:
        for strategy_name in strategies:
            logger.info("")
            logger.info("━" * 60)
            logger.info("开始评估策略 %s", strategy_name)
            logger.info("━" * 60)

            strategy_obj = strategy_objs[strategy_name]
            preparse_fn = None
            if strategy_name == "A":
                preparse_fn = lambda todo, s=strategy_obj: s.preparse(todo, concurrency=args.concurrency)

            metrics = run_strategy(
                strategy_name=strategy_name,
                strategy_obj=strategy_obj,
                samples=samples,
                k_values=k_values,
                existing=existing,
                raw_out_f=raw_out_f,
                max_k=max_k,
                preparse_fn=preparse_fn,
            )
            all_strategy_metrics[strategy_name] = metrics

    # ── 聚合指标 ─────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("所有策略跑完，开始聚合指标…")
    agg = aggregate_metrics(all_strategy_metrics, k_values, strategies)

    # ── 控制台打印整体结果 ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 召回策略评估结果（整体平均）")
    print("=" * 60)
    header = f"{'指标':<15}" + "".join(f"{'策略' + s:<12}" for s in strategies)
    print(header)
    print("-" * (15 + 12 * len(strategies)))
    for k in k_values:
        for metric_name in [f"recall@{k}", f"mrr@{k}"]:
            row = f"{metric_name:<15}"
            for s in strategies:
                val = agg[s]["overall"].get(metric_name, 0.0)
                row += f"{val:<12.4f}"
            print(row)
    print("=" * 60)

    # ── 写报告 ───────────────────────────────────────────────────
    summary_path = os.path.join(EVAL_DIR, "eval_metrics_summary.md")
    detail_path  = os.path.join(EVAL_DIR, "eval_metrics_detail.csv")
    write_summary_markdown(agg, k_values, strategies, summary_path)
    write_detail_csv(all_strategy_metrics, k_values, strategies, samples, detail_path)

    logger.info("")
    logger.info("✅ 评估完成！输出文件：")
    logger.info("   原始结果: %s", RAW_RESULTS_PATH)
    logger.info("   汇总报告: %s", summary_path)
    logger.info("   明细 CSV: %s", detail_path)


if __name__ == "__main__":
    main()
