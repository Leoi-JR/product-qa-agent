"""
Benchmark 测试集生成脚本

按"反向构造 + 多路召回扩充 + LLM 判定"的方法学生成 RAG 检索评估测试集。
产出 data/benchmark_dataset.jsonl，每行一个样本：
    {"query": "...", "query_type": "...", "anchor_id": "prod_xxx",
     "relevant_ids": ["prod_xxx", ...], "candidate_pool_size": N, "judge_model": "..."}

六个阶段，每阶段产出中间文件到 data/benchmark_work/，支持 --stage 断点续跑：
    1. 分层采样商品           → sampled_products.parquet
    2. LLM 反向构造 query     → queries_raw.jsonl
    3. 三路召回建候选池       → candidate_pools.jsonl
    4. judge 并发筛选         → judged_samples.jsonl
    5. 整理输出测试集         → benchmark_dataset.jsonl
    6. 抽检报告               → inspection_report.md

用法：
    conda run -n py312 python scripts/build_benchmark_dataset.py
    conda run -n py312 python scripts/build_benchmark_dataset.py --stage 4
    conda run -n py312 python scripts/build_benchmark_dataset.py --samples-per-type 2 --queries-per-product 3  # 冒烟测试
"""

import os
import sys
import json
import time
import pickle
import logging
import argparse
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from openai import OpenAI

# ── 路径定位（脚本可从任意目录运行）─────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
WORK_DIR = os.path.join(DATA_DIR, "benchmark_work")
DATA_PATH = os.path.join(DATA_DIR, "cleaned_products.parquet")
SPARSE_INDEX_PATH = os.path.join(WORK_DIR, "sparse_index.pkl")
OUTPUT_PATH = os.path.join(DATA_DIR, "benchmark_dataset.jsonl")

# 把项目根加入 sys.path，以便复用 src/ 的 HybridRetriever 和 parse_query
sys.path.insert(0, BASE_DIR)

# 提前加载 .env：下面的 QIANFAN_* 配置在【模块导入时】就读 os.environ，
# 必须在读取之前 load_dotenv，否则 .env 里的 key 不会进配置常量
from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── LLM 后端：百度千帆 DeepSeek-V3.2 ──
# 限流说明：
#   - deepseek-v3.2（非 think）：5000 RPM，默认 LLM_RPM=2000（留 60% 余量防 429）
#   - deepseek-v3.2-think：      60 RPM，  使用 think 时请在 .env 设 LLM_RPM=54
# 全局 _RateLimiter 把请求发起速率钉在 LLM_RPM 以下，并发只用来掩盖单次延迟。
QIANFAN_BASE_URL = os.environ.get("QIANFAN_BASE_URL", "https://qianfan.baidubce.com/v2")
QIANFAN_API_KEY = os.environ.get("QIANFAN_API_KEY", "")
QIANFAN_MODEL = os.environ.get("QIANFAN_MODEL", "deepseek-v3.2")
# 限流目标：默认对应非 think 模型（5000 RPM），留 60% 余量取 2000
LLM_RPM = int(os.environ.get("LLM_RPM", "2000"))
# 思考模式：非 think 模型默认关闭；如使用 think 模型，设 LLM_THINKING=true
LLM_THINKING = os.environ.get("LLM_THINKING", "false").lower() in ("1", "true", "yes")
# 并发：非 think 模型响应约 1-3s，32 路并发可维持 ~1000 RPM 有效吞吐
LLM_CONCURRENCY = int(os.environ.get("LLM_CONCURRENCY", "32"))
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "4"))

# query 生成 / parse_query：千帆 endpoint + Langfuse 可观测
LLM_BASE_URL = QIANFAN_BASE_URL
LLM_API_KEY = QIANFAN_API_KEY
LLM_MODEL = QIANFAN_MODEL
# judge：千帆 endpoint + 原生 OpenAI client（不上报 Langfuse，避免海量 judge 调用刷屏）
JUDGE_BASE_URL = QIANFAN_BASE_URL
JUDGE_API_KEY = QIANFAN_API_KEY
JUDGE_MODEL = QIANFAN_MODEL


class _RateLimiter:
    """请求发起限速器：保证两次 create() 发起间隔 ≥ 60/RPM 秒。

    千帆 60 RPM 是硬上限，瓶颈是发起速率而非并发数。本类把发起速率钉死，
    LLM_CONCURRENCY 只负责掩盖单次延迟。阶段2/3/4 顺序执行、共用同一实例，
    全流程都不会超 RPM。
    """

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


# 全局限速器：所有千帆调用（query 生成 / parse / judge）共用，统一不超过 LLM_RPM
_RATE_LIMITER = _RateLimiter(LLM_RPM)


def _retry_after_seconds(exc):
    """从异常提取 429 退避秒数；非 429 返回 None。"""
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
    return 30.0  # 千帆配额每 0-60s 刷新，保守等 30s


def _llm_chat(client, *, model, messages, timeout=120, **kwargs):
    """统一千帆调用：限速器节流 + 429 退避 + thinking 开关。

    所有 LLM 调用都走这里，保证全局不超 RPM、429 自动恢复。
    外层函数仍保留各自的 JSON 解析重试（这里只管 HTTP 层）。
    """
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


# ════════════════════════════════════════════════════════════════
# 阶段 1：分层采样
# ════════════════════════════════════════════════════════════════

# 五种 query 类型，对应不同的采样规则
# 规则用 DataFrame 的布尔过滤表达，挑出"适合生成该类 query"的商品池
BIG_CATEGORIES = {"AMAZON FASHION", "Amazon Home"}  # 大类太宽泛，vague 类要避开


def stratified_sample(df: pd.DataFrame, n_per_type: int, seed: int) -> pd.DataFrame:
    """按 query 类型反推分层采样。返回带 sample_type 列的 DataFrame。

    每类从合适的商品池里随机抽 n_per_type 个，不重复。
    """
    rng = random.Random(seed)
    sampled_rows = []
    used_indices = set()

    def _sample_from(mask, n, type_name):
        """从 mask 命中的商品池里抽 n 个未用过的。"""
        candidates = df.index[mask & ~df.index.isin(used_indices)].tolist()
        rng.shuffle(candidates)
        picked = candidates[:n]
        used_indices.update(picked)
        logger.info("  %s: 候选池 %d, 抽取 %d", type_name, len(candidates), len(picked))
        if len(picked) < n:
            logger.warning("  %s: 数量不足，只抽到 %d（目标 %d）", type_name, len(picked), n)
        return picked

    logger.info("阶段1: 分层采样（每类目标 %d）", n_per_type)

    # brand_model: brand 非空 + title 含数字/代数词
    mask_bm = (
        df["brand"].notna()
        & (df["brand"].astype(str).str.strip() != "")
        & df["title"].astype(str).str.contains(r"\d", regex=True, na=False)
    )
    for idx in _sample_from(mask_bm, n_per_type, "brand_model"):
        sampled_rows.append((idx, "brand_model"))

    # natural_desc: description 较长（>200 字符）
    desc_len = df["description"].astype(str).str.len()
    mask_nd = desc_len > 200
    for idx in _sample_from(mask_nd, n_per_type, "natural_desc"):
        sampled_rows.append((idx, "natural_desc"))

    # cross_lang: 随机抽（任何商品都可能被中文搜）
    mask_cl = pd.Series(True, index=df.index)
    for idx in _sample_from(mask_cl, n_per_type, "cross_lang"):
        sampled_rows.append((idx, "cross_lang"))

    # constrained: price > 0 + detail_color 非空
    mask_co = (
        df["price"].notna()
        & (df["price"] > 0)
        & df["detail_color"].notna()
        & (df["detail_color"].astype(str).str.strip() != "")
    )
    for idx in _sample_from(mask_co, n_per_type, "constrained"):
        sampled_rows.append((idx, "constrained"))

    # vague: main_category 较具体（非大类）
    mask_va = ~df["main_category"].astype(str).isin(BIG_CATEGORIES)
    for idx in _sample_from(mask_va, n_per_type, "vague"):
        sampled_rows.append((idx, "vague"))

    result_df = df.loc[[r[0] for r in sampled_rows]].copy()
    result_df["sample_type"] = [r[1] for r in sampled_rows]
    result_df["sample_id"] = [f"prod_{i}" for i in result_df.index]
    logger.info("阶段1 完成: 共采样 %d 个商品", len(result_df))
    return result_df


# ════════════════════════════════════════════════════════════════
# 阶段 2：LLM 反向构造 query（智谱 GLM）
# ════════════════════════════════════════════════════════════════

QUERY_GEN_SYSTEM_PROMPT = """你是电商搜索测试工程师。给定一个商品，生成 5 个"用户可能搜到这个商品"的查询。

要求：覆盖以下 5 种类型，每种恰好 1 条，查询要自然、真实、多样化。

类型定义（query_type 字段值）：
- brand_model: 精确品牌或型号（如 "Nike Air Max 90"）
- natural_desc: 自然语言描述需求（如 "跑步穿的轻便运动鞋"）
- cross_lang: 中英混合或跨语言（如 "耐克气垫鞋"）
- constrained: 带价格/颜色/尺寸等约束（如 "200美元以内的红色跑鞋"）
- vague: 模糊的开放需求（如 "适合马拉松的装备"）

约束：
1. 每条 query 都应该能合理地检索到这个商品（用户搜它，这个商品该出现在结果里）
2. constrained 类的约束要基于商品实际属性（价格、颜色等），不要瞎编
3. cross_lang 类要让中文 query 配英文商品（或反之），体现跨语言匹配
4. query 长度 5-25 个字/词，不要太长也不要太短

严格输出 JSON，格式：{"queries": [{"query": "...", "query_type": "brand_model"}, ...]}（共 5 条）
不要输出任何其他内容。"""


def _truncate(s, n):
    """安全截断字符串到 n 字符。"""
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    return s[:n].replace("\n", " ").strip()


def build_query_gen_messages(product: dict) -> list:
    """构造 query 生成的 messages。product 是 sampled_df 的一行转 dict。"""
    user_content = f"""商品信息：
标题：{_truncate(product.get('title'), 200)}
品牌：{_truncate(product.get('brand'), 50) or '无'}
价格：{product.get('price', '未知')}
大类：{_truncate(product.get('main_category'), 50)}
颜色：{_truncate(product.get('detail_color'), 30) or '无'}
描述：{_truncate(product.get('description'), 300)}

请生成 5 条覆盖上述类型的查询，输出 JSON。"""
    return [
        {"role": "system", "content": QUERY_GEN_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate_queries_for_product(client, product: dict, max_retries: int = 2) -> list:
    """对单个商品生成 5 条 query。失败返回空列表。"""
    messages = build_query_gen_messages(product)
    for attempt in range(max_retries + 1):
        try:
            resp = _llm_chat(
                client, model=LLM_MODEL, messages=messages,
                temperature=0.7,  # 稍高温度增加多样性
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            if not content:
                raise ValueError("空 content")
            data = json.loads(content)
            queries = data.get("queries", [])
            if isinstance(queries, list) and len(queries) > 0:
                return queries
            raise ValueError(f"queries 字段异常: {data}")
        except Exception as e:
            if attempt < max_retries:
                logger.warning("  query 生成重试 (%d/%d): %s", attempt + 1, max_retries, e)
                time.sleep(1)
            else:
                logger.error("  query 生成失败: %s", e)
                return []
    return []


def stage2_generate_queries(sampled_df: pd.DataFrame, client,
                            concurrency: int = LLM_CONCURRENCY) -> list:
    """对采样商品【并发】生成 query。返回 list of {query, query_type, anchor_id}。"""
    products = []
    for _, row in sampled_df.iterrows():
        p = row.to_dict()
        p["sample_id"] = row["sample_id"]
        products.append(p)

    logger.info("阶段2: 并发生成 query（共 %d 商品，并发 %d）", len(products), concurrency)

    def _worker(prod):
        # generate_queries_for_product 内部已有重试，失败返回 []
        return prod["sample_id"], generate_queries_for_product(client, prod)

    all_queries = []
    t0 = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_worker, p): p for p in products}
        for future in as_completed(futures):
            try:
                sample_id, queries = future.result()
            except Exception as e:
                logger.warning("  query 生成 future 异常: %s", e)
                completed += 1
                continue
            for q in queries:
                query_text = q.get("query", "").strip()
                query_type = q.get("query_type", "").strip()
                if query_text and query_type:
                    all_queries.append({
                        "query": query_text,
                        "query_type": query_type,
                        "anchor_id": sample_id,
                    })
            completed += 1
            if completed % 10 == 0 or completed == len(products):
                logger.info("  进度 %d/%d (%.1fs), 累计 query %d",
                            completed, len(products), time.time() - t0, len(all_queries))
    logger.info("阶段2 完成: 共 %d 条 query", len(all_queries))
    return all_queries


# ════════════════════════════════════════════════════════════════
# 阶段 3：三路召回
# ════════════════════════════════════════════════════════════════

def _parse_query_llm(client, query: str, model: str, max_retries: int = 2):
    """用 LLM 解析 query 为 {semantic_query, extracted}。带重试，失败返回 None。

    内联复用 src/query_parser 的 PARSER_SYSTEM_PROMPT（只读 import，不改动生产代码）。
    生产 query_parser 用智谱且无重试；benchmark 走千帆，HTTP 层限速/429 由 _llm_chat
    兜底，这里只兜 JSON 解析异常，避免单次异常让该 query 丢失 dense 路。
    """
    from src.query_parser import PARSER_SYSTEM_PROMPT, PARSER_USER_TEMPLATE
    messages = [
        {"role": "system", "content": PARSER_SYSTEM_PROMPT},
        {"role": "user", "content": PARSER_USER_TEMPLATE.format(query=query)},
    ]
    for attempt in range(max_retries + 1):
        try:
            resp = _llm_chat(
                client, model=model, messages=messages, temperature=0.1,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            if not content:
                raise ValueError("空 content")
            data = json.loads(content)
            extracted = data.get("extracted", {})
            for key in ["price_min", "price_max", "brand", "color", "category"]:
                extracted.setdefault(key, None)
            return {"semantic_query": data.get("semantic_query", ""), "extracted": extracted}
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
            else:
                logger.warning("  parse 失败 (query=%s...): %s", query[:30], e)
                return None
    return None


class DenseRoute:
    """Dense 路：parse_query（千帆）+ HybridRetriever（完整链路，含字段加权）。

    parse 阶段在阶段3 开始时【并发预解析】所有 query 并缓存，retrieve 直接查缓存，
    避免在召回循环里串行调 LLM。千帆限 54 RPM，~750 条 query 预解析约 14min
    （_RateLimiter 保证不超限，并发只掩盖单次延迟）。
    """

    def __init__(self, retriever, parser_client, model: str = LLM_MODEL):
        self.retriever = retriever
        self.parser_client = parser_client
        self.model = model
        self._parsed_cache = {}  # query -> parsed dict (或 None)

    def preparse(self, queries: list, concurrency: int = LLM_CONCURRENCY):
        """并发预解析所有 query，填入缓存。"""
        unique_queries = list({q["query"] for q in queries})
        logger.info("Dense: 并发预解析 %d 条 query（并发 %d）",
                    len(unique_queries), concurrency)
        t0 = time.time()
        completed = 0
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(_parse_query_llm, self.parser_client, q, self.model): q
                for q in unique_queries
            }
            for future in as_completed(futures):
                q = futures[future]
                try:
                    self._parsed_cache[q] = future.result()
                except Exception as e:
                    logger.warning("  preparse future 异常: %s", e)
                    self._parsed_cache[q] = None
                completed += 1
                if completed % 50 == 0 or completed == len(unique_queries):
                    logger.info("  preparse 进度 %d/%d (%.1fs)",
                                completed, len(unique_queries), time.time() - t0)
        ok = sum(1 for v in self._parsed_cache.values() if v)
        logger.info("Dense 预解析完成: %d/%d 成功", ok, len(unique_queries))

    def retrieve(self, query: str, top_k: int) -> list:
        """返回 doc id 列表（top_k 个）。失败返回空列表。"""
        parsed = self._parsed_cache.get(query)
        if parsed is None:  # 缓存未命中或之前失败，兜底再试一次
            parsed = _parse_query_llm(self.parser_client, query, self.model)
        if not parsed:
            return []
        try:
            results = self.retriever.retrieve(
                semantic_query=parsed["semantic_query"],
                extracted=parsed["extracted"],
                top_k=top_k,
            )
            return [r["id"] for r in results]
        except Exception as e:
            logger.warning("  Dense 路召回失败 (query=%s...): %s", query[:30], e)
            return []


class BM25Route:
    """BM25 路：用 rank_bm25 对全库 title 建索引。简单空格分词 + 小写。"""

    def __init__(self, df: pd.DataFrame):
        logger.info("BM25: 建索引（%d 条商品）", len(df))
        from rank_bm25 import BM25Okapi
        # 用 title 建索引，空格分词 + 小写（商品标题基本是英文）
        self.ids = [f"prod_{i}" for i in df.index]
        self.corpus = [
            str(t).lower().split() for t in df["title"].fillna("")
        ]
        self.bm25 = BM25Okapi(self.corpus)
        logger.info("BM25 索引建完")

    def retrieve(self, query: str, top_k: int) -> list:
        tokens = query.lower().split()
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        # 取 top_k 的索引（按分数降序）
        top_idx = np.argsort(-scores)[:top_k]
        return [self.ids[i] for i in top_idx if scores[i] > 0]


class SparseRoute:
    """M3-sparse 路：用 BGE-M3 的 lexical_weights（稀疏向量）做召回。

    启动时对全库 search_text 编码并缓存到 pkl。查询时 sparse 编码 + 点积打分。
    """

    def __init__(self, model, df: pd.DataFrame, cache_path: str, rebuild: bool = False):
        self.model = model
        self.ids = [f"prod_{i}" for i in df.index]
        self.cache_path = cache_path

        # 缓存里存的是倒排索引：{token_id_str: [(doc_idx, weight), ...]}
        if not rebuild and os.path.exists(cache_path):
            logger.info("Sparse: 加载缓存倒排索引 %s", cache_path)
            with open(cache_path, "rb") as f:
                self.inverted_index, cached_ids = pickle.load(f)
            if cached_ids == self.ids:
                logger.info("Sparse 缓存命中（%d 个 token）", len(self.inverted_index))
                return
            logger.warning("Sparse 缓存 id 不匹配，重新编码")

        logger.info("Sparse: 对全库编码 lexical_weights（%d 条）", len(df))
        texts = df["search_text"].fillna("").astype(str).tolist()
        # batch_size=128 + max_length=1024：经实测在 A6000 上 ~165 条/秒
        # max_length=1024 覆盖 P99 商品长度（实测 P99=1302），极少截断
        batch_size = 128
        max_length = 1024
        # 先收集正向索引（每条商品的 sparse dict），再转成倒排索引
        all_sparse = []  # list of {token_id_str: weight}
        t0 = time.time()
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            out = model.encode(
                batch, return_dense=False, return_sparse=True,
                return_colbert_vecs=False, max_length=max_length,
            )
            # lexical_weights 是 list[defaultdict{token_id_str: float16}]
            for lw in out["lexical_weights"]:
                # 转 {str: float}，统一类型
                clean = {str(k): float(v) for k, v in lw.items()}
                all_sparse.append(clean)
            if (i // batch_size + 1) % 10 == 0:
                logger.info("  Sparse 编码进度 %d/%d (%.1fs, %.1f 条/秒)",
                            i + len(batch), len(texts), time.time() - t0,
                            (i + len(batch)) / (time.time() - t0))
        # 正向索引 → 倒排索引：{token_id: [(doc_idx, weight), ...]}
        # 这样查询时只需遍历 query token 命中的商品，复杂度 O(命中商品数) 而非 O(全库)
        logger.info("Sparse: 构建倒排索引...")
        inverted_index = defaultdict(list)
        for doc_idx, doc_lw in enumerate(all_sparse):
            for token_id, weight in doc_lw.items():
                inverted_index[token_id].append((doc_idx, weight))
        self.inverted_index = dict(inverted_index)
        del all_sparse, inverted_index  # 释放正向索引内存
        # 缓存倒排索引
        with open(cache_path, "wb") as f:
            pickle.dump((self.inverted_index, self.ids), f)
        logger.info("Sparse 编码完成并缓存（耗时 %.1fs，%d 个 token）",
                    time.time() - t0, len(self.inverted_index))

    def retrieve(self, query: str, top_k: int) -> list:
        """查询：query 的 token 查倒排索引，只对命中商品累加分数。

        复杂度 O(query_token数 × 平均每token命中商品数)，远快于遍历全库。
        """
        # 编码 query
        out = self.model.encode(
            [query], return_dense=False, return_sparse=True,
            return_colbert_vecs=False, max_length=256,  # query 很短，256 够
        )
        q_lw = out["lexical_weights"][0]
        q_sparse = {str(k): float(v) for k, v in q_lw.items()}
        if not q_sparse:
            return []
        # 查倒排索引：对 query 的每个 token，取出包含该 token 的商品，累加点积
        scores = defaultdict(float)  # doc_idx -> score
        for token_id, q_weight in q_sparse.items():
            postings = self.inverted_index.get(token_id)
            if not postings:
                continue
            for doc_idx, doc_weight in postings:
                scores[doc_idx] += q_weight * doc_weight
        if not scores:
            return []
        # 取 top_k
        sorted_docs = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [self.ids[idx] for idx, _ in sorted_docs]


def stage3_build_candidate_pools(queries: list, routes: dict, top_k: int,
                                concurrency: int = LLM_CONCURRENCY) -> list:
    """三路召回 + 去重合并候选池。

    routes: {"dense": DenseRoute, "bm25": BM25Route, "sparse": SparseRoute}
    返回 list of {query, query_type, anchor_id, candidate_ids, route_hits}
    """
    logger.info("阶段3: 三路召回建候选池（%d 条 query，每路 top-%d）",
                len(queries), top_k)
    # Dense 路先并发预解析所有 query（bm25/sparse 无需 LLM）
    if "dense" in routes:
        routes["dense"].preparse(queries, concurrency)
    results = []
    t0 = time.time()
    for i, q in enumerate(queries):
        query_text = q["query"]
        route_hits = {}
        for route_name, route in routes.items():
            ids = route.retrieve(query_text, top_k)
            route_hits[route_name] = ids
        # 去重合并（保持 anchor_id 在前）
        seen = set()
        merged = []
        # anchor_id 先入
        if q["anchor_id"] not in seen:
            merged.append(q["anchor_id"])
            seen.add(q["anchor_id"])
        for route_name in ["dense", "bm25", "sparse"]:
            for did in route_hits.get(route_name, []):
                if did not in seen:
                    merged.append(did)
                    seen.add(did)
        results.append({
            "query": query_text,
            "query_type": q["query_type"],
            "anchor_id": q["anchor_id"],
            "candidate_ids": merged,
            "route_hits": route_hits,
        })
        if (i + 1) % 50 == 0 or i == len(queries) - 1:
            logger.info("  进度 %d/%d (%.1fs), 平均候选池 %.1f",
                        i + 1, len(queries), time.time() - t0,
                        np.mean([len(r["candidate_ids"]) for r in results]))
    avg_pool = np.mean([len(r["candidate_ids"]) for r in results])
    logger.info("阶段3 完成: 平均候选池 %.1f 个商品", avg_pool)
    return results


# ════════════════════════════════════════════════════════════════
# 阶段 4：judge 并发筛选
# ════════════════════════════════════════════════════════════════

JUDGE_SYSTEM_PROMPT = """你是电商商品相关性判断员。判断给定的商品是否与用户的查询相关（即用户搜这个 query 时，这个商品是否应该出现在结果里）。

判断标准：
- "相关"：商品能满足 query 表达的需求（品牌、类型、用途、约束条件大致匹配）
- "不相关"：商品与 query 需求明显不符（类型不对、约束冲突、风马牛不相及）
- 边界 case 倾向判"相关"（宁可多召回，让下游筛选）

严格只输出 JSON：{"relevant": true/false, "reason": "一句话理由"}
不要输出任何其他内容。"""


def build_judge_messages(query: str, doc_info: dict) -> list:
    user_content = f"""用户查询：{query}

商品信息：
标题：{_truncate(doc_info.get('title'), 200)}
品牌：{_truncate(doc_info.get('brand'), 50) or '无'}
价格：{doc_info.get('price', '未知')}
大类：{_truncate(doc_info.get('main_category'), 50)}
描述：{_truncate(doc_info.get('description'), 200)}

请判断相关性，输出 JSON。"""
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def judge_single(client, query: str, doc_id: str, doc_info: dict,
                 max_retries: int = 3) -> dict:
    """对单个 (query, doc) 判断相关性。带重试。"""
    messages = build_judge_messages(query, doc_info)
    for attempt in range(max_retries):
        try:
            resp = _llm_chat(
                client, model=JUDGE_MODEL, messages=messages,
                temperature=0, response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            if not content:
                raise ValueError("空 content（可能只有 reasoning_content）")
            data = json.loads(content)
            relevant = data.get("relevant")
            if isinstance(relevant, bool):
                return {
                    "doc_id": doc_id,
                    "relevant": relevant,
                    "reason": _truncate(data.get("reason", ""), 100),
                }
            # 兼容字符串 "true"/"false"
            if isinstance(relevant, str):
                return {
                    "doc_id": doc_id,
                    "relevant": relevant.lower() in ("true", "yes", "1"),
                    "reason": _truncate(data.get("reason", ""), 100),
                }
            raise ValueError(f"relevant 字段异常: {data}")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))  # 指数退避
            else:
                logger.warning("  judge 失败 (query=%s..., doc=%s): %s",
                               query[:20], doc_id, e)
                return {"doc_id": doc_id, "relevant": False,
                        "reason": f"judge_error: {e}"}
    return {"doc_id": doc_id, "relevant": False, "reason": "max_retries_exceeded"}


def stage4_judge_candidates(pools: list, df: pd.DataFrame, client,
                            concurrency: int) -> list:
    """并发 judge 所有 (query, candidate_doc) 对。"""
    # 构造所有 judge 任务
    tasks = []  # (pool_idx, query, doc_id, doc_info)
    for pool_idx, pool in enumerate(pools):
        query = pool["query"]
        for doc_id in pool["candidate_ids"]:
            # doc_id 格式 prod_{i}，i 是原 df 行索引
            try:
                row_idx = int(doc_id.split("_")[1])
                doc_info = df.iloc[row_idx].to_dict()
            except (IndexError, ValueError):
                doc_info = {"title": doc_id}
            tasks.append((pool_idx, query, doc_id, doc_info))

    logger.info("阶段4: judge %d 个 (query, doc) 对（并发 %d）",
                len(tasks), concurrency)

    # 并发执行
    results_by_pool = defaultdict(dict)  # pool_idx -> {doc_id: judge_result}
    completed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(judge_single, client, query, doc_id, doc_info): (pool_idx, doc_id)
            for pool_idx, query, doc_id, doc_info in tasks
        }
        for future in as_completed(futures):
            pool_idx, doc_id = futures[future]
            try:
                result = future.result()
                results_by_pool[pool_idx][doc_id] = result
            except Exception as e:
                logger.warning("  judge future 异常: %s", e)
                results_by_pool[pool_idx][doc_id] = {
                    "doc_id": doc_id, "relevant": False, "reason": f"future_error: {e}"
                }
            completed += 1
            if completed % 100 == 0 or completed == len(tasks):
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                logger.info("  进度 %d/%d (%.1fs, %.1f/s)",
                            completed, len(tasks), elapsed, rate)

    # 组装结果
    judged = []
    for pool_idx, pool in enumerate(pools):
        judge_map = results_by_pool[pool_idx]
        relevant_ids = [
            doc_id for doc_id in pool["candidate_ids"]
            if judge_map.get(doc_id, {}).get("relevant", False)
        ]
        judged.append({
            "query": pool["query"],
            "query_type": pool["query_type"],
            "anchor_id": pool["anchor_id"],
            "candidate_ids": pool["candidate_ids"],
            "route_hits": pool["route_hits"],
            "relevant_ids": relevant_ids,
            "judge_details": judge_map,
        })
    avg_rel = np.mean([len(j["relevant_ids"]) for j in judged])
    logger.info("阶段4 完成: 平均 relevant_ids %.1f 个", avg_rel)
    return judged


# ════════════════════════════════════════════════════════════════
# 阶段 5：整理输出
# ════════════════════════════════════════════════════════════════

def stage5_assemble_dataset(judged: list) -> list:
    """组装最终测试集格式。"""
    dataset = []
    for j in judged:
        dataset.append({
            "query": j["query"],
            "query_type": j["query_type"],
            "anchor_id": j["anchor_id"],
            "relevant_ids": j["relevant_ids"],
            "candidate_pool_size": len(j["candidate_ids"]),
            "judge_model": JUDGE_MODEL,
        })
    logger.info("阶段5 完成: %d 条测试样本", len(dataset))
    return dataset


# ════════════════════════════════════════════════════════════════
# 阶段 6：抽检报告
# ════════════════════════════════════════════════════════════════

def stage6_write_inspection_report(dataset: list, judged: list, df: pd.DataFrame,
                                    output_path: str, seed: int):
    """生成抽检报告：
    - inspection_report.md           : 统计摘要 + CSV 文件说明（概览用）
    - inspection_query_quality.csv   : Query 质量抽检，含 LLM 生成 query 时看到的完整商品信息
    - inspection_judge_accuracy.csv  : Judge 准确率抽检，含 judge 判断时看到的完整商品信息 + 理由
    """
    rng = random.Random(seed + 1)
    work_dir = os.path.dirname(output_path)
    query_csv_path = os.path.join(work_dir, "inspection_query_quality.csv")
    judge_csv_path = os.path.join(work_dir, "inspection_judge_accuracy.csv")

    # ── 统计 ──────────────────────────────────────────────────────
    by_type = defaultdict(int)
    for d in dataset:
        by_type[d["query_type"]] += 1
    avg_pool = np.mean([d["candidate_pool_size"] for d in dataset])
    avg_rel = np.mean([len(d["relevant_ids"]) for d in dataset])

    # ── 辅助：从 df 取商品完整字段 ───────────────────────────────
    def _get_product_fields(prod_id: str) -> dict:
        try:
            row_idx = int(prod_id.split("_")[1])
            row = df.iloc[row_idx]
            return {
                "title":         _truncate(str(row.get("title", "") or ""), 200),
                "brand":         _truncate(str(row.get("brand", "") or "无"), 50),
                "price":         row.get("price", "未知"),
                "main_category": _truncate(str(row.get("main_category", "") or ""), 50),
                "detail_color":  _truncate(str(row.get("detail_color", "") or "无"), 30),
                "description":   _truncate(str(row.get("description", "") or ""), 300),
            }
        except (IndexError, ValueError):
            return {k: "?" for k in
                    ["title", "brand", "price", "main_category", "detail_color", "description"]}

    # ── CSV 1：Query 质量抽检（每类 6 条）───────────────────────
    # LLM 生成 query 时看到：title/brand/price/main_category/detail_color/description
    # 人工应通过同样的信息判断 query 是否合理
    query_rows = []
    for t in sorted(by_type.keys()):
        type_samples = [d for d in dataset if d["query_type"] == t]
        rng.shuffle(type_samples)
        for d in type_samples[:6]:
            fields = _get_product_fields(d["anchor_id"])
            query_rows.append({
                "query_type":              t,
                "query":                   d["query"],
                "anchor_id":               d["anchor_id"],
                "anchor_title":            fields["title"],
                "anchor_brand":            fields["brand"],
                "anchor_price":            fields["price"],
                "anchor_main_category":    fields["main_category"],
                "anchor_detail_color":     fields["detail_color"],
                "anchor_description":      fields["description"],
                "人工判断（合理/不合理）": "",
            })
    pd.DataFrame(query_rows).to_csv(query_csv_path, index=False, encoding="utf-8-sig")
    logger.info("Query 质量抽检 CSV: %s（%d 行）", query_csv_path, len(query_rows))

    # ── CSV 2：Judge 准确率抽检（随机抽 30 条）──────────────────
    # judge 判断时看到：title/brand/price/main_category/description
    # 人工应通过同样的信息判断 judge 是否判断正确
    all_judge_pairs = []
    for j in judged:
        for doc_id, detail in j["judge_details"].items():
            fields = _get_product_fields(doc_id)
            all_judge_pairs.append({
                "query":               j["query"],
                "doc_id":              doc_id,
                "title":               fields["title"],
                "brand":               fields["brand"],
                "price":               fields["price"],
                "main_category":       fields["main_category"],
                "description":         fields["description"],
                "judge_relevant":      detail.get("relevant", False),
                "judge_reason":        _truncate(str(detail.get("reason", "")), 200),
                "人工判断（正确/错误）": "",
            })
    rng.shuffle(all_judge_pairs)
    pd.DataFrame(all_judge_pairs[:30]).to_csv(judge_csv_path, index=False, encoding="utf-8-sig")
    logger.info("Judge 准确率抽检 CSV: %s（30 行）", judge_csv_path)

    # ── MD：仅保留统计摘要 + 操作说明 ───────────────────────────
    lines = []
    lines.append("# Benchmark 测试集抽检报告\n\n")
    lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"Judge 模型: {JUDGE_MODEL}\n")
    lines.append(f"随机种子: {seed}\n\n")

    lines.append("## 1. 数据统计\n\n")
    lines.append(f"- 总样本数: {len(dataset)}\n")
    lines.append("- 各 query 类型分布:\n")
    for t, c in sorted(by_type.items()):
        lines.append(f"  - {t}: {c}\n")
    lines.append(f"- 平均候选池大小: {avg_pool:.1f}\n")
    lines.append(f"- 平均 relevant_ids 大小: {avg_rel:.1f}\n\n")

    lines.append("## 2. 人工抽检文件\n\n")
    lines.append("> 用 Excel / Numbers / LibreOffice 打开 CSV，在最后一列填写判断结果。\n\n")

    lines.append("### Query 质量抽检\n\n")
    lines.append(f"文件：`{os.path.basename(query_csv_path)}`（共 {len(query_rows)} 行，每类 6 条）\n\n")
    lines.append("| 列名 | 说明 |\n|---|---|\n")
    lines.append("| query | LLM 生成的测试查询 |\n")
    lines.append("| anchor_title/brand/price/main_category/detail_color/description"
                 " | LLM 生成 query 时看到的完整商品信息 |\n")
    lines.append("| 人工判断（合理/不合理） | 填写：用户搜该 query 时，anchor 商品是否应出现在结果里 |\n\n")

    lines.append("### Judge 准确率抽检\n\n")
    lines.append(f"文件：`{os.path.basename(judge_csv_path)}`（共 30 行，随机抽样）\n\n")
    lines.append("| 列名 | 说明 |\n|---|---|\n")
    lines.append("| title/brand/price/main_category/description | judge 判断时看到的完整商品信息 |\n")
    lines.append("| judge_relevant | judge 的判定（True=相关，False=不相关）|\n")
    lines.append("| judge_reason | judge 给出的判断理由 |\n")
    lines.append("| 人工判断（正确/错误） | 填写：你认为 judge 判断是否正确 |\n\n")

    lines.append("## 3. 如何使用结果\n\n")
    lines.append("- **Query 质量**：若某类「不合理」占比 > 30%，需调整阶段2 prompt 并重跑\n")
    lines.append("- **Judge 准确率**：若「错误」占比 > 30%（准确率 < 70%），"
                 "需换 judge 模型或调整 `JUDGE_SYSTEM_PROMPT` 并重跑阶段4\n")
    lines.append("- **relevant_ids 过少**：若平均 relevant_ids 接近 1，说明候选池召回不足或 judge 过严\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    logger.info("阶段6 完成:\n  MD:        %s\n  Query CSV: %s\n  Judge CSV: %s",
                output_path, query_csv_path, judge_csv_path)


# ════════════════════════════════════════════════════════════════
# 中间文件读写
# ════════════════════════════════════════════════════════════════

def save_jsonl(data: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="生成 RAG 检索 benchmark 测试集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4, 5, 6],
                        help="从哪个阶段开始（断点续跑，默认1=从头）")
    parser.add_argument("--samples-per-type", type=int, default=30,
                        help="每类采样商品数（默认30，冒烟用2）")
    parser.add_argument("--topk-per-route", type=int, default=20,
                        help="每路召回 top-K（默认20）")
    parser.add_argument("--judge-concurrency", type=int, default=LLM_CONCURRENCY,
                        help="judge 并发数（默认 %(default)s，与 LLM_CONCURRENCY 保持一致）")
    parser.add_argument("--llm-concurrency", type=int, default=LLM_CONCURRENCY,
                        help="query 生成 / parse_query 的并发数（默认 %(default)s）。"
                             "与 LLM_RPM 配合使用；非 think 模型建议 32，think 模型建议 8")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认42）")
    parser.add_argument("--rebuild-sparse-index", action="store_true",
                        help="强制重建 sparse 索引（默认复用缓存）")
    args = parser.parse_args()

    os.makedirs(WORK_DIR, exist_ok=True)

    # 千帆 key 自检（阶段2/3/4 都要用；.env 已在模块顶部加载）
    if args.stage <= 4 and not QIANFAN_API_KEY:
        logger.error("QIANFAN_API_KEY 未设置：请在 .env 填入百度千帆 API Key")
        sys.exit(1)

    # 加载全量数据（多阶段共用）
    logger.info("加载商品数据: %s", DATA_PATH)
    df = pd.read_parquet(DATA_PATH)
    logger.info("共 %d 条商品", len(df))

    # 中间文件路径
    f_sampled = os.path.join(WORK_DIR, "sampled_products.parquet")
    f_queries = os.path.join(WORK_DIR, "queries_raw.jsonl")
    f_pools = os.path.join(WORK_DIR, "candidate_pools.jsonl")
    f_judged = os.path.join(WORK_DIR, "judged_samples.jsonl")
    f_report = os.path.join(WORK_DIR, "inspection_report.md")

    # ── 阶段 1：分层采样 ──
    sampled_df = None
    if args.stage <= 1:
        sampled_df = stratified_sample(df, args.samples_per_type, args.seed)
        sampled_df.to_parquet(f_sampled)
        logger.info("已保存: %s", f_sampled)
    else:
        sampled_df = pd.read_parquet(f_sampled)
        logger.info("已加载采样商品: %d 条", len(sampled_df))

    # ── 阶段 2：query 生成 ──
    queries = None
    if args.stage <= 2:
        # query 生成用千帆（走 Langfuse 可观测），并发化；.env 已在模块顶部加载
        from langfuse.openai import OpenAI as LangfuseOpenAI
        gen_client = LangfuseOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        # 连通性自检（走 _llm_chat，顺带验证限速器/thinking 配置）
        try:
            test = _llm_chat(
                gen_client, model=LLM_MODEL,
                messages=[{"role": "user", "content": "回复 ok"}],
                max_tokens=10, timeout=30,
            )
            logger.info("LLM 连通性测试 (%s): %s", LLM_BASE_URL,
                        test.choices[0].message.content)
        except Exception as e:
            logger.error("LLM 连接失败 (%s): %s", LLM_BASE_URL, e)
            sys.exit(1)
        queries = stage2_generate_queries(sampled_df, gen_client, args.llm_concurrency)
        save_jsonl(queries, f_queries)
        logger.info("已保存: %s", f_queries)
    else:
        queries = load_jsonl(f_queries)
        logger.info("已加载 query: %d 条", len(queries))

    # ── 阶段 3：三路召回 ──
    pools = None
    if args.stage <= 3:
        logger.info("阶段3: 初始化三路召回器...")
        # Dense 路：parse 用千帆，HybridRetriever 走完整字段加权链路；.env 已在顶部加载
        from src.hybrid_retriever import HybridRetriever
        from langfuse.openai import OpenAI as LangfuseOpenAI
        parser_client = LangfuseOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        logger.info("加载 HybridRetriever（首次启动需几十秒）...")
        retriever = HybridRetriever()
        dense_route = DenseRoute(retriever, parser_client, LLM_MODEL)

        # BM25 路
        bm25_route = BM25Route(df)

        # Sparse 路：复用 retriever 的 model（避免重复加载 M3）
        sparse_route = SparseRoute(
            retriever.model, df, SPARSE_INDEX_PATH,
            rebuild=args.rebuild_sparse_index,
        )

        routes = {"dense": dense_route, "bm25": bm25_route, "sparse": sparse_route}
        pools = stage3_build_candidate_pools(queries, routes, args.topk_per_route,
                                             args.llm_concurrency)
        save_jsonl(pools, f_pools)
        logger.info("已保存: %s", f_pools)
    else:
        pools = load_jsonl(f_pools)
        logger.info("已加载候选池: %d 条 query", len(pools))

    # ── 阶段 4：judge 筛选 ──
    judged = None
    if args.stage <= 4:
        judge_client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)
        # 测试 judge 连通性
        try:
            test = _llm_chat(
                judge_client, model=JUDGE_MODEL,
                messages=[{"role": "user", "content": "回复 ok"}],
                max_tokens=10, timeout=30,
            )
            logger.info("Judge 连通性测试: %s", test.choices[0].message.content)
        except Exception as e:
            logger.error("Judge 连接失败 (%s): %s", JUDGE_BASE_URL, e)
            sys.exit(1)
        judged = stage4_judge_candidates(pools, df, judge_client, args.judge_concurrency)
        save_jsonl(judged, f_judged)
        logger.info("已保存: %s", f_judged)
    else:
        judged = load_jsonl(f_judged)
        logger.info("已加载 judge 结果: %d 条", len(judged))

    # ── 阶段 5：整理输出 ──
    dataset = stage5_assemble_dataset(judged)
    save_jsonl(dataset, OUTPUT_PATH)
    logger.info("阶段5 已保存最终测试集: %s", OUTPUT_PATH)

    # ── 阶段 6：抽检报告 ──
    stage6_write_inspection_report(dataset, judged, df, f_report, args.seed)

    logger.info("=" * 60)
    logger.info("全部完成！")
    logger.info("  测试集: %s", OUTPUT_PATH)
    logger.info("  抽检报告: %s", f_report)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
