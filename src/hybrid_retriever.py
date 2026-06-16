"""
混合检索模块 — 多字段加权融合打分

打分逻辑：
  最终分数 = w_field × 字段分 + w_text × 语义分
  - 字段分：对 LLM 提取到的每个字段（brand/color/category），
            计算查询值向量与商品字段值向量的余弦相似度，缺失字段给保守低分 0.2。
            多字段时取平均。
  - 语义分：semantic_query 与商品 search_text 的余弦相似度。
  - 两项分别 min-max 归一化到 [0,1] 再加权。
  - price 作为硬过滤（不在分数计算中）。

默认权重：w_field = w_text = 0.5
"""

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
import chromadb
from FlagEmbedding import BGEM3FlagModel
from langfuse import observe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 路径定位（运行时模块，基于项目根目录）─────────────────
# src/hybrid_retriever.py 向上两级到根目录
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_BASE_DIR, "data")
FIELD_EMBEDDINGS_PATH = os.path.join(_DATA_DIR, "field_embeddings.npz")
PRODUCT_INDICES_PATH = os.path.join(_DATA_DIR, "product_field_indices.parquet")
CHROMA_DIR = os.path.join(_DATA_DIR, "chroma_db")
COLLECTION_NAME = "products"

# 字段配置
FIELDS = ["brand", "detail_color", "main_category"]
FIELD_LABEL = {"brand": "brand", "detail_color": "color", "main_category": "category"}

MISSING_FIELD_SCORE = 0.2     # 商品该字段缺失时的保守低分
DEFAULT_W_FIELD = 0.5
DEFAULT_W_TEXT = 0.5
DEFAULT_TOP_K = 5


class HybridRetriever:
    def __init__(self, model_path: str = os.path.join(_BASE_DIR, "models", "bge-m3")):
        logger.info("加载字段向量表: %s", FIELD_EMBEDDINGS_PATH)
        data = np.load(FIELD_EMBEDDINGS_PATH, allow_pickle=True)
        self.field_values = {}      # field -> list[str]
        self.field_embeddings = {}  # field -> np.ndarray (N, 1024)
        for field in FIELDS:
            self.field_values[field] = list(data[f"{field}_values"])
            self.field_embeddings[field] = data[f"{field}_embeddings"].astype(np.float32)

        logger.info("加载商品字段索引: %s", PRODUCT_INDICES_PATH)
        self.indices_df = pd.read_parquet(PRODUCT_INDICES_PATH)
        self.n_products = len(self.indices_df)

        logger.info("加载 ChromaDB: %s", CHROMA_DIR)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.chroma_client.get_collection(COLLECTION_NAME)

        logger.info("加载 BGE-M3 模型（锁定 cuda:3）...")
        self.model = BGEM3FlagModel(model_path, use_fp16=True, devices=["cuda:3"])
        logger.info("HybridRetriever 初始化完成，共 %d 条商品", self.n_products)

    def _cosine_matrix(self, query_vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
        """query_vec (D,) 与 mat (N, D) 每行的余弦相似度。"""
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
        return mat_norm @ q_norm   # (N,)

    def _compute_field_scores(
        self, extracted: dict, all_idx_mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """计算字段分。返回 (n_products,) 数组；若无字段提取则返回 None。"""
        active_fields = []
        query_vecs = {}
        for field in FIELDS:
            label = FIELD_LABEL[field]
            val = extracted.get(label)
            if val:
                q_vec = self.model.encode([val], max_length=64)["dense_vecs"][0].astype(np.float32)
                query_vecs[field] = q_vec
                active_fields.append(field)

        if not active_fields:
            return None

        # 对每个 active 字段，先算 (n_unique,) 的相似度，再通过 idx 广播到 (n_products,)
        per_field_score = np.zeros((len(active_fields), self.n_products), dtype=np.float32)

        for i, field in enumerate(active_fields):
            mat = self.field_embeddings[field]                 # (n_unique, D)
            sims_unique = self._cosine_matrix(query_vecs[field], mat)  # (n_unique,)

            idx = self.indices_df[f"{field}_idx"].to_numpy()   # (n_products,)
            valid = idx >= 0

            # 有效商品用其字段值的相似度，无效商品给 MISSING_FIELD_SCORE
            scores = np.full(self.n_products, MISSING_FIELD_SCORE, dtype=np.float32)
            scores[valid] = sims_unique[idx[valid]]
            per_field_score[i] = scores

        # 多字段取平均
        return per_field_score.mean(axis=0)

    def _compute_text_scores(self, semantic_query: str) -> np.ndarray:
        """通过 ChromaDB 查询获取语义分（返回全部商品的余弦相似度）。"""
        # ChromaDB 一次最多返回 collection 总数，这里我们查全部
        q_vec = self.model.encode([semantic_query], max_length=8192)["dense_vecs"][0].astype(np.float32)

        # 直接从 ChromaDB 拉 Top-N（避免拉全部向量占内存），N 取较大值用于归一化
        # 这里改用拉取向量做暴力计算，保证字段分和语义分维度一致
        result = self.collection.get(include=["embeddings"])
        all_emb = np.array(result["embeddings"], dtype=np.float32)
        all_ids = result["ids"]

        sims = self._cosine_matrix(q_vec, all_emb)
        return all_ids, sims

    def _min_max_normalize(self, arr: np.ndarray) -> np.ndarray:
        """归一化到 [0,1]。"""
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-8:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    @observe(name="hybrid_retrieve", as_type="retriever")
    def retrieve(
        self,
        semantic_query: str,
        extracted: dict,
        top_k: int = DEFAULT_TOP_K,
        w_field: float = DEFAULT_W_FIELD,
        w_text: float = DEFAULT_W_TEXT,
        n_candidates: int = 200,
    ) -> list[dict]:
        """
        执行混合检索。

        策略：先从 ChromaDB 取语义 Top-N 候选（n_candidates），
        然后在这批候选上计算字段分并融合，最后返回 Top-K。
        这样避免对全量 117K 商品做字段广播计算。
        """
        # 1. 编码 semantic_query
        q_vec = self.model.encode([semantic_query], max_length=8192)["dense_vecs"][0].astype(np.float32)

        # 2. 从 ChromaDB 取语义 Top-N 候选
        logger.info("从 ChromaDB 取语义 Top-%d 候选", n_candidates)
        query_result = self.collection.query(
            query_embeddings=q_vec.tolist(),
            n_results=min(n_candidates, self.n_products),
            include=["embeddings", "metadatas", "documents", "distances"],
        )
        cand_ids = query_result["ids"][0]
        cand_metas = query_result["metadatas"][0]
        cand_emb = np.array(query_result["embeddings"][0], dtype=np.float32)
        n_cand = len(cand_ids)
        logger.info("候选数: %d", n_cand)

        # 3. 候选集上的语义分（chromadb query 返回的是 distance=1-sim，转回 sim）
        cand_text_sims = 1.0 - np.array(query_result["distances"][0], dtype=np.float32)

        # 4. 候选集上的字段分
        # 把候选 id 映射到 product indices_df 的行
        id_to_df_idx = {f"prod_{i}": i for i in range(self.n_products)}
        cand_df_idxs = np.array([id_to_df_idx[i] for i in cand_ids], dtype=np.int64)

        field_scores = self._compute_field_scores_on_candidates(
            extracted, cand_df_idxs
        )

        # 5. 价格硬过滤
        price_min = extracted.get("price_min")
        price_max = extracted.get("price_max")
        price_mask = np.ones(n_cand, dtype=bool)
        for i, meta in enumerate(cand_metas):
            p = meta.get("price", -1.0)
            if p is None or p < 0:
                continue  # 价格缺失的商品保留
            if price_min is not None and p < price_min:
                price_mask[i] = False
            if price_max is not None and p > price_max:
                price_mask[i] = False
        logger.info("价格过滤后候选: %d / %d", price_mask.sum(), n_cand)

        # 6. 归一化 + 加权
        text_norm = self._min_max_normalize(cand_text_sims)
        if field_scores is not None:
            field_norm = self._min_max_normalize(field_scores)
            final_scores = w_field * field_norm + w_text * text_norm
        else:
            logger.info("无字段提取，使用纯语义分")
            final_scores = text_norm

        # 7. 应用价格过滤
        final_scores = np.where(price_mask, final_scores, -1.0)

        # 8. Top-K
        top_idx = np.argsort(-final_scores)[:top_k]
        results = []
        for i in top_idx:
            if final_scores[i] < 0:
                continue
            meta = cand_metas[i]
            results.append({
                "id": cand_ids[i],
                "title": meta.get("title", ""),
                "brand": meta.get("brand", ""),
                "main_category": meta.get("main_category", ""),
                "price": meta.get("price", -1.0),
                "average_rating": meta.get("average_rating", 0.0),
                "image": meta.get("image", ""),
                "final_score": float(final_scores[i]),
                "text_score": float(cand_text_sims[i]),
                "field_score": float(field_scores[i]) if field_scores is not None else None,
            })
        return results

    def _compute_field_scores_on_candidates(
        self, extracted: dict, cand_df_idxs: np.ndarray
    ) -> Optional[np.ndarray]:
        """对候选集计算字段分。"""
        active_fields = []
        query_vecs = {}
        for field in FIELDS:
            label = FIELD_LABEL[field]
            val = extracted.get(label)
            if val:
                q_vec = self.model.encode([val], max_length=64)["dense_vecs"][0].astype(np.float32)
                query_vecs[field] = q_vec
                active_fields.append(field)

        if not active_fields:
            return None

        n_cand = len(cand_df_idxs)
        per_field_score = np.zeros((len(active_fields), n_cand), dtype=np.float32)

        for i, field in enumerate(active_fields):
            mat = self.field_embeddings[field]
            sims_unique = self._cosine_matrix(query_vecs[field], mat)

            # 候选商品在该字段的 idx
            all_idx = self.indices_df[f"{field}_idx"].to_numpy()
            cand_idx = all_idx[cand_df_idxs]
            valid = cand_idx >= 0

            scores = np.full(n_cand, MISSING_FIELD_SCORE, dtype=np.float32)
            scores[valid] = sims_unique[cand_idx[valid]]
            per_field_score[i] = scores

        return per_field_score.mean(axis=0)


if __name__ == "__main__":
    # 手动构造解析结果测试（不依赖 LLM）
    retriever = HybridRetriever()

    test_cases = [
        {
            "name": "红色徒步背包（多字段）",
            "semantic_query": "A red backpack suitable for hiking and outdoor adventures",
            "extracted": {"price_min": None, "price_max": 200, "brand": None, "color": "red", "category": None},
        },
        {
            "name": "纯语义（蓝牙耳机）",
            "semantic_query": "Wireless bluetooth earbuds for running and sports",
            "extracted": {"price_min": None, "price_max": None, "brand": None, "color": None, "category": None},
        },
        {
            "name": "Nike 运动鞋（品牌+品类）",
            "semantic_query": "Nike athletic running shoes for men",
            "extracted": {"price_min": None, "price_max": None, "brand": "Nike", "color": None, "category": None},
        },
    ]

    for tc in test_cases:
        print("\n" + "=" * 60)
        print(f"测试: {tc['name']}")
        print(f"semantic_query: {tc['semantic_query']}")
        print(f"extracted: {tc['extracted']}")
        results = retriever.retrieve(tc["semantic_query"], tc["extracted"], top_k=5)
        for i, r in enumerate(results):
            print(f"  #{i+1} [{r['final_score']:.4f}] {r['title'][:50]} | brand={r['brand']} | price={r['price']} | cat={r['main_category']}")
