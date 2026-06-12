"""
BGE-M3 Embedding + ChromaDB 构建
对 cleaned_products.parquet 的 search_text 生成向量，存入 ChromaDB。
"""

import os
import time
import logging

import pandas as pd
import torch
from FlagEmbedding import BGEM3FlagModel
import chromadb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 路径定位（脚本可从任意目录运行）─────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── 配置 ──────────────────────────────────────────────
DATA_PATH = os.path.join(DATA_DIR, "cleaned_products.parquet")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
MODEL_PATH = os.path.join(MODELS_DIR, "bge-m3")
COLLECTION_NAME = "products"
BATCH_SIZE = 64          # GPU 批次大小（RTX A6000 48GB 显存足够）
MAX_LENGTH = 8192         # BGE-M3 最大 token 数


def load_data():
    logger.info("加载清洗后的数据: %s", DATA_PATH)
    df = pd.read_parquet(DATA_PATH)
    logger.info("共 %d 条记录", len(df))
    return df


def build_metadata(df: pd.DataFrame) -> list[dict]:
    """为每条记录构建 ChromaDB metadata。"""
    metadata_list = []
    for _, row in df.iterrows():
        meta = {
            "title": str(row.get("title", ""))[:500],
            "brand": str(row.get("brand", "unknown")),
            "main_category": str(row.get("main_category", "unknown")),
            "subcategory": str(row.get("subcategory", "")),
            "price": float(row["price"]) if pd.notna(row["price"]) else -1.0,
            "average_rating": float(row["average_rating"]) if pd.notna(row.get("average_rating")) else 0.0,
            "image": str(row.get("image", ""))[:300],
        }
        metadata_list.append(meta)
    return metadata_list


def embed_and_store(df: pd.DataFrame):
    logger.info("加载 BGE-M3 模型...")
    model = BGEM3FlagModel(MODEL_PATH, use_fp16=True, device="cuda")
    logger.info("模型加载完成")

    texts = df["search_text"].fillna("").tolist()
    ids = [f"prod_{i}" for i in range(len(texts))]
    metadata = build_metadata(df)

    # 初始化 ChromaDB（持久化到磁盘）
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # 如果 collection 已存在则先删除（保证幂等）
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        logger.info("已删除旧 collection")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("ChromaDB collection '%s' 已创建", COLLECTION_NAME)

    # 分批 encode 并写入
    total = len(texts)
    start_time = time.time()
    encoded_total = 0

    logger.info("开始生成 embedding，共 %d 条，批次大小 %d", total, BATCH_SIZE)

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_texts = texts[batch_start:batch_end]
        batch_ids = ids[batch_start:batch_end]
        batch_meta = metadata[batch_start:batch_end]

        # BGE-M3 encode
        embeddings = model.encode(
            batch_texts,
            max_length=MAX_LENGTH,
            batch_size=len(batch_texts),
        )["dense_vecs"]

        # 确保为 float32 list（ChromaDB 要求）
        emb_list = embeddings.astype("float32").tolist()

        collection.add(
            ids=batch_ids,
            embeddings=emb_list,
            documents=batch_texts,
            metadatas=batch_meta,
        )

        encoded_total += len(batch_texts)
        if batch_start % (BATCH_SIZE * 50) == 0 or batch_end == total:
            elapsed = time.time() - start_time
            speed = encoded_total / elapsed if elapsed > 0 else 0
            logger.info(
                "进度: %d/%d (%.1f%%), 速度: %.0f 条/秒",
                encoded_total, total, encoded_total / total * 100, speed,
            )

    elapsed = time.time() - start_time
    logger.info("Embedding 完成！共 %d 条，耗时 %.1f 秒 (%.0f 条/秒)", total, elapsed, total / elapsed)
    logger.info("ChromaDB 持久化目录: %s", CHROMA_DIR)

    return collection


def test_retrieval(collection):
    """用中英文查询测试检索效果。"""
    test_queries = [
        "red hiking backpack under 200 dollars",         # 英文 - 结构化条件
        "200美元以内的红色登山背包",                       # 中文 - 跨语言
        "wireless bluetooth earbuds for running",         # 英文 - 语义
        "适合户外露营的防水帐篷",                          # 中文 - 语义场景
        "lightweight laptop for students",                # 英文 - 用途
    ]

    logger.info("=" * 60)
    logger.info("检索效果测试")
    logger.info("=" * 60)

    model = BGEM3FlagModel(MODEL_PATH, use_fp16=True, device="cuda")

    for query in test_queries:
        q_emb = model.encode([query], max_length=MAX_LENGTH)["dense_vecs"].astype("float32").tolist()

        results = collection.query(
            query_embeddings=q_emb,
            n_results=3,
        )

        logger.info("\n查询: %s", query)
        for i, (doc_id, meta, dist) in enumerate(
            zip(results["ids"][0], results["metadatas"][0], results["distances"][0])
        ):
            logger.info(
                "  #%d [%s] %s | brand=%s, price=%.2f, dist=%.4f",
                i + 1, doc_id, meta["title"][:60], meta["brand"], meta["price"], dist,
            )


if __name__ == "__main__":
    df = load_data()
    collection = embed_and_store(df)
    test_retrieval(collection)
