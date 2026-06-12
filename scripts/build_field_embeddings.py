"""
字段唯一值向量表预计算
为 brand / color / category 的唯一值生成 BGE-M3 向量，存为 npz。
同时保存每个商品的字段值索引，供查询时快速 lookup。
"""

import os
import time
import logging

import time
import logging

import numpy as np
import pandas as pd
from FlagEmbedding import BGEM3FlagModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 路径定位（脚本可从任意目录运行）─────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models", "bge-m3")

DATA_PATH = os.path.join(DATA_DIR, "cleaned_products.parquet")
OUTPUT_EMBEDDINGS = os.path.join(DATA_DIR, "field_embeddings.npz")
OUTPUT_INDICES = os.path.join(DATA_DIR, "product_field_indices.parquet")

FIELDS = ["brand", "detail_color", "main_category"]
MISSING_TOKEN = ""           # NaN / 空字符串统一映射到空，对应索引 -1
BATCH_SIZE = 256


def collect_unique_values(df: pd.DataFrame) -> dict[str, list[str]]:
    """提取每个字段的唯一非空值列表。"""
    unique_map = {}
    for field in FIELDS:
        s = df[field].fillna(MISSING_TOKEN).astype(str)
        s = s.str.strip()
        # 过滤掉空、unknown 占位
        s = s.replace({"": np.nan, "unknown": np.nan, "nan": np.nan}).dropna()
        unique_vals = sorted(s.unique().tolist())
        unique_map[field] = unique_vals
        logger.info("%s: %d 个唯一值", field, len(unique_vals))
    return unique_map


def embed_unique_values(model, values: list[str]) -> np.ndarray:
    """对唯一值列表批量编码，返回 (N, 1024) fp16 数组。"""
    if not values:
        return np.zeros((0, 1024), dtype=np.float16)
    logger.info("编码 %d 个唯一值...", len(values))
    start = time.time()
    embs = model.encode(
        values,
        max_length=64,           # 字段值都很短，64 tokens 足够
        batch_size=BATCH_SIZE,
    )["dense_vecs"].astype(np.float16)
    logger.info("完成，耗时 %.1f 秒", time.time() - start)
    return embs


def build_product_indices(df: pd.DataFrame, value_to_idx: dict[str, dict[str, int]]) -> pd.DataFrame:
    """为每个商品构建字段值索引（-1 表示缺失）。"""
    out = pd.DataFrame(index=df.index)
    for field in FIELDS:
        s = df[field].fillna(MISSING_TOKEN).astype(str).str.strip()
        s = s.replace({"": np.nan, "unknown": np.nan, "nan": np.nan})
        mapping = value_to_idx[field]
        out[field + "_idx"] = s.map(lambda v: mapping.get(v, -1)).astype(np.int32)
    return out


def main():
    logger.info("加载清洗后的数据: %s", DATA_PATH)
    df = pd.read_parquet(DATA_PATH)

    logger.info("加载 BGE-M3 模型...")
    model = BGEM3FlagModel(MODEL_PATH, use_fp16=True, device="cuda")

    # 1. 收集唯一值
    unique_map = collect_unique_values(df)

    # 2. 编码每个字段的唯一值
    embeddings_map = {}
    value_to_idx = {}
    for field, values in unique_map.items():
        embs = embed_unique_values(model, values)
        embeddings_map[field + "_values"] = np.array(values, dtype=object)
        embeddings_map[field + "_embeddings"] = embs
        value_to_idx[field] = {v: i for i, v in enumerate(values)}

    # 3. 构建商品→字段值索引
    logger.info("构建商品字段索引...")
    indices_df = build_product_indices(df, value_to_idx)

    # 4. 保存
    logger.info("保存字段向量表到 %s", OUTPUT_EMBEDDINGS)
    np.savez(OUTPUT_EMBEDDINGS, **embeddings_map)

    logger.info("保存商品字段索引到 %s", OUTPUT_INDICES)
    indices_df.to_parquet(OUTPUT_INDICES)

    # 验证
    logger.info("=" * 50)
    logger.info("验证:")
    for field in FIELDS:
        n_unique = len(embeddings_map[field + "_values"])
        n_with_value = (indices_df[field + "_idx"] >= 0).sum()
        logger.info(
            "  %s: %d 唯一值, %d/%d 商品有值 (%.1f%%)",
            field, n_unique, n_with_value, len(df), n_with_value / len(df) * 100,
        )


if __name__ == "__main__":
    main()
