"""
电商商品智能问答 Agent — 数据治理脚本

原始数据: Amazon 商品数据 (train-00000-of-00001.parquet)
治理目标: 为 Agent 的混合检索（结构化过滤 + 语义检索）准备高质量数据

治理内容:
  1. 缺失值处理 — 关键字段缺失的记录标记/删除
  2. store 字段清洗 — 提取品牌名，去除无关信息
  3. 文本清洗 — 去 HTML 标签、特殊字符、多余空白
  4. features 字段解析 — ndarray → 拼接文本
  5. categories 字段解析 — ndarray → 提取子类别
  6. details 字段解析 — 字典字符串 → 提取关键属性
  7. 价格异常处理 — 0/极端值标记
  8. 拼接搜索文本 — title + description + features → 统一检索文本
  9. 输出治理后数据 + 数据质量报告

输出:
  - data/cleaned_products.parquet  治理后的数据
  - data/governance_report.txt     数据质量报告
"""

import pandas as pd
import numpy as np
import re
import os
from datetime import datetime

# ============================================================
# 路径定位（脚本可从任意目录运行）
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ============================================================
# 配置
# ============================================================
INPUT_FILE = os.path.join(DATA_DIR, "train-00000-of-00001.parquet")
OUTPUT_FILE = os.path.join(DATA_DIR, "cleaned_products.parquet")
REPORT_FILE = os.path.join(DATA_DIR, "governance_report.txt")

# 价格合理范围
PRICE_MIN = 0.01
PRICE_MAX = 50000.0

# ============================================================
# 工具函数
# ============================================================
report_lines = []

def log(msg: str, print_too: bool = True):
    """同时输出到控制台和报告"""
    if print_too:
        print(msg)
    report_lines.append(msg)


def clean_text(text: str) -> str:
    """
    清洗文本:
    - 去除 HTML 标签
    - 去除多余空白（换行、制表符、连续空格）
    - 保留基本标点
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 去除 HTML 标签
    text = re.sub(r"<[^>]+>", "", text)

    # 替换 HTML 实体
    html_entities = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&#39;": "'", "&nbsp;": " ",
    }
    for entity, char in html_entities.items():
        text = text.replace(entity, char)

    # 去除制表符、换行符，统一为空格
    text = re.sub(r"[\t\r\n]+", " ", text)

    # 多个连续空格合并为一个
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def extract_brand_from_store(store_str: str) -> str:
    """
    从 store 字段提取品牌名。
    store 字段样例:
      - "Scritti Politti   Format: Audio CD"       → "Scritti Politti"
      - "38 Special  (Contributor)    Format: ..."  → "38 Special"
      - "Susan Boyle   Format: Audio CD"            → "Susan Boyle"
      - "Nike"                                       → "Nike"
    """
    if not isinstance(store_str, str) or not store_str.strip():
        return ""

    # 去掉 "Format: ..." 及其后内容
    brand = re.split(r"\s+Format\s*:", store_str)[0]

    # 去掉 "(Contributor)"、"(Artist)" 等括号标注
    brand = re.sub(r"\s*\((?:Contributor|Artist|Author|Manufacturer)\)", "", brand)

    # 去掉多余的逗号及后面的其他品牌/贡献者（以逗号分隔的多个贡献者，取第一个作为品牌）
    # 例: "Brand A (Contributor),     Brand B (Contributor)" → "Brand A"
    brand = brand.split(",")[0]

    return brand.strip()


def parse_details(details_str: str) -> dict:
    """
    解析 details 字典字符串，提取关键属性。
    输入样例: "{'Package Dimensions': '5.55 x 4.92 x 0.51 inches', ...}"
    输出: {"package_dimensions": "5.55 x 4.92 x 0.51 inches", ...}
    """
    if not isinstance(details_str, str) or not details_str.strip():
        return {}

    # 用 eval 安全替代：ast.literal_eval
    try:
        import ast
        details_dict = ast.literal_eval(details_str)
        if not isinstance(details_dict, dict):
            return {}
    except (ValueError, SyntaxError):
        return {}

    # 标准化 key：小写 + 下划线
    result = {}
    key_map = {
        "Package Dimensions": "package_dimensions",
        "Item Weight": "item_weight",
        "Date First Available": "date_first_available",
        "Country of Origin": "country_of_origin",
        "Number of discs": "number_of_discs",
        "Run time": "run_time",
        "Brand": "brand",
        "Material": "material",
        "Color": "color",
        "Size": "size",
        "Weight": "weight",
    }
    for k, v in details_dict.items():
        std_key = key_map.get(k, k.lower().replace(" ", "_"))
        result[std_key] = str(v).strip()

    return result


def ndarray_to_str(val) -> str:
    """将 ndarray/list 字段转为拼接字符串"""
    if isinstance(val, np.ndarray):
        val = val.tolist()
    if isinstance(val, list):
        # 过滤空字符串
        items = [str(v).strip() for v in val if str(v).strip()]
        return " | ".join(items)
    if isinstance(val, str):
        return val.strip()
    return ""


# ============================================================
# 主流程
# ============================================================
def main():
    start_time = datetime.now()
    log("=" * 70)
    log(f"数据治理开始 — {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    # ----------------------------------------------------------
    # Step 0: 加载数据
    # ----------------------------------------------------------
    log("\n" + "=" * 70)
    log("Step 0: 加载原始数据")
    log("=" * 70)
    df = pd.read_parquet(INPUT_FILE)
    log(f"  原始数据: {df.shape[0]} 行 × {df.shape[1]} 列")
    original_count = len(df)

    # ----------------------------------------------------------
    # Step 1: 缺失值处理
    # ----------------------------------------------------------
    log("\n" + "=" * 70)
    log("Step 1: 缺失值处理")
    log("=" * 70)

    # 1a. title 和 description 是核心字段，缺失的记录删除
    before = len(df)
    df = df.dropna(subset=["title"]).copy()
    after = len(df)
    log(f"  [title 缺失] 删除 {before - after} 条 (title 为空)")

    before = len(df)
    df = df[df["description"].notna() & (df["description"].astype(str).str.strip() != "")].copy()
    after = len(df)
    log(f"  [description 缺失/空] 删除 {before - after} 条")

    # 1b. price 缺失 30.6%，不删除，标记为 NaN（后续 Agent 查询时做过滤提示）
    price_null = df["price"].isnull().sum()
    log(f"  [price 缺失] {price_null} 条 ({price_null/len(df)*100:.1f}%) — 保留，标记为无价格")

    # 1c. main_category 缺失 21.2%，用 categories 字段的第一级补全
    cat_filled = 0
    for idx in df[df["main_category"].isnull()].index:
        cats = df.loc[idx, "categories"]
        if isinstance(cats, np.ndarray) and len(cats) > 0:
            df.loc[idx, "main_category"] = str(cats[0])
            cat_filled += 1
        elif isinstance(cats, list) and len(cats) > 0:
            df.loc[idx, "main_category"] = str(cats[0])
            cat_filled += 1
    remaining_null = df["main_category"].isnull().sum()
    log(f"  [main_category 缺失] 用 categories 第一级补全 {cat_filled} 条，仍缺失 {remaining_null} 条")
    # 剩余无法补全的标记为 "Unknown"
    df["main_category"] = df["main_category"].fillna("Unknown")
    log(f"  [main_category] 剩余缺失 → 标记为 'Unknown'")

    log(f"  处理后数据量: {len(df)} 行 (原始 {original_count} 行)")

    # ----------------------------------------------------------
    # Step 2: store 字段清洗 → 提取品牌名
    # ----------------------------------------------------------
    log("\n" + "=" * 70)
    log("Step 2: store 字段清洗 → 提取品牌名")
    log("=" * 70)

    df["brand"] = df["store"].apply(lambda x: extract_brand_from_store(x) if pd.notna(x) else "")
    brand_empty = (df["brand"] == "").sum()
    log(f"  提取 brand 字段完成")
    log(f"  brand 为空: {brand_empty} 条 ({brand_empty/len(df)*100:.1f}%)")

    # 展示几个样例
    sample_idx = df[df["brand"] != ""].head(5).index
    for idx in sample_idx:
        log(f"    store: '{df.loc[idx, 'store'][:60]}' → brand: '{df.loc[idx, 'brand']}'", print_too=False)
    log(f"  样例: (见报告文件)")

    # ----------------------------------------------------------
    # Step 3: 文本清洗
    # ----------------------------------------------------------
    log("\n" + "=" * 70)
    log("Step 3: 文本清洗 (title, description)")
    log("=" * 70)

    df["title_clean"] = df["title"].apply(clean_text)
    df["description_clean"] = df["description"].apply(clean_text)

    title_len_before = df["title"].str.len().mean()
    title_len_after = df["title_clean"].str.len().mean()
    desc_len_before = df["description"].str.len().mean()
    desc_len_after = df["description_clean"].str.len().mean()

    log(f"  title 平均长度: {title_len_before:.0f} → {title_len_after:.0f}")
    log(f"  description 平均长度: {desc_len_before:.0f} → {desc_len_after:.0f}")

    # 检查是否有清洗后变成空的
    title_empty = (df["title_clean"] == "").sum()
    desc_empty = (df["description_clean"] == "").sum()
    log(f"  title 清洗后为空: {title_empty}")
    log(f"  description 清洗后为空: {desc_empty}")

    # ----------------------------------------------------------
    # Step 4: features 字段解析
    # ----------------------------------------------------------
    log("\n" + "=" * 70)
    log("Step 4: features 字段解析")
    log("=" * 70)

    df["features_text"] = df["features"].apply(ndarray_to_str)
    feat_empty = (df["features_text"] == "").sum()
    feat_has_content = len(df) - feat_empty
    log(f"  features 有内容: {feat_has_content} ({feat_has_content/len(df)*100:.1f}%)")
    log(f"  features 为空: {feat_empty} ({feat_empty/len(df)*100:.1f}%)")

    # features 文本长度
    feat_lengths = df[df["features_text"] != ""]["features_text"].str.len()
    if len(feat_lengths) > 0:
        log(f"  features 文本长度: min={feat_lengths.min()}, median={feat_lengths.median():.0f}, "
            f"mean={feat_lengths.mean():.0f}, max={feat_lengths.max()}")

    # ----------------------------------------------------------
    # Step 5: categories 字段解析
    # ----------------------------------------------------------
    log("\n" + "=" * 70)
    log("Step 5: categories 字段解析")
    log("=" * 70)

    df["categories_text"] = df["categories"].apply(ndarray_to_str)
    cat_empty = (df["categories_text"] == "").sum()
    log(f"  categories 有内容: {len(df) - cat_empty} ({(len(df)-cat_empty)/len(df)*100:.1f}%)")
    log(f"  categories 为空: {cat_empty} ({cat_empty/len(df)*100:.1f}%)")

    # 提取子类别（最后一层）
    def get_subcategory(cats):
        if isinstance(cats, np.ndarray):
            cats = cats.tolist()
        if isinstance(cats, list) and len(cats) > 0:
            return str(cats[-1]).strip()
        return ""

    df["subcategory"] = df["categories"].apply(get_subcategory)
    subcat_count = df["subcategory"].nunique()
    log(f"  子类别(subcategory)唯一数: {subcat_count}")

    # ----------------------------------------------------------
    # Step 6: details 字段解析
    # ----------------------------------------------------------
    log("\n" + "=" * 70)
    log("Step 6: details 字段解析")
    log("=" * 70)

    details_parsed = df["details"].apply(parse_details)
    details_df = pd.DataFrame(details_parsed.tolist(), index=df.index)

    # 提取几个关键属性到主 DataFrame
    key_attrs = ["color", "material", "size", "brand", "country_of_origin"]
    for attr in key_attrs:
        if attr in details_df.columns:
            df[f"detail_{attr}"] = details_df[attr].fillna("")
        else:
            df[f"detail_{attr}"] = ""

    for attr in key_attrs:
        non_empty = (df[f"detail_{attr}"] != "").sum()
        log(f"  detail_{attr}: 有值 {non_empty} ({non_empty/len(df)*100:.1f}%)")

    # details 拼接为文本（用于检索）
    def details_to_text(d):
        if not isinstance(d, str):
            return ""
        parsed = parse_details(d)
        if not parsed:
            return ""
        parts = [f"{k}: {v}" for k, v in parsed.items() if v]
        return " | ".join(parts)

    df["details_text"] = df["details"].apply(details_to_text)

    # ----------------------------------------------------------
    # Step 7: 价格异常处理
    # ----------------------------------------------------------
    log("\n" + "=" * 70)
    log("Step 7: 价格异常处理")
    log("=" * 70)

    price_valid = df["price"].notna()
    price_zero = (df["price"] == 0).sum()
    price_extreme = (df["price"] > PRICE_MAX).sum()

    log(f"  价格=0: {price_zero} 条 → 标记为 NaN")
    log(f"  价格>{PRICE_MAX}: {price_extreme} 条 → 标记为 NaN")
    log(f"  价格范围限制: [{PRICE_MIN}, {PRICE_MAX}]")

    # 将异常价格置为 NaN
    df.loc[df["price"] == 0, "price"] = np.nan
    df.loc[df["price"] > PRICE_MAX, "price"] = np.nan

    price_final_valid = df["price"].notna().sum()
    log(f"  治理后有效价格: {price_final_valid} ({price_final_valid/len(df)*100:.1f}%)")

    # ----------------------------------------------------------
    # Step 8: 拼接统一搜索文本
    # ----------------------------------------------------------
    log("\n" + "=" * 70)
    log("Step 8: 拼接统一搜索文本 (search_text)")
    log("=" * 70)

    def build_search_text(row):
        """拼接所有文本信息为统一检索文本"""
        parts = []

        # 商品名称（权重最高）
        if row["title_clean"]:
            parts.append(f"Product: {row['title_clean']}")

        # 品牌
        if row.get("brand", ""):
            parts.append(f"Brand: {row['brand']}")

        # 描述
        if row["description_clean"]:
            parts.append(f"Description: {row['description_clean']}")

        # 卖点/特性
        if row.get("features_text", ""):
            parts.append(f"Features: {row['features_text']}")

        # 分类
        if row.get("categories_text", ""):
            parts.append(f"Categories: {row['categories_text']}")

        # 详情中的关键属性
        detail_parts = []
        for attr in ["color", "material", "size"]:
            val = row.get(f"detail_{attr}", "")
            if val:
                detail_parts.append(f"{attr}: {val}")
        if detail_parts:
            parts.append("Details: " + ", ".join(detail_parts))

        return "\n".join(parts)

    df["search_text"] = df.apply(build_search_text, axis=1)

    search_lengths = df["search_text"].str.len()
    log(f"  search_text 长度分布:")
    log(f"    min={search_lengths.min()}, median={search_lengths.median():.0f}, "
        f"mean={search_lengths.mean():.0f}, max={search_lengths.max()}")
    search_empty = (df["search_text"].str.strip() == "").sum()
    log(f"  search_text 为空: {search_empty}")

    # ----------------------------------------------------------
    # Step 9: 选择输出字段 & 保存
    # ----------------------------------------------------------
    log("\n" + "=" * 70)
    log("Step 9: 整理输出字段 & 保存")
    log("=" * 70)

    output_cols = [
        # 原始核心字段
        "title", "description", "main_category", "price",
        "average_rating", "rating_number",
        # 清洗/增强字段
        "title_clean", "description_clean",
        "brand",
        "subcategory",
        "features_text",
        "categories_text",
        "details_text",
        "search_text",
        # detail 提取的关键属性
        "detail_color", "detail_material", "detail_size",
        "detail_brand", "detail_country_of_origin",
        # 原始字段（保留供参考）
        "store", "image",
    ]

    df_out = df[output_cols].copy()

    # 保存
    df_out.to_parquet(OUTPUT_FILE, index=False)
    log(f"  治理后数据已保存: {OUTPUT_FILE}")
    log(f"  最终数据: {df_out.shape[0]} 行 × {df_out.shape[1]} 列")

    # ----------------------------------------------------------
    # 数据质量摘要
    # ----------------------------------------------------------
    log("\n" + "=" * 70)
    log("数据治理摘要")
    log("=" * 70)
    log(f"  原始数据:   {original_count} 行")
    log(f"  治理后数据: {len(df_out)} 行 (删除 {original_count - len(df_out)} 条, "
        f"{(original_count - len(df_out))/original_count*100:.2f}%)")
    log(f"")
    log(f"  字段完整性:")
    for col in output_cols:
        if df_out[col].dtype == "object":
            non_empty = (df_out[col].notna() & (df_out[col].astype(str).str.strip() != "")).sum()
        else:
            non_empty = df_out[col].notna().sum()
        pct = non_empty / len(df_out) * 100
        log(f"    {col:35s} {non_empty:>7d} / {len(df_out)} ({pct:.1f}%)")

    log(f"\n  main_category 分布 (治理后):")
    for cat, count in df_out["main_category"].value_counts().head(15).items():
        log(f"    {str(cat):40s} {count:>6d} ({count/len(df_out)*100:.1f}%)")

    log(f"\n  价格统计 (治理后):")
    price_series = df_out["price"].dropna()
    if len(price_series) > 0:
        log(f"    有效: {len(price_series)}, 缺失: {df_out['price'].isnull().sum()}")
        log(f"    range: [{price_series.min():.2f}, {price_series.max():.2f}]")
        log(f"    mean={price_series.mean():.2f}, median={price_series.median():.2f}")

    # ----------------------------------------------------------
    # 保存报告
    # ----------------------------------------------------------
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    log(f"\n治理完成 — 耗时 {duration:.1f} 秒")
    log("=" * 70)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n报告已保存: {REPORT_FILE}")


if __name__ == "__main__":
    main()
