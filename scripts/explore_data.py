"""临时脚本：探索 parquet 数据的结构和内容样式（修复版）"""

import os
import pandas as pd
import numpy as np

# ── 路径定位 ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(BASE_DIR, "data", "train-00000-of-00001.parquet")

df = pd.read_parquet(PARQUET_PATH)
print(f"数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")

# ============================================================
# 1. 字段名及类型
# ============================================================
print("\n" + "=" * 60)
print("1. 字段名及类型")
print("=" * 60)
for col in df.columns:
    # 检测实际存储的元素类型
    sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
    elem_type = type(sample).__name__ if sample is not None else "all-null"
    print(f"  {col:20s} dtype={str(df[col].dtype):>10s}  元素类型={elem_type}")

# ============================================================
# 2. 前 3 条数据逐字段展示
# ============================================================
print("\n" + "=" * 60)
print("2. 前 3 条数据逐字段展示")
print("=" * 60)
for col in df.columns:
    print(f"\n--- {col} ---")
    for i in range(min(3, len(df))):
        val = df[col].iloc[i]
        val_str = str(val)
        if len(val_str) > 300:
            val_str = val_str[:300] + f" ... (共 {len(str(val))} 字符)"
        print(f"  [{i}] {val_str}")

# ============================================================
# 3. 缺失值统计
# ============================================================
print("\n" + "=" * 60)
print("3. 缺失值统计")
print("=" * 60)
null_counts = df.isnull().sum()
null_pct = (df.isnull().sum() / len(df) * 100).round(2)
null_df = pd.DataFrame({"缺失数量": null_counts, "缺失比例%": null_pct})
null_df = null_df[null_df["缺失数量"] > 0].sort_values("缺失比例%", ascending=False)
if len(null_df) > 0:
    print(null_df.to_string())
else:
    print("  无缺失值")

# ============================================================
# 4. 每个字段的唯一值数量（安全处理 list/dict/array 类型）
# ============================================================
print("\n" + "=" * 60)
print("4. 每个字段的唯一值数量")
print("=" * 60)
for col in df.columns:
    series = df[col].dropna()
    if len(series) == 0:
        print(f"  {col:20s} unique=0 (全空)")
        continue
    sample = series.iloc[0]
    # 对 list/dict/ndarray 类型，转为字符串再计算
    if isinstance(sample, (list, dict, np.ndarray)):
        n_unique = series.astype(str).nunique()
    else:
        n_unique = series.nunique()
    print(f"  {col:20s} unique={n_unique:>8d}")

# ============================================================
# 5. main_category 分布
# ============================================================
print("\n" + "=" * 60)
print("5. main_category 分布（前 20）")
print("=" * 60)
cat_counts = df["main_category"].value_counts()
for cat, count in cat_counts.head(20).items():
    pct = count / len(df) * 100
    print(f"  {str(cat):40s} {count:>6d} ({pct:.1f}%)")
print(f"  {'NaN':40s} {df['main_category'].isnull().sum():>6d} ({df['main_category'].isnull().sum()/len(df)*100:.1f}%)")

# ============================================================
# 6. price 分布
# ============================================================
print("\n" + "=" * 60)
print("6. price 统计")
print("=" * 60)
price = df["price"].dropna()
print(f"  有效价格数量: {len(price)} / {len(df)} ({len(price)/len(df)*100:.1f}%)")
print(f"  min={price.min():.2f}, max={price.max():.2f}, mean={price.mean():.2f}, median={price.median():.2f}")
print(f"  价格=0: {(price == 0).sum()}, 价格<0: {(price < 0).sum()}, 价格>10000: {(price > 10000).sum()}")
print(f"\n  价格区间分布:")
bins = [0, 10, 25, 50, 100, 250, 500, 1000, 5000, 100000]
for i in range(len(bins) - 1):
    count = ((price >= bins[i]) & (price < bins[i+1])).sum()
    print(f"    ${bins[i]:>6} ~ ${bins[i+1]:<6}: {count:>6d} ({count/len(price)*100:.1f}%)")

# ============================================================
# 7. 文本字段长度分布
# ============================================================
print("\n" + "=" * 60)
print("7. 文本字段长度分布")
print("=" * 60)
text_cols = ["title", "description"]
for col in text_cols:
    series = df[col].dropna().astype(str)
    lengths = series.str.len()
    print(f"\n  --- {col} ---")
    print(f"    非空数量: {len(series)}")
    print(f"    长度: min={lengths.min()}, median={lengths.median():.0f}, "
          f"mean={lengths.mean():.0f}, max={lengths.max()}")
    empty = (series.str.strip() == "").sum()
    print(f"    空字符串: {empty}")

# ============================================================
# 8. features 字段分析
# ============================================================
print("\n" + "=" * 60)
print("8. features 字段分析")
print("=" * 60)
feat = df["features"].dropna()
print(f"  非空数量: {len(feat)}")

# 统计空列表 vs 非空列表
empty_list_count = 0
non_empty_count = 0
total_feat_count = 0
max_feat_len = 0
sample_feats = []

for idx, val in feat.items():
    if isinstance(val, (list, np.ndarray)):
        if len(val) == 0:
            empty_list_count += 1
        else:
            non_empty_count += 1
            total_feat_count += len(val)
            max_feat_len = max(max_feat_len, len(val))
            if len(sample_feats) < 3:
                sample_feats.append((idx, val))
    else:
        # 可能是字符串或其他类型
        if str(val).strip() == "" or str(val) == "[]":
            empty_list_count += 1
        else:
            non_empty_count += 1
            if len(sample_feats) < 3:
                sample_feats.append((idx, val))

print(f"  空列表 []: {empty_list_count}")
print(f"  有内容的: {non_empty_count}")
if non_empty_count > 0:
    print(f"  平均 features 数量: {total_feat_count / non_empty_count:.1f}")
    print(f"  最大 features 数量: {max_feat_len}")
    print(f"\n  有内容的 features 样例:")
    for idx, val in sample_feats:
        print(f"    [idx={idx}]")
        if isinstance(val, (list, np.ndarray)):
            for j, f in enumerate(val[:3]):
                f_str = str(f)
                if len(f_str) > 150:
                    f_str = f_str[:150] + "..."
                print(f"      feature[{j}]: {f_str}")
        else:
            print(f"      {str(val)[:200]}")

# ============================================================
# 9. details 字段分析
# ============================================================
print("\n" + "=" * 60)
print("9. details 字段分析（前 5 条）")
print("=" * 60)
details = df["details"].dropna()
print(f"  非空数量: {len(details)}")
# 看看 details 里有哪些 key
from collections import Counter
all_keys = Counter()
for val in details:
    if isinstance(val, dict):
        all_keys.update(val.keys())
print(f"\n  details 中最常见的 key（前 15）:")
for key, count in all_keys.most_common(15):
    print(f"    {key:40s} 出现 {count:>6d} 次")
print(f"\n  details 样例:")
for i in range(min(3, len(details))):
    val = details.iloc[i]
    print(f"    [{i}] {val}")

# ============================================================
# 10. categories 字段分析
# ============================================================
print("\n" + "=" * 60)
print("10. categories 字段分析")
print("=" * 60)
cats = df["categories"].dropna()
empty_cat = 0
non_empty_cat = 0
sample_cats = []
for val in cats:
    if isinstance(val, (list, np.ndarray)):
        if len(val) == 0:
            empty_cat += 1
        else:
            non_empty_cat += 1
            if len(sample_cats) < 5:
                sample_cats.append(val)
    else:
        if str(val).strip() in ("", "[]"):
            empty_cat += 1
        else:
            non_empty_cat += 1
print(f"  空列表 []: {empty_cat}")
print(f"  有内容的: {non_empty_cat}")
if sample_cats:
    print(f"\n  有内容的样例:")
    for i, cat in enumerate(sample_cats):
        cat_str = str(cat)
        if len(cat_str) > 200:
            cat_str = cat_str[:200] + "..."
        print(f"    [{i}] {cat_str}")

# ============================================================
# 11. store 字段样例
# ============================================================
print("\n" + "=" * 60)
print("11. store 字段样例（前 10 个非空值）")
print("=" * 60)
stores = df["store"].dropna()
for i in range(min(10, len(stores))):
    print(f"  [{i}] {str(stores.iloc[i])[:120]}")

# ============================================================
# 12. image 字段样例
# ============================================================
print("\n" + "=" * 60)
print("12. image 字段样例（前 3 个）")
print("=" * 60)
images = df["image"].dropna()
for i in range(min(3, len(images))):
    val = images.iloc[i]
    val_str = str(val)
    if len(val_str) > 150:
        val_str = val_str[:150] + "..."
    print(f"  [{i}] {val_str}")

print("\n" + "=" * 60)
print("探索完成")
print("=" * 60)
