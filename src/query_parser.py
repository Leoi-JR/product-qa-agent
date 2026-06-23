"""
查询解析模块
用 LLM 将用户自然语言查询解析为：
  - semantic_query: 用于向量检索的自然语言句子（英文）
  - extracted: 结构化字段（price_min/max, brand, color, category），未提及为 null
"""

import json
import os
import logging
from typing import Optional

from langfuse.openai import OpenAI
from langfuse import observe
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

QIANFAN_API_KEY = os.environ.get("QIANFAN_API_KEY", "")
QIANFAN_BASE_URL = os.environ.get("QIANFAN_BASE_URL", "")
QIANFAN_MODEL = os.environ.get("QIANFAN_MODEL", "")


PARSER_SYSTEM_PROMPT = """你是一个商品查询解析器。用户的输入是关于电商商品的自然语言查询（可能是中文或英文）。

你的任务是把用户的查询解析为两部分：
1. semantic_query：把用户的意图润色为一句适合向量检索的完整英文描述句（不是关键词堆叠），用于和商品描述做语义相似度匹配。
2. extracted：从查询中提取的结构化过滤条件。

提取规则：
- price_min / price_max：价格区间（数字，无单位）。如"200美元以内" → price_max=200。未提及则 null。
- brand：品牌名。用户提到品牌才填，用英文标准写法。未提及则 null。
- color：颜色。用户提到颜色才填，用英文单词。未提及则 null。
- category：商品**大类**（不是具体商品类型）。只有当用户明确提到宽泛的商品大类时才填。例如：
  - "电子产品" / "electronics" → category="Electronics"
  - "服装" / "clothing" → category="Clothing"
  - "户外用品" / "outdoor gear" → category="Sports Outdoors"
  具体商品类型（如"背包"、"耳机"、"鞋子"、"帐篷"）**不要**填到 category，这些已经包含在 semantic_query 里了。未提及大类则 null。

注意事项：
- 只提取用户**明确提到**的条件，不要猜测或推断。
- semantic_query 必须是完整的英文句子，例如 "A red backpack suitable for hiking and outdoor adventures"，而不是 "red backpack hiking"。
- semantic_query 中不要包含价格信息（价格用 extracted 字段表达）。
- 如果用户没有提到任何结构化条件，extracted 全部为 null，semantic_query 仍然要生成。

严格输出以下 JSON 格式，不要有任何额外文字：
{
  "semantic_query": "...",
  "extracted": {
    "price_min": null,
    "price_max": null,
    "brand": null,
    "color": null,
    "category": null
  }
}"""


PARSER_USER_TEMPLATE = """用户查询：{query}

请解析为 JSON："""


def get_llm_client() -> OpenAI:
    if not QIANFAN_API_KEY:
        raise ValueError(
            "未配置 QIANFAN_API_KEY，请在 .env 文件中设置（参考 .env.example）"
        )
    if not QIANFAN_BASE_URL:
        raise ValueError(
            "未配置 QIANFAN_BASE_URL，请在 .env 文件中设置（参考 .env.example）"
        )
    return OpenAI(api_key=QIANFAN_API_KEY, base_url=QIANFAN_BASE_URL)


@observe(name="parse_query", as_type="chain")
def parse_query(query: str, client: Optional[OpenAI] = None) -> dict:
    """解析用户查询，返回 {semantic_query, extracted}。"""
    if client is None:
        client = get_llm_client()

    logger.info("解析查询: %s", query)

    response = client.chat.completions.create(
        model=QIANFAN_MODEL,
        messages=[
            {"role": "system", "content": PARSER_SYSTEM_PROMPT},
            {"role": "user", "content": PARSER_USER_TEMPLATE.format(query=query)},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    logger.info("LLM 原始输出: %s", content)

    try:
        result = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error("JSON 解析失败: %s", e)
        raise

    # 兜底：确保 extracted 字段完整
    extracted = result.get("extracted", {})
    for key in ["price_min", "price_max", "brand", "color", "category"]:
        if key not in extracted:
            extracted[key] = None
    result["extracted"] = extracted
    result["original_query"] = query

    return result


if __name__ == "__main__":
    # 测试样例
    test_queries = [
        "200美元以内适合徒步的红色背包",
        "推荐一款蓝牙耳机",
        "Nike 的运动鞋，预算100美元",
        "waterproof jacket under 100 dollars",
        "好看的",
    ]

    client = get_llm_client()
    for q in test_queries:
        print("\n" + "=" * 60)
        result = parse_query(q, client)
        print(f"semantic_query: {result['semantic_query']}")
        print(f"extracted: {result['extracted']}")
