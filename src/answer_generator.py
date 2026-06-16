"""
答案生成模块
基于检索结果（Top-K 商品）和原始用户查询，用 LLM 生成带推荐理由的中文回答。
"""

import os
import logging
from typing import Optional

from langfuse.openai import OpenAI
from langfuse import observe
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY", "")
ZHIPU_BASE_URL = os.environ.get("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
ZHIPU_MODEL = os.environ.get("ZHIPU_MODEL", "glm-4-flash")


GENERATOR_SYSTEM_PROMPT = """你是一个电商商品推荐助手。用户提出了一个商品查询，系统已经检索出最相关的几款商品。

你的任务：
1. 用中文回答用户
2. 从检索结果中挑选最符合用户需求的商品进行推荐
3. 每个推荐要包含：商品名称、品牌、价格、简短的推荐理由（结合用户需求说明为什么推荐这款）
4. 如果检索结果中有明显不符合的，可以不推荐或只推荐最相关的
5. 回答风格要自然、有帮助，像真人导购
6. 价格为 -1 或缺失时，标注为"价格待定"

输出格式（Markdown）：
针对您的需求，为您推荐以下商品：

**1. 商品名称**
- 品牌：XXX
- 价格：$XXX
- 推荐理由：XXX（结合用户提到的场景、预算、偏好等说明）

**2. 商品名称**
...

如果检索结果与用户需求完全不相关，诚实告知并建议调整查询。"""


def get_llm_client() -> OpenAI:
    if not ZHIPU_API_KEY:
        raise ValueError(
            "未配置 ZHIPU_API_KEY，请在 .env 文件中设置（参考 .env.example）"
        )
    return OpenAI(api_key=ZHIPU_API_KEY, base_url=ZHIPU_BASE_URL)


def format_products_for_prompt(products: list[dict]) -> str:
    """把检索结果格式化为 LLM 可读的商品列表。"""
    if not products:
        return "（无匹配商品）"

    lines = []
    for i, p in enumerate(products, 1):
        price_str = f"${p['price']:.2f}" if p.get("price", -1) > 0 else "价格待定"
        rating = p.get("average_rating", 0)
        rating_str = f"{rating:.1f}" if rating > 0 else "无评分"
        lines.append(
            f"{i}. 商品名: {p['title']}\n"
            f"   品牌: {p['brand']} | 价格: {price_str} | 评分: {rating_str} | 类目: {p['main_category']}"
        )
    return "\n".join(lines)


@observe(name="generate_answer", as_type="chain")
def generate_answer(
    user_query: str,
    products: list[dict],
    client: Optional[OpenAI] = None,
) -> str:
    """生成推荐回答。"""
    if client is None:
        client = get_llm_client()

    if not products:
        return (
            f"抱歉，没有找到与「{user_query}」相关的商品。"
            "您可以尝试调整查询条件，比如放宽价格范围或更换关键词。"
        )

    products_text = format_products_for_prompt(products)
    user_prompt = (
        f"用户的原始查询：{user_query}\n\n"
        f"系统检索到的候选商品（按相关性排序）：\n{products_text}\n\n"
        f"请根据用户需求生成推荐回答："
    )

    logger.info("生成回答，候选商品数: %d", len(products))

    response = client.chat.completions.create(
        model=ZHIPU_MODEL,
        messages=[
            {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )

    answer = response.choices[0].message.content
    logger.info("回答生成完成")
    return answer


@observe(name="generate_answer_stream", as_type="chain")
def generate_answer_stream(
    user_query: str,
    products: list[dict],
    client: Optional[OpenAI] = None,
):
    """流式生成回答，逐 token yield。"""
    if client is None:
        client = get_llm_client()

    if not products:
        yield (
            f"抱歉，没有找到与「{user_query}」相关的商品。"
            "您可以尝试调整查询条件，比如放宽价格范围或更换关键词。"
        )
        return

    products_text = format_products_for_prompt(products)
    user_prompt = (
        f"用户的原始查询：{user_query}\n\n"
        f"系统检索到的候选商品（按相关性排序）：\n{products_text}\n\n"
        f"请根据用户需求生成推荐回答："
    )

    logger.info("流式生成回答，候选商品数: %d", len(products))

    response = client.chat.completions.create(
        model=ZHIPU_MODEL,
        messages=[
            {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        stream=True,
    )

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

    logger.info("流式生成完成")


if __name__ == "__main__":
    # 用模拟商品数据测试（不依赖检索）
    mock_products = [
        {
            "title": "Nike Air Max AP Running Shoes",
            "brand": "Nike",
            "price": 89.99,
            "average_rating": 4.5,
            "main_category": "AMAZON FASHION",
        },
        {
            "title": "Nike Men's Gymnastics Shoes",
            "brand": "Nike",
            "price": 55.97,
            "average_rating": 4.3,
            "main_category": "AMAZON FASHION",
        },
    ]

    client = get_llm_client()
    answer = generate_answer("Nike 的运动鞋，预算100美元", mock_products, client)
    print(answer)
