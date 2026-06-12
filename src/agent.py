"""
商品智能问答 Agent — 主程序
交互式对话，串联 LLM 查询解析 → 混合检索 → LLM 答案生成。

启动后会一次性加载所有模型（BGE-M3、ChromaDB、字段向量表），
然后进入交互循环，每轮查询复用已加载的模型。
"""

import sys
import json
import logging

from openai import OpenAI
from src.query_parser import parse_query, get_llm_client as get_parser_client
from src.hybrid_retriever import HybridRetriever
from src.answer_generator import generate_answer, generate_answer_stream

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ProductAgent:
    def __init__(self):
        logger.info("=" * 60)
        logger.info("初始化商品问答 Agent")
        logger.info("=" * 60)

        # 1. LLM client（查询解析 + 答案生成共用）
        self.llm_client = get_parser_client()
        logger.info("LLM 客户端初始化完成")

        # 2. 混合检索器（内含 BGE-M3 模型 + ChromaDB + 字段向量表）
        self.retriever = HybridRetriever()
        logger.info("检索器初始化完成")
        logger.info("=" * 60)
        logger.info("Agent 就绪，输入查询开始对话")
        logger.info("=" * 60)

    def chat(self, user_query: str, top_k: int = 5, verbose: bool = True) -> str:
        """单轮对话：查询 → 检索 → 回答。"""
        # 1. 查询解析
        parsed = parse_query(user_query, self.llm_client)
        if verbose:
            print(f"\n[解析] semantic_query: {parsed['semantic_query']}")
            print(f"[解析] extracted: {parsed['extracted']}")

        # 2. 混合检索
        results = self.retriever.retrieve(
            semantic_query=parsed["semantic_query"],
            extracted=parsed["extracted"],
            top_k=top_k,
        )

        if verbose:
            print(f"\n[检索] Top-{top_k} 候选:")
            for i, r in enumerate(results):
                score_str = f"final={r['final_score']:.3f}"
                field_str = f"field={r['field_score']:.3f}" if r["field_score"] is not None else "field=N/A"
                print(f"  #{i+1} [{score_str}, {field_str}] {r['title'][:50]} | {r['brand']} | ${r['price']}")

        # 3. 答案生成
        answer = generate_answer(user_query, results, self.llm_client)
        return answer

    def chat_stream(self, user_query: str, top_k: int = 5):
        """流式对话：yield SSE 事件 dict（event/data）。

        事件类型：
          - status: 进度提示
          - parsed: Agent 解析结果（semantic_query + extracted）
          - retrieved: Top-K 商品列表
          - token: 答案 token（流式）
          - done: 完成
          - error: 出错
        """
        try:
            # 1. 查询解析
            yield {"event": "status", "data": "正在解析查询..."}
            parsed = parse_query(user_query, self.llm_client)
            yield {
                "event": "parsed",
                "data": json.dumps({
                    "semantic_query": parsed["semantic_query"],
                    "extracted": parsed["extracted"],
                }, ensure_ascii=False),
            }

            # 2. 混合检索
            yield {"event": "status", "data": "正在检索商品..."}
            results = self.retriever.retrieve(
                semantic_query=parsed["semantic_query"],
                extracted=parsed["extracted"],
                top_k=top_k,
            )
            yield {
                "event": "retrieved",
                "data": json.dumps(results, ensure_ascii=False),
            }

            # 3. 流式答案生成
            yield {"event": "status", "data": "正在生成回答..."}
            for token in generate_answer_stream(user_query, results, self.llm_client):
                yield {"event": "token", "data": token}

            yield {"event": "done", "data": ""}
        except Exception as e:
            logger.error("流式处理出错: %s", e, exc_info=True)
            yield {"event": "error", "data": str(e)}


def main():
    try:
        agent = ProductAgent()
    except Exception as e:
        logger.error("初始化失败: %s", e)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("欢迎使用商品智能问答 Agent")
    print("输入你的查询（中文/英文均可），输入 'quit' 或 'exit' 退出")
    print("=" * 60 + "\n")

    # 预设示例查询（方便快速测试）
    examples = [
        "200美元以内适合徒步的红色背包",
        "推荐一款蓝牙耳机",
        "Nike 的运动鞋，预算100美元",
        "waterproof jacket under 100 dollars",
        "适合户外露营的防水帐篷",
    ]

    while True:
        try:
            user_input = input("🛒 你的查询> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if user_input == "examples":
            print("示例查询：")
            for i, ex in enumerate(examples, 1):
                print(f"  {i}. {ex}")
            continue

        try:
            answer = agent.chat(user_input)
            print("\n" + "=" * 60)
            print("🤖 Agent 回答：")
            print("=" * 60)
            print(answer)
            print("\n" + "=" * 60 + "\n")
        except Exception as e:
            logger.error("处理查询时出错: %s", e)
            print(f"\n❌ 出错了: {e}\n")


if __name__ == "__main__":
    main()
