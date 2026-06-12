"""
商品智能问答 Agent — FastAPI Web 服务

启动时一次性加载 ProductAgent（BGE-M3 + ChromaDB + 字段向量表常驻内存），
通过 SSE 流式返回查询结果。

启动：
    conda run -n py312 uvicorn web.server:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import logging

# 确保能 import 项目根目录的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agent import ProductAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="商品智能问答 Agent")

# 模型常驻：启动时一次性加载
logger.info("=" * 60)
logger.info("初始化 ProductAgent（模型常驻内存）")
logger.info("=" * 60)
agent = ProductAgent()
logger.info("=" * 60)
logger.info("Agent 初始化完成，服务就绪")
logger.info("=" * 60)


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/")
async def index():
    """返回前端页面。"""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """SSE 流式接口。"""
    query = req.query.strip()
    if not query:
        return JSONResponse({"error": "查询不能为空"}, status_code=400)

    logger.info("收到查询: %s", query)

    def event_generator():
        for event in agent.chat_stream(query, top_k=req.top_k):
            # SSE 协议：data 内的换行要拆成多行，每行前缀 "data: "
            data = event["data"]
            lines = data.split("\n")
            data_block = "\n".join(f"data: {line}" for line in lines)
            yield f"event: {event['event']}\n{data_block}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲（如果有反代）
        },
    )


@app.get("/api/health")
async def health():
    """健康检查。"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
