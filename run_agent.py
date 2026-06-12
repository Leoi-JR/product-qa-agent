"""
商品智能问答 Agent — CLI 入口

便捷启动入口，避免每次都要输入 `python -m src.agent`。

用法：
    conda run -n py312 python run_agent.py
"""

from src.agent import main

if __name__ == "__main__":
    main()
