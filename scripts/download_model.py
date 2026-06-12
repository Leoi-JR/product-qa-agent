"""
下载 BGE-M3 模型到本地目录。
用法:
    # 使用代理下载
    python download_model.py --proxy http://127.0.0.1:7890

    # 不使用代理（如果能直连 HuggingFace）
    python download_model.py
"""

import argparse
import os

from huggingface_hub import snapshot_download

# ── 路径定位 ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "models", "bge-m3")


def main():
    parser = argparse.ArgumentParser(description="下载 BGE-M3 模型")
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="HTTP 代理地址，如 http://127.0.0.1:7890",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"本地保存目录（默认: {DEFAULT_MODEL_DIR}）",
    )
    args = parser.parse_args()

    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy
        print(f"使用代理: {args.proxy}")

    print(f"下载 BAAI/bge-m3 → {args.local_dir}")
    path = snapshot_download(
        repo_id="BAAI/bge-m3",
        local_dir=args.local_dir,
    )
    print(f"下载完成: {path}")


if __name__ == "__main__":
    main()
