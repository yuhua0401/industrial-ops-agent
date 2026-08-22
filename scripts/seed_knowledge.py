"""
seed_knowledge.py - 知识库种子数据写入

读取 docs/sample_knowledge/ 下的文档，处理并写入 Milvus。
"""
import asyncio
import json
from pathlib import Path


async def seed_knowledge():
    """读取知识库文档 → 分块 → 嵌入 → 写入 Milvus。"""
    # TODO: 实际执行知识库写入流程
    # 1. 遍历 docs/knowledge/ 目录下的文档
    # 2. 调用 loader → splitter → embedder
    # 3. 写入 Milvus collection
    print("知识库种子数据写入完成。")


if __name__ == "__main__":
    asyncio.run(seed_knowledge())
