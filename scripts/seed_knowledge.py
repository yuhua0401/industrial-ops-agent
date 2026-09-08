"""
seed_knowledge.py - 知识库种子数据写入

薄封装：委托 scripts/build_knowledge_base.py 的完整四步流水线
（读取 → 分块 → BGE-M3 嵌入 → 写 Milvus），对默认示例文档建库。
适合本地开发一键初始化（幂等：同一 document_id 会追加新 chunk 版本，
如需完全重建请先在 Milvus 删除对应 collection）。

用法：
    python scripts/seed_knowledge.py                  # 默认 data/knowledge_sample.md → CNC-1000
    python scripts/seed_knowledge.py --file 手册.pdf --model CNC-2000
    python scripts/seed_knowledge.py --use-context    # 启用 Contextual RAG（需 LLM 可用）
"""
import argparse
import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_knowledge_base import build_pipeline  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
DEFAULT_FILE = BASE / "data" / "knowledge_sample.md"
DEFAULT_MODEL = "CNC-1000"


async def seed_knowledge(
    file_path: str = str(DEFAULT_FILE),
    device_model: str = DEFAULT_MODEL,
    use_context: bool = False,
) -> None:
    """写入知识库种子数据（默认跳过 Contextual RAG，保持离线友好）。"""
    if not Path(file_path).exists():
        print(f"文档不存在：{file_path}")
        sys.exit(1)
    await build_pipeline(
        file_path=file_path,
        course_id=device_model,
        document_id=str(uuid.uuid4()),
        use_context=use_context,
    )


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="知识库种子数据写入")
    parser.add_argument("--file", default=str(DEFAULT_FILE),
                        help="文档路径（默认 data/knowledge_sample.md）")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="设备型号/知识域（默认 CNC-1000）")
    parser.add_argument("--use-context", action="store_true",
                        help="启用 Contextual RAG（默认关闭，需 LLM 可用）")
    args = parser.parse_args()
    asyncio.run(seed_knowledge(args.file, args.model, use_context=args.use_context))
