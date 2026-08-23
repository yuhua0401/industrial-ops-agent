"""
build_knowledge_base.py - 知识库建库流水线

完整流程（四步）：
    Step 1  读取文档            loader.load_document
    Step 2  智能分块            splitter.split_documents
    Step 2.5 Contextual RAG    contextual.add_context（可选，LLM 并发）
    Step 3  BGE-M3 嵌入         embedder.embed_chunks（dense + sparse 双向量）
    Step 4  写入 Milvus        writer.KnowledgeBaseClient.write_document

用法：
    python scripts/build_knowledge_base.py <文件路径> [--course-id UUID] [--no-context]
示例：
    python scripts/build_knowledge_base.py data/sample.md
    python scripts/build_knowledge_base.py 手册.pdf --course-id CNC-1000 --no-context
"""
import argparse
import asyncio
import sys
import uuid
from pathlib import Path

# 把项目根目录加入模块搜索路径，确保能 import backend.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import get_settings  # noqa: E402
from backend.knowledge_base import contextual, embedder, loader, splitter, writer  # noqa: E402


async def build_pipeline(
    file_path: str,
    course_id: str,
    document_id: str,
    tenant_id: str = "tenant_default",
    version: str = "1.0",
    use_context: bool = True,
) -> None:
    """
    知识库建库完整流水线。

    Args:
        file_path:   待处理的文档路径（.pdf / .docx / .xlsx / .md）
        course_id:   产品/设备知识域标识（Milvus 过滤分组，通常用设备型号）
        document_id: 文档 UUID（更新同一文档时需保留此 ID）
        tenant_id:   租户 ID，用于 Milvus 多租户过滤
        version:     文档版本号
        use_context: 是否启用 Contextual RAG 上下文增强
    """
    print(f"\n{'='*55}")
    print(" 知识库构建")
    print(f" 文件      ：{file_path}")
    print(f" 知识域 ID ：{course_id}")
    print(f" 文档 ID   ：{document_id}")
    print(f" 租户      ：{tenant_id}")
    print(f" Contextual RAG：{'启用' if use_context else '跳过'}")
    print(f"{'='*55}\n")

    # Step 1：读取
    print("📖 Step 1/4  读取文档…")
    docs = loader.load_document(file_path)

    # Step 2：分块
    print("\n✂️  Step 2/4  智能分块…")
    chunks = splitter.split_documents(docs, file_path)

    # Step 2.5：Contextual RAG（可选）
    if use_context and chunks:
        concurrency = get_settings().contextual_concurrency
        print(f"\n🧠 Step 2.5  Contextual RAG 上下文增强"
              f"（并发={concurrency}）…")
        chunks = await contextual.add_context(chunks, docs)

    # Step 3：嵌入
    print("\n🔢 Step 3/4  BGE-M3 嵌入…")
    doc_chunks = embedder.embed_chunks(
        chunks,
        course_id=course_id,
        document_id=document_id,
        tenant_id=tenant_id,
        version=version,
    )

    # Step 4：写入 Milvus
    print("\n💾 Step 4/4  写入 Milvus…")
    client = writer.KnowledgeBaseClient()
    written = client.write_document(doc_chunks)
    total = client.count()

    print(f"\n🎉 完成！写入 {written} 个 chunk，collection 现有 {total} 个 chunk")
    print(f"   document_id = {document_id}")
    print("   ⚠️  更新此文档时请保留此 document_id")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="知识库建库流水线")
    parser.add_argument("file_path", help="待处理文档路径（.pdf / .docx / .xlsx / .md）")
    parser.add_argument("--course-id", default="", help="知识域标识（默认用设备型号分组）")
    parser.add_argument("--document-id", default="", help="文档 UUID（更新时复用；默认自动生成）")
    parser.add_argument("--tenant-id", default="tenant_default", help="租户 ID")
    parser.add_argument("--version", default="1.0", help="文档版本号")
    parser.add_argument("--no-context", action="store_true", help="跳过 Contextual RAG")
    args = parser.parse_args()

    course_id = args.course_id or Path(args.file_path).stem
    doc_id = args.document_id or str(uuid.uuid4())

    asyncio.run(build_pipeline(
        file_path=args.file_path,
        course_id=course_id,
        document_id=doc_id,
        tenant_id=args.tenant_id,
        version=args.version,
        use_context=not args.no_context,
    ))
