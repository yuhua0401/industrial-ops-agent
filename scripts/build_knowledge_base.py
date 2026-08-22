"""
build_knowledge_base.py - 知识库建库流水线

完整流程（四步，到嵌入为止）：
    Step 1  读取文档            loader.load_document
    Step 2  智能分块            splitter.split_documents
    Step 2.5 Contextual RAG    contextual.add_context（可选，LLM 并发）
    Step 3  BGE-M3 嵌入         embedder.embed_chunks（dense + sparse 双向量）

Milvus 写入（Step 4）暂缺：KnowledgeBaseClient 尚未在 backend/core/ 实现，
待其就绪后再在此处接入 write_to_milvus。

用法：
    python scripts/build_knowledge_base.py
（直接修改下方 __main__ 中的常量，或改为从命令行参数读取。）
"""
import asyncio
import sys
import uuid
from pathlib import Path

# 把项目根目录加入模块搜索路径，确保能 import backend.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import get_settings  # noqa: E402
from backend.knowledge_base import contextual, embedder, loader, splitter  # noqa: E402


async def build_pipeline(
    file_path: str,
    course_id: str,
    document_id: str,
    tenant_id: str = "tenant_default",
    version: str = "1.0",
    use_context: bool = True,
) -> None:
    """
    知识库建库完整流水线（到嵌入为止）。

    Args:
        file_path:   待处理的文档路径（.pdf / .docx / .xlsx / .md）
        course_id:   产品/设备知识域 UUID（Milvus 过滤分组）
        document_id: 文档 UUID（更新同一文档时需保留此 ID）
        tenant_id:   租户 ID，用于 Milvus 多租户过滤
        version:     文档版本号
        use_context: 是否启用 Contextual RAG 上下文增强
    """
    print(f"\n{'='*55}")
    print(f" 知识库构建")
    print(f" 文件      ：{file_path}")
    print(f" 知识域 ID ：{course_id}")
    print(f" 文档 ID   ：{document_id}")
    print(f" 租户      ：{tenant_id}")
    print(f" Contextual RAG：{'启用' if use_context else '跳过'}")
    print(f"{'='*55}\n")

    # Step 1：读取
    print("📖 Step 1/3  读取文档…")
    docs = loader.load_document(file_path)

    # Step 2：分块
    print("\n✂️  Step 2/3  智能分块…")
    chunks = splitter.split_documents(docs, file_path)

    # Step 2.5：Contextual RAG（可选）
    if use_context and chunks:
        concurrency = get_settings().contextual_concurrency
        print(f"\n🧠 Step 2.5  Contextual RAG 上下文增强"
              f"（并发={concurrency}）…")
        chunks = await contextual.add_context(chunks, docs)

    # Step 3：嵌入
    print("\n🔢 Step 3/3  BGE-M3 嵌入…")
    doc_chunks = embedder.embed_chunks(
        chunks,
        course_id=course_id,
        document_id=document_id,
        tenant_id=tenant_id,
        version=version,
    )

    # Step 4：写入 Milvus（TODO：待 KnowledgeBaseClient 实现后接入）
    print("\n💾 Step 4  写入 Milvus…（暂缺：KnowledgeBaseClient 未实现，跳过）")

    print(f"\n🎉 完成！共生成 {len(doc_chunks)} 个 chunk")
    print(f"   document_id = {document_id}")
    print(f"   ⚠️  更新此文档时请保留此 document_id")


if __name__ == "__main__":
    FILE_PATH   = "./samples/sample2.md"     # 替换为实际文档路径
    COURSE_ID   = "3e76aeed-5e01-4aa7-be8d-2055d12b9ea7"   # 产品/设备知识域 UUID
    DOCUMENT_ID = None                        # None = 自动生成；更新时填入上次输出的 ID
    TENANT_ID   = "tenant_default"
    VERSION     = "1.0"
    USE_CONTEXT = True                        # False = 跳过 Contextual RAG

    doc_id = DOCUMENT_ID or str(uuid.uuid4())

    asyncio.run(build_pipeline(
        file_path=FILE_PATH,
        course_id=COURSE_ID,
        document_id=doc_id,
        tenant_id=TENANT_ID,
        version=VERSION,
        use_context=USE_CONTEXT,
    ))
