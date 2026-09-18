"""
contextual - Contextual RAG 上下文增强

为分块后的每个 chunk 用 LLM 生成一句“定位描述”，拼接到 chunk 文本前方，
使向量同时编码“在哪里”和“说了什么”两层信息，改善检索效果。

独立成模块的原因：
  - 单一职责：分块（splitter.py）与 LLM 上下文增强是两个独立关注点；
  - 可复用：建库流水线（scripts/build_knowledge_base.py）与文档增量重建都会调用 add_context；
  - 依赖清晰：本模块只依赖 LLM 工厂（backend.core.llm_factory），不依赖分块细节。
"""
import asyncio

from langchain_core.documents import Document

from backend.config import get_settings
from backend.core.llm_factory import get_llm
from backend.core.logger import get_logger

logger = get_logger(__name__)
_settings = get_settings()

# ── 常量（默认值可在 config.py 的 Settings 中覆盖）────────────────

CONTEXTUAL_CHUNK_PROMPT = """\
<document>
{document_text}
</document>

以下是需要在整个文档中定位的 chunk：
<chunk>
{chunk_content}
</chunk>

请用一句简洁的中文，描述这段内容在整个文档中的位置和作用，以便改善检索效果。
只输出这一句描述，不要加任何前缀或标签。"""


# ── Contextual RAG 上下文增强 ────────────────────────────────

async def generate_chunk_context(
    llm,
    document_text: str,
    chunk_content: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """
    用 LLM 为单个 chunk 生成一句定位描述。

    失败时返回空字符串，调用方保留原始 chunk 文本（降级处理）。

    Args:
        llm:           DeepSeek LLM 实例（via get_llm）
        document_text: 整篇文档全文（截断至 contextual_fulltext_limit 字）
        chunk_content: 当前 chunk 的原始文本
        semaphore:     并发限流（最多 contextual_concurrency 个 LLM 请求同时进行）
    """
    async with semaphore:
        try:
            from langchain_core.messages import HumanMessage
            prompt = CONTEXTUAL_CHUNK_PROMPT.format(
                document_text=document_text,
                chunk_content=chunk_content,
            )
            resp = await llm.ainvoke([HumanMessage(content=prompt)])
            ctx = (
                resp.text
                if hasattr(resp, "text") and not callable(resp.text)
                else str(resp.content)
            ).strip()
            return ctx
        except Exception as e:
            logger.warning("contextual.chunk_failed", error=str(e))
            return ""


async def add_context(
    chunks: list[Document],
    docs: list[Document],
    concurrency: int = 0,          # 0 = 使用配置默认值
) -> list[Document]:
    """
    Contextual RAG：并发为所有 chunk 生成上下文描述，拼接到 chunk 文本前方。

    拼接后格式：
        "<上下文描述一句话>\\n\\n<原始 chunk 文本>"

    拼接后再做嵌入（embedder.embed_chunks），向量同时编码"在哪里"和"说了什么"两层信息。

    Args:
        chunks:      splitter.split_documents() 输出的 list[Document]
        docs:        loader.load_document() 输出的原始 list[Document]（用于构建全文参考）
        concurrency: 最大并发 LLM 请求数；0 表示用配置默认值（contextual_concurrency）

    Returns:
        page_content 已被就地修改（拼接上下文）的 list[Document]
    """
    concurrency = concurrency or _settings.contextual_concurrency

    # 拼接全文供 LLM 参考（截断 contextual_fulltext_limit 字，避免超出模型 context 长度）
    full_doc_text = "\n\n".join(d.page_content for d in docs)[:_settings.contextual_fulltext_limit]

    llm       = get_llm("knowledge", temperature=0)
    semaphore = asyncio.Semaphore(concurrency)

    # 并发调用 LLM，为每个 chunk 生成上下文描述
    contexts = await asyncio.gather(*[
        generate_chunk_context(llm, full_doc_text, c.page_content, semaphore)
        for c in chunks
    ])

    enriched = 0
    for chunk, ctx in zip(chunks, contexts, strict=False):
        if ctx:
            chunk.page_content = f"{ctx}\n\n{chunk.page_content}"
            enriched += 1

    logger.info("contextual.done", enriched=enriched, total=len(chunks))
    return chunks
