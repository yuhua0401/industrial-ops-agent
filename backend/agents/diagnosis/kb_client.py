"""
kb_client - 知识库检索适配层

故障诊断 Agent（⑤号）不直接碰 knowledge_base/ 内部实现，
统一通过本适配层调用知识库引擎（③号维护的 KnowledgeBaseRetriever）。

注意：③号 的 hybrid_retrieve 是 retriever.py 的「模块级异步函数」
（内部完成 BGE-M3 向量化 + Hybrid 召回），而非 KnowledgeBaseRetriever 的实例方法。
本层直接委托该模块级函数，避免误把函数当方法调用导致 AttributeError。

知识库未就绪时，search/rerank 抛错 → 由 nodes.load_diag_tree_node 捕获并优雅降级。
"""
from __future__ import annotations

# 直接运行（python kb_client.py）时项目根不在 sys.path，这里补上；作为包导入时无副作用。
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.core.logger import get_logger

logger = get_logger(__name__)


class KBClient:
    """知识库检索客户端：封装 hybrid_retrieve（Hybrid 召回）与 BGEReranker（可选精排）。"""

    @classmethod
    async def search(
        cls,
        query: str,
        device_model: str | None = None,
        top_k: int = 10,
    ) -> list[dict]:
        """混合召回（稠密+稀疏 → WeightedRanker），返回 [{content, source, score, metadata}, ...]。

        device_model 作为 course_id 过滤条件，缩小到该设备型号的知识域。
        """
        from backend.knowledge_base.retriever import hybrid_retrieve
        return await hybrid_retrieve(
            query=query,
            device_model=device_model,
            top_k=top_k,
        )

    @classmethod
    async def rerank(cls, docs: list[dict], query: str, top_k: int = 5) -> list[dict]:
        """BGE-Reranker 精排（③号 reranker.py）。

        返回与 search 相同的 dict 形态（content / score / source / metadata）。
        模型未就绪或加载失败时降级为原序截断，不中断流程。
        """
        try:
            from backend.knowledge_base.reranker import BGEReranker
            reranker = BGEReranker.get_instance()
            ranked, _ = reranker.rerank_with_confidence(
                query=query,
                documents=docs,
                top_k=top_k,
            )
            return [
                {
                    "content": d.content,
                    "score":   d.score,
                    "source":  d.metadata.get("source_name", ""),
                    "metadata": d.metadata,
                }
                for d in ranked
            ]
        except Exception as e:
            logger.warning("kb_client.rerank_failed", error=str(e), fallback=True)
            return docs[:top_k]


# ──────────────────────────────────────────────────────────────
# 直接运行演示：python kb_client.py
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path

    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    async def demo():
        print("=== KBClient.search（真实调用知识库检索器）===")
        try:
            hits = await KBClient.search(query="E001 电机不转", device_model="CNC-1000", top_k=10)
            print(f"检索成功，命中 {len(hits)} 条:")
            for h in hits[:3]:
                print(f"  - [{h.get('source')}] {h.get('content', '')[:60]}")
            if not hits:
                print("  （知识库当前为空：Milvus 无该 course_id 的文档，或检索链路未联通）")
        except Exception as e:
            print(f"知识库未就绪，search 抛错（由 nodes.load_diag_tree_node 捕获降级）: {type(e).__name__}: {e}")

    asyncio.run(demo())
