"""
retriever - Hybrid 检索器

结合稠密向量检索 + 稀疏向量检索 + WeightedRanker 融合排序（召回阶段）。

返回契约（供 agents/knowledge/nodes.py 消费）：
    每项 dict 含
        content:   chunk 文本
        score:     WeightedRanker 加权融合排序信号（非概率）
        source:    来源标注（source_name）
        metadata:  source_name / chunk_type / course_id / document_id / chunk_index

说明：
  - 本模块只做【召回】，精排见 reranker.py（BGEReranker.rerank_with_confidence）。
  - pymilvus 为可选依赖，缺失时模块仍可导入，但检索会抛 ImportError。
"""
import os
from typing import Optional

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

try:
    from pymilvus import AnnSearchRequest, MilvusClient, WeightedRanker
    _PYMILVUS_AVAILABLE = True
except ImportError:   # 未安装 pymilvus 时降级，避免模块导入即崩
    _PYMILVUS_AVAILABLE = False

DEFAULT_COLLECTION = "equipment_knowledge"
VECTOR_TOP_K = 10    # Hybrid 召回的候选数量，传给 Reranker 精排
ANN_EF       = 64    # HNSW 搜索时候选集大小（精度/速度平衡点）


class KnowledgeBaseRetriever:
    """知识库检索器，封装 Milvus Hybrid 检索流程（召回阶段）。"""

    def __init__(self, collection_name: str = DEFAULT_COLLECTION):
        self.collection_name = collection_name
        settings = get_settings()
        self._client = MilvusClient(
            uri=f"http://{settings.milvus_host}:{settings.milvus_port}"
        )

    @staticmethod
    def _build_filter(tenant_id: str, course_id: Optional[str] = None) -> str:
        """构建 Milvus bool 过滤表达式，对字符串做转义防止注入。"""
        safe_tenant = tenant_id.replace('"', '\\"')
        expr = f'tenant_id == "{safe_tenant}"'
        if course_id:
            safe_course = course_id.replace('"', '\\"')
            expr += f' and course_id == "{safe_course}"'
        return expr

    def hybrid_search(
        self,
        query_embedding: list[float],
        query_sparse: dict,
        tenant_id: str,
        course_id: Optional[str] = None,
        top_k: int = VECTOR_TOP_K,
    ) -> list[dict]:
        """Dense + Sparse 双路 ANN → WeightedRanker 融合，返回候选文档。

        Args:
            query_embedding: Dense Query 向量（来自 BGEMEmbedder.encode_query）
            query_sparse:    Sparse Query 向量（{token_id: weight}）
            tenant_id:       租户 ID（过滤）
            course_id:       产品/设备知识域 ID（可选，缩小范围）
            top_k:           召回数量（融合后同样取 top_k）

        Returns:
            候选文档列表，每项含 content / score / source / metadata。
            score 是 WeightedRanker 的排序信号，不是概率。
        """
        if not _PYMILVUS_AVAILABLE:
            raise ImportError("pymilvus 未安装，无法执行 Hybrid 检索。请先 pip install pymilvus")

        filters = self._build_filter(tenant_id, course_id)

        # Dense 检索：COSINE 匹配 BGE-M3 dense 向量（L2 归一化后等价余弦相似度）
        dense_req = AnnSearchRequest(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": ANN_EF}},
            limit=top_k,
            expr=filters,
        )

        # Sparse 检索：IP（内积）是 BGE-M3 lexical_weights 的标准度量
        sparse_req = AnnSearchRequest(
            data=[query_sparse],
            anns_field="sparse_embedding",
            param={"metric_type": "IP"},
            limit=top_k,
            expr=filters,
        )

        output_fields = [
            "content", "source_name", "chunk_type",
            "course_id", "document_id", "chunk_index",
        ]

        try:
            # 两路结果在 Milvus 服务端并行检索，WeightedRanker(0.7, 0.3) 加权融合
            results = self._client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_req, sparse_req],
                ranker=WeightedRanker(0.7, 0.3),
                limit=top_k,
                output_fields=output_fields,
            )
        except Exception as e:
            logger.error("retriever.hybrid_failed", error=str(e))
            return []

        candidates = []
        for hit in results[0]:
            entity = hit.get("entity") or {}
            candidates.append({
                "content": entity.get("content") or "",
                "score":   hit.get("distance") or 0.0,
                "source":  entity.get("source_name") or "",
                "metadata": {
                    "source_name": entity.get("source_name") or "",
                    "chunk_type":  entity.get("chunk_type")  or "text",
                    "course_id":   entity.get("course_id")   or "",
                    "document_id": entity.get("document_id") or "",
                    "chunk_index": entity.get("chunk_index") or 0,
                },
            })

        logger.info("retriever.hybrid_done", candidates=len(candidates))
        return candidates


async def hybrid_retrieve(
    query: str,
    device_model: Optional[str] = None,
    top_k: int = VECTOR_TOP_K,
) -> list[dict]:
    """模块级薄封装：向量化 Query + Hybrid 召回。

    对齐 agents/knowledge/nodes.py 的预期调用方式：
        docs = await hybrid_retrieve(query, device_model=device_model, top_k=5)

    Args:
        query:        用户问题文本
        device_model: 设备型号（作为 course_id 过滤条件，缩小检索范围）
        top_k:        召回数量

    Returns:
        同 KnowledgeBaseRetriever.hybrid_search 的 dict 列表。
    """
    from backend.knowledge_base.embedder import BGEMEmbedder

    embedder = BGEMEmbedder.get_instance()
    dense_vec, sparse_vec = embedder.encode_query(query)

    retriever = KnowledgeBaseRetriever()
    return retriever.hybrid_search(
        query_embedding=dense_vec,
        query_sparse=sparse_vec,
        tenant_id="tenant_default",
        course_id=device_model or None,
        top_k=top_k,
    )
