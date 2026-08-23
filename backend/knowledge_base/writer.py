"""
writer - Milvus 写入端（KnowledgeBaseClient）

建库流水线 Step 4：把 embedder.embed_chunks() 产出的 DocumentChunk 写入 Milvus。

Collection Schema（与 retriever.py 的检索字段完全对齐）：
    equipment_knowledge
    ├── 主键 id:      VARCHAR（MD5，chunk 全局唯一）
    ├── 向量 embedding:        FLOAT_VECTOR(1024)   ← BGE-M3 dense（COSINE 度量）
    ├── 向量 sparse_embedding: SPARSE_FLOAT_VECTOR  ← BGE-M3 lexical weights（IP 度量）
    └── 标量字段: content / source_name / chunk_type / course_id / document_id /
                  chunk_index / version / tenant_id / updated_at

用法：
    client = KnowledgeBaseClient()
    await client.write_document(doc_chunks)   # 幂等：同 id 覆盖（删旧插新）
"""
from __future__ import annotations

from typing import Optional

from pymilvus import DataType, MilvusClient

from backend.config import get_settings
from backend.core.logger import get_logger
from backend.knowledge_base.embedder import DocumentChunk

logger = get_logger(__name__)

DEFAULT_COLLECTION = "equipment_knowledge"
DENSE_DIM = 1024          # BGE-M3 dense 向量维度


class KnowledgeBaseClient:
    """Milvus 写入端：建 collection + 写 DocumentChunk。"""

    def __init__(self, collection_name: str = DEFAULT_COLLECTION):
        self.collection_name = collection_name
        settings = get_settings()
        self._client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")

    # ── 建 collection（幂等：已存在则跳过）────────────────────
    def ensure_collection(self) -> None:
        """确保 collection 存在，不存在则创建 Schema + 双向量索引。"""
        if self._client.has_collection(self.collection_name):
            logger.info("milvus.collection_exists", collection=self.collection_name)
            return

        schema = self._client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
            description="工业设备运维知识库（BGE-M3 dense+sparse 混合检索）",
        )
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=16384)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=DENSE_DIM)
        schema.add_field(field_name="sparse_embedding", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="source_name", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="chunk_type", datatype=DataType.VARCHAR, max_length=16)
        schema.add_field(field_name="course_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
        schema.add_field(field_name="version", datatype=DataType.VARCHAR, max_length=16)
        schema.add_field(field_name="tenant_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="updated_at", datatype=DataType.INT64)

        self._client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
        )

        # 双向量索引：dense 用 HNSW/COSINE，sparse 用 IP
        # 用 prepare_index_params().add_index() 兼容 pymilvus 2.5/3.0
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_name="idx_embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        index_params.add_index(
            field_name="sparse_embedding",
            index_name="idx_sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        self._client.create_index(
            collection_name=self.collection_name,
            index_params=index_params,
        )

        logger.info("milvus.collection_created", collection=self.collection_name)

    # ── 写入文档（幂等：同 id 覆盖）───────────────────────────
    def write_document(self, chunks: list[DocumentChunk]) -> int:
        """把一组 DocumentChunk 写入 Milvus。

        幂等语义：chunk id = MD5(content + document_id + chunk_index)，内容不变时 id 稳定，
        同一 document_id 重建时按 id 覆盖，避免重复插入。
        """
        if not chunks:
            logger.warning("milvus.write_empty")
            return 0

        self.ensure_collection()

        rows = [
            {
                "id":               c.id,
                "content":          c.content,
                "embedding":        c.embedding,
                "sparse_embedding": c.sparse_embedding,
                "source_name":      c.source_name,
                "chunk_type":       c.chunk_type,
                "course_id":        c.course_id,
                "document_id":      c.document_id,
                "chunk_index":      c.chunk_index,
                "version":          c.version,
                "tenant_id":        c.tenant_id,
                "updated_at":       c.updated_at,
            }
            for c in chunks
        ]

        # 分批写入，避免单次请求过大
        batch = 100
        total = 0
        for i in range(0, len(rows), batch):
            sub = rows[i:i + batch]
            self._client.upsert(collection_name=self.collection_name, data=sub)
            total += len(sub)
            logger.info("milvus.write_progress", written=total, total=len(rows))

        self._client.flush(collection_name=self.collection_name)
        logger.info("milvus.write_done", count=total, collection=self.collection_name)
        return total

    def delete_document(self, document_id: str, tenant_id: str = "tenant_default") -> None:
        """按 document_id 删除该文档的所有 chunk（重建文档前调用）。"""
        if not self._client.has_collection(self.collection_name):
            return
        expr = f'document_id == "{document_id}" and tenant_id == "{tenant_id}"'
        self._client.delete(collection_name=self.collection_name, filter=expr)
        logger.info("milvus.document_deleted", document_id=document_id)

    def count(self) -> int:
        """当前 collection 的 chunk 总数。"""
        if not self._client.has_collection(self.collection_name):
            return 0
        return self._client.get_collection_stats(self.collection_name).get("row_count", 0)
