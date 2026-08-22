"""
embedder - BGE-M3 嵌入模型封装

支持稠密向量（Dense）和稀疏向量（Sparse）两种嵌入。
"""

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from backend.config import get_settings
from backend.core.logger import get_logger
from langchain_core.documents import Document

logger = get_logger(__name__)
backend_path = os.path.dirname(os.path.dirname(__file__))


# ──────────────────────────────────────────────────────────────
# BGE-M3 本地嵌入模型（进程内单例，dense + sparse 双输出）
# ──────────────────────────────────────────────────────────────

class BGEMEmbedder:
    """
    BGE-M3 本地嵌入模型单例。

    一次推理同时输出：
      - dense 向量（1024 维浮点数组，用于语义相似度检索）
      - sparse 向量（{token_id: weight} 字典，用于关键词精确检索）

    进程内单例：首次调用 get_instance() 时加载模型（约5-15秒），
    后续调用直接返回同一实例，不重复加载。

    用法：
        embedder = BGEMEmbedder.get_instance()
        dense, sparse = embedder.encode_query("这台设备的故障代码含义是什么？")
    """

    _instance: Optional["BGEMEmbedder"] = None   # 单例持有

    def __init__(self, model_path: str):
        # ── 兼容性补丁1：FlagEmbedding 1.3.x 依赖 transformers 内部函数 ──
        # transformers>=5.0 移除了 is_torch_fx_available，
        # 但当前锁定 transformers==4.51.0 不受影响。
        # 此补丁作为保险，避免未来升级时报 ImportError。
        import importlib.util as _ilu
        from transformers.utils import import_utils as _tf_iu
        if not hasattr(_tf_iu, "is_torch_fx_available"):
            _tf_iu.is_torch_fx_available = (
                lambda: _ilu.find_spec("torch.fx") is not None
            )

        # ── 兼容性补丁2：修复 XLMRobertaModel 不接受 dtype 参数的问题 ──
        # 某些 transformers 版本中 XLMRobertaModel.__init__() 不接受 dtype 关键字参数，
        # 但 FlagEmbedding 内部会传入该参数，导致 TypeError。
        # 通过子类覆盖，在调用父类前丢弃 dtype 参数。
        from transformers.models.xlm_roberta import modeling_xlm_roberta as _xlm
        _OriginalXLMRoberta = _xlm.XLMRobertaModel

        class _PatchedXLMRobertaModel(_OriginalXLMRoberta):
            def __init__(self, config, **kwargs):
                kwargs.pop("dtype", None)  # 丢弃 FlagEmbedding 传入的 dtype
                super().__init__(config, **kwargs)

        _xlm.XLMRobertaModel = _PatchedXLMRobertaModel

        import torch
        from FlagEmbedding import BGEM3FlagModel

        logger.info("bge_m3.loading", model_path=model_path)

        # ── fp16 仅在 CUDA 上启用，MPS（Apple M系列）不启用 ──
        # MPS 在 BGE-M3 attention 矩阵乘法上会触发 LLVM ERROR，
        # CPU 模式下用 fp32，速度稍慢但稳定。
        _use_fp16 = torch.cuda.is_available()

        self._model = BGEM3FlagModel(
            model_name_or_path=model_path,
            use_fp16=_use_fp16,
        )
        logger.info("bge_m3.loaded", use_fp16=_use_fp16)

    @classmethod
    def get_instance(cls) -> "BGEMEmbedder":
        """获取单例（首次调用时加载模型，后续复用）"""
        if cls._instance is None:
            bge3_path = os.path.join(backend_path, get_settings().bge_m3_model_path)
            cls._instance = BGEMEmbedder(bge3_path)
        return cls._instance

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,   # None = 使用配置 bge_m3_batch_size
    ) -> tuple[list[list[float]], list[dict]]:
        """
        批量编码文本，同时返回 dense 和 sparse 两种向量。

        Args:
            texts:      待编码的文本列表
            batch_size: 单次推理批大小，越大速度越快但显存占用越多；
                        None 表示用配置 bge_m3_batch_size

        Returns:
            (dense_vecs, sparse_vecs)
              dense_vecs:  list of 1024-dim float 向量，每项对应 texts[i]
              sparse_vecs: list of {token_id: weight} 字典，每项对应 texts[i]
        """
        settings = get_settings()
        output = self._model.encode(
            texts,
            batch_size=batch_size or settings.bge_m3_batch_size,
            max_length=settings.bge_m3_max_length,   # 默认 8192，覆盖大多数 chunk
            return_dense=True,          # 输出稠密语义向量
            return_sparse=True,         # 输出稀疏关键词向量
            return_colbert_vecs=False,  # ColBERT 多向量表示，本项目不用
        )

        dense_vecs = output["dense_vecs"].tolist()   # numpy → Python list

        # sparse: numpy.float16 → Python float
        # 必须转换！LangGraph MemorySaver 用 msgpack 序列化 State，
        # msgpack 不支持 numpy.float16，会在运行时抛 TypeError。
        sparse_vecs = [
            {int(k): float(v) for k, v in d.items()}
            for d in output["lexical_weights"]
        ]

        return dense_vecs, sparse_vecs

    def encode_query(self, text: str) -> tuple[list[float], dict]:
        """
        编码单条查询，返回 (dense_vec, sparse_vec)。

        查询时调用此方法（而非 encode），batch_size=1 避免不必要的 padding。

        Returns:
            (dense_vec, sparse_vec)
              dense_vec:  1024-dim float 列表
              sparse_vec: {token_id: weight} 字典
        """
        dense_list, sparse_list = self.encode([text], batch_size=1)
        return dense_list[0], sparse_list[0]


# ──────────────────────────────────────────────────────────────
# 数据类：DocumentChunk（建库写入 Milvus 的数据结构）
# ──────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """
    准备写入 Milvus 的单个文档块，字段与 Milvus Schema 一一对应。

    id:               全局唯一 ID（MD5 of content + document_id + chunk_index）
    content:          chunk 文本（Contextual RAG 模式下含 LLM 生成的上下文描述前缀）
    embedding:        Dense 向量（BGE-M3，1024 维）
    sparse_embedding: Sparse 向量（{token_id: weight}，BGE-M3 lexical weights）
    source_name:      来源标注（检索结果展示用，如 "XX设备手册 > 第3章 > 3.1 故障代码"）
    """
    id:               str
    content:          str
    embedding:        list[float]
    sparse_embedding: dict
    course_id:        str
    document_id:      str
    source_name:      str
    chunk_type:       str                  # "text" / "code" / "table"
    chunk_index:      int
    version:          str
    tenant_id:        str = "tenant_default"
    updated_at:       int = field(default_factory=lambda: int(time.time()))


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────

def generate_chunk_id(content: str, document_id: str, chunk_index: int) -> str:
    """
    生成 chunk 全局唯一 ID（MD5 散列）。

    用 document_id + chunk_index + content 前缀组合，确保：
    - 同一文档不同位置的 chunk 不冲突
    - 内容不变时 ID 稳定（幂等重建时不会重复插入）

    注意：KnowledgeBaseClient 尚未在 backend/core/ 实现，
    此函数暂为模块级工具函数；待其实现后再考虑迁移为静态方法。
    """
    raw = f"{document_id}_{chunk_index}_{content[:50]}"
    return hashlib.md5(raw.encode()).hexdigest()


def embed_chunks(
    chunks: list[Document],
    course_id: str,
    document_id: str,
    tenant_id: str = "tenant_default",
    version: str = "1.0",
) -> list[DocumentChunk]:
    """
    对 split_documents() 产出的 chunk 列表做 BGE-M3 嵌入，返回 DocumentChunk 列表。

    BGE-M3 推理为 CPU / GPU-bound，按配置 bge_m3_batch_size 批量处理：
    - 减少模型推理次数（每次推理有固定启动开销）
    - 控制显存/内存峰值（整批一次性推理会爆显存）

    Args:
        chunks:      split_documents() 返回的 list[Document]
        course_id:   所属产品/设备知识域的 UUID（设备型号分组，Milvus 过滤用）
        document_id: 文档的 UUID（用于 Milvus 幂等更新，删旧插新）
        tenant_id:   租户 ID，用于 Milvus 多租户过滤
        version:     文档版本号

    Returns:
        list[DocumentChunk]，每项包含 dense + sparse 向量，可直接写入 Milvus
    """
    embedder = BGEMEmbedder.get_instance()   # 单例，首次调用加载模型
    all_doc_chunks: list[DocumentChunk] = []

    batch_size = get_settings().bge_m3_batch_size
    total = len(chunks)
    for batch_start in range(0, total, batch_size):
        batch = chunks[batch_start: batch_start + batch_size]
        texts = [c.page_content for c in batch]

        # BGE-M3 批量推理：同时拿到 dense 和 sparse
        dense_vecs, sparse_vecs = embedder.encode(texts, batch_size=batch_size)

        for i, (chunk, dense, sparse) in enumerate(zip(batch, dense_vecs, sparse_vecs)):
            global_index = batch_start + i    # 在整个文档中的顺序编号

            all_doc_chunks.append(DocumentChunk(
                id=generate_chunk_id(chunk.page_content, document_id, global_index),
                content=chunk.page_content,
                embedding=dense,
                sparse_embedding=sparse,
                course_id=course_id,
                document_id=document_id,
                source_name=chunk.metadata.get("source_name", ""),
                chunk_type=chunk.metadata.get("chunk_type", "text"),
                chunk_index=global_index,
                version=version,
                tenant_id=tenant_id,
            ))

        done = min(batch_start + batch_size, total)
        logger.info("embed.progress", done=done, total=total)

    logger.info("embed.done", count=len(all_doc_chunks))
    return all_doc_chunks
