"""
reranker - BGE Reranker 精排

对检索结果进行重排序，提升 Top-K 准确率。
"""
import os
from dataclasses import dataclass
from typing import Optional

import torch
from sentence_transformers import CrossEncoder

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)
backend_path = os.path.dirname(os.path.dirname(__file__))
_settings = get_settings()
RERANK_MAX_INPUT_CHARS = _settings.reranker_max_input_chars   # 截断过长文档，防止超出 CrossEncoder max_length


@dataclass
class RankedDocument:
    """精排后的单个文档结果"""
    content:        str    # 文档文本
    score:          float  # BGE-Reranker 输出的相关性概率 [0, 1]
    original_index: int    # 在原始召回列表中的位置（0 起）
    metadata:       dict   # 来源元数据（source_name / chunk_type / 知识域 ID 等）


class BGEReranker:
    """
    BGE-Reranker-v2-m3 精排服务（单例）。

    对 Hybrid 召回的候选文档做 CrossEncoder 精排，
    直接返回 [0, 1] 置信度，无需额外归一化。

    用法：
        reranker = BGEReranker.get_instance()
        docs, confidence = reranker.rerank_with_confidence(
            query="这台设备的故障代码含义是什么？",
            documents=candidates,
            top_k=3,
        )
    """

    _instance: Optional["BGEReranker"] = None

    def __init__(self):
        os.environ["ACCELERATE_USE_META_DEVICE"] = "0"
        settings = get_settings()
        model_path = os.path.join(backend_path, settings.reranker_model_path)

        use_local = (
            os.path.exists(model_path)
            and os.path.isdir(model_path)
            and any(f.endswith((".bin", ".safetensors", ".json")) for f in os.listdir(model_path))
        )
        model_id = model_path if use_local else "BAAI/bge-reranker-v2-m3"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("reranker.loading", model_id=model_id, device=device)
        # CrossEncoder max_length 需与 reranker_max_input_chars 对齐，
        # 避免输入被 RERANK_MAX_INPUT_CHARS 截断后仍超出模型窗口。
        self._model = CrossEncoder(model_id, device=device, max_length=RERANK_MAX_INPUT_CHARS)
        logger.info("reranker.loaded", model_id=model_id)

    @classmethod
    def get_instance(cls) -> "BGEReranker":
        """获取单例，首次调用时加载模型"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def rerank_with_confidence(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 3,
    ) -> tuple[list[RankedDocument], float]:
        """
        精排并返回置信度。

        Args:
            query:     用户 Query
            documents: 候选文档列表，每项含 "content" 字段
            top_k:     返回文档数量

        Returns:
            (ranked_docs, confidence)
            ranked_docs:  按相关性降序排列的 RankedDocument，长度 <= top_k
            confidence:   Top-1 文档的 BGE 相关性概率 [0, 1]。
                          阈值见 config.reranker_confidence_threshold（默认 0.75）：
                          ≥ 阈值 → 高置信度，直接走 LLM 生成；
                          < 阈值 → 低置信度，触发兜底
        """
        if not documents:
            return [], 0.0

        # CrossEncoder 输入：(query, document) 对，截断过长文档
        pairs = [
            (query, (doc.get("content") or "")[:RERANK_MAX_INPUT_CHARS])
            for doc in documents
        ]

        # CrossEncoder 默认 sigmoid 激活，predict() 直接输出 [0, 1] 概率
        scores: list[float] = self._model.predict(pairs).tolist()

        ranked = sorted(
            [
                RankedDocument(
                    content=documents[i].get("content", ""),
                    score=scores[i],
                    original_index=i,
                    metadata=documents[i].get("metadata", {}),
                )
                for i in range(len(documents))
            ],
            key=lambda x: x.score,
            reverse=True,
        )

        top_results = ranked[:top_k]
        confidence = top_results[0].score if top_results else 0.0

        logger.info(
            "reranker.done",
            candidates=len(documents),
            top_k=top_k,
            confidence=round(confidence, 4),
        )

        return top_results, confidence
