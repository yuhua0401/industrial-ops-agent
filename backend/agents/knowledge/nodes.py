"""
nodes - 产品知识库 Agent（RAG 检索）

三个节点：
  1. retrieve — 向量检索 + 精排
  2. generate — LLM 基于检索结果生成回答
  3. fallback — 降级处理
"""
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.knowledge.state import KnowledgeState, KnowledgeResult
from backend.agents.knowledge.prompts import SYSTEM_PROMPT, KNOWLEDGE_GENERATE_PROMPT
from backend.core.llm_factory import get_structured_llm
from backend.core.logger import get_logger

logger = get_logger(__name__)


async def retrieve_node(state: KnowledgeState) -> dict:
    """节点①：从知识库检索相关文档。

    对接 Milvus 向量库进行 Hybrid 召回（Dense + Sparse 融合）。
    检索失败或 pymilvus 未安装时降级返回空列表，不影响下游节点。
    """
    query = state["query"]
    device_model = state.get("device_model", "")

    try:
        from backend.knowledge_base.retriever import hybrid_retrieve
        docs = await hybrid_retrieve(query, device_model=device_model or None, top_k=5)
    except Exception as e:
        logger.warning("knowledge.retrieve_failed", error=str(e))
        docs = []

    if not docs:
        logger.info("knowledge.retrieve_empty", query=query)
    else:
        logger.info("knowledge.retrieve_done", query=query, hits=len(docs))

    return {"retrieved_docs": docs}


async def generate_node(state: KnowledgeState) -> dict:
    """节点②：基于检索结果生成回答。

    有检索结果 → 基于上下文回答
    无检索结果 → 告知无法回答 + 建议转人工
    """
    docs = state.get("retrieved_docs", [])
    query = state["query"]

    if docs:
        context = "\n\n".join(
            f"【来源：{d.get('source', '未知')}】\n{d.get('content', '')}"
            for d in docs[:3]
        )
    else:
        context = "（知识库中未找到相关信息）"

    prompt = KNOWLEDGE_GENERATE_PROMPT.format(query=query, context=context)
    structured_llm = get_structured_llm("knowledge", KnowledgeResult)

    for attempt in range(2):
        try:
            result: KnowledgeResult = await structured_llm.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            if result is None:
                raise ValueError("structured output returned None")
            return {"knowledge_result": result.model_dump()}
        except Exception as e:
            if attempt == 0:
                logger.warning("knowledge.generate_retry", error=str(e))

    # 降级
    logger.warning("knowledge.generate_failed", action="fallback")
    return {
        "knowledge_result": KnowledgeResult(
            answer="抱歉，我目前无法回答这个问题，已为您转接人工客服。",
            confidence="low",
        ).model_dump()
    }
