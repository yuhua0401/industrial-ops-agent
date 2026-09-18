"""
knowledge - 产品知识问答 API

设备智能客服 — Agent② 产品知识库 RAG 问答接口。
支持非流式（/chat）与 SSE 流式（/chat/stream）两种模式，
提供对话历史查询（/sessions/{session_id}/history）。

SSE 事件类型：
  progress  问答流水线进度提示（检索知识库... / 生成回答中...）
  token     流式回答 token
  meta      回答元数据（answer_mode / sources / confidence / related_questions）
  done      流结束信号
  error     异常
"""

import json

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from backend.agents.knowledge.graph import build_knowledge_graph
from backend.agents.knowledge.prompts import KNOWLEDGE_GENERATE_PROMPT, SYSTEM_PROMPT
from backend.agents.knowledge.state import KnowledgeResult
from backend.core.llm_factory import get_llm, get_structured_llm
from backend.core.logger import get_logger
from backend.dependencies import get_current_user

router = APIRouter()
logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# 请求 / 响应模型
# ═══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """知识问答请求。"""
    session_id:  str       = Field(..., description="会话 ID")
    message:     str       = Field(..., min_length=1, max_length=2000, description="用户问题")
    device_model: str      = Field(default="", description="设备型号（可选，限定检索范围）")
    top_k:       int       = Field(default=5, ge=1, le=20, description="检索返回文档数")


class ChatResponse(BaseModel):
    """知识问答响应（非流式）。"""
    session_id:        str        = Field(..., description="会话 ID")
    answer:            str        = Field(..., description="AI 回答")
    answer_mode:       str        = Field(default="rag", description="rag / direct / fallback")
    confidence:        str        = Field(default="high", description="高 / 中 / 低")
    sources:           list[str]  = Field(default_factory=list, description="引用来源")
    related_questions: list[str]  = Field(default_factory=list, description="推荐追问")


class SessionMessage(BaseModel):
    """会话中的单条消息。"""
    role:       str    # "user" / "assistant"
    content:    str
    created_at: str    = ""


class HistoryResponse(BaseModel):
    """会话历史响应。"""
    session_id:  str
    messages:    list[SessionMessage]
    total_turns: int


# ═══════════════════════════════════════════════════════════════
# 非流式问答
# ═══════════════════════════════════════════════════════════════

@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    产品知识问答（非流式）。

    流程：retrieve → generate → 返回结构化结果。
    """
    graph = build_knowledge_graph()

    initial_state = {
        "messages":      [HumanMessage(content=req.message)],
        "session_id":    req.session_id,
        "device_model":  req.device_model,
        "query":         req.message,
        "retrieved_docs": [],
        "knowledge_result": None,
    }

    try:
        result = await graph.ainvoke(initial_state)
    except Exception as e:
        logger.error("knowledge.chat_error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "KNOWLEDGE_ERROR", "message": str(e)},
        ) from e

    kr = result.get("knowledge_result") or {}
    return ChatResponse(
        session_id=req.session_id,
        answer=kr.get("answer", "抱歉，暂时无法回答您的问题。"),
        answer_mode="rag" if result.get("retrieved_docs") else "fallback",
        confidence=kr.get("confidence", "low"),
        sources=kr.get("sources", []),
        related_questions=kr.get("related_questions", []),
    )


# ═══════════════════════════════════════════════════════════════
# SSE 流式问答
# ═══════════════════════════════════════════════════════════════

@router.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    产品知识问答流式接口（SSE）。

    事件序列：
        progress（检索中）→ progress（生成中）→ token × N → meta → done
    """

    async def event_generator():
        # ── 阶段一：知识库检索 ────────────────────────────────
        yield _sse({"type": "progress", "stage": "检索知识库中..."})

        docs: list[dict] = []
        try:
            from backend.knowledge_base.retriever import hybrid_retrieve
            docs = await hybrid_retrieve(
                query=req.message,
                device_model=req.device_model or None,
                top_k=req.top_k,
            )
        except Exception as e:
            logger.warning("knowledge.stream_retrieve_failed", error=str(e))
            docs = []

        # ── 阶段二：流式 LLM 生成 ──────────────────────────────
        yield _sse({"type": "progress", "stage": "生成回答中..."})

        if docs:
            context = "\n\n".join(
                f"【来源：{d.get('source', d.get('metadata', {}).get('source_name', '未知'))}】\n"
                f"{d.get('content', '')}"
                for d in docs[:3]
            )
            answer_mode = "rag"
        else:
            context = "（知识库中未找到相关信息，请基于通用知识回答并建议联系人工客服）"
            answer_mode = "direct"

        prompt = KNOWLEDGE_GENERATE_PROMPT.format(query=req.message, context=context)
        llm = get_llm("knowledge", temperature=0, streaming=True)

        full_answer = ""
        try:
            async for chunk in llm.astream([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]):
                if hasattr(chunk, 'content') and chunk.content:
                    full_answer += chunk.content
                    yield _sse({"type": "token", "content": chunk.content})
        except Exception as e:
            logger.error("knowledge.stream_generate_error", error=str(e), exc_info=True)
            if not full_answer:
                yield _sse({
                    "type": "error",
                    "message": "生成回答失败，请稍后重试或使用非流式接口。",
                })
                return

        # ── 阶段三：结构化提取（置信度 / 来源 / 推荐追问）──────
        confidence = "high"
        sources: list[str] = []
        related: list[str] = []

        if docs and full_answer:
            try:
                structured_llm = get_structured_llm("knowledge", KnowledgeResult)
                result: KnowledgeResult = await structured_llm.ainvoke([
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=f"问题：{req.message}\n回答：{full_answer}\n请提取来源和置信度。"),
                ])
                if result:
                    confidence = result.confidence
                    sources = result.sources
                    related = result.related_questions
            except Exception as e:
                logger.warning("knowledge.stream_structured_failed", error=str(e))
                # 结构化提取失败不阻断流程，使用默认值
                sources = [
                    d.get("source", d.get("metadata", {}).get("source_name", "未知来源"))
                    for d in docs[:3]
                ]

        # ── 元数据 + 结束 ─────────────────────────────────────
        yield _sse({
            "type":               "meta",
            "session_id":         req.session_id,
            "answer_mode":        answer_mode,
            "confidence":         confidence,
            "sources":            sources,
            "related_questions":  related,
        })
        yield _sse({"type": "done"})

    return EventSourceResponse(event_generator())


# ═══════════════════════════════════════════════════════════════
# 会话历史
# ═══════════════════════════════════════════════════════════════

@router.get("/sessions/{session_id}/history", response_model=HistoryResponse)
async def get_session_history(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    获取知识问答会话的历史消息。

    从数据库 conversations 表读取历史对话记录。
    """
    from sqlalchemy import text as sa_text

    from backend.dependencies import AsyncSessionLocal

    messages: list[SessionMessage] = []
    total_turns = 0

    try:
        async with AsyncSessionLocal() as db_session:
            result = await db_session.execute(
                sa_text(
                    "SELECT message, reply, created_at "
                    "FROM conversations "
                    "WHERE session_id = :sid AND customer_id = :cid "
                    "ORDER BY created_at ASC"
                ),
                {"sid": session_id, "cid": current_user["user_id"]},
            )
            rows = result.fetchall()
            for row in rows:
                created_at = row.created_at.isoformat() if hasattr(row.created_at, 'isoformat') else str(row.created_at)
                messages.append(SessionMessage(
                    role="user", content=row.message, created_at=created_at,
                ))
                messages.append(SessionMessage(
                    role="assistant", content=row.reply or "", created_at=created_at,
                ))
            total_turns = sum(1 for m in messages if m.role == "user")
    except Exception as e:
        logger.warning("knowledge.get_history_db_error", error=str(e))

    return HistoryResponse(
        session_id=session_id,
        messages=messages,
        total_turns=total_turns,
    )


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def _sse(data: dict) -> dict:
    """把 dict 包成 sse_starlette 格式。"""
    return {"data": json.dumps(data, ensure_ascii=False)}
