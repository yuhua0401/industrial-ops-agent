"""
state - 产品知识库 Agent 的状态定义
"""
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class KnowledgeResult(BaseModel):
    """知识库检索与生成的输出。"""
    answer: str = Field(description="基于知识库的最终回答")
    sources: list[str] = Field(default_factory=list, description="引用来源，如[手册-第三章-第2节]")
    confidence: str = Field(default="high", description="高/中/低")
    related_questions: list[str] = Field(default_factory=list, description="猜你想问的相关问题")


class KnowledgeState(TypedDict):
    """知识库 Agent 的状态。"""
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    device_model: str
    query: str                          # 客户问题
    retrieved_docs: list[dict]          # 检索到的文档片段
    knowledge_result: Optional[dict]    # KnowledgeResult.model_dump()
