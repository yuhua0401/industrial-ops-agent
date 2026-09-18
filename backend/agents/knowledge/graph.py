"""
graph - 产品知识库 Agent 的 LangGraph 图定义
"""
from langgraph.graph import END, StateGraph

from backend.agents.knowledge.nodes import generate_node, retrieve_node
from backend.agents.knowledge.state import KnowledgeState


def build_knowledge_graph() -> StateGraph:
    workflow = StateGraph(KnowledgeState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()
