"""
graph - 产品知识库 Agent 的 LangGraph 图定义
"""
from langgraph.graph import StateGraph, END
from backend.agents.knowledge.state import KnowledgeState
from backend.agents.knowledge.nodes import retrieve_node, generate_node


def build_knowledge_graph() -> StateGraph:
    workflow = StateGraph(KnowledgeState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()
