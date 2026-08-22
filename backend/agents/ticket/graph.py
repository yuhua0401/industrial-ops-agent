"""
graph - 工单管理 Agent 的 LangGraph 图定义
"""
from langgraph.graph import StateGraph, END
from backend.agents.ticket.state import TicketState
from backend.agents.ticket.nodes import create_ticket_node


def build_ticket_graph() -> StateGraph:
    workflow = StateGraph(TicketState)
    workflow.add_node("create_ticket", create_ticket_node)
    workflow.set_entry_point("create_ticket")
    workflow.add_edge("create_ticket", END)
    return workflow.compile()
