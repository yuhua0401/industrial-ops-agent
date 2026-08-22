"""
graph - 售后协调 Agent 的 LangGraph 图定义
"""
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from backend.agents.after_sale.state import AfterSaleState
from backend.agents.after_sale.nodes import query_warranty_node, order_part_node
from backend.agents.after_sale.tools import query_warranty, check_part_stock, create_appointment


def build_after_sale_graph() -> StateGraph:
    workflow = StateGraph(AfterSaleState)
    workflow.add_node("query_warranty", query_warranty_node)
    workflow.add_node("order_part", order_part_node)
    workflow.add_node("tools", ToolNode([query_warranty, check_part_stock, create_appointment]))
    workflow.set_entry_point("query_warranty")
    workflow.add_edge("query_warranty", END)
    return workflow.compile()
