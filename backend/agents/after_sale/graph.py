"""
graph - 售后协调 Agent 的 LangGraph 图定义

拓扑（修复死节点）：
    START → route_request_type
        ├─(warranty)    → query_warranty_node → END
        ├─(parts)       → order_part_node → END
        └─(appointment) → prepare_appointment_node → tools(ToolNode) → END
"""
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from backend.agents.after_sale.nodes import (
    finalize_appointment_node,
    order_part_node,
    prepare_appointment_node,
    query_warranty_node,
    route_request_type_node,
)
from backend.agents.after_sale.state import AfterSaleState
from backend.agents.after_sale.tools import check_part_stock, create_appointment, query_warranty


def _route_by_request_type(state: AfterSaleState) -> str:
    """按请求类型路由到对应处理节点。"""
    request_type = state.get("request_type", "appointment")
    return {"warranty": "query_warranty", "parts": "order_part"}.get(
        request_type, "prepare_appointment"
    )


def build_after_sale_graph():
    """构建并编译售后协调 Agent 的 LangGraph 状态图。"""
    workflow = StateGraph(AfterSaleState)
    workflow.add_node("route_request_type",     route_request_type_node)
    workflow.add_node("query_warranty",         query_warranty_node)
    workflow.add_node("order_part",             order_part_node)
    workflow.add_node("prepare_appointment",    prepare_appointment_node)
    workflow.add_node(
        "tools", ToolNode([query_warranty, check_part_stock, create_appointment]),
    )
    workflow.add_node("finalize_appointment",   finalize_appointment_node)

    workflow.set_entry_point("route_request_type")
    workflow.add_conditional_edges(
        "route_request_type",
        _route_by_request_type,
        {
            "query_warranty":      "query_warranty",
            "order_part":          "order_part",
            "prepare_appointment": "prepare_appointment",
        },
    )
    workflow.add_edge("query_warranty", END)
    workflow.add_edge("order_part", END)
    workflow.add_edge("prepare_appointment", "tools")
    workflow.add_edge("tools", "finalize_appointment")
    workflow.add_edge("finalize_appointment", END)
    return workflow.compile()
