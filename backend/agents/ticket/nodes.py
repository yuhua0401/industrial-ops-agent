"""
nodes - 工单管理 Agent 的节点函数

流程：
  1. create_ticket_node — 结构化归一化工单数据 → 真实落库（tickets + ticket_logs 审计）
"""
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.ticket.repo import create_ticket_via_factory
from backend.agents.ticket.state import TicketSchema, TicketState
from backend.core.llm_factory import get_structured_llm
from backend.core.logger import get_logger

logger = get_logger(__name__)


async def create_ticket_node(state: TicketState) -> dict:
    """节点①：创建工单（结构化归一化 + 真实落库）。

    归一化策略：LLM 只负责补全/整理描述类字段，**不覆盖调用方传入的权威数据**。
    通过 `{**llm_result, **ticket_data}` 合并，原始 ticket_data 优先，
    避免 LLM 结构化输出丢弃 customer_id / diagnosis_result 等关键字段。
    """
    ticket_data = state.get("ticket_data", {}) or {}

    # ── 第一步：用 LLM 把零散的 ticket_data 归一化成结构化 TicketSchema ──
    llm_normalized: dict | None = None
    try:
        structured_llm = get_structured_llm("ticket", TicketSchema)
        prompt = f"请将以下工单信息整理为结构化工单。\n\n工单信息：{ticket_data}"
        for attempt in range(2):
            try:
                result: TicketSchema = await structured_llm.ainvoke([
                    SystemMessage(content="你是一位工单管理系统的数据录入员，负责将客户信息整理为标准化工单。"),
                    HumanMessage(content=prompt),
                ])
                if result is None:
                    raise ValueError("structured output returned None")
                llm_normalized = result.model_dump()
                break
            except Exception as e:
                if attempt == 0:
                    logger.warning("ticket.structured_retry", error=str(e))
    except Exception as e:
        logger.warning("ticket.structured_failed", error=str(e))

    # ── 第二步：合并归一化结果与原始数据（原始数据优先，不丢关键字段）──
    ticket_dict: dict = dict(ticket_data)
    if llm_normalized is not None:
        # 原始数据覆盖 LLM 输出：调用方传入的字段更权威（customer_id/diagnosis_result 等）
        ticket_dict = {**llm_normalized, **ticket_data}
        logger.info("ticket.structured_merged")
    else:
        ticket_dict.setdefault("fault_description", ticket_data.get("fault_description", ""))
        ticket_dict.setdefault("severity", "medium")
        ticket_dict.setdefault("category", "repair")
        logger.info("ticket.structured_fallback_to_raw")

    # ── 第三步：真实落库（tickets 表 + ticket_logs 审计）──
    create_result = await create_ticket_via_factory(ticket_dict, operator="system")
    if create_result["created"]:
        ticket_dict["ticket_id"] = create_result["ticket_id"]
        logger.info("ticket.created", ticket_id=create_result["ticket_id"])
        return {"ticket": ticket_dict, "created": True, "ticket_id": create_result["ticket_id"]}

    logger.warning("ticket.create_failed", error=create_result.get("error"))
    return {"created": False, "ticket": ticket_dict, "ticket_id": ""}
