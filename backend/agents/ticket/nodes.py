"""
nodes - 工单管理 Agent 的节点函数

流程：
  1. extract_ticket_info — 从对话中提取工单所需信息
  2. create_ticket — 调用工单系统 API 创建工单
"""
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.ticket.state import TicketState, TicketSchema
from backend.core.llm_factory import get_structured_llm
from backend.core.logger import get_logger

logger = get_logger(__name__)


async def create_ticket_node(state: TicketState) -> dict:
    """节点①：创建工单（结构化生成 + API 写入）。"""
    ticket_data = state.get("ticket_data", {})

    # 用 LLM 把零散的 ticket_data 整理成结构化 TicketSchema
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
            ticket_dict = result.model_dump()

            # TODO: 实际调用工单系统 API 创建工单
            # from backend.api.ticket import create_ticket_api
            # ticket_dict["ticket_id"] = await create_ticket_api(ticket_dict)
            ticket_dict["ticket_id"] = f"TK-{hash(str(ticket_dict)) % 1000000:06d}"

            logger.info("ticket.created", ticket_id=ticket_dict["ticket_id"])
            return {"ticket": ticket_dict, "created": True, "ticket_id": ticket_dict["ticket_id"]}
        except Exception as e:
            if attempt == 0:
                logger.warning("ticket.create_retry", error=str(e))

    logger.warning("ticket.create_failed")
    return {"created": False, "ticket": TicketSchema(fault_description="创建失败").model_dump()}
