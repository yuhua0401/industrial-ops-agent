"""
nodes - 售后协调 Agent 的节点函数

通过 Tool Calling 调用外部系统 API 完成各类售后操作。
拓扑：route_request_type → (query_warranty | order_part | prepare_appointment → tools)
"""
from langchain_core.messages import AIMessage, HumanMessage

from backend.agents.after_sale.state import AfterSaleState
from backend.agents.after_sale.tools import (
    check_part_stock,
    create_appointment,
    query_warranty_from_db,
)
from backend.core.llm_factory import get_llm
from backend.core.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# 请求类型分类（纯函数 + LLM 兜底）
# ──────────────────────────────────────────────────────────────

_WARRANTY_KEYWORDS = ("保修", "质保", "在保", "过保", "保修期", "保修状态")
_PARTS_KEYWORDS    = ("配件", "备件", "零件", "库存", "发货", "更换件")
_APPOINTMENT_KEYWORDS = ("预约", "上门", "维修", "派人", "师傅", "工程师")


def _classify_request_type(text: str) -> str:
    """关键词规则分类售后请求类型，返回 warranty / parts / appointment。

    命中多个关键词时按优先级：保修 > 配件 > 预约（由关键词顺序天然决定）。
    """
    t = text or ""
    if any(k in t for k in _WARRANTY_KEYWORDS):
        return "warranty"
    if any(k in t for k in _PARTS_KEYWORDS):
        return "parts"
    if any(k in t for k in _APPOINTMENT_KEYWORDS):
        return "appointment"
    return ""


async def _llm_classify(state: AfterSaleState) -> str:
    """LLM 兜底分类：关键词未命中时交给大模型判断。失败默认 appointment。"""
    try:
        llm = get_llm("after_sale", temperature=0)
        prompt = (
            "判断以下售后诉求的类型，只输出一个词：warranty（保修/质保）、parts（配件/备件）、"
            f"appointment（预约上门维修）。\n用户诉求：{state.get('message', '')}\n输出："
        )
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        text = resp.text if hasattr(resp, "text") else str(resp)
        text = text.strip().strip("。.,，").lower()
        if text in ("warranty", "parts", "appointment"):
            return text
        if "保修" in text or "质保" in text:
            return "warranty"
        if "配件" in text or "备件" in text:
            return "parts"
        return "appointment"
    except Exception as e:
        logger.warning("after_sale.llm_classify_failed", error=str(e))
        return "appointment"


async def route_request_type_node(state: AfterSaleState) -> dict:
    """路由节点：判断售后请求类型（warranty / parts / appointment）。

    优先使用 state.request_type（API 可显式指定），否则关键词规则，最后 LLM 兜底。
    """
    request_type = state.get("request_type", "") or ""
    if not request_type:
        request_type = _classify_request_type(state.get("message", ""))
    if not request_type:
        request_type = await _llm_classify(state)

    logger.info("after_sale.route_request_type", request_type=request_type)
    return {"request_type": request_type}


# ──────────────────────────────────────────────────────────────
# 节点：保修查询 / 配件订购 / 预约准备
# ──────────────────────────────────────────────────────────────

async def query_warranty_node(state: AfterSaleState) -> dict:
    """节点：查询保修信息（devices 表）。"""
    device_sn = state.get("device_sn", "")
    if not device_sn:
        return {"warranty_info": {"status": "unknown", "error": "缺少设备序列号"}}

    info = await query_warranty_from_db(device_sn)
    logger.info("after_sale.warranty_query", device_sn=device_sn, status=info.get("status"))
    return {"warranty_info": info}


async def order_part_node(state: AfterSaleState) -> dict:
    """节点：配件订购（备件表未建，走 check_part_stock 桩查询）。"""
    part = state.get("part_order", {}) or {}
    try:
        stock_text = await check_part_stock.ainvoke({"part_no": part.get("part_no", "未知配件")})
    except Exception as e:
        logger.warning("after_sale.part_stock_failed", error=str(e))
        stock_text = "配件库存查询暂时不可用"
    return {"part_order": {**part, "stock_info": stock_text}}


async def prepare_appointment_node(state: AfterSaleState) -> dict:
    """节点：LLM 生成 create_appointment 的 tool_calls 并追加进 messages。

    ToolNode 消费 messages 中的带 tool_calls 的 AIMessage，因此必须真正 append。
    """
    llm = get_llm("after_sale", temperature=0)
    tool_llm = llm.bind_tools([create_appointment])
    prompt = (
        "用户想预约上门维修服务。请调用 create_appointment 工具创建预约。\n"
        f"设备序列号：{state.get('device_sn', '')}\n"
        f"用户诉求：{state.get('message', '')}\n"
        "若用户未提供地址或时间，用空字符串占位，不要编造。"
    )
    try:
        resp = await tool_llm.ainvoke([HumanMessage(content=prompt)])
        aim = resp if isinstance(resp, AIMessage) else AIMessage(content=str(resp))
    except Exception as e:
        logger.warning("after_sale.prepare_appointment_failed", error=str(e))
        aim = AIMessage(content="预约服务暂时不可用，请稍后重试或联系人工客服")

    return {"messages": [aim]}


async def finalize_appointment_node(state: AfterSaleState) -> dict:
    """节点：ToolNode 执行 create_appointment 后，把结果提取到 appointment_info。

    ToolNode 把 ToolMessage 追加进 state.messages，本节点从最后一条 ToolMessage
    解析预约确认信息（预约号 / 时间 / 地址），供 API 层渲染。
    """
    appointment_info: dict | None = None
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            content = msg.content
            # create_appointment 工具返回 "已为您预约上门服务，预约号 APPT-..."
            if content.startswith("已为您预约") or "预约号" in content:
                import re

                m = re.search(r"APPT-\d{8}-\d{6}", content)
                appointment_info = {
                    "appointment_id": m.group(0) if m else "",
                    "reply": content,
                }
                break
    return {"appointment_info": appointment_info}
