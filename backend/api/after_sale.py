"""
after_sale - 售后协调 API

设备智能客服 — Agent⑤ 售后协调接口。
调用 build_after_sale_graph() 完成保修查询 / 配件库存 / 预约上门。
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.agents.after_sale.graph import build_after_sale_graph
from backend.core.logger import get_logger
from backend.dependencies import get_current_user

router = APIRouter()
logger = get_logger(__name__)


class AfterSaleRequest(BaseModel):
    """售后协调请求。"""
    session_id:  str   = Field(..., description="会话 ID")
    customer_id: str   = Field(default="", description="客户 ID（可选）")
    device_sn:   str   = Field(default="", description="设备序列号")
    message:     str   = Field(..., min_length=1, max_length=2000, description="用户诉求")
    request_type: str  = Field(
        default="", description="显式指定类型：warranty / parts / appointment（可选）",
    )


class AfterSaleResponse(BaseModel):
    """售后协调响应。"""
    status:          str        = Field(..., description="completed / error")
    request_type:    str        = Field(default="", description="识别到的售后类型")
    warranty_info:   dict | None = Field(default=None, description="保修信息")
    part_order:      dict | None = Field(default=None, description="配件订购结果")
    appointment_info: dict | None = Field(default=None, description="预约结果")
    reply:           str        = Field(default="", description="给用户的回复文本")
    error:           str        = Field(default="", description="错误信息（仅 error 状态）")


@router.post("", response_model=AfterSaleResponse)
async def run_after_sale(
    req: AfterSaleRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    售后协调（非流式）。

    流程：route_request_type → (query_warranty | order_part | prepare_appointment → tools)。
    """
    graph = build_after_sale_graph()

    initial_state: dict = {
        "messages":       [],
        "session_id":     req.session_id,
        "customer_id":    req.customer_id or current_user["user_id"],
        "device_sn":      req.device_sn,
        "message":        req.message,
        "request_type":   req.request_type,
        "warranty_info":  None,
        "part_order":     None,
        "appointment_time": None,
        "appointment_info": None,
        "stock_info":     None,
        "reply":          "",
        "service_completed": False,
    }

    try:
        result = await graph.ainvoke(initial_state)
    except Exception as e:
        logger.error("after_sale.api_error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "AFTER_SALE_ERROR", "message": str(e)},
        ) from e

    request_type = result.get("request_type", "")
    warranty = result.get("warranty_info")
    part_order = result.get("part_order")
    appointment = result.get("appointment_info")

    if request_type == "warranty" and warranty:
        if warranty.get("status") == "in_warranty":
            reply = (
                f"设备 {warranty.get('device_sn', '')} 目前处于保修期内，"
                f"保修截止 {warranty.get('warranty_end', '')}，"
                f"覆盖范围：{warranty.get('coverage', '')}。"
            )
        elif warranty.get("status") == "out_of_warranty":
            reply = (
                f"设备 {warranty.get('device_sn', '')} 已过保修期"
                f"（截止 {warranty.get('warranty_end', '')}）。"
                "您可以选择付费维修或预约上门服务。"
            )
        else:
            reply = f"保修信息查询失败：{warranty.get('error', '未知原因')}"
    elif request_type == "parts" and part_order:
        reply = part_order.get("stock_info", "配件库存信息如下。")
    elif request_type == "appointment" and appointment:
        reply = (
            f"预约已创建，预约号 {appointment.get('appointment_id', '')}，"
            f"时间 {appointment.get('scheduled_time', '')}。"
        )
    else:
        reply = "已收到您的售后诉求，如需进一步帮助请联系人工客服。"
        if result.get("warranty_info") and result["warranty_info"].get("error"):
            reply = f"保修查询失败：{result['warranty_info']['error']}"

    return AfterSaleResponse(
        status="completed",
        request_type=request_type,
        warranty_info=warranty,
        part_order=part_order,
        appointment_info=appointment,
        reply=reply,
    )
