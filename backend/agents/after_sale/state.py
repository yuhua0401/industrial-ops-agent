"""
state - 售后协调 Agent 的状态定义
"""
from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class WarrantyInfo(BaseModel):
    """保修信息。"""
    device_model: str = Field(description="设备型号")
    device_sn: str = Field(description="序列号")
    purchase_date: str = Field(description="购买日期")
    warranty_end: str = Field(description="保修截止日期")
    status: str = Field(description="保修状态：in_warranty / out_of_warranty / grace_period")
    coverage: str = Field(description="保修范围")


class PartOrder(BaseModel):
    """配件订购信息。"""
    part_no: str = Field(description="配件编号")
    part_name: str = Field(description="配件名称")
    quantity: int = Field(default=1, description="数量")
    price: float = Field(default=0.0, description="单价")
    stock: int = Field(default=0, description="库存数量")
    estimated_delivery: str = Field(default="", description="预计发货时间")


class AfterSaleState(TypedDict):
    """售后协调 Agent 的状态。"""
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    customer_id: str
    device_sn: str
    message: str                         # 用户诉求原文（路由分类依据）
    request_type: str                    # warranty / parts / appointment / followup
    warranty_info: dict | None
    part_order: dict | None
    appointment_time: str | None
    appointment_info: dict | None
    stock_info: dict | None
    reply: str                           # 最终回复文本（API 层渲染用）
    service_completed: bool
