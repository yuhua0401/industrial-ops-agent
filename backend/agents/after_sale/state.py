"""
state - 售后协调 Agent 的状态定义
"""
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


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
    request_type: str                    # warranty / parts / appointment / followup
    warranty_info: Optional[dict]
    part_order: Optional[dict]
    appointment_time: Optional[str]
    service_completed: bool
