"""
state - 工单管理 Agent 的状态定义
"""
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from datetime import datetime


class TicketSchema(BaseModel):
    """工单 Schema。"""
    ticket_id: str = Field(default="", description="工单号，创建后由系统生成")
    customer_name: str = Field(default="", description="客户姓名/企业名")
    customer_contact: str = Field(default="", description="联系方式")
    device_model: str = Field(default="", description="设备型号")
    device_sn: str = Field(default="", description="设备序列号")
    fault_description: str = Field(description="故障描述")
    severity: str = Field(default="medium", description="严重程度：low/medium/high/critical")
    category: str = Field(default="repair", description="工单类别：repair/installation/consultation/complaint")
    status: str = Field(default="pending", description="状态：pending/dispatched/processing/waiting_parts/resolved/closed/cancelled")
    assigned_engineer: str = Field(default="", description="指派的工程师")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    notes: str = Field(default="", description="备注")


class TicketState(TypedDict):
    """工单管理 Agent 的状态。"""
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    ticket_data: Optional[dict]      # 工单原始数据（从诊断/咨询中提取）
    ticket: Optional[dict]           # TicketSchema.model_dump()
    created: bool                    # 是否已创建工单
    ticket_id: Optional[str]         # 工单号
