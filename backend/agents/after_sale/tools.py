"""
tools - 售后协调 Agent 的 Tool 定义

用于 LangChain Tool Calling，LLM 根据客户需求调用对应的工具。
"""
from langchain_core.tools import tool


@tool
def query_warranty(device_sn: str) -> str:
    """查询设备保修信息。传入设备序列号，返回保修状态和截止日期。"""
    # TODO: 对接 ERP 系统
    return f"设备 {device_sn} 保修状态：保修期内，保修截止 2026-01-15"


@tool
def check_part_stock(part_no: str) -> str:
    """查询配件库存。传入配件编号，返回库存数量和预计发货时间。"""
    # TODO: 对接 WMS 系统
    return f"配件 {part_no} 库存：10 件，预计发货：3-5 个工作日"


@tool
def create_appointment(device_sn: str, customer_address: str, prefer_time: str) -> str:
    """创建上门服务预约。返回预约确认信息和工程师预计到达时间。"""
    # TODO: 对接预约调度系统
    return f"已为您预约上门服务，设备 {device_sn}，地址 {customer_address}，时间 {prefer_time}。"
