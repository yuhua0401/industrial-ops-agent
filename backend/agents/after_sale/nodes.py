"""
nodes - 售后协调 Agent 的节点函数

通过 Tool Calling 调用外部系统 API 完成各类售后操作。
"""
from backend.core.logger import get_logger

logger = get_logger(__name__)


async def query_warranty_node(state: AfterSaleState) -> dict:
    """节点①：查询保修信息。"""
    device_sn = state.get("device_sn", "")
    if not device_sn:
        return {"warranty_info": {"status": "unknown", "error": "缺少设备序列号"}}

    # TODO: 调用 ERP/CRM 接口查询保修信息
    # warranty = await call_erp_api(f"query_warranty?sn={device_sn}")
    logger.info("after_sale.warranty_query", device_sn=device_sn)

    # 桩数据
    return {
        "warranty_info": {
            "device_sn": device_sn,
            "purchase_date": "2024-01-15",
            "warranty_end": "2026-01-15",
            "status": "out_of_warranty",
            "coverage": "整机保修",
        }
    }


async def order_part_node(state: AfterSaleState) -> dict:
    """节点②：配件订购。"""
    part = state.get("part_order", {})
    # TODO: 调用 WMS 接口查询库存 + 下订单
    logger.info("after_sale.part_order", part=part)
    return {"part_order": {**part, "estimated_delivery": "3-5个工作日", "stock": 10}}
