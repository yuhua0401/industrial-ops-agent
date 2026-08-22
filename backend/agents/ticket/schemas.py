"""
schemas - 工单相关的 Schema 定义

工单状态机流转规则（完整运维状态流，覆盖派单/等备件/已解决）：
  pending → dispatched → processing → waiting_parts → resolved → closed
     ↓           ↓              ↓            ↓
  cancelled  cancelled      cancelled    cancelled
"""
# 工单状态常量：唯一权威状态机定义（与 db/models.py Ticket.status、state.py TicketSchema 保持一致）
TICKET_STATUS_FLOW = {
    "pending":       ["dispatched", "cancelled"],
    "dispatched":    ["processing", "cancelled"],
    "processing":    ["waiting_parts", "resolved", "cancelled"],
    "waiting_parts": ["processing", "resolved", "cancelled"],
    "resolved":      ["closed"],
    "closed":        [],
    "cancelled":     [],
}
