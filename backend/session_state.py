"""
session_state - 会话状态注册表（进程级）

记录对话流中需要跨请求保留的标记。作用域与诊断图进程级 MemorySaver 一致：
- 诊断 interrupt/resume 本身由 graph.get_state(config).next 判定（权威来源）；
- 本注册表只记录「pipeline 模式中断后 resume 是否要继续 diagnosis→ticket→after_sale」。

使用场景（api/chat.py）：
  1. 首次 pipeline 启动时 set(session_id, pipeline_after=True)。
  2. 诊断中断返回 __interrupt__ 后，用户下一条消息 resume 时据此判断走 _stream_pipeline。
  3. 完成后 clear(session_id)。
"""
from dataclasses import dataclass, field


@dataclass
class SessionState:
    """单个会话的跨请求标记。"""
    pipeline_after: bool = False    # 中断后 resume 是否要继续 pipeline（诊断→工单→售后）
    route_label: str = ""           # 最近一次 LLM 路由 label（调试/追踪用）
    ticket_id: str = ""             # pipeline 中已创建的工单号（供售后阶段引用）


_registry: dict[str, SessionState] = {}


def get(session_id: str) -> SessionState:
    """获取会话状态，不存在则创建默认（返回缓存对象，可原地修改）。"""
    if session_id not in _registry:
        _registry[session_id] = SessionState()
    return _registry[session_id]


def set_pipeline_after(session_id: str, value: bool = True) -> None:
    """设置 pipeline_after 标记。"""
    get(session_id).pipeline_after = value


def set_ticket_id(session_id: str, ticket_id: str) -> None:
    """记录 pipeline 已创建的工单号。"""
    get(session_id).ticket_id = ticket_id


def clear(session_id: str) -> None:
    """清空会话状态。"""
    _registry.pop(session_id, None)
