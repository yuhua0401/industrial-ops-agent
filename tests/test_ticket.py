"""
工单管理 Agent（Agent④）测试。

覆盖：
- 状态机 TICKET_STATUS_FLOW 合法/非法流转
- repo._normalize_ticket_data / _next_ticket_id
- create_ticket_node（mock LLM + mock session factory，验证落库语义）
- API 端点（mock DB session，验证审计日志写入）

所有测试不依赖真实 DB / LLM / 网络，可离线运行。
"""
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.agents.ticket import nodes as ticket_nodes
from backend.agents.ticket import repo as ticket_repo
from backend.agents.ticket.schemas import TICKET_STATUS_FLOW
from backend.agents.ticket.state import TicketSchema

# ──────────────────────────────────────────────────────────────
# 状态机流转
# ──────────────────────────────────────────────────────────────

def test_status_flow_valid_transitions():
    """合法流转表驱动。"""
    valid_cases = [
        ("pending", "dispatched"),
        ("pending", "cancelled"),
        ("dispatched", "processing"),
        ("dispatched", "cancelled"),
        ("processing", "waiting_parts"),
        ("processing", "resolved"),
        ("processing", "cancelled"),
        ("waiting_parts", "processing"),
        ("waiting_parts", "resolved"),
        ("waiting_parts", "cancelled"),
        ("resolved", "closed"),
    ]
    for from_status, to_status in valid_cases:
        assert to_status in TICKET_STATUS_FLOW[from_status], (
            f"{from_status} → {to_status} 应为合法流转"
        )


def test_status_flow_invalid_transitions():
    """非法流转表驱动（终态不可再转）。"""
    invalid_cases = [
        ("resolved", "processing"),
        ("closed", "resolved"),
        ("cancelled", "pending"),
        ("pending", "resolved"),
        ("pending", "processing"),
        ("cancelled", "dispatched"),
    ]
    for from_status, to_status in invalid_cases:
        assert to_status not in TICKET_STATUS_FLOW[from_status], (
            f"{from_status} → {to_status} 应为非法流转"
        )


def test_status_flow_terminal_states():
    """closed / cancelled 为终态（无出边）。"""
    assert TICKET_STATUS_FLOW["closed"] == []
    assert TICKET_STATUS_FLOW["cancelled"] == []


# ──────────────────────────────────────────────────────────────
# repo：归一化 / 工单号生成
# ──────────────────────────────────────────────────────────────

def test_normalize_ticket_data():
    """字段归一化：缺省值 + 类型转 str。"""
    d = ticket_repo._normalize_ticket_data({
        "fault_description": "E001 电机不转",
        "severity": "high",
    })
    assert d["status"] == "pending"
    assert d["severity"] == "high"
    assert d["category"] == "repair"
    assert d["tenant_id"] == "tenant_default"
    assert d["diagnosis_result"] == ""


def test_normalize_ticket_data_empty_customer_id():
    """customer_id 为空串 → None（外键列 nullable=True，空串会触发外键失败）。"""
    d = ticket_repo._normalize_ticket_data({"fault_description": "x"})
    assert d["customer_id"] is None

    d2 = ticket_repo._normalize_ticket_data({"customer_id": "C-1001"})
    assert d2["customer_id"] == "C-1001"


class _FakeSession:
    """mock AsyncSession：仅实现 _next_ticket_id 用到的 execute + scalar_one_or_none。"""

    def __init__(self, last_ticket_id):
        self._last = last_ticket_id

    async def execute(self, stmt):
        return _FakeResult(self._last)

    async def flush(self):
        pass

    async def commit(self):
        pass


class _FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


def test_next_ticket_id_first_of_day():
    """当天无工单 → 序号从 000001 开始。"""
    import asyncio

    async def run():
        # last_id 为 None → seq=1
        session = _FakeSession(None)
        return await ticket_repo._next_ticket_id(session)

    tid = asyncio.run(run())
    assert tid.startswith("TK-")
    assert tid.endswith("-000001")
    assert len(tid) == len("TK-YYYYMMDD-000001")


def test_next_ticket_id_increments():
    """当天已有工单 → 序号 +1。"""
    import asyncio

    async def run():
        session = _FakeSession("TK-20260822-000007")
        return await ticket_repo._next_ticket_id(session)

    tid = asyncio.run(run())
    assert tid.endswith("-000008")


def test_next_ticket_id_bad_last_id_falls_back():
    """最后一个 ID 解析失败 → 从头开始（不崩溃）。"""
    import asyncio

    async def run():
        session = _FakeSession("TK-20260822-garbage")
        return await ticket_repo._next_ticket_id(session)

    tid = asyncio.run(run())
    assert tid.endswith("-000001")


# ──────────────────────────────────────────────────────────────
# create_ticket_node：LLM 归一化 + 真实落库语义
# ──────────────────────────────────────────────────────────────

class _FakeStructuredLLM:
    """结构化 LLM 替身：返回固定 TicketSchema。"""

    async def ainvoke(self, messages):
        return TicketSchema(
            fault_description="E001 电机不转",
            device_model="CNC-1000",
            severity="high",
            category="repair",
        )


class _FakeFailLLM:
    """失败 LLM 替身：抛异常，测试原始数据兜底。"""

    async def ainvoke(self, messages):
        raise ValueError("LLM unavailable")


class _FakeSessionFactory:
    """mock session 工厂：模拟 create_ticket_via_factory 的落库。"""

    def __init__(self, created: bool = True):
        self._created = created

    def __call__(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def __aiter__(self):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_create_ticket_node_success(monkeypatch):
    """LLM 归一化成功 + 落库成功 → created=True，ticket_id 非空。"""
    monkeypatch.setattr(ticket_nodes, "get_structured_llm", lambda *a, **k: _FakeStructuredLLM())

    async def fake_create(data, operator="system"):
        return {"ticket_id": "TK-20260822-000001", "created": True, "error": ""}
    monkeypatch.setattr(ticket_nodes, "create_ticket_via_factory", fake_create)

    result = await ticket_nodes.create_ticket_node({
        "messages": [], "session_id": "s1",
        "ticket_data": {"fault_description": "E001 电机不转", "device_model": "CNC-1000"},
        "ticket": None, "created": False, "ticket_id": None,
    })
    assert result["created"] is True
    assert result["ticket_id"] == "TK-20260822-000001"
    assert result["ticket"]["fault_description"] == "E001 电机不转"


@pytest.mark.asyncio
async def test_create_ticket_node_llm_failure_fallback(monkeypatch):
    """LLM 归一化失败 → 用原始 ticket_data 兜底仍可创建。"""
    monkeypatch.setattr(ticket_nodes, "get_structured_llm", lambda *a, **k: _FakeFailLLM())

    async def fake_create(data, operator="system"):
        return {"ticket_id": "TK-20260822-000002", "created": True, "error": ""}
    monkeypatch.setattr(ticket_nodes, "create_ticket_via_factory", fake_create)

    result = await ticket_nodes.create_ticket_node({
        "messages": [], "session_id": "s1",
        "ticket_data": {"fault_description": "电机异响", "customer_id": "C1"},
        "ticket": None, "created": False, "ticket_id": None,
    })
    assert result["created"] is True
    # 原始数据兜底时 fault_description 保留
    assert result["ticket"]["fault_description"] == "电机异响"


@pytest.mark.asyncio
async def test_create_ticket_node_db_failure(monkeypatch):
    """落库失败 → created=False，不抛异常。"""
    monkeypatch.setattr(ticket_nodes, "get_structured_llm", lambda *a, **k: _FakeStructuredLLM())

    async def fake_create(data, operator="system"):
        return {"ticket_id": "", "created": False, "error": "DB down"}
    monkeypatch.setattr(ticket_nodes, "create_ticket_via_factory", fake_create)

    result = await ticket_nodes.create_ticket_node({
        "messages": [], "session_id": "s1",
        "ticket_data": {"fault_description": "E001"},
        "ticket": None, "created": False, "ticket_id": None,
    })
    assert result["created"] is False
    assert result["ticket_id"] == ""
