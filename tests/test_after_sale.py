"""
售后协调 Agent（Agent⑤）测试。

覆盖：
- _classify_request_type 关键词路由
- query_warranty_node（DB 正常 / 异常降级）
- create_appointment_record 写表（mock session）
- graph 三边拓扑（warranty / parts / appointment）
- check_part_stock 桩

所有测试不依赖真实 DB / LLM / 网络，可离线运行。
"""
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.agents.after_sale import nodes as after_sale_nodes
from backend.agents.after_sale import tools as after_sale_tools
from backend.agents.after_sale.graph import _route_by_request_type, build_after_sale_graph

# ──────────────────────────────────────────────────────────────
# 关键词路由
# ──────────────────────────────────────────────────────────────

def test_classify_warranty():
    assert after_sale_nodes._classify_request_type("帮我查下这台设备的保修状态") == "warranty"
    assert after_sale_nodes._classify_request_type("设备还在质保期内吗") == "warranty"
    assert after_sale_nodes._classify_request_type("查一下保修期") == "warranty"


def test_classify_parts():
    assert after_sale_nodes._classify_request_type("这个配件有库存吗") == "parts"
    assert after_sale_nodes._classify_request_type("想买个备件") == "parts"
    assert after_sale_nodes._classify_request_type("配件多久能发货") == "parts"


def test_classify_appointment():
    assert after_sale_nodes._classify_request_type("想预约师傅上门维修") == "appointment"
    assert after_sale_nodes._classify_request_type("派人过来看看") == "appointment"
    assert after_sale_nodes._classify_request_type("预约工程师") == "appointment"


def test_classify_priority():
    """多关键词时优先级：保修 > 配件 > 预约。"""
    assert after_sale_nodes._classify_request_type("查保修顺便预约上门") == "warranty"


def test_classify_empty():
    assert after_sale_nodes._classify_request_type("") == ""
    assert after_sale_nodes._classify_request_type("随便聊聊") == ""


# ──────────────────────────────────────────────────────────────
# route_request_type_node：LLM 兜底分支
# ──────────────────────────────────────────────────────────────

class _FakeLLM:
    async def ainvoke(self, messages):
        return _FakeResp("appointment")


class _FakeResp:
    def __init__(self, text):
        self.text = text


@pytest.mark.asyncio
async def test_route_node_keyword_hit():
    """关键词命中直接路由，不调 LLM。"""
    state = {"message": "查保修", "request_type": "", "device_sn": "SN1"}
    result = await after_sale_nodes.route_request_type_node(state)
    assert result["request_type"] == "warranty"


@pytest.mark.asyncio
async def test_route_node_explicit_type(monkeypatch):
    """API 显式指定 request_type 优先。"""
    state = {"message": "随便说", "request_type": "parts", "device_sn": "SN1"}
    monkeypatch.setattr(after_sale_nodes, "get_llm", lambda *a, **k: _FakeLLM())
    result = await after_sale_nodes.route_request_type_node(state)
    assert result["request_type"] == "parts"


@pytest.mark.asyncio
async def test_route_node_llm_fallback(monkeypatch):
    """关键词未命中 → LLM 兜底。"""
    state = {"message": "设备有点问题", "request_type": "", "device_sn": "SN1"}
    monkeypatch.setattr(after_sale_nodes, "get_llm", lambda *a, **k: _FakeLLM())
    result = await after_sale_nodes.route_request_type_node(state)
    assert result["request_type"] == "appointment"


# ──────────────────────────────────────────────────────────────
# query_warranty_node：DB 正常 / 异常降级
# ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_warranty_node_in_warranty(monkeypatch):
    async def fake_query(sn):
        return {"device_sn": sn, "status": "in_warranty", "warranty_end": "2026-01-15"}
    monkeypatch.setattr(after_sale_nodes, "query_warranty_from_db", fake_query)
    result = await after_sale_nodes.query_warranty_node({"device_sn": "SN1"})
    assert result["warranty_info"]["status"] == "in_warranty"


@pytest.mark.asyncio
async def test_query_warranty_node_out_of_warranty(monkeypatch):
    async def fake_query(sn):
        return {"device_sn": sn, "status": "out_of_warranty", "warranty_end": "2024-01-15"}
    monkeypatch.setattr(after_sale_nodes, "query_warranty_from_db", fake_query)
    result = await after_sale_nodes.query_warranty_node({"device_sn": "SN1"})
    assert result["warranty_info"]["status"] == "out_of_warranty"


@pytest.mark.asyncio
async def test_query_warranty_node_missing_sn():
    result = await after_sale_nodes.query_warranty_node({"device_sn": ""})
    assert result["warranty_info"]["status"] == "unknown"


@pytest.mark.asyncio
async def test_query_warranty_node_db_error(monkeypatch):
    async def fake_query(sn):
        return {"status": "unknown", "error": "服务不可用"}
    monkeypatch.setattr(after_sale_nodes, "query_warranty_from_db", fake_query)
    result = await after_sale_nodes.query_warranty_node({"device_sn": "SN1"})
    assert result["warranty_info"]["status"] == "unknown"
    assert "error" in result["warranty_info"]


# ──────────────────────────────────────────────────────────────
# create_appointment_record：写表（mock session）
# ──────────────────────────────────────────────────────────────

class _FakeApptSession:
    def __init__(self):
        self.added = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        pass


class _FakeApptFactory:
    def __init__(self):
        self.session = _FakeApptSession()

    def __call__(self):
        return self.session


@pytest.mark.asyncio
async def test_create_appointment_record_writes(monkeypatch):
    fake_factory = _FakeApptFactory()
    monkeypatch.setattr(after_sale_tools, "_get_session_factory", lambda: fake_factory)

    record = await after_sale_tools.create_appointment_record(
        device_sn="SN1", customer_id="C1", scheduled_time="2026-08-25T10:00:00", address="北京市",
    )
    assert record["appointment_id"].startswith("APPT-")
    assert record["status"] == "scheduled"
    assert record["device_sn"] == "SN1"
    # 断言写入了 AfterSaleAppointment ORM 对象
    assert len(fake_factory.session.added) == 1
    obj = fake_factory.session.added[0]
    assert obj.device_sn == "SN1"
    assert obj.customer_id == "C1"


@pytest.mark.asyncio
async def test_create_appointment_record_invalid_time(monkeypatch):
    """非法时间字符串 → 兜底为当前时间+3天（不崩溃）。"""
    fake_factory = _FakeApptFactory()
    monkeypatch.setattr(after_sale_tools, "_get_session_factory", lambda: fake_factory)

    record = await after_sale_tools.create_appointment_record(
        device_sn="SN1", customer_id="C1", scheduled_time="not-a-date", address="",
    )
    assert record["appointment_id"].startswith("APPT-")


# ──────────────────────────────────────────────────────────────
# graph：三边拓扑
# ──────────────────────────────────────────────────────────────

def test_route_by_request_type():
    assert _route_by_request_type({"request_type": "warranty"}) == "query_warranty"
    assert _route_by_request_type({"request_type": "parts"}) == "order_part"
    assert _route_by_request_type({"request_type": "appointment"}) == "prepare_appointment"
    assert _route_by_request_type({"request_type": "unknown"}) == "prepare_appointment"


def test_graph_builds():
    g = build_after_sale_graph()
    node_names = list(g.get_graph().nodes.keys())
    assert "route_request_type" in node_names
    assert "query_warranty" in node_names
    assert "order_part" in node_names
    assert "prepare_appointment" in node_names
    assert "tools" in node_names


# ──────────────────────────────────────────────────────────────
# check_part_stock 桩
# ──────────────────────────────────────────────────────────────

def test_check_part_stock_stub():
    result = after_sale_tools.check_part_stock.invoke({"part_no": "P001"})
    assert "P001" in result
    assert "库存" in result
