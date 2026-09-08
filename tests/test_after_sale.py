"""
售后协调 Agent（Agent⑤）测试。

覆盖：
- _classify_request_type 关键词路由
- query_warranty_node（DB 正常 / 异常降级）
- create_appointment_record 写表（mock session）
- graph 三边拓扑（warranty / parts / appointment）
- query_part_stock_from_db / check_part_stock（parts 表真查库）

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
# order_part_node：配件编号提取 + 兜底
# ──────────────────────────────────────────────────────────────

def test_extract_part_no():
    assert after_sale_nodes._extract_part_no("主轴轴承 BRG-6204 还有库存吗") == "BRG-6204"
    assert after_sale_nodes._extract_part_no("VFD-7K5 多久能发货") == "VFD-7K5"
    assert after_sale_nodes._extract_part_no("轴承坏了想买个备件") == ""
    assert after_sale_nodes._extract_part_no("") == ""


def test_extract_device_sn():
    assert after_sale_nodes._extract_device_sn("帮我查一下 SN-AC-0001 的保修") == "SN-AC-0001"
    assert after_sale_nodes._extract_device_sn("设备 sn-ac-0002 过保了吗") == "SN-AC-0002"
    assert after_sale_nodes._extract_device_sn("我的设备坏了") == ""


@pytest.mark.asyncio
async def test_query_warranty_node_sn_from_message(monkeypatch):
    """state 无 device_sn 时从消息提取（对话流式链路的兜底路径）。"""
    captured = {}

    async def fake_query(sn):
        captured["sn"] = sn
        return {"device_sn": sn, "status": "in_warranty", "warranty_end": "2027-01-01"}

    monkeypatch.setattr(after_sale_nodes, "query_warranty_from_db", fake_query)
    result = await after_sale_nodes.query_warranty_node(
        {"device_sn": "", "message": "帮我查一下 SN-AC-0001 还在保修期吗"}
    )
    assert captured["sn"] == "SN-AC-0001"
    assert result["warranty_info"]["status"] == "in_warranty"


@pytest.mark.asyncio
async def test_order_part_node_extracts_from_message(monkeypatch):
    """part_order 为空时从消息提取配件编号并查库存。"""
    captured = {}

    async def fake_stock(payload):
        captured.update(payload)
        return "配件 深沟球轴承（BRG-6204）库存 25 件"

    monkeypatch.setattr(after_sale_nodes, "check_part_stock",
                        type("T", (), {"ainvoke": staticmethod(fake_stock)}))
    result = await after_sale_nodes.order_part_node(
        {"message": "主轴轴承 BRG-6204 还有库存吗", "part_order": {}}
    )
    assert captured["part_no"] == "BRG-6204"
    assert "库存" in result["part_order"]["stock_info"]
    assert result["part_order"]["part_no"] == "BRG-6204"


@pytest.mark.asyncio
async def test_order_part_node_no_part_no():
    """消息无配件编号 → 引导用户提供，不调工具。"""
    result = await after_sale_nodes.order_part_node(
        {"message": "想买个备件", "part_order": {}}
    )
    assert "配件编号" in result["part_order"]["stock_info"]


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
# query_part_stock_from_db / check_part_stock（parts 表真查库）
# ──────────────────────────────────────────────────────────────

class _FakePart:
    """模拟 parts 表一行记录。"""

    def __init__(self, **kw):
        self.part_no = kw.get("part_no", "BRG-6204")
        self.name = kw.get("name", "深沟球轴承 6204")
        self.stock_qty = kw.get("stock_qty", 25)
        self.lead_time_days = kw.get("lead_time_days", 3)
        self.price = kw.get("price", 45.0)
        self.device_models = kw.get("device_models", "CNC-1000")


class _FakeResult:
    def __init__(self, obj):
        self._obj = obj

    def scalar_one_or_none(self):
        return self._obj


class _FakeQuerySession:
    def __init__(self, obj):
        self._obj = obj

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def execute(self, stmt):
        return _FakeResult(self._obj)


class _FakeQueryFactory:
    def __init__(self, obj):
        self.session = _FakeQuerySession(obj)

    def __call__(self):
        return self.session


@pytest.mark.asyncio
async def test_query_part_stock_in_stock(monkeypatch):
    monkeypatch.setattr(
        after_sale_tools, "_get_session_factory", lambda: _FakeQueryFactory(_FakePart())
    )
    info = await after_sale_tools.query_part_stock_from_db("BRG-6204")
    assert info["status"] == "in_stock"
    assert info["stock_qty"] == 25


@pytest.mark.asyncio
async def test_query_part_stock_out_of_stock(monkeypatch):
    monkeypatch.setattr(
        after_sale_tools,
        "_get_session_factory",
        lambda: _FakeQueryFactory(_FakePart(stock_qty=0)),
    )
    info = await after_sale_tools.query_part_stock_from_db("VFD-7K5")
    assert info["status"] == "out_of_stock"


@pytest.mark.asyncio
async def test_query_part_stock_not_found(monkeypatch):
    monkeypatch.setattr(
        after_sale_tools, "_get_session_factory", lambda: _FakeQueryFactory(None)
    )
    info = await after_sale_tools.query_part_stock_from_db("NO-SUCH-PART")
    assert info["status"] == "not_found"


@pytest.mark.asyncio
async def test_query_part_stock_db_error(monkeypatch):
    class _BrokenSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def execute(self, stmt):
            raise RuntimeError("db down")

    class _BrokenFactory:
        def __call__(self):
            return _BrokenSession()

    monkeypatch.setattr(
        after_sale_tools, "_get_session_factory", lambda: _BrokenFactory()
    )
    info = await after_sale_tools.query_part_stock_from_db("BRG-6204")
    assert info["status"] == "unknown"
    assert "error" in info


@pytest.mark.asyncio
async def test_check_part_stock_tool_in_stock(monkeypatch):
    monkeypatch.setattr(
        after_sale_tools, "_get_session_factory", lambda: _FakeQueryFactory(_FakePart())
    )
    text = await after_sale_tools.check_part_stock.ainvoke({"part_no": "BRG-6204"})
    assert "BRG-6204" in text
    assert "库存" in text
    assert "25" in text


@pytest.mark.asyncio
async def test_check_part_stock_tool_out_of_stock(monkeypatch):
    monkeypatch.setattr(
        after_sale_tools,
        "_get_session_factory",
        lambda: _FakeQueryFactory(_FakePart(part_no="VFD-7K5", name="变频器 7.5kW", stock_qty=0)),
    )
    text = await after_sale_tools.check_part_stock.ainvoke({"part_no": "VFD-7K5"})
    assert "缺货" in text


@pytest.mark.asyncio
async def test_check_part_stock_tool_not_found(monkeypatch):
    monkeypatch.setattr(
        after_sale_tools, "_get_session_factory", lambda: _FakeQueryFactory(None)
    )
    text = await after_sale_tools.check_part_stock.ainvoke({"part_no": "NO-SUCH"})
    assert "未找到" in text
