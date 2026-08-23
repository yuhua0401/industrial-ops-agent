"""
Supervisor 主编排测试。

覆盖：
- _build_ticket_data 纯函数字段映射
- _render_diagnosis_report 渲染
- run_diagnosis 返回 AgentResult（mock 图）
- pipeline 串联（mock run_ticket / run_after_sale）
- 图缓存（诊断图 MemorySaver 单例）

所有测试不依赖真实 LLM / DB / 网络，可离线运行。
"""
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.supervisor import (
    AgentResult,
    Supervisor,
    _build_ticket_data,
    _render_diagnosis_report,
)

# ──────────────────────────────────────────────────────────────
# 纯函数
# ──────────────────────────────────────────────────────────────

def test_build_ticket_data_mapping():
    report = {
        "conclusion": "电机过载保护触发",
        "need_ticket": True,
        "ticket_reason": "需上门处理",
    }
    context = {
        "user_input": "E001 电机不转",
        "customer_id": "C1",
        "customer_name": "华东钢铁",
        "device_model": "CNC-1000",
        "device_sn": "SN001",
    }
    td = _build_ticket_data(report, context)
    assert td["fault_description"] == "E001 电机不转"
    assert td["diagnosis_result"] == "电机过载保护触发"
    assert td["category"] == "repair"
    assert td["customer_id"] == "C1"
    assert td["customer_name"] == "华东钢铁"
    assert td["notes"] == "需上门处理"
    assert td["tenant_id"] == "tenant_default"


def test_build_ticket_data_defaults():
    td = _build_ticket_data({}, {})
    assert td["severity"] == "medium"
    assert td["category"] == "repair"
    assert td["fault_description"] == ""


def test_render_diagnosis_report_full():
    report = {
        "conclusion": "电机过载",
        "causes": [{"desc": "负载过大", "probability": "高"}],
        "solutions": [{"step": "断电盘车", "need_skill": False}],
        "need_ticket": True,
    }
    text = _render_diagnosis_report(report)
    assert "电机过载" in text
    assert "负载过大" in text
    assert "断电盘车" in text
    assert "工单" in text


def test_render_diagnosis_report_empty():
    """空 report 渲染：不含原因/方案/工单建议（结论为空串）。"""
    text = _render_diagnosis_report({})
    assert "可能原因" not in text
    assert "处理方案" not in text
    assert "工单" not in text


# ──────────────────────────────────────────────────────────────
# Supervisor：图缓存
# ──────────────────────────────────────────────────────────────

def test_supervisor_singleton():
    from backend.supervisor import get_supervisor
    s1 = get_supervisor()
    s2 = get_supervisor()
    assert s1 is s2


def test_diagnosis_graph_cached():
    s = Supervisor()
    g1 = s.get_diagnosis_graph()
    g2 = s.get_diagnosis_graph()
    assert g1 is g2


def test_diagnosis_graph_has_memory_checkpointer():
    """诊断图必须带 checkpointer（interrupt/resume 依赖）。"""
    s = Supervisor()
    g = s.get_diagnosis_graph()
    assert g.checkpointer is not None


# ──────────────────────────────────────────────────────────────
# run_diagnosis：mock 图 ainvoke
# ──────────────────────────────────────────────────────────────

class _FakeGraph:
    def __init__(self, result):
        self._result = result

    async def ainvoke(self, *args, **kwargs):
        return self._result


@pytest.mark.asyncio
async def test_run_diagnosis_completed(monkeypatch):
    s = Supervisor()
    result_state = {
        "structured_output": {
            "conclusion": "电机过载",
            "causes": [],
            "solutions": [],
            "need_ticket": True,
            "confidence": 0.85,
            "ticket_reason": "需上门",
        },
        "need_ticket": True,
    }
    monkeypatch.setattr(s, "get_diagnosis_graph", lambda: _FakeGraph(result_state))

    result = await s.run_diagnosis("E001 电机不转", "s1")
    assert result.need_ticket is True
    assert result.confidence == 0.85
    assert "电机过载" in result.reply


@pytest.mark.asyncio
async def test_run_diagnosis_interrupted(monkeypatch):
    """诊断返回 __interrupt__ → interrupted=True + payload。"""
    s = Supervisor()
    from langgraph.types import Interrupt
    result_state = {"__interrupt__": [Interrupt(value={"question": "故障时满载吗？"})]}
    monkeypatch.setattr(s, "get_diagnosis_graph", lambda: _FakeGraph(result_state))

    result = await s.run_diagnosis("电机不转", "s1")
    assert result.interrupted is True
    assert result.interrupt_payload == {"question": "故障时满载吗？"}


# ──────────────────────────────────────────────────────────────
# run_pipeline：串联诊断 → 工单 → 售后
# ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_pipeline_chains(monkeypatch):
    """need_ticket=True → run_ticket 被调 → run_after_sale 被调。"""
    s = Supervisor()

    async def fake_diag(*a, **k):
        return AgentResult(
            reply="诊断结论",
            structured={
                "conclusion": "电机过载",
                "need_ticket": True,
                "confidence": 0.8,
            },
            agent_chain=["diagnosis"],
            need_ticket=True,
            confidence=0.8,
        )

    async def fake_ticket(*a, **k):
        return AgentResult(
            reply="工单已创建 TK-xxx", ticket_id="TK-xxx", agent_chain=["ticket"],
        )

    async def fake_after(*a, **k):
        return AgentResult(reply="售后信息", agent_chain=["after_sale"])

    monkeypatch.setattr(s, "run_diagnosis", fake_diag)
    monkeypatch.setattr(s, "run_ticket", fake_ticket)
    monkeypatch.setattr(s, "run_after_sale", fake_after)

    result = await s.run_pipeline("E001 不转", "s1")
    assert result.agent_chain == ["diagnosis", "ticket", "after_sale"]
    assert result.ticket_id == "TK-xxx"
    assert result.need_ticket is True
    assert "工单" in result.reply


@pytest.mark.asyncio
async def test_run_pipeline_no_ticket(monkeypatch):
    """need_ticket=False → 不建工单不售后。"""
    s = Supervisor()

    async def fake_diag(*a, **k):
        return AgentResult(
            reply="诊断结论",
            structured={"conclusion": "正常", "need_ticket": False, "confidence": 0.9},
            agent_chain=["diagnosis"], need_ticket=False, confidence=0.9,
        )

    async def _should_not_call(*a, **k):
        raise AssertionError("不应调用")

    monkeypatch.setattr(s, "run_diagnosis", fake_diag)
    monkeypatch.setattr(s, "run_ticket", _should_not_call)
    monkeypatch.setattr(s, "run_after_sale", _should_not_call)

    result = await s.run_pipeline("E001 不转", "s1")
    assert result.agent_chain == ["diagnosis"]
    assert result.ticket_id is None


@pytest.mark.asyncio
async def test_run_pipeline_interrupt(monkeypatch):
    """诊断中断 → pipeline 直接返回，不建单。"""
    s = Supervisor()

    async def fake_diag(*a, **k):
        return AgentResult(
            reply="", structured={}, agent_chain=["diagnosis"],
            interrupted=True, interrupt_payload={"question": "满载吗？"},
        )

    async def _should_not_call(*a, **k):
        raise AssertionError("不应调用")

    monkeypatch.setattr(s, "run_diagnosis", fake_diag)
    monkeypatch.setattr(s, "run_ticket", _should_not_call)

    result = await s.run_pipeline("电机不转", "s1")
    assert result.interrupted is True
    assert result.interrupt_payload == {"question": "满载吗？"}
