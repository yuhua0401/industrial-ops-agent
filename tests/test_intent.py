"""
意图识别与路由测试（api/chat.py）。

覆盖：
- _pre_filter 五类规则拦截（问候/感谢/道别/身份/功能）
- _llm_route 路由（合法 label / 非法 label 降级 / LLM 抛错降级）
- _extract_ticket_id

所有测试不依赖真实 LLM / 网络，可离线运行。
"""
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.api import chat as chat_api

# ──────────────────────────────────────────────────────────────
# _pre_filter 五类规则拦截
# ──────────────────────────────────────────────────────────────

def test_pre_filter_hello():
    assert chat_api._pre_filter("你好") is not None
    assert chat_api._pre_filter("您好") is not None
    assert chat_api._pre_filter("hi") is not None
    assert chat_api._pre_filter("在吗") is not None


def test_pre_filter_thanks():
    assert chat_api._pre_filter("谢谢") is not None
    assert chat_api._pre_filter("辛苦了") is not None
    assert chat_api._pre_filter("明白了") is not None


def test_pre_filter_bye():
    assert chat_api._pre_filter("再见") is not None
    assert chat_api._pre_filter("拜拜") is not None
    assert chat_api._pre_filter("886") is not None


def test_pre_filter_identity():
    assert chat_api._pre_filter("你是谁") is not None
    assert chat_api._pre_filter("你叫什么名字") is not None


def test_pre_filter_capability():
    assert chat_api._pre_filter("你能做什么") is not None
    assert chat_api._pre_filter("使用说明") is not None
    assert chat_api._pre_filter("help") is not None


def test_pre_filter_no_match():
    assert chat_api._pre_filter("设备报E001电机不转了") is None
    assert chat_api._pre_filter("查一下工单进度") is None


# ──────────────────────────────────────────────────────────────
# _llm_route 路由
# ──────────────────────────────────────────────────────────────

class _FakeLLMResp:
    def __init__(self, text):
        self.text = text


class _FakeLLM:
    def __init__(self, text):
        self._text = text

    async def ainvoke(self, messages):
        return _FakeLLMResp(self._text)


def _make_fake_llm(text: str):
    return _FakeLLM(text)


@pytest.mark.asyncio
async def test_llm_route_valid_labels(monkeypatch):
    """合法 label 映射正确。"""
    cases = {
        '{"label": "diagnosis", "reason": "故障"}': "diagnosis",
        '{"label": "knowledge", "reason": "问知识"}': "knowledge",
        '{"label": "ticket", "reason": "查工单"}': "ticket",
        '{"label": "after_sale", "reason": "售后"}': "after_sale",
        '{"label": "pipeline", "reason": "报修"}': "pipeline",
        '{"label": "clarify", "reason": "不清楚"}': "clarify",
    }
    for raw, expected in cases.items():
        monkeypatch.setattr(
            chat_api, "get_llm",
            lambda *a, _raw=raw, **k: _make_fake_llm(_raw),
        )
        result = await chat_api._llm_route("测试消息")
        assert result.label == expected
        assert result.agent_type.value in {"knowledge", "diagnosis", "ticket", "after_sale"}


@pytest.mark.asyncio
async def test_llm_route_markdown_wrapped(monkeypatch):
    """LLM 输出被 markdown 代码块包裹 → 剥壳解析。"""
    raw = '```json\n{"label": "diagnosis", "reason": "故障"}\n```'
    monkeypatch.setattr(chat_api, "get_llm", lambda *a, **k: _make_fake_llm(raw))
    result = await chat_api._llm_route("E001")
    assert result.label == "diagnosis"


@pytest.mark.asyncio
async def test_llm_route_invalid_label_fallback(monkeypatch):
    """非法 label → 降级 knowledge。"""
    raw = '{"label": "unknown_thing", "reason": "x"}'
    monkeypatch.setattr(chat_api, "get_llm", lambda *a, **k: _make_fake_llm(raw))
    result = await chat_api._llm_route("测试")
    assert result.label == "knowledge"


@pytest.mark.asyncio
async def test_llm_route_exception_fallback(monkeypatch):
    """LLM 调用抛错 → 降级 knowledge（不抛）。"""

    class _FailLLM:
        async def ainvoke(self, messages):
            raise RuntimeError("LLM down")

    monkeypatch.setattr(chat_api, "get_llm", lambda *a, **k: _FailLLM())
    result = await chat_api._llm_route("测试")
    assert result.label == "knowledge"


# ──────────────────────────────────────────────────────────────
# _extract_ticket_id
# ──────────────────────────────────────────────────────────────

def test_extract_ticket_id():
    assert chat_api._extract_ticket_id("工单号 TK-20260822-000123 查进度") == "TK-20260822-000123"
    assert chat_api._extract_ticket_id("没有工单号") == ""
    assert chat_api._extract_ticket_id("TK-20260822-123") == ""
