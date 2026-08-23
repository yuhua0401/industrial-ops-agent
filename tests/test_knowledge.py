"""
知识库 Agent（Agent②）测试。

覆盖：
- retrieve_node（检索成功 / 抛错降级为空列表）
- generate_node（结构化输出成功 / 失败降级）
- graph 连接

所有测试不依赖真实 Milvus / LLM / 网络，可离线运行。
"""
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.agents.knowledge import nodes as knowledge_nodes
from backend.agents.knowledge.state import KnowledgeResult

# ──────────────────────────────────────────────────────────────
# retrieve_node
# ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retrieve_node_success(monkeypatch):
    async def fake_retrieve(query, device_model=None, top_k=5):
        return [{"content": "E001 电机过载排查手册", "source": "手册", "score": 0.9}]
    monkeypatch.setattr(
        "backend.knowledge_base.retriever.hybrid_retrieve", fake_retrieve,
    )
    result = await knowledge_nodes.retrieve_node({"query": "E001 不转", "device_model": "CNC"})
    assert len(result["retrieved_docs"]) == 1
    assert result["retrieved_docs"][0]["source"] == "手册"


@pytest.mark.asyncio
async def test_retrieve_node_failure_degrades(monkeypatch):
    async def fake_retrieve(*a, **k):
        raise RuntimeError("Milvus down")
    monkeypatch.setattr(
        "backend.knowledge_base.retriever.hybrid_retrieve", fake_retrieve,
    )
    result = await knowledge_nodes.retrieve_node({"query": "E001", "device_model": ""})
    assert result["retrieved_docs"] == []


# ──────────────────────────────────────────────────────────────
# generate_node
# ──────────────────────────────────────────────────────────────

class _FakeStructuredLLM:
    def __init__(self, result=None, fail=False):
        self._result = result
        self._fail = fail

    async def ainvoke(self, messages):
        if self._fail:
            raise ValueError("LLM down")
        return self._result


@pytest.mark.asyncio
async def test_generate_node_success(monkeypatch):
    fake = _FakeStructuredLLM(result=KnowledgeResult(
        answer="电机过载需断电检查", confidence="high", sources=["手册-第三章"],
    ))
    monkeypatch.setattr(knowledge_nodes, "get_structured_llm", lambda *a, **k: fake)

    state = {"query": "E001", "retrieved_docs": [{"content": "x", "source": "手册"}]}
    result = await knowledge_nodes.generate_node(state)
    kr = result["knowledge_result"]
    assert kr["answer"] == "电机过载需断电检查"
    assert kr["confidence"] == "high"


@pytest.mark.asyncio
async def test_generate_node_no_docs(monkeypatch):
    fake = _FakeStructuredLLM(result=KnowledgeResult(
        answer="知识库未找到，已转人工", confidence="low",
    ))
    monkeypatch.setattr(knowledge_nodes, "get_structured_llm", lambda *a, **k: fake)

    state = {"query": "E001", "retrieved_docs": []}
    result = await knowledge_nodes.generate_node(state)
    assert "人工" in result["knowledge_result"]["answer"]


@pytest.mark.asyncio
async def test_generate_node_failure_fallback(monkeypatch):
    """LLM 连续失败 → 返回降级固定文案。"""
    fake = _FakeStructuredLLM(fail=True)
    monkeypatch.setattr(knowledge_nodes, "get_structured_llm", lambda *a, **k: fake)

    state = {"query": "E001", "retrieved_docs": [{"content": "x", "source": "手册"}]}
    result = await knowledge_nodes.generate_node(state)
    assert "转接人工客服" in result["knowledge_result"]["answer"]
    assert result["knowledge_result"]["confidence"] == "low"


# ──────────────────────────────────────────────────────────────
# graph 连接
# ──────────────────────────────────────────────────────────────

def test_knowledge_graph_builds():
    from backend.agents.knowledge.graph import build_knowledge_graph
    g = build_knowledge_graph()
    node_names = list(g.get_graph().nodes.keys())
    assert "retrieve" in node_names
    assert "generate" in node_names
