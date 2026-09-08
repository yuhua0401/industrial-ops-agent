"""
故障诊断 Agent（Agent③）测试。

覆盖范围：
- DiagTree 诊断树（故障码精确匹配 / 现象模糊匹配 / YAML 加载）
- nodes 纯函数（故障码提取、现象切分、三轨匹配、追问选择、模板兜底）
- 异步节点（mock LLM / 知识库，验证降级路径不中断流程）
- graph 条件路由

所有测试不依赖真实 LLM / 知识库 / 网络，可离线运行。
"""
# -*- coding: utf-8 -*-
import json
import sys
import types
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

# 直接运行（python tests/test_diagnosis.py）时项目根不在 sys.path，这里补上；
# pytest 运行时项目根已在路径中，重复插入无害。
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.agents.diagnosis import nodes
from backend.agents.diagnosis.diag_tree import DiagNode, DiagTree, RootCause, Solution
from backend.agents.diagnosis.graph import _route_after_check
from backend.agents.diagnosis.kb_client import KBClient
from backend.agents.diagnosis.state import DiagnosisReport, DiagnosisResult, DiagnosisStep


# ──────────────────────────────────────────────────────────────
# 测试工具：构造诊断树 / 节点
# ──────────────────────────────────────────────────────────────

def _make_node(**kwargs) -> DiagNode:
    """构造一个诊断树节点（默认贴近 data/diag_tree.yaml 的 D001）。"""
    base = dict(
        node_id="D001",
        name="电机过载保护触发",
        fault_code="E001",
        phenomena=["电机不转", "过载报警", "电机发热", "异响"],
        priority=2,
        causes=[RootCause(desc="负载过大或卡死", probability="高"),
                RootCause(desc="供电电压过低", probability="中")],
        solutions=[Solution(step="断电后手动盘车", need_skill=False),
                   Solution(step="万用表检测三相供电电压", need_skill=True)],
        need_ticket=False,
        questions=["故障时设备是否处于满载状态？", "电机外壳是否明显发烫？"],
    )
    base.update(kwargs)
    return DiagNode(**base)


def _make_tree(*nodes_list: DiagNode) -> DiagTree:
    """构造内存诊断树（避免依赖真实 YAML 数据文件）。"""
    tree = DiagTree()
    for n in nodes_list:
        tree.add_node(n)
    return tree


def _make_state(**overrides) -> dict:
    """构造最小可用的 DiagnosisState（TypedDict 的 dict 形态）。"""
    base = dict(
        session_id="s-test",
        device_model="CNC-1000",
        fault_description="",
        user_input="电机不转了，出现过载报警",
        fault_code=None,
        phenomena=["电机不转", "过载报警"],
        turn=0,
        messages=[],
    )
    base.update(overrides)
    return base


# ──────────────────────────────────────────────────────────────
# DiagTree：故障码精确匹配
# ──────────────────────────────────────────────────────────────

def test_exact_match_hit():
    """故障码精确命中（大小写不敏感）。"""
    tree = _make_tree(_make_node())
    node = tree.exact_match("e001")
    assert node is not None
    assert node.node_id == "D001"
    assert node.fault_code == "E001"


def test_exact_match_miss():
    """未命中或入参为 None 时返回 None。"""
    tree = _make_tree(_make_node())
    assert tree.exact_match("E999") is None
    assert tree.exact_match(None) is None
    assert tree.exact_match("") is None


def test_exact_match_after_load_yaml(tmp_path):
    """load_yaml 从 YAML 加载节点并支持精确匹配。"""
    yaml_path = tmp_path / "tree.yaml"
    yaml_path.write_text(
        "[]\n",  # 空列表场景先行
        encoding="utf-8",
    )
    tree = DiagTree()
    tree.load_yaml(str(yaml_path))
    assert tree.get_tree()["nodes"] == []


# ──────────────────────────────────────────────────────────────
# DiagTree：现象模糊匹配（覆盖率）
# ──────────────────────────────────────────────────────────────

def test_fuzzy_match_coverage():
    """覆盖率 = 命中现象数 / 输入现象总数，按分数降序。"""
    node_a = _make_node(node_id="A", phenomena=["电机不转", "过载报警", "电机发热"])
    node_b = _make_node(node_id="B", phenomena=["电机不转"])
    tree = _make_tree(node_a, node_b)

    matches = tree.fuzzy_match(["电机不转", "过载报警"], threshold=0.5)
    assert len(matches) == 2
    # A 覆盖 2/2=1.0，B 覆盖 1/2=0.5 → A 排在前面
    assert [n.node_id for n, _ in matches] == ["A", "B"]
    assert matches[0][1] == pytest.approx(1.0)
    assert matches[1][1] == pytest.approx(0.5)


def test_fuzzy_match_threshold():
    """低于阈值的结果被剔除。"""
    node = _make_node(phenomena=["电机不转", "过载报警", "电机发热", "异响"])
    tree = _make_tree(node)
    # 输入 3 个现象只有 1 个命中 → 覆盖率 1/3 < 0.5
    matches = tree.fuzzy_match(["电机不转", "通讯中断", "COM丢失"], threshold=0.5)
    assert matches == []


def test_fuzzy_match_empty_input():
    """空现象列表返回空，不报错。"""
    tree = _make_tree(_make_node())
    assert tree.fuzzy_match([]) == []
    assert tree.fuzzy_match([""]) == []


# ──────────────────────────────────────────────────────────────
# nodes 纯函数：故障码 / 现象提取
# ──────────────────────────────────────────────────────────────

def test_extract_fault_code_prefers_arg():
    """入参故障码优先于文本正则提取，并转大写。"""
    assert nodes._extract_fault_code("设备报E001错误", "e-102") == "E-102"


def test_extract_fault_code_from_text():
    """无入参时从文本正则提取故障码。"""
    assert nodes._extract_fault_code("设备报错E001，电机不转", None) == "E001"


def test_extract_fault_code_none():
    """无法提取时返回 None。"""
    assert nodes._extract_fault_code("电机不转", None) is None
    assert nodes._extract_fault_code("", None) is None
    assert nodes._extract_fault_code(None, None) is None


def test_extract_phenomena_split_and_stopwords():
    """按标点切分并去除停用词。"""
    # "设备"、"出现"、"问题" 均为停用词 → 全部剔除
    assert nodes._extract_phenomena("设备，出现，问题") == []


def test_extract_phenomena_keep_fragments():
    """非停用词片段被保留（含汉字，长度不超过 12），口语词缀被剥离。"""
    words = nodes._extract_phenomena("电机不转了，出现过载报警")
    # 词缀剥离后返回主干：'电机不转'（"了"被剥离）；'出现过载报警' 保留
    assert "电机不转" in words
    assert "出现过载报警" in words


def test_extract_phenomena_filter_long():
    """超过 12 字的片段被过滤。"""
    long_text = "这个故障现象的描述非常详细而且特别长"
    assert nodes._extract_phenomena(long_text) == []


def test_extract_phenomena_empty():
    """空输入返回空列表。"""
    assert nodes._extract_phenomena("") == []
    assert nodes._extract_phenomena(None) == []


def test_extract_phenomena_atoms_survive_long_input():
    """回归：长输入的碎片段落不得把词典原子词挤出 6 个槽位。

    原子词（如 '主轴不转'）与诊断树节点现象词同源，若被碎片挤掉，
    模糊匹配轨将整体失效（只能落 LLM 兜底轨）。
    """
    long_input = (
        "今天早上车间开机之后操作人员反映这台机床的主轴不转了，"
        "而且过载报警一直响个不停，另外冷却液也时有时无，请帮忙看看"
    )
    words = nodes._extract_phenomena(long_input)
    assert "主轴不转" in words, f"原子词被碎片挤出: {words}"
    # 原子词应排在碎片段落之前
    assert words.index("主轴不转") < len(words) - 1 or len(words) == 1


# ──────────────────────────────────────────────────────────────
# nodes 纯函数：三轨匹配
# ──────────────────────────────────────────────────────────────

def test_run_exact_track_hit():
    """第一轨：故障码命中返回带置信度与节点数据的匹配结果。"""
    tree = _make_tree(_make_node())
    result = nodes._run_exact_track(tree, "E001")
    assert result is not None
    assert result["source"] == "exact"
    assert result["confidence"] == 0.95
    assert result["node_id"] == "D001"


def test_run_exact_track_miss():
    """第一轨：未命中返回 None。"""
    tree = _make_tree(_make_node())
    assert nodes._run_exact_track(tree, "E999") is None
    assert nodes._run_exact_track(tree, None) is None


def test_run_fuzzy_track_sorted():
    """第二轨：模糊匹配结果按置信度降序且标记 source=fuzzy。"""
    # 注意：不能使用通用词（如"异响"）做区分——它们会被 _GENERIC_PHENOMENA 过滤
    node_a = _make_node(node_id="A", phenomena=["电机不转", "过载报警", "主轴发热"])
    node_b = _make_node(node_id="B", phenomena=["电机不转", "过载报警"])
    tree = _make_tree(node_a, node_b)
    matches = nodes._run_fuzzy_track(tree, ["电机不转", "过载报警", "主轴发热"])
    # A 覆盖率 3/3=1.0，B 覆盖率 2/3≈0.67（均高于 FUZZY_THRESHOLD=0.5）
    assert matches[0]["node_id"] == "A"
    assert matches[1]["node_id"] == "B"
    assert matches[0]["source"] == "fuzzy"
    assert matches[0]["confidence"] == pytest.approx(1.0)
    # 置信度在 _run_fuzzy_track 中 round(s, 4)
    assert matches[1]["confidence"] == pytest.approx(0.6667)


# ──────────────────────────────────────────────────────────────
# nodes 纯函数：证据上下文组装
# ──────────────────────────────────────────────────────────────

def test_format_candidates_all_tracks():
    """三轨结果全部格式化进上下文。"""
    exact = {"name": "电机过载", "confidence": 0.95,
             "data": {"causes": [{"desc": "负载过大"}], "solutions": [{"step": "盘车"}]}}
    fuzzy = [{"name": "过载报警", "confidence": 0.8}]
    hyps = [{"desc": "轴承磨损", "confidence": 0.5}]
    text = nodes._format_candidates(exact, fuzzy, hyps)
    assert "[精确命中]" in text
    assert "[模糊命中]" in text
    assert "[LLM假设]" in text
    assert "负载过大" in text


def test_format_candidates_empty():
    """无任何命中时给出兜底提示。"""
    text = nodes._format_candidates(None, [], [])
    assert "无诊断树命中" in text


# ──────────────────────────────────────────────────────────────
# nodes 纯函数：追问选择
# ──────────────────────────────────────────────────────────────

def test_pick_clarify_question_first_unasked():
    """取节点预设问题中第一个未问过的。"""
    exact = {"data": {"questions": ["Q1", "Q2"]}}
    state = _make_state(exact_match=exact, clarify_question=None)
    assert nodes._pick_clarify_question(state) == "Q1"


def test_pick_clarify_question_skip_asked():
    """已问过的问题被跳过，返回下一个。"""
    exact = {"data": {"questions": ["Q1", "Q2"]}}
    state = _make_state(exact_match=exact, clarify_question="Q1")
    assert nodes._pick_clarify_question(state) == "Q2"


def test_pick_clarify_question_turn_limit():
    """追问轮次到达上限后不再生成追问。"""
    exact = {"data": {"questions": ["Q1"]}}
    state = _make_state(exact_match=exact, turn=nodes.MAX_CLARIFY_TURNS)
    assert nodes._pick_clarify_question(state) is None


def test_pick_clarify_question_no_node():
    """无命中节点时返回 None（转人工）。"""
    state = _make_state(exact_match=None, fuzzy_matches=[])
    assert nodes._pick_clarify_question(state) is None


# ──────────────────────────────────────────────────────────────
# nodes 纯函数：模板兜底
# ──────────────────────────────────────────────────────────────

def test_template_fallback_with_hit():
    """有命中节点时，兜底报告直接渲染节点 causes/solutions。"""
    exact = {"node_id": "D001", "name": "电机过载保护触发", "confidence": 0.95,
             "data": {"causes": [{"desc": "负载过大", "probability": "高"}],
                      "solutions": [{"step": "断电盘车", "need_skill": False}],
                      "need_ticket": False}}
    report = nodes._template_fallback(_make_state(exact_match=exact), has_tree_hit=True)
    assert report["_template"] is True
    assert "电机过载保护触发" in report["conclusion"]
    assert report["causes"][0]["desc"] == "负载过大"
    assert report["need_ticket"] is False


def test_template_fallback_no_hit():
    """无命中节点时兜底转人工。"""
    report = nodes._template_fallback(_make_state(exact_match=None, fuzzy_matches=[]), has_tree_hit=False)
    assert report["need_ticket"] is True
    assert report["confidence"] == 0.0
    assert report["ticket_reason"]


def test_template_fallback_need_ticket():
    """节点标记需上门时，兜底报告生成 ticket_reason。"""
    exact = {"node_id": "D002", "name": "变频器过流", "confidence": 0.95,
             "data": {"causes": [], "solutions": [], "need_ticket": True}}
    report = nodes._template_fallback(_make_state(exact_match=exact), has_tree_hit=True)
    assert report["need_ticket"] is True
    assert "上门" in report["ticket_reason"]


# ──────────────────────────────────────────────────────────────
# 异步节点：parse_input
# ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parse_input_node():
    """解析输入：提取故障码 + 现象，图片置空不报错。"""
    state = _make_state(user_input="设备报E001，电机不转了")
    result = await nodes.parse_input_node(state)
    assert result["fault_code"] == "E001"
    # 词缀剥离：'电机不转了' → '电机不转'
    assert "电机不转" in result["phenomena"]
    assert result["image_desc"] is None
    assert result["user_input"] == "设备报E001，电机不转了"


@pytest.mark.asyncio
async def test_parse_input_node_fallback_to_fault_description():
    """无 user_input 时回退到项目原有 fault_description 字段。"""
    state = _make_state(user_input="", fault_description="电机不转，过载报警")
    result = await nodes.parse_input_node(state)
    assert result["user_input"] == "电机不转，过载报警"
    assert "电机不转" in result["phenomena"]


@pytest.mark.asyncio
async def test_parse_input_node_with_image():
    """带图片时 image_desc 保持 None（视觉未接入不阻断流程）。"""
    state = _make_state(user_input="电机不转", image="base64:xxx")
    result = await nodes.parse_input_node(state)
    assert result["image_desc"] is None


# ──────────────────────────────────────────────────────────────
# 异步节点：load_diag_tree（知识库检索 + 降级）
# ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_load_diag_tree_node_kb_failure(monkeypatch):
    """知识库抛错时优雅降级：kb_hits 置空，流程不中断。"""
    async def _boom(*args, **kwargs):
        raise RuntimeError("milvus down")

    monkeypatch.setattr(nodes.KBClient, "search", _boom)
    result = await nodes.load_diag_tree_node(_make_state(user_input="电机不转", fault_code="E001"))
    assert result["kb_hits"] == []


@pytest.mark.asyncio
async def test_load_diag_tree_node_ok(monkeypatch):
    """知识库正常返回时透传检索结果。"""
    async def _ok(*args, **kwargs):
        return [{"content": "电机过载排查手册", "source": "kb", "score": 0.9}]

    monkeypatch.setattr(nodes.KBClient, "search", _ok)
    result = await nodes.load_diag_tree_node(_make_state(user_input="电机不转", fault_code="E001"))
    assert len(result["kb_hits"]) == 1
    assert result["kb_hits"][0]["content"] == "电机过载排查手册"


@pytest.mark.asyncio
async def test_load_diag_tree_node_empty_query_guard(monkeypatch):
    """user_input/fault_code 全空时 query 不落空（防嵌入器空串报错）。"""
    captured = {}

    async def _capture(query, device_model=None, top_k=10):
        captured["query"] = query
        captured["device_model"] = device_model
        return []

    monkeypatch.setattr(nodes.KBClient, "search", _capture)
    await nodes.load_diag_tree_node(_make_state(user_input="", fault_code=None, device_model="CNC-1000"))
    assert captured["query"] == "CNC-1000"
    assert captured["device_model"] == "CNC-1000"


# ──────────────────────────────────────────────────────────────
# KBClient：知识库适配层（对接 ③号 模块级 hybrid_retrieve）
# ──────────────────────────────────────────────────────────────

def _make_kb_doc(content: str = "手册", source: str = "kb", score: float = 0.9) -> dict:
    return {"content": content, "source": source, "score": score,
            "metadata": {"source_name": source}}


@pytest.mark.asyncio
async def test_kb_client_search_delegates_to_module_hybrid_retrieve(monkeypatch):
    """KBClient.search 委托给 retriever.py 模块级 hybrid_retrieve（而非类实例方法）。"""
    captured = {}

    async def _fake_hybrid_retrieve(query, device_model=None, top_k=10):
        captured.update(query=query, device_model=device_model, top_k=top_k)
        return [_make_kb_doc()]

    monkeypatch.setattr("backend.knowledge_base.retriever.hybrid_retrieve", _fake_hybrid_retrieve)
    hits = await KBClient.search(query="E001 电机不转", device_model="CNC-1000", top_k=7)

    assert captured == {"query": "E001 电机不转", "device_model": "CNC-1000", "top_k": 7}
    assert hits[0]["content"] == "手册"
    assert hits[0]["metadata"]["source_name"] == "kb"


def _fake_reranker_module(monkeypatch, cls) -> None:
    """在 sys.modules 注入假的 backend.knowledge_base.reranker 模块。

    避免触发真实模块导入（reranker.py 模块级 get_settings() 需要 .env.local，
    且导入 torch / sentence_transformers 很慢），仅验证 KBClient 的委托契约。
    """
    fake_mod = types.ModuleType("backend.knowledge_base.reranker")
    fake_mod.BGEReranker = cls
    monkeypatch.setitem(sys.modules, "backend.knowledge_base.reranker", fake_mod)


@pytest.mark.asyncio
async def test_kb_client_rerank_success(monkeypatch):
    """rerank 走 BGEReranker.rerank_with_confidence，返回与 search 一致的 dict 形态。"""
    docs = [_make_kb_doc(content="A", score=0.1), _make_kb_doc(content="B", score=0.9)]

    class _FakeReranker:
        @staticmethod
        def get_instance():
            return _FakeReranker()

        def rerank_with_confidence(self, query, documents, top_k):
            # 契约对齐 BGEReranker：返回 RankedDocument 对象（.content / .score / .metadata）
            ranked = sorted(
                [types.SimpleNamespace(
                    content=d["content"], score=d["score"],
                    metadata={"source_name": d["source"]}) for d in documents],
                key=lambda r: r.score,
                reverse=True,
            )
            return ranked[:top_k], ranked[0].score

    _fake_reranker_module(monkeypatch, _FakeReranker)
    result = await KBClient.rerank(docs, query="测试", top_k=1)

    assert len(result) == 1
    assert result[0]["content"] == "B"
    assert result[0]["score"] == 0.9
    assert result[0]["source"] == "kb"


@pytest.mark.asyncio
async def test_kb_client_rerank_fallback(monkeypatch):
    """rerank 模型未就绪时降级为原序截断，不抛错。"""
    class _Boom:
        @staticmethod
        def get_instance():
            raise RuntimeError("reranker model unavailable")

    _fake_reranker_module(monkeypatch, _Boom)
    docs = [_make_kb_doc(content="A"), _make_kb_doc(content="B"), _make_kb_doc(content="C")]
    result = await KBClient.rerank(docs, query="测试", top_k=2)

    assert [d["content"] for d in result] == ["A", "B"]


# ──────────────────────────────────────────────────────────────
# 异步节点：run_diag_tracks（三轨并行容错）
# ──────────────────────────────────────────────────────────────

class _FakeLLM:
    """最小可用的 LLM 替身，可配置返回内容或抛错。"""

    def __init__(self, content: str = "", error: Exception | None = None):
        self._content = content
        self._error = error
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        if self._error:
            raise self._error
        return AIMessage(content=self._content)


@pytest.mark.asyncio
async def test_run_diag_tracks_node_llm_failure(monkeypatch):
    """LLM 轨失败不影响精确/模糊轨（gather 容错模式）。"""
    tree = _make_tree(_make_node())
    monkeypatch.setattr(nodes, "_diag_tree", tree)
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: _FakeLLM(error=RuntimeError("llm down")))

    state = _make_state(user_input="设备报E001，电机不转", fault_code="E001", phenomena=["电机不转", "过载报警"])
    result = await nodes.run_diag_tracks_node(state)

    assert result["exact_match"] is not None
    assert result["exact_match"]["node_id"] == "D001"
    assert len(result["fuzzy_matches"]) >= 1
    assert result["llm_hypotheses"] == []  # 降级为空列表


@pytest.mark.asyncio
async def test_run_diag_tracks_node_all_ok(monkeypatch):
    """三轨全通：LLM 轨解析 JSON 假设。"""
    tree = _make_tree(_make_node())
    monkeypatch.setattr(nodes, "_diag_tree", tree)
    llm = _FakeLLM(content='{"hypotheses": [{"desc": "轴承磨损", "confidence": 0.6}]}')
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm)

    state = _make_state(user_input="电机不转，异响", phenomena=["电机不转", "异响"])
    result = await nodes.run_diag_tracks_node(state)

    assert result["llm_hypotheses"][0]["desc"] == "轴承磨损"
    assert result["llm_hypotheses"][0]["confidence"] == pytest.approx(0.6)


# ──────────────────────────────────────────────────────────────
# 异步节点：generate_report（LLM 失败 → 模板兜底）
# ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_report_node_llm_invalid_json(monkeypatch):
    """LLM 返回非法 JSON 时走模板兜底，不中断流程。"""
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: _FakeLLM(content="这不是 JSON"))

    exact = {"node_id": "D001", "name": "电机过载保护触发", "confidence": 0.95,
             "data": {"causes": [{"desc": "负载过大", "probability": "高"}],
                      "solutions": [{"step": "断电盘车", "need_skill": False}],
                      "need_ticket": False}}
    state = _make_state(
        user_input="电机不转",
        fault_code="E001",
        phenomena=["电机不转"],
        exact_match=exact,
        fuzzy_matches=[],
        llm_hypotheses=[],
        kb_hits=[],
        evidence_context="",
    )
    result = await nodes.generate_report_node(state)

    assert result["fallback_used"] is True
    assert result["report"]["_template"] is True
    assert "电机过载保护触发" in result["report"]["conclusion"]


@pytest.mark.asyncio
async def test_generate_report_node_llm_valid(monkeypatch):
    """LLM 返回合法 JSON 时通过 Pydantic 校验并生成报告。"""
    valid_json = json_report({"conclusion": "电机过载保护触发", "need_ticket": False, "confidence": 0.9})
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: _FakeLLM(content=valid_json))

    exact = {"node_id": "D001", "name": "电机过载保护触发", "confidence": 0.95,
             "data": {"causes": [], "solutions": [], "need_ticket": False}}
    state = _make_state(
        user_input="电机不转", fault_code="E001", phenomena=["电机不转"],
        exact_match=exact, fuzzy_matches=[], llm_hypotheses=[], kb_hits=[],
        evidence_context="",
    )
    result = await nodes.generate_report_node(state)

    assert result["report"]["conclusion"] == "电机过载保护触发"
    # 树命中时置信度不允许超过 0.95
    assert result["report"]["confidence"] == pytest.approx(0.9)
    # LLM 成功路径不带 _template 标记，也不算降级
    assert result["report"].get("_template") is not True
    assert result["fallback_used"] is False


@pytest.mark.asyncio
async def test_generate_report_node_no_tree_hit_valid_llm_not_fallback(monkeypatch):
    """无树命中但 LLM 正常出报告：是纯 LLM 推理而非降级（fallback_used=False）。"""
    valid_json = json_report({"conclusion": "推断结论", "need_ticket": False, "confidence": 0.3})
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: _FakeLLM(content=valid_json))

    state = _make_state(
        user_input="电机不转", fault_code=None, phenomena=["电机不转"],
        exact_match=None, fuzzy_matches=[], llm_hypotheses=[{"desc": "x", "confidence": 0.3}],
        kb_hits=[], evidence_context="",
    )
    result = await nodes.generate_report_node(state)

    assert result["fallback_used"] is False
    assert result["report"].get("_template") is not True
    # 无树命中 → 置信度被压到 LLM_HYP_CONFIDENCE 以内
    assert result["report"]["confidence"] <= nodes.LLM_HYP_CONFIDENCE


# ──────────────────────────────────────────────────────────────
# 异步节点：check_sufficiency（是否追问）
# ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_sufficiency_high_confidence_no_clarify(monkeypatch):
    """置信度足够时直接出报告，不追问。"""
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: _FakeLLM())
    state = _make_state(report={"confidence": 0.9, "need_ticket": False})
    result = await nodes.check_sufficiency_node(state)
    assert result["clarify_question"] is None


@pytest.mark.asyncio
async def test_check_sufficiency_low_confidence_uses_node_question(monkeypatch):
    """置信度不足且树命中时，优先用节点预设问题追问。"""
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: _FakeLLM())
    exact = {"data": {"questions": ["故障时设备是否处于满载状态？"]}}
    state = _make_state(
        report={"confidence": 0.3, "need_ticket": False},
        exact_match=exact,
        fuzzy_matches=[],
        turn=0,
    )
    result = await nodes.check_sufficiency_node(state)
    assert result["clarify_question"] == "故障时设备是否处于满载状态？"


# ──────────────────────────────────────────────────────────────
# 异步节点：apply_clarify_answer（合并用户回答，重新诊断）
# ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_apply_clarify_answer_node():
    """合并用户回答：追加描述、更新故障码、turn+1、清空旧诊断结果。"""
    state = _make_state(
        user_input="电机不转",
        fault_code="E001",
        clarify_answer={"answer": "满载运行时报的", "fault_code": "E099"},
    )
    result = await nodes.apply_clarify_answer_node(state)

    assert result["user_input"] == "电机不转 满载运行时报的"
    assert result["fault_code"] == "E099"
    assert result["turn"] == 1
    assert result["clarify_answer"] is None
    assert result["report"] is None
    assert result["exact_match"] is None


@pytest.mark.asyncio
async def test_apply_clarify_answer_node_keep_fault_code():
    """用户未补充故障码时保留原故障码。"""
    state = _make_state(
        user_input="电机不转",
        fault_code="E001",
        clarify_answer={"answer": "是的"},
    )
    result = await nodes.apply_clarify_answer_node(state)
    assert result["fault_code"] == "E001"


# ──────────────────────────────────────────────────────────────
# 异步节点：route_next（结果路由）
# ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_route_next_node_need_ticket():
    """报告标记 need_ticket=True 时路由到工单流程。"""
    report = {"need_ticket": True, "conclusion": "需上门", "confidence": 0.3}
    result = await nodes.route_next_node(_make_state(report=report))
    assert result["need_ticket"] is True
    assert result["finished"] is True
    assert result["structured_output"] == report


@pytest.mark.asyncio
async def test_route_next_node_finished():
    """报告无需工单时正常结束。"""
    result = await nodes.route_next_node(_make_state(report={"need_ticket": False}))
    assert result["need_ticket"] is False
    assert result["finished"] is True


# ──────────────────────────────────────────────────────────────
# graph：条件路由
# ──────────────────────────────────────────────────────────────

def test_route_after_check_ask_clarify():
    """有追问问题 → 走 ask_clarify 分支。"""
    assert _route_after_check({"clarify_question": "电机外壳是否发烫？"}) == "ask_clarify"


def test_route_after_check_route_next():
    """无追问问题 → 直接结束路由。"""
    assert _route_after_check({"clarify_question": None}) == "route_next"
    assert _route_after_check({}) == "route_next"


def test_build_diagnosis_graph():
    """构建诊断图不报错，且包含关键节点。"""
    graph = __import__("backend.agents.diagnosis.graph", fromlist=["build_diagnosis_graph"]).build_diagnosis_graph()
    assert graph is not None


# ──────────────────────────────────────────────────────────────
# state：Pydantic 模型校验
# ──────────────────────────────────────────────────────────────

def test_diagnosis_result_model():
    """DiagnosisResult / DiagnosisStep 模型可用。"""
    result = DiagnosisResult(
        possible_causes=["负载过大"],
        steps=[DiagnosisStep(step_no=1, action="断电盘车", expected_result="电机可转动")],
        required_tools=["万用表"],
        severity="medium",
        recommendation="自行处理",
    )
    assert result.steps[0].step_no == 1
    assert result.required_tools == ["万用表"]


def test_diagnosis_report_model_validation():
    """DiagnosisReport 拒绝非法字段（如缺失必填项）。"""
    with pytest.raises(Exception):
        DiagnosisReport(conclusion="x")  # 缺 causes / solutions / need_ticket / confidence


# ──────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────

def json_report(overrides: dict) -> str:
    """构造一段合法的诊断报告 JSON（模拟 LLM 输出）。"""
    base = {
        "conclusion": "诊断结论",
        "causes": [{"desc": "负载过大", "probability": "高"}],
        "solutions": [{"step": "断电盘车", "need_skill": False}],
        "need_ticket": False,
        "confidence": 0.8,
        "ticket_reason": "",
    }
    base.update(overrides)
    return json.dumps(base, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────
# 直接运行入口：python test_diagnosis.py
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
