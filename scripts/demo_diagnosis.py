"""
故障诊断 Agent（Agent③）演示脚本 —— 离线运行，无需真实 LLM / 知识库 / 网络。

第一部分：逐函数效果演示（诊断树匹配、输入解析、三轨匹配、上下文组装、追问、模板兜底、各节点）
第二部分：完整端到端流程（LangGraph 图 invoke，含「追问 interrupt → 用户回答 → 重新诊断」）

运行方式：
    python scripts/demo_diagnosis.py
或在 PyCharm 中直接右键 Run。
"""
import asyncio
import json
import sys
from pathlib import Path

# Windows 终端默认 GBK，强制用 UTF-8 输出避免中文乱码
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 保证从任意 cwd 都能找到 backend 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage
from langgraph.types import Command

from backend.agents.diagnosis import nodes
from backend.agents.diagnosis import prompts as diag_prompts
from backend.agents.diagnosis.diag_tree import DiagTree
from backend.agents.diagnosis.graph import build_diagnosis_graph

SEP = "=" * 72


# ──────────────────────────────────────────────────────────────
# 假 LLM：按 prompt 关键字返回对应内容，可配置行为
# ──────────────────────────────────────────────────────────────

class FakeLLM:
    """离线替身：识别三类 prompt（推理假设 / 诊断报告 / 追问问题）。"""

    def __init__(self, first_report_confidence: float = 0.9):
        self.first_report_confidence = first_report_confidence
        self.calls = 0
        self.json_calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        text = messages[-1].content
        if "hypotheses" in text:
            return AIMessage(content=json.dumps({
                "hypotheses": [
                    {"desc": "负载过大或机械卡死", "confidence": 0.6},
                    {"desc": "供电电压波动", "confidence": 0.3},
                ]
            }, ensure_ascii=False))
        if "JSON" in text:
            self.json_calls += 1
            # 第一次出报告给低置信度（触发追问）；之后给高置信度（直接出报告）
            conf = self.first_report_confidence if self.json_calls == 1 else 0.9
            return AIMessage(content=json.dumps({
                "conclusion": "电机过载保护触发，建议优先排查负载与供电",
                "causes": [
                    {"desc": "负载过大或机械卡死", "probability": "高"},
                    {"desc": "供电电压过低", "probability": "中"},
                ],
                "solutions": [
                    {"step": "断电后手动盘车确认是否卡死", "need_skill": False},
                    {"step": "万用表检测三相供电电压", "need_skill": True},
                ],
                "need_ticket": False,
                "confidence": conf,
                "ticket_reason": "",
            }, ensure_ascii=False))
        if "追问" in text:
            return AIMessage(content="故障发生时设备是否处于满载状态？")
        return AIMessage(content="（推理分析：故障现象与诊断树 D001 节点吻合，根因大概率是负载过大。）")


def _fake_llm(fake: FakeLLM):
    """让 nodes 模块使用假 LLM。"""
    nodes.get_llm = lambda *a, **k: fake


def _fake_kb(hits: list[dict] | None = None):
    """让 nodes 模块使用假知识库检索。"""
    async def _search(*a, **k):
        return hits if hits is not None else [
            {"content": "电机过载排查手册：先检查负载，再测三相电压……", "source": "kb", "score": 0.9}
        ]
    nodes.KBClient.search = _search


def _base_state(**kw) -> dict:
    """构造完整 DiagnosisState 的 dict 形态。"""
    s = {
        "messages": [],
        "session_id": "demo-session",
        "device_model": "CNC-1000",
        "fault_description": "",
        "collected_symptoms": [],
        "current_step": 0,
        "diagnosis_result": None,
        "resolved": False,
        "user_input": "",
        "fault_code": None,
        "image": None,
        "image_desc": None,
        "phenomena": [],
        "kb_hits": [],
        "exact_match": None,
        "fuzzy_matches": [],
        "llm_hypotheses": [],
        "evidence_context": "",
        "report": None,
        "confidence": 0.0,
        "clarify_question": None,
        "clarify_answer": None,
        "turn": 0,
        "need_ticket": False,
        "finished": False,
        "fallback_used": False,
        "structured_output": None,
    }
    s.update(kw)
    return s


# ──────────────────────────────────────────────────────────────
# 第一部分：逐函数效果
# ──────────────────────────────────────────────────────────────

def demo_functions():
    print(SEP)
    print("第一部分：逐函数效果演示")
    print(SEP)

    # ── 0. Prompt 模板：4 个模板格式化效果 ──────────────────
    print("\n【0】Prompt 模板")
    print("── SYSTEM_PROMPT（人设前缀，无占位符）──")
    print(diag_prompts.SYSTEM_PROMPT)
    print("\n── REASON_PROMPT（格式化示例）──")
    print(diag_prompts.REASON_PROMPT.format(
        user_input="电机不转了，出现过载报警",
        fault_code="E001",
        phenomena="电机不转、过载报警",
        matched_nodes="[精确命中] 电机过载保护触发 (置信度 0.95)",
        kb_hits="电机过载排查手册：先检查负载，再测三相电压……",
    ))
    print("\n── DIAGNOSIS_REPORT_PROMPT（格式化示例）──")
    print(diag_prompts.DIAGNOSIS_REPORT_PROMPT.format(
        user_input="电机不转了，出现过载报警",
        fault_code="E001",
        phenomena="电机不转、过载报警",
        matched_nodes="[精确命中] 电机过载保护触发 (置信度 0.95)",
        kb_hits="电机过载排查手册：先检查负载，再测三相电压……",
        reasoning_trace="故障现象与 D001 节点吻合，根因大概率是负载过大。",
    ))
    print("\n── CLARIFY_PROMPT（格式化示例）──")
    print(diag_prompts.CLARIFY_PROMPT.format(
        user_input="电机不转了",
        phenomena="电机不转",
        candidates="[精确命中] 电机过载保护触发 (置信度 0.95)",
    ))

    # ── 1. 诊断树：加载 / 精确匹配 / 模糊匹配 ────────────────
    print("\n【1】诊断树 DiagTree")
    tree = DiagTree()
    tree.load_yaml(str(Path(__file__).resolve().parents[1] / "data/diag_tree_full.yaml"))
    print(f"  YAML 加载节点数: {len(tree.get_tree()['nodes'])}")
    node = tree.exact_match("e001")
    print(f"  exact_match('e001') → {node.node_id} {node.name} (大小写不敏感)")
    print(f"  exact_match('E999') → {tree.exact_match('E999')}")
    fuzzy = tree.fuzzy_match(["电机不转", "过载报警"], threshold=0.5)
    print(f"  fuzzy_match(['电机不转','过载报警'], 0.5) → {[(n.node_id, s) for n, s in fuzzy]}")

    # ── 2. 输入解析：故障码 + 现象提取 ──────────────────────
    print("\n【2】输入解析（纯函数）")
    print(f"  _extract_fault_code('设备报E001，电机不转', None)"
          f" → {nodes._extract_fault_code('设备报E001，电机不转', None)}")
    print(f"  _extract_fault_code('设备报错', 'e-102')"
          f" → {nodes._extract_fault_code('设备报错', 'e-102')} (入参优先+转大写)")
    print(f"  _extract_fault_code('电机不转', None)               → {nodes._extract_fault_code('电机不转', None)}")
    print(f"  _extract_phenomena('设备，出现，问题')"
          f" → {nodes._extract_phenomena('设备，出现，问题')} (全是停用词)")
    print(f"  _extract_phenomena('电机不转了，出现过载报警')"
          f" → {nodes._extract_phenomena('电机不转了，出现过载报警')}")

    # ── 3. 三轨匹配 ─────────────────────────────────────────
    print("\n【3】三轨匹配（run_diag_tracks_node，含 LLM 轨容错）")
    exact = nodes._run_exact_track(tree, "E001")
    print(f"  轨1 _run_exact_track(tree,'E001') → source={exact['source']}"
          f" node={exact['node_id']} conf={exact['confidence']}")
    print(f"  轨1 _run_exact_track(tree,'E999') → {nodes._run_exact_track(tree, 'E999')}")
    fuzzy_res = nodes._run_fuzzy_track(tree, ["电机不转", "过载报警"])
    print(f"  轨2 _run_fuzzy_track → {[(m['node_id'], m['confidence']) for m in fuzzy_res]}")

    async def _demo_llm_ok():
        _fake_llm(FakeLLM())
        _fake_kb()
        return await nodes.run_diag_tracks_node(_base_state(
            user_input="电机不转，过载报警", phenomena=["电机不转", "过载报警"]))
    res_ok = asyncio.run(_demo_llm_ok())
    print(f"  轨3 LLM 正常 → llm_hypotheses={res_ok['llm_hypotheses']}")

    async def _demo_llm_fail():
        class _Boom:
            async def ainvoke(self, msgs):
                raise RuntimeError("llm down")
        _fake_llm(_Boom())
        return await nodes.run_diag_tracks_node(_base_state(
            user_input="电机不转", phenomena=["电机不转"]))
    res_fail = asyncio.run(_demo_llm_fail())
    print(f"  轨3 LLM 抛错 → llm_hypotheses={res_fail['llm_hypotheses']} (降级为空，不中断)")

    # ── 4. 上下文组装 ───────────────────────────────────────
    print("\n【4】上下文组装（assemble_context_node）")
    async def _demo_ctx():
        _fake_llm(FakeLLM())
        _fake_kb()
        state = _base_state(
            user_input="设备报E001，电机不转", fault_code="E001", phenomena=["电机不转", "过载报警"])
        s1 = await nodes.parse_input_node(state)
        state.update(s1)
        s2 = await nodes.load_diag_tree_node(state)
        state.update(s2)
        s3 = await nodes.run_diag_tracks_node(state)
        state.update(s3)
        s4 = await nodes.assemble_context_node(state)
        return s4["evidence_context"]
    ctx = asyncio.run(_demo_ctx())
    print("  evidence_context 示例：")
    for line in ctx.splitlines()[:6]:
        print(f"    | {line}")

    # ── 5. 追问选择 ─────────────────────────────────────────
    print("\n【5】追问选择（_pick_clarify_question）")
    exact = {"data": {"questions": ["故障时满载吗？", "电机发烫吗？"]}}
    state_q = _base_state(exact_match=exact, clarify_question=None)
    print(f"  未问过 → {nodes._pick_clarify_question(state_q)}")
    state_q2 = _base_state(exact_match=exact, clarify_question="故障时满载吗？")
    print(f"  已问过 Q1 → {nodes._pick_clarify_question(state_q2)}")
    state_q3 = _base_state(exact_match=exact, turn=nodes.MAX_CLARIFY_TURNS)
    print(f"  turn 达上限 → {nodes._pick_clarify_question(state_q3)} (返回 None → 转人工)")
    state_q4 = _base_state(exact_match=None, fuzzy_matches=[])
    print(f"  无命中节点 → {nodes._pick_clarify_question(state_q4)}")

    # ── 6. 模板兜底 ─────────────────────────────────────────
    print("\n【6】模板兜底（_template_fallback，LLM 失败时的降级）")
    exact = {"node_id": "D001", "name": "电机过载保护触发", "confidence": 0.95,
             "data": {"causes": [{"desc": "负载过大", "probability": "高"}],
                      "solutions": [{"step": "断电盘车", "need_skill": False}],
                      "need_ticket": False}}
    fb = nodes._template_fallback(_base_state(exact_match=exact), has_tree_hit=True)
    print(f"  树命中 → conclusion={fb['conclusion']!r},"
          f" need_ticket={fb['need_ticket']}, _template={fb.get('_template')}")
    fb2 = nodes._template_fallback(_base_state(exact_match=None, fuzzy_matches=[]), has_tree_hit=False)
    print(f"  无命中 → need_ticket={fb2['need_ticket']},"
          f" confidence={fb2['confidence']}, reason={fb2['ticket_reason']!r}")

    # ── 7. 各节点单独调用（parse / load_diag_tree 降级 / route） ──
    print("\n【7】节点单独效果")
    async def _demo_nodes():
        _fake_llm(FakeLLM())
        _fake_kb()
        p = await nodes.parse_input_node(_base_state(user_input="设备报E001，电机不转了"))
        print(f"  parse_input_node → fault_code={p['fault_code']}, phenomena={p['phenomena']}")

        # 知识库降级
        async def _boom(*a, **k):
            raise RuntimeError("milvus down")
        nodes.KBClient.search = _boom
        kb_fail = await nodes.load_diag_tree_node(_base_state(user_input="电机不转", fault_code="E001"))
        print(f"  load_diag_tree_node (KB 抛错) → kb_hits={kb_fail['kb_hits']} (降级为空)")

        r = await nodes.route_next_node(_base_state(report={"need_ticket": True, "conclusion": "需上门"}))
        print(f"  route_next_node (need_ticket=True) → finished={r['finished']}, need_ticket={r['need_ticket']}")
    asyncio.run(_demo_nodes())


# ──────────────────────────────────────────────────────────────
# 第二部分：完整端到端流程
# ──────────────────────────────────────────────────────────────

async def run_scenario(name: str, user_input: str, resume: dict | None, first_conf: float):
    """跑一个完整诊断流程场景。resume 为追问后的用户回答（None 表示不追问）。"""
    print("\n" + SEP)
    print(f"【场景】{name}")
    print(f"  用户输入: {user_input!r}")
    print(SEP)

    _fake_llm(FakeLLM(first_report_confidence=first_conf))
    _fake_kb()
    graph = build_diagnosis_graph()
    config = {"configurable": {"thread_id": f"demo-{name}"}}

    state = _base_state(user_input=user_input)
    result = await graph.ainvoke(state, config)

    # 检查是否暂停在追问点
    interrupts = result.get("__interrupt__", [])
    if interrupts:
        payload = interrupts[0].value
        print("\n  >> 图在追问点暂停（interrupt）:")
        print(f"      问题: {payload['question']}")
        print(f"      附带证据: {payload['evidence'][:60]}...")
        print(f"  >> 用户回答: {resume!r}")
        result = await graph.ainvoke(Command(resume=resume), config)

    print("\n── 最终结果 ──")
    print(f"  need_ticket : {result.get('need_ticket')}")
    print(f"  finished    : {result.get('finished')}")
    report = result.get("report") or {}
    print(f"  结论        : {report.get('conclusion')}")
    print(f"  置信度      : {report.get('confidence')}")
    for c in report.get("causes", []):
        print(f"    - 可能原因: {c['desc']} ({c['probability']})")
    for s in report.get("solutions", []):
        skill = "需专业人员" if s["need_skill"] else "可自行处理"
        print(f"    - 处理方案: {s['step']} [{skill}]")
    print(f"  追问轮数    : {result.get('turn')}")
    return result


async def demo_full_flow():
    print("\n" + SEP)
    print("第二部分：完整端到端流程（LangGraph 图）")
    print(SEP)

    # 场景 A：带故障码 → 第一轨精确命中 → 置信度高 → 直接出报告，不追问
    await run_scenario(
        name="A",
        user_input="设备报E001，电机不转了",
        resume=None,
        first_conf=0.9,
    )

    # 场景 B：只有现象、无故障码 → 第一轮置信度低 → 追问 → 用户补充 → 重新诊断
    await run_scenario(
        name="B",
        user_input="电机不转了，出现过载报警",
        resume={"answer": "满载运行时报的，面板显示E001", "fault_code": "E001"},
        first_conf=0.3,
    )


if __name__ == "__main__":
    demo_functions()
    asyncio.run(demo_full_flow())
    print("\n演示结束。")
