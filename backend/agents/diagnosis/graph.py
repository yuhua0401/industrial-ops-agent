"""
graph - 故障诊断 Agent（Agent③）的 LangGraph 图定义

迁移自 EduAgent 课件 06-试卷批改 Agent（exam/graph.py）的
「线性链 + HitL interrupt + MemorySaver」模式，并保留项目原有
build_diagnosis_graph() 入口（由 Supervisor/④号 集成调用）。

图拓扑：
    START → parse_input → load_diag_tree → run_diag_tracks → assemble_context
          → generate_report → check_sufficiency
                                    ├─(追问)→ ask_clarify [interrupt]
                                    │            → apply_clarify_answer → run_diag_tracks（循环）
                                    └─(结束)→ route_next → END

注意：有 checkpointer（MemorySaver），追问循环的暂停/恢复依赖跨请求状态持久化。
"""
# 直接运行（python graph.py）时项目根不在 sys.path，这里补上；作为包导入时无副作用。
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from backend.agents.diagnosis.state import DiagnosisState
from backend.agents.diagnosis.nodes import (
    parse_input_node,
    load_diag_tree_node,
    run_diag_tracks_node,
    assemble_context_node,
    generate_report_node,
    check_sufficiency_node,
    ask_clarify_node,
    apply_clarify_answer_node,
    route_next_node,
)


def _route_after_check(state: DiagnosisState) -> str:
    """
    check_sufficiency 后的条件路由：
        - 有 clarify_question → 追问路径（interrupt 等待用户回答）
        - 否则 → 直接路由结束
    """
    if state.get("clarify_question"):
        return "ask_clarify"
    return "route_next"


def build_diagnosis_graph():
    """构建并编译故障诊断 Agent 的 LangGraph 状态图。"""
    builder = StateGraph(DiagnosisState)

    # ── 注册节点 ──────────────────────────────────────────────
    builder.add_node("parse_input",          parse_input_node)
    builder.add_node("load_diag_tree",       load_diag_tree_node)
    builder.add_node("run_diag_tracks",      run_diag_tracks_node)
    builder.add_node("assemble_context",     assemble_context_node)
    builder.add_node("generate_report",      generate_report_node)
    builder.add_node("check_sufficiency",    check_sufficiency_node)
    builder.add_node("ask_clarify",          ask_clarify_node)
    builder.add_node("apply_clarify_answer", apply_clarify_answer_node)
    builder.add_node("route_next",           route_next_node)

    # ── 主链（线性）──────────────────────────────────────────
    builder.add_edge(START,                "parse_input")
    builder.add_edge("parse_input",        "load_diag_tree")
    builder.add_edge("load_diag_tree",     "run_diag_tracks")
    builder.add_edge("run_diag_tracks",    "assemble_context")
    builder.add_edge("assemble_context",   "generate_report")
    builder.add_edge("generate_report",    "check_sufficiency")

    # ── 条件分支：追问 vs 结束 ──────────────────────────────
    builder.add_conditional_edges(
        "check_sufficiency",
        _route_after_check,
        {"ask_clarify": "ask_clarify", "route_next": "route_next"},
    )

    # ── 追问循环：合并用户回答后重新诊断 ─────────────────────
    builder.add_edge("ask_clarify",           "apply_clarify_answer")
    builder.add_edge("apply_clarify_answer",  "run_diag_tracks")

    builder.add_edge("route_next", END)

    # ── 编译，绑定模块级单例 MemorySaver（追问暂停/恢复需要）────────
    # 注意：不能在函数内新建 MemorySaver——每次 build 都产生空 checkpoint 存储，
    # interrupt 暂停后下一次请求无法 resume。必须复用进程级单例；
    # 生产环境建议切换为 AsyncPostgresSaver（跨进程/重启持久化），见 requirements。
    checkpointer = _get_checkpointer()
    return builder.compile(checkpointer=checkpointer)


# 进程级 checkpointer 单例：保证同一进程内所有 build_diagnosis_graph() 复用同一
# checkpoint 存储，追问循环的暂停/恢复跨请求成立。
# 通过 .env.local 的 CHECKPOINTER_BACKEND 切换：
#   memory（默认，进程内）| postgres（AsyncPostgresSaver，跨进程/重启持久化）
_checkpointer = None


def _get_checkpointer():
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    from backend.config import get_settings
    if get_settings().checkpointer_backend.lower() == "postgres":
        try:
            import asyncio

            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            saver = AsyncPostgresSaver.from_conn_string(_pg_conn_string())
            # AsyncPostgresSaver 需异步 setup 建表；构建图发生在同步上下文，
            # 用一次性事件循环完成初始化
            asyncio.new_event_loop().run_until_complete(saver.setup())
            _checkpointer = saver
            return _checkpointer
        except Exception as e:
            # PG 不可用/依赖缺失时降级 memory 并告警，不阻断图构建
            from backend.core.logger import get_logger
            get_logger(__name__).warning(
                "diagnosis.postgres_checkpointer_failed_fallback_memory", error=str(e))

    _checkpointer = MemorySaver()
    return _checkpointer


def _pg_conn_string() -> str:
    """从配置拼 PostgreSQL 连接串（与 settings.database_url 同源字段）。"""
    from backend.config import get_settings
    s = get_settings()
    return f"postgresql://{s.db_user}:{s.db_password}@{s.db_host}:{s.db_port}/{s.db_name}"


# ──────────────────────────────────────────────────────────────
# 直接运行演示：python graph.py（注入假 LLM / 假知识库，离线可跑）
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    import json
    import sys
    from pathlib import Path

    from langchain_core.messages import AIMessage
    from langgraph.types import Command

    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from backend.agents.diagnosis import nodes  # noqa: E402

    class _FakeLLM:
        """离线替身：第一次 JSON 报告给 first_conf（模拟低置信度触发追问），之后给 0.9。"""

        def __init__(self, first_conf: float = 0.9):
            self.first_conf = first_conf
            self.json_calls = 0

        async def ainvoke(self, messages):
            text = messages[-1].content
            if "hypotheses" in text:
                return AIMessage(content=json.dumps({
                    "hypotheses": [{"desc": "负载过大或机械卡死", "confidence": 0.6}]
                }, ensure_ascii=False))
            if "JSON" in text:
                self.json_calls += 1
                conf = self.first_conf if self.json_calls == 1 else 0.9
                return AIMessage(content=json.dumps({
                    "conclusion": "电机过载保护触发，建议优先排查负载与供电",
                    "causes": [{"desc": "负载过大或机械卡死", "probability": "高"}],
                    "solutions": [{"step": "断电后手动盘车确认是否卡死", "need_skill": False}],
                    "need_ticket": False,
                    "confidence": conf,
                    "ticket_reason": "",
                }, ensure_ascii=False))
            if "追问" in text:
                return AIMessage(content="故障发生时设备是否处于满载状态？")
            return AIMessage(content="故障现象与诊断树 D001 节点吻合，根因大概率是负载过大。")

    async def _fake_search(*args, **kwargs):
        return [{"content": "电机过载排查手册：先检查负载，再测三相电压……", "source": "kb", "score": 0.9}]

    def _base_state(user_input: str) -> dict:
        return {
            "messages": [], "session_id": "demo", "device_model": "CNC-1000",
            "fault_description": "", "collected_symptoms": [], "current_step": 0,
            "diagnosis_result": None, "resolved": False,
            "user_input": user_input, "fault_code": None, "image": None,
            "image_desc": None, "phenomena": [], "kb_hits": [],
            "exact_match": None, "fuzzy_matches": [], "llm_hypotheses": [],
            "evidence_context": "", "report": None, "confidence": 0.0,
            "clarify_question": None, "clarify_answer": None, "turn": 0,
            "need_ticket": False, "finished": False, "fallback_used": False,
            "structured_output": None,
        }

    async def demo():
        nodes.KBClient.search = _fake_search
        graph = build_diagnosis_graph()

        print("=== 场景 A：带故障码 → 第一轨精确命中 → 置信度高 → 直接出报告（不追问）===")
        fake_a = _FakeLLM(first_conf=0.9)
        nodes.get_llm = lambda *a, **k: fake_a  # noqa: E731
        result_a = await graph.ainvoke(
            _base_state("设备报E001，电机不转了"),
            {"configurable": {"thread_id": "demo-graph-a"}},
        )
        report_a = result_a.get("report") or {}
        print(f"  conclusion = {report_a.get('conclusion')}")
        print(f"  confidence = {report_a.get('confidence')}, need_ticket = {result_a.get('need_ticket')}, "
              f"turn = {result_a.get('turn')}, __interrupt__ = {bool(result_a.get('__interrupt__'))}")

        print("\n=== 场景 B：无故障码 → 第一轮低置信度 → 追问 → 用户回答 → 重新诊断 ===")
        fake_b = _FakeLLM(first_conf=0.3)
        nodes.get_llm = lambda *a, **k: fake_b  # noqa: E731
        result_b = await graph.ainvoke(
            _base_state("电机不转了，出现过载报警"),
            {"configurable": {"thread_id": "demo-graph-b"}},
        )
        interrupts = result_b.get("__interrupt__", [])
        if interrupts:
            print(f"  interrupt 追问: {interrupts[0].value['question']}")
            result_b = await graph.ainvoke(
                Command(resume={"answer": "满载运行时报的，面板显示E001", "fault_code": "E001"}),
                {"configurable": {"thread_id": "demo-graph-b"}},
            )
        report_b = result_b.get("report") or {}
        print(f"  conclusion = {report_b.get('conclusion')}")
        print(f"  confidence = {report_b.get('confidence')}, need_ticket = {result_b.get('need_ticket')}, "
              f"turn = {result_b.get('turn')}")

    asyncio.run(demo())
