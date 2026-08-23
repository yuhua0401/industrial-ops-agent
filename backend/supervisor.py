"""
supervisor - 主编排（服务类）

将 4 个 Agent 图（知识库 / 故障诊断 / 工单 / 售后）编排为可被统一对话入口调用的服务。

形态选择：**服务类而非巨型 LangGraph 主编排图**。理由：
- 诊断图已绑定进程级 MemorySaver 单例 checkpointer，interrupt/resume 的
  `Command(resume=...)` 控制必须留在调用方（chat.py 执行器 / diagnosis API），
  服务类让 ainvoke/astream 与 thread 控制透明可测；
- 4 个子 Agent 的 State 类型差异大，硬塞进一张大 StateGraph 会产生巨型 TypedDict，
  且父图 + 子图 checkpointer 嵌套有歧义。

用法：
    from backend.supervisor import get_supervisor
    supervisor = get_supervisor()
    result = await supervisor.run_diagnosis("设备报E001电机不转了", session_id="s1")
    async for step in supervisor.stream_diagnosis(...): ...
"""
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.types import Command

from backend.core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class AgentResult:
    """单个 Agent 执行结果（非流式 API 返回）。"""
    reply: str                     # 给用户的最终回复文本
    structured: dict = field(default_factory=dict)   # Agent 结构化输出（报告/知识结果/工单）
    agent_chain: list[str] = field(default_factory=list)  # 经过的 Agent 链路
    interrupted: bool = False      # 是否在 Human-in-the-Loop 追问处暂停
    interrupt_payload: dict | None = None   # 追问内容 {question, ...}
    need_ticket: bool = False      # 诊断是否建议建工单
    ticket_id: str | None = None   # pipeline 已创建的工单号
    confidence: float = 0.0        # 结果置信度
    request_type: str = ""         # 售后请求类型


@dataclass
class AgentStep:
    """流式执行步骤（供 chat.py SSE 消费）。"""
    type: str                      # "node_update" | "interrupt" | "result" | "error"
    node: str | None = None     # 节点名（node_update 时）
    payload: dict | None = None # 步骤数据
    text: str = ""                 # 文本内容（result 时）


# ═══════════════════════════════════════════════════════════════
# 纯函数：诊断报告 → 工单数据
# ═══════════════════════════════════════════════════════════════

def _build_ticket_data(report: dict, context: dict) -> dict:
    """把诊断报告（structured_output）映射为工单创建数据（纯函数，便于测试）。

    context 含 user_input / customer_id / device_model / device_sn / tenant_id。
    """
    return {
        "customer_id":      context.get("customer_id", ""),
        "customer_name":    context.get("customer_name", ""),
        "customer_contact": context.get("customer_contact", ""),
        "device_model":     context.get("device_model", ""),
        "device_sn":        context.get("device_sn", ""),
        "fault_description": context.get("user_input", ""),
        "diagnosis_result": report.get("conclusion", ""),
        "severity":         context.get("severity", "medium"),
        "category":         "repair",
        "notes":            report.get("ticket_reason", ""),
        "tenant_id":        context.get("tenant_id", "tenant_default"),
    }


def _render_diagnosis_report(report: dict) -> str:
    """把诊断报告渲染成给用户看的文本。"""
    conclusion = report.get("conclusion", "")
    causes = report.get("causes", []) or []
    solutions = report.get("solutions", []) or []
    parts = [f"**诊断结论**：{conclusion}\n"]
    if causes:
        lines = [f"- {c.get('desc', '')}（{c.get('probability', '')}）" for c in causes[:4]]
        parts.append("**可能原因**：\n" + "\n".join(lines))
    if solutions:
        lines = [f"{i}. {s.get('step', '')}" for i, s in enumerate(solutions[:6], 1)]
        parts.append("**处理方案**：\n" + "\n".join(lines))
    if report.get("need_ticket"):
        parts.append("**建议**：需要创建报修工单安排进一步处理。")
    return "\n\n".join(parts)


def _render_knowledge_result(result: dict) -> str:
    """把知识库 Agent 结果渲染成文本。"""
    answer = result.get("answer", "")
    sources = result.get("sources", []) or []
    related = result.get("related_questions", []) or []
    parts = [answer]
    if sources:
        parts.append("**参考来源**：" + "；".join(sources[:5]))
    if related:
        parts.append("**猜你想问**：\n" + "\n".join(f"- {q}" for q in related[:3]))
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# Supervisor 服务类
# ═══════════════════════════════════════════════════════════════

class Supervisor:
    """主编排服务：编排 4 个 Agent 图。"""

    def __init__(self) -> None:
        self._graphs: dict[str, object] = {}

    # ── 图缓存 ──────────────────────────────────────────────
    def get_knowledge_graph(self):
        from backend.agents.knowledge.graph import build_knowledge_graph
        if "knowledge" not in self._graphs:
            self._graphs["knowledge"] = build_knowledge_graph()
        return self._graphs["knowledge"]

    def get_diagnosis_graph(self):
        from backend.agents.diagnosis.graph import build_diagnosis_graph
        if "diagnosis" not in self._graphs:
            # build_diagnosis_graph 内部已保证 MemorySaver 进程级单例
            self._graphs["diagnosis"] = build_diagnosis_graph()
        return self._graphs["diagnosis"]

    def get_ticket_graph(self):
        from backend.agents.ticket.graph import build_ticket_graph
        if "ticket" not in self._graphs:
            self._graphs["ticket"] = build_ticket_graph()
        return self._graphs["ticket"]

    def get_after_sale_graph(self):
        from backend.agents.after_sale.graph import build_after_sale_graph
        if "after_sale" not in self._graphs:
            self._graphs["after_sale"] = build_after_sale_graph()
        return self._graphs["after_sale"]

    # ── 诊断初始 state 构造 ─────────────────────────────────
    @staticmethod
    def _diagnosis_initial_state(
        message: str, session_id: str, customer_id: str = "",
        device_model: str = "", fault_code: str | None = None,
    ) -> dict:
        return {
            "messages": [HumanMessage(content=message)],
            "session_id": session_id,
            "device_model": device_model,
            "fault_description": "",
            "collected_symptoms": [],
            "current_step": 0,
            "diagnosis_result": None,
            "resolved": False,
            "user_input": message,
            "fault_code": fault_code,
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

    @staticmethod
    def _config(session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}}

    # ── 非流式 API ──────────────────────────────────────────

    async def run_knowledge(
        self, message: str, session_id: str, device_model: str = "",
    ) -> AgentResult:
        graph = self.get_knowledge_graph()
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "session_id": session_id,
            "device_model": device_model,
            "query": message,
            "retrieved_docs": [],
            "knowledge_result": None,
        }
        result = await graph.ainvoke(initial_state)
        kr = result.get("knowledge_result") or {}
        return AgentResult(
            reply=_render_knowledge_result(kr),
            structured=kr,
            agent_chain=["knowledge"],
            confidence=_conf_to_float(kr.get("confidence", "low")),
        )

    async def run_diagnosis(
        self, message: str, session_id: str, customer_id: str = "",
        device_model: str = "", fault_code: str | None = None, image: str | None = None,
    ) -> AgentResult:
        graph = self.get_diagnosis_graph()
        initial_state = self._diagnosis_initial_state(
            message, session_id, customer_id, device_model, fault_code,
        )
        if image:
            initial_state["image"] = image

        result = await graph.ainvoke(initial_state, self._config(session_id))

        # interrupt 检测：返回的 state 里带 __interrupt__ 字段
        interrupts = result.get("__interrupt__")
        if interrupts:
            payload = (
                dict(interrupts[0].value)
                if hasattr(interrupts[0], "value")
                else dict(interrupts[0])
            )
            return AgentResult(
                reply="", structured={}, agent_chain=["diagnosis"],
                interrupted=True, interrupt_payload=payload,
            )

        report = result.get("structured_output") or result.get("report") or {}
        return AgentResult(
            reply=_render_diagnosis_report(report),
            structured=report,
            agent_chain=["diagnosis"],
            need_ticket=bool(result.get("need_ticket", report.get("need_ticket", False))),
            confidence=float(report.get("confidence", 0.0)),
        )

    async def run_ticket(
        self, session_id: str, customer_id: str, ticket_data: dict,
    ) -> AgentResult:
        graph = self.get_ticket_graph()
        initial_state: dict = {
            "messages": [],
            "session_id": session_id,
            "ticket_data": ticket_data,
            "ticket": None,
            "created": False,
            "ticket_id": None,
        }
        result = await graph.ainvoke(initial_state)
        ticket_id = result.get("ticket_id") or ""
        created = bool(result.get("created"))
        if created:
            reply = f"报修工单已创建，工单号 **{ticket_id}**。"
        else:
            reply = "工单创建失败，请稍后重试或联系人工客服。"
        return AgentResult(
            reply=reply,
            structured=result.get("ticket") or {},
            agent_chain=["ticket"],
            ticket_id=ticket_id or None,
        )

    async def run_after_sale(
        self, session_id: str, customer_id: str, device_sn: str,
        request_type: str, message: str = "",
    ) -> AgentResult:
        graph = self.get_after_sale_graph()
        initial_state: dict = {
            "messages": [],
            "session_id": session_id,
            "customer_id": customer_id,
            "device_sn": device_sn,
            "message": message,
            "request_type": request_type,
            "warranty_info": None,
            "part_order": None,
            "appointment_time": None,
            "appointment_info": None,
            "stock_info": None,
            "reply": "",
            "service_completed": False,
        }
        result = await graph.ainvoke(initial_state)
        request_type = result.get("request_type", request_type)
        reply = _render_after_sale_result(result)
        return AgentResult(
            reply=reply,
            structured={
                "request_type": request_type,
                "warranty_info": result.get("warranty_info"),
                "part_order": result.get("part_order"),
                "appointment_info": result.get("appointment_info"),
            },
            agent_chain=["after_sale"],
            request_type=request_type,
        )

    async def run_pipeline(
        self, message: str, session_id: str, customer_id: str = "",
        device_model: str = "", device_sn: str = "",
    ) -> AgentResult:
        """pipeline：诊断 → (need_ticket) 工单 → 售后。"""
        diag = await self.run_diagnosis(
            message, session_id, customer_id=customer_id, device_model=device_model,
        )
        if diag.interrupted:
            return diag
        if not diag.need_ticket:
            return diag

        ticket_data = _build_ticket_data(diag.structured, {
            "user_input": message, "customer_id": customer_id,
            "device_model": device_model, "device_sn": device_sn,
        })
        ticket = await self.run_ticket(session_id, customer_id, ticket_data)
        if not ticket.ticket_id:
            return diag

        after = await self.run_after_sale(
            session_id, customer_id, device_sn=device_sn,
            request_type="warranty", message=message,
        )
        combined_reply = (
            f"{_render_diagnosis_report(diag.structured)}\n\n"
            f"📋 **工单**：{ticket.reply}\n\n"
            f"🛠️ **售后**：{after.reply}"
        )
        return AgentResult(
            reply=combined_reply,
            structured=diag.structured,
            agent_chain=["diagnosis", "ticket", "after_sale"],
            need_ticket=True,
            ticket_id=ticket.ticket_id,
            confidence=diag.confidence,
        )

    # ── 流式 API（供 chat.py SSE）────────────────────────────

    async def stream_diagnosis(
        self, message: str, session_id: str, customer_id: str = "",
        device_model: str = "", fault_code: str | None = None,
        resume: bool = False, resume_answer: str = "",
    ) -> AsyncIterator[AgentStep]:
        """流式执行诊断图，产出 AgentStep。

        resume=True 时用 Command(resume=...) 恢复被 interrupt 暂停的线程。
        """
        graph = self.get_diagnosis_graph()
        config = self._config(session_id)

        if resume:
            command = Command(resume={"answer": resume_answer, "fault_code": None})
        else:
            command = self._diagnosis_initial_state(
                message, session_id, customer_id, device_model, fault_code,
            )

        final_state: dict = {}
        try:
            async for mode, data in graph.astream(
                command, config, stream_mode=["updates", "values"],
            ):
                if mode == "updates":
                    for node, upd in data.items():
                        if "__interrupt__" in upd:
                            interrupts = upd["__interrupt__"]
                            payload = (
                dict(interrupts[0].value)
                if hasattr(interrupts[0], "value")
                else dict(interrupts[0])
            )
                            yield AgentStep(type="interrupt", node=node, payload=payload)
                            return
                        yield AgentStep(type="node_update", node=node, payload=upd)
                elif mode == "values":
                    final_state = data
        except Exception as e:
            logger.error("supervisor.diagnosis_stream_error", error=str(e), exc_info=True)
            yield AgentStep(type="error", payload={"error": str(e)})
            return

        report = final_state.get("structured_output") or final_state.get("report") or {}
        yield AgentStep(
            type="result",
            payload={
                "report": report,
                "need_ticket": bool(
                    final_state.get("need_ticket", report.get("need_ticket", False))
                ),
                "confidence": float(report.get("confidence", 0.0)),
                "interrupted": bool(final_state.get("__interrupt__")),
            },
            text=_render_diagnosis_report(report),
        )


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def _conf_to_float(conf: str) -> float:
    """"高/中/低" → 0.9 / 0.6 / 0.3。"""
    return {"高": 0.9, "中": 0.6, "低": 0.3}.get(conf, 0.5)


def _render_after_sale_result(result: dict) -> str:
    """渲染售后图结果。"""
    request_type = result.get("request_type", "")
    warranty = result.get("warranty_info")
    part_order = result.get("part_order")
    appointment = result.get("appointment_info")

    if request_type == "warranty" and warranty:
        if warranty.get("status") == "in_warranty":
            return (
                f"设备 {warranty.get('device_sn', '')} 处于保修期内，"
                f"保修截止 {warranty.get('warranty_end', '')}。"
            )
        if warranty.get("status") == "out_of_warranty":
            return (
                f"设备 {warranty.get('device_sn', '')} 已过保修期"
                f"（截止 {warranty.get('warranty_end', '')}），可付费维修或预约上门。"
            )
        return f"保修信息查询失败：{warranty.get('error', '未知原因')}"
    if request_type == "parts" and part_order:
        return part_order.get("stock_info", "配件库存信息。")
    if request_type == "appointment" and appointment:
        return (
            f"预约号 {appointment.get('appointment_id', '')}，"
            f"时间 {appointment.get('scheduled_time', '')}。"
        )
    return "已收到售后诉求，如需进一步帮助请联系人工客服。"


# ── 模块级单例 ─────────────────────────────────────────────────
_supervisor: Supervisor | None = None


def get_supervisor() -> Supervisor:
    """获取 Supervisor 进程级单例。"""
    global _supervisor
    if _supervisor is None:
        _supervisor = Supervisor()
    return _supervisor
