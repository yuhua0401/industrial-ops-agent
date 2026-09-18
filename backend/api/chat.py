"""
chat - 统一对话入口 API

设备智能客服系统的唯一对话入口。
流程：规则前置拦截 → LLM 路由判断 → 推送 routing_decision → Agent 执行 → 流式返回

SSE 事件类型：
  routing_decision  路由决策结果（agent_type / confidence / reason / execution_mode）
  progress          Agent 执行进度提示（"检索知识库中..."等）
  token             流式回答 token
  guidance          需引导跳转的意图（如"请上传故障图片"）
  pipeline_plan     多 Agent 协同计划（故障报修全流程）
  meta              回答完毕后的元数据（sources / confidence 等）
  done              流结束信号
  error             异常
"""

import json
import re
from dataclasses import dataclass
from enum import StrEnum

from fastapi import APIRouter
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from backend import session_state
from backend.core.llm_factory import get_llm
from backend.core.logger import get_logger
from backend.supervisor import get_supervisor

router = APIRouter()
logger = get_logger(__name__)
supervisor = get_supervisor()


# ═══════════════════════════════════════════════════════════════
# Agent 类型枚举（对齐 Supervisor 状态机）
# ═══════════════════════════════════════════════════════════════

class AgentType(StrEnum):
    """Agent 类型枚举，与 Supervisor 路由表保持一致。"""
    KNOWLEDGE  = "knowledge"    # Agent② 产品知识问答
    DIAGNOSIS  = "diagnosis"    # Agent③ 故障诊断
    TICKET     = "ticket"       # Agent④ 工单管理
    AFTER_SALE = "after_sale"   # Agent⑤ 售后协调


class ExecutionMode(StrEnum):
    """执行模式。"""
    SINGLE   = "single"     # 单 Agent 执行
    PIPELINE = "pipeline"   # 多 Agent 串联（诊断 → 工单 → 售后）
    CLARIFY  = "clarify"    # 意图不明，需追问


# ── Agent 中文名映射（路由卡片展示用）──────────────────────────
_AGENT_DISPLAY: dict[AgentType, str] = {
    AgentType.KNOWLEDGE:  "产品知识问答",
    AgentType.DIAGNOSIS:  "故障诊断",
    AgentType.TICKET:     "工单管理",
    AgentType.AFTER_SALE: "售后协调",
}

# ── 各 Agent 节点进度提示（映射 LangGraph 真实节点名）────────────
_PROGRESS_LABELS: dict[str, str] = {
    # 知识问答
    "retrieve":        "检索知识库中...",
    "rerank":          "精排文档中...",
    "generate_rag":    "生成回答中...",
    # 故障诊断图节点
    "parse_input":     "解析故障描述中...",
    "load_diag_tree":  "加载诊断树与知识库中...",
    "run_diag_tracks": "三轨并行诊断中...",
    "assemble_context": "组装诊断证据中...",
    "generate_report": "生成诊断结论中...",
    "check_sufficiency": "评估诊断置信度...",
    "ask_clarify":     "等待补充信息...",
    "apply_clarify_answer": "合并补充信息，重新诊断...",
    "route_next":      "处理诊断结果...",
    # 工单
    "create_ticket":   "创建工单中...",
    "query_ticket":    "查询工单中...",
    # 售后图节点
    "route_request_type": "识别售后诉求中...",
    "query_warranty":  "查询保修信息中...",
    "order_part":      "查询配件库存中...",
    "prepare_appointment": "准备预约服务...",
    "tools":           "调用售后工具中...",
}
_GENERATE_NODES = frozenset({
    "generate_rag", "generate_direct", "generate_general",
    "generate_diag", "generate_ticket", "generate_after_sale",
})


# ═══════════════════════════════════════════════════════════════
# 规则前置拦截：五类社交/元场景（零 Token，不调 LLM）
# ═══════════════════════════════════════════════════════════════

_STRIP_TAIL_RE = re.compile(r"[\s!！?？。~～,.，。]+$")

# ── 类别一：问候 ──────────────────────────────────────────────
_HELLO_KEYWORDS = frozenset([
    "你好", "您好", "hi", "hello", "hey", "哈喽", "嗨",
    "在吗", "在不在", "在线吗", "有人吗",
])

# ── 类别二：感谢 ──────────────────────────────────────────────
_THANKS_KEYWORDS = frozenset([
    "谢谢", "感谢", "多谢", "谢了", "非常感谢", "万分感谢",
    "辛苦了", "辛苦", "麻烦了",
    "太棒了", "太好了", "厉害", "厉害了",
    "好的好的", "明白了", "懂了", "知道了", "收到",
])

# ── 类别三：道别 ──────────────────────────────────────────────
_BYE_KEYWORDS = frozenset([
    "再见", "拜拜", "拜", "88", "886", "bye", "goodbye", "byebye",
    "下次见", "下次再聊", "先走了", "先撤了",
])

# ── 类别四：身份询问（正则）───────────────────────────────────
_IDENTITY_RE = re.compile(
    r"(你|您)(是谁|叫什么|的名字|是什么|是.*AI|是.*机器人|是.*助手)"
    r"|介绍.{0,4}(你自己|自己|一下)"
    r"|你是谁"
    r"|你叫(啥|什么名)",
    re.IGNORECASE,
)

# ── 类别五：功能询问（正则）───────────────────────────────────
_CAPABILITY_RE = re.compile(
    r"(你|您)(能|可以|会).{0,6}(做|帮|干)"
    r"|(你|您).{0,4}(功能|用途|能力|特点)"
    r"|怎么(用|使用)(你|您|这个)?"
    r"|(使用说明|帮助菜单|help|usage)"
    r"|你能帮(我|忙)吗",
    re.IGNORECASE,
)

# ── 五类回复模板 ──────────────────────────────────────────────

_REPLY_HELLO = (
    "您好！我是设备智能服务助手，专注于工业设备运维支持。\n\n"
    "我可以帮您：\n"
    "- **产品知识问答**：直接提问设备参数、操作规范、维护保养等问题，AI 从知识库检索解答\n"
    "- **故障诊断**：描述故障现象或提供错误代码，AI 自动推理根因并给出解决方案\n"
    "- **工单管理**：查询工单进度、催单、或由诊断结论自动生成报修工单\n"
    "- **售后协调**：查询保修状态、配件库存、预约上门维修服务\n\n"
    "直接告诉我您的需求，我会自动路由到最合适的处理流程。"
)

_REPLY_THANKS = (
    "不客气，很高兴能帮到您！\n\n"
    "如果设备还有其他问题，随时告诉我。"
)

_REPLY_BYE = (
    "再见！如有设备问题请随时联系，祝您生产顺利！"
)

_REPLY_IDENTITY = (
    "我是**设备智能服务助手**，一套面向工业设备运维的多 Agent 智能系统。\n\n"
    "我由以下专业 Agent 协同构成：\n"
    "- **产品知识 Agent**：基于 RAG 知识库，7×24 即时解答设备相关技术问题\n"
    "- **故障诊断 Agent**：基于诊断树推理引擎，输入故障现象即可定位根因\n"
    "- **工单管理 Agent**：自动创建/查询/更新报修工单，全流程追踪\n"
    "- **售后协调 Agent**：查询保修、核对配件库存、预约上门服务\n\n"
    "遇到设备故障时，我会自动串联诊断→工单→售后全流程，无需您反复描述问题。\n\n"
    "请问有什么可以帮到您？"
)

_REPLY_CAPABILITY = (
    "我能为您提供以下服务：\n\n"
    "**单 Agent 直达**\n"
    "- 直接提问设备问题 → 产品知识问答（RAG 知识库检索）\n"
    "- 描述故障现象/错误代码 → 故障诊断（诊断树推理 + 方案输出）\n"
    "- 查工单/催进度 → 工单管理\n"
    "- 查保修/约上门 → 售后协调\n\n"
    "**多 Agent 协同**\n"
    "- 「设备报 E05 错误，需要维修」→ 自动串联 故障诊断 → 创建工单 → 售后协调\n"
    "- 一次描述，全流程自动推进，无需反复说明\n\n"
    "直接告诉我您遇到的问题，我会自动路由到最合适的处理流程。"
)


def _pre_filter(text: str) -> str | None:
    """
    规则前置拦截，五类社交/元场景零 Token 直接返回模板回复。
    命中返回模板字符串；未命中返回 None，继续走 LLM 路由。
    """
    t = text.strip()
    t_lower = _STRIP_TAIL_RE.sub("", t.lower())

    if t_lower in _HELLO_KEYWORDS:
        return _REPLY_HELLO
    if t_lower in _THANKS_KEYWORDS:
        return _REPLY_THANKS
    if t_lower in _BYE_KEYWORDS:
        return _REPLY_BYE
    if _IDENTITY_RE.search(t):
        return _REPLY_IDENTITY
    if _CAPABILITY_RE.search(t):
        return _REPLY_CAPABILITY

    return None


# ═══════════════════════════════════════════════════════════════
# LLM 路由：将用户输入归类到 6 类之一
# ═══════════════════════════════════════════════════════════════

_ROUTE_PROMPT = """判断用户需求应路由到哪个功能模块。

可选功能：
- knowledge    : 产品知识问答（设备参数、操作规范、维护保养、技术文档等常规问题）
- diagnosis    : 故障诊断（用户描述了故障现象、错误代码、设备异常，需要诊断根因）
- ticket       : 工单管理（查询已有工单的进度、催单、修改工单信息）
- after_sale   : 售后协调（查询保修期、配件库存、预约上门维修、退换货等售后需求）
- pipeline     : 故障报修全流程（用户明确表示设备坏了需要修，同时涉及诊断+报修+售后）
- clarify      : 意图不明确，无法判断用户想做什么，需要追问

严格按以下 JSON 格式返回，不要有其他内容：
{{"label": "功能名", "reason": "一句话说明判断依据", "confidence": 0.0到1.0的小数}}

confidence 为你对本次分类的把握度：
- 命中强特征（明确的故障码/工单号/配件编号/保修询问）给 0.9 以上
- 语义清晰但需要理解意图给 0.7-0.9
- 模糊或存在多种理解给 0.4-0.7
- 几乎无法判断给 0.4 以下
- 用户没有表达任何具体需求（如"随便问问"、无实质内容的闲聊）→ label 选 clarify 且 confidence ≤ 0.5

用户输入：{message}"""

# label → AgentType / ExecutionMode 映射
_LABEL_TO_AGENT: dict[str, AgentType] = {
    "knowledge":    AgentType.KNOWLEDGE,
    "diagnosis":    AgentType.DIAGNOSIS,
    "ticket":       AgentType.TICKET,
    "after_sale":   AgentType.AFTER_SALE,
    "pipeline":     AgentType.DIAGNOSIS,   # pipeline 以诊断入口
    "clarify":      AgentType.KNOWLEDGE,   # clarify 用知识问答兜底追问
}

_LABEL_TO_MODE: dict[str, ExecutionMode] = {
    "knowledge":    ExecutionMode.SINGLE,
    "diagnosis":    ExecutionMode.SINGLE,
    "ticket":       ExecutionMode.SINGLE,
    "after_sale":   ExecutionMode.SINGLE,
    "pipeline":     ExecutionMode.PIPELINE,
    "clarify":      ExecutionMode.CLARIFY,
}

_VALID_LABELS = frozenset(_LABEL_TO_AGENT.keys())


@dataclass
class _RouteResult:
    """LLM 路由结果，对齐前端 routing_decision 事件字段。"""
    label:          str
    agent_type:     AgentType
    execution_mode: ExecutionMode
    confidence:     float
    reason:         str


async def _llm_route(message: str) -> _RouteResult:
    """
    调用 LLM 对用户输入做跨 Agent 路由判断。

    异常时降级返回 knowledge（产品知识问答），不阻断 SSE 流。
    """
    try:
        llm = get_llm("intent", temperature=0)
        resp = await llm.ainvoke([
            HumanMessage(content=_ROUTE_PROMPT.format(message=message))
        ])
        raw = resp.text.strip() if hasattr(resp, 'text') else str(resp)
        # 容错：LLM 可能在 JSON 外包裹 markdown 代码块
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        parsed = json.loads(raw)
        label = parsed.get("label", "knowledge").strip().lower()
        reason = parsed.get("reason", "LLM 路由判断")
        confidence = _clamp_confidence(parsed.get("confidence"), default=0.7)

        if label not in _VALID_LABELS:
            logger.warning("unified_chat.llm_route_unknown_label", label=label, fallback="knowledge")
            label = "knowledge"
            confidence = min(confidence, 0.5)   # 分类不可信，压低置信度

        logger.info("unified_chat.llm_route_result", label=label,
                    confidence=round(confidence, 3), reason=reason)

    except Exception as e:
        logger.warning("unified_chat.llm_route_failed", error=str(e), fallback="knowledge")
        label = "knowledge"
        reason = "路由判断异常，默认转入产品知识问答"
        confidence = 0.5

    return _RouteResult(
        label=label,
        agent_type=_LABEL_TO_AGENT[label],
        execution_mode=_LABEL_TO_MODE[label],
        confidence=confidence,
        reason=reason,
    )


def _clamp_confidence(value, default: float = 0.7) -> float:
    """把 LLM 自评置信度收敛到 [0,1]；缺失/非法用 default。"""
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return default


# ═══════════════════════════════════════════════════════════════
# Pipeline 计划（故障报修全流程）
# ═══════════════════════════════════════════════════════════════

_PIPELINE_PLAN = {
    "type":   "pipeline_plan",
    "title":  "故障报修全流程",
    "intro":  "已为您规划「故障报修全流程」，系统将自动串联诊断 → 工单 → 售后，"
              "无需您反复描述问题。",
    "steps": [
        {
            "step":         1,
            "agent_type":   "diagnosis",
            "label":        "故障诊断",
            "desc":         "分析故障现象，定位根因并给出解决方案",
            "action_label": "开始诊断",
            "action_url":   "",
            "tip":          "请准备好设备型号和故障现象描述",
        },
        {
            "step":         2,
            "agent_type":   "ticket",
            "label":        "创建工单",
            "desc":         "基于诊断结论自动生成报修工单，无需重复填写故障信息",
            "action_label": "查看工单",
            "action_url":   "/ticket",
            "tip":          "诊断结论将自动填充到工单",
        },
        {
            "step":         3,
            "agent_type":   "after_sale",
            "label":        "售后协调",
            "desc":         "查询保修状态、核对配件库存、预约上门维修时间",
            "action_label": "进入售后",
            "action_url":   "/after-sale",
            "tip":          "工单创建后自动触发",
        },
    ],
}


# ═══════════════════════════════════════════════════════════════
# 请求/响应模型
# ═══════════════════════════════════════════════════════════════

class UnifiedChatRequest(BaseModel):
    """统一对话请求。"""
    session_id:  str   = Field(..., description="会话 ID")
    message:     str   = Field(..., min_length=1, max_length=2000, description="用户输入")
    customer_id: str   = Field(default="", description="客户 ID（可选，用于关联历史工单）")
    device_model: str  = Field(default="", description="设备型号（可选，辅助故障诊断）")
    device_sn:   str   = Field(default="", description="设备序列号（可选，售后查保修用）")
    image:       str   = Field(default="", max_length=7_000_000,
                               description="故障图片 base64 dataURL（可选，带图直达故障诊断）")


# ═══════════════════════════════════════════════════════════════
# SSE 工具函数
# ═══════════════════════════════════════════════════════════════

def _sse(data: dict) -> dict:
    """把 dict 包成 sse_starlette 格式：{"data": "<JSON字符串>"}。"""
    return {"data": json.dumps(data, ensure_ascii=False)}


# ═══════════════════════════════════════════════════════════════
# 统一对话 SSE 端点
# ═══════════════════════════════════════════════════════════════

@router.post("/stream")
async def unified_chat_stream(req: UnifiedChatRequest):
    """
    设备智能客服统一流式接口（SSE）。

    请求：{session_id, message, customer_id?, device_model?}
    响应：text/event-stream

    流程：
        1. 规则前置拦截（五类社交/元场景，零 Token 直接返回）
        2. LLM 路由判断（knowledge / diagnosis / ticket / after_sale / pipeline / clarify）
        3. 推送 routing_decision 事件（前端显示路由卡片）
        4. 按路由结果分发：
           - knowledge   → 流式执行知识问答 Agent
           - diagnosis   → 流式执行故障诊断 Agent
           - ticket      → 处理工单查询/催单
           - after_sale  → 处理售后咨询
           - pipeline    → 推送故障报修全流程计划，然后依次执行诊断
           - clarify     → 推送追问提示
    """

    async def event_generator():
        # ── Step 0a：诊断中断恢复（resume）───────────────────
        # 若同一 thread_id 的诊断图停在 ask_clarify（get_state().next 非空），
        # 则本次消息视为对追问的回答，跳过规则拦截与 LLM 路由直接恢复。
        diag_graph = supervisor.get_diagnosis_graph()
        config = {"configurable": {"thread_id": req.session_id}}
        if diag_graph.get_state(config).next:
            st = session_state.get(req.session_id)
            if st.pipeline_after:
                async for event in _stream_pipeline_agent(req, resume=True):
                    yield event
            else:
                async for event in _stream_diagnosis_agent(req, resume=True):
                    yield event
            yield _sse({"type": "done"})
            return

        # ── Step 0b：带图片 → 直达故障诊断 ────────────────────
        # 图片通常伴随故障上报，跳过规则拦截与 LLM 路由；
        # 视觉描述在诊断图 parse_input 节点内生成（失败自动降级纯文字）。
        if req.image:
            yield _sse({
                "type":           "routing_decision",
                "agent_type":     AgentType.DIAGNOSIS.value,
                "agent_display":  _AGENT_DISPLAY[AgentType.DIAGNOSIS],
                "confidence":     0.9,
                "reason":         "检测到图片输入，直达故障诊断",
                "execution_mode": ExecutionMode.SINGLE.value,
            })
            async for event in _stream_diagnosis_agent(req):
                yield event
            yield _sse({"type": "done"})
            return

        # ── Step 0：规则前置拦截（零 Token）────────────────────
        pre_reply = _pre_filter(req.message)
        if pre_reply is not None:
            yield _sse({"type": "token", "content": pre_reply})
            yield _sse({"type": "done"})
            return

        # ── Step 1：LLM 路由判断 ──────────────────────────────
        decision = await _llm_route(req.message)

        # ── Step 2：推送路由决策卡片 ──────────────────────────
        yield _sse({
            "type":           "routing_decision",
            "agent_type":     decision.agent_type.value,
            "agent_display":  _AGENT_DISPLAY.get(decision.agent_type, ""),
            "confidence":     round(decision.confidence, 4),
            "reason":         decision.reason,
            "execution_mode": decision.execution_mode.value,
        })

        label = decision.label

        # ── Step 3a：knowledge → 产品知识问答 ─────────────────
        if label == "knowledge":
            async for event in _stream_knowledge_agent(req):
                yield event

        # ── Step 3b：diagnosis → 故障诊断 ─────────────────────
        elif label == "diagnosis":
            async for event in _stream_diagnosis_agent(req):
                yield event

        # ── Step 3c：ticket → 工单管理 ────────────────────────
        elif label == "ticket":
            async for event in _stream_ticket_agent(req):
                yield event

        # ── Step 3d：after_sale → 售后协调 ────────────────────
        elif label == "after_sale":
            async for event in _stream_after_sale_agent(req):
                yield event

        # ── Step 3e：pipeline → 故障报修全流程 ────────────────
        elif label == "pipeline":
            async for event in _stream_pipeline_agent(req):
                yield event

        # ── Step 3f：clarify → 追问提示 ───────────────────────
        else:
            yield _sse({
                "type":    "guidance",
                "message": (
                    "您的问题我还不太确定应该如何处理，能否描述得更具体一些？\n\n"
                    "例如：\n"
                    "- 想了解设备参数或操作规范 → 直接提问即可\n"
                    "- 设备出现故障 → 描述故障现象或提供错误代码\n"
                    "- 查询工单进度 → 提供工单编号\n"
                    "- 售后需求 → 说明需要查保修、买配件还是预约上门"
                ),
                "action_label": "",
                "action_url":   "",
            })

        yield _sse({"type": "done"})

    return EventSourceResponse(event_generator())


# ═══════════════════════════════════════════════════════════════
# 各 Agent 流式执行器
# ═══════════════════════════════════════════════════════════════

async def _stream_knowledge_agent(req: UnifiedChatRequest):
    """
    流式执行产品知识问答 Agent（Agent②）。

    流程：retrieve → generate（调用知识库 Agent 图，astream updates/values 双模式）。
    """
    try:
        graph = supervisor.get_knowledge_graph()
        initial_state = {
            "messages": [HumanMessage(content=req.message)],
            "session_id": req.session_id,
            "device_model": req.device_model,
            "query": req.message,
            "retrieved_docs": [],
            "knowledge_result": None,
        }

        final_state: dict = {}
        async for mode, data in graph.astream(
            initial_state, config=None, stream_mode=["updates", "values"],
        ):
            if mode == "updates":
                for node in data:
                    label = _PROGRESS_LABELS.get(node)
                    if label:
                        yield _sse({"type": "progress", "stage": label})
            elif mode == "values":
                final_state = data

        kr = final_state.get("knowledge_result") or {}
        answer = kr.get("answer", "抱歉，暂时无法回答您的问题。")
        yield _sse({"type": "token", "content": answer})
        yield _sse({
            "type":              "meta",
            "answer_mode":       "rag" if final_state.get("retrieved_docs") else "fallback",
            "confidence":        kr.get("confidence", "low"),
            "sources":           kr.get("sources", []),
            "related_questions": kr.get("related_questions", []),
        })

    except Exception as e:
        logger.error("unified_chat.knowledge_stream_error", error=str(e), exc_info=True)
        yield _sse({"type": "error", "message": "知识问答服务异常，请稍后重试"})


async def _stream_diagnosis_agent(req: UnifiedChatRequest, resume: bool = False):
    """
    流式执行故障诊断 Agent（Agent③）。

    流程：parse_input → load_diag_tree → run_diag_tracks → ... → route_next。
    resume=True 时用 Command(resume=...) 恢复被 interrupt 暂停的线程。
    """
    try:
        if resume:
            yield _sse({"type": "progress", "stage": "正在结合您的补充信息重新诊断..."})
            async for step in supervisor.stream_diagnosis(
                req.message, req.session_id,
                customer_id=req.customer_id, device_model=req.device_model,
                resume=True, resume_answer=req.message,
            ):
                for event in _diagnosis_step_to_sse(step):
                    yield event
            return

        if req.image:
            yield _sse({"type": "progress", "stage": "识别故障图片中..."})
        async for step in supervisor.stream_diagnosis(
            req.message, req.session_id,
            customer_id=req.customer_id, device_model=req.device_model,
            image=req.image or None,
        ):
            for event in _diagnosis_step_to_sse(step):
                yield event

    except Exception as e:
        logger.error("unified_chat.diagnosis_stream_error", error=str(e), exc_info=True)
        yield _sse({"type": "error", "message": "故障诊断服务异常，请稍后重试"})


def _diagnosis_step_to_sse(step):
    """把 Supervisor 诊断流式 step 转成 SSE 事件（generator）。"""
    if step.type == "node_update":
        label = _PROGRESS_LABELS.get(step.node or "")
        if label:
            yield _sse({"type": "progress", "stage": label})
    elif step.type == "interrupt":
        payload = step.payload or {}
        question = payload.get("question", "请补充更多故障信息。")
        yield _sse({
            "type": "guidance",
            "message": question,
            "action_label": "",
            "action_url": "",
        })
    elif step.type == "result":
        payload = step.payload or {}
        report = payload.get("report") or {}
        text = step.text or ""
        # 分片推送渲染结果
        for i in range(0, len(text), 200):
            yield _sse({"type": "token", "content": text[i:i + 200]})
        yield _sse({
            "type":             "meta",
            "answer_mode":      "diagnosis",
            "confidence":       payload.get("confidence", 0.0),
            "need_ticket":      payload.get("need_ticket", False),
            "diagnosis_summary": report.get("conclusion", "")[:500],
        })
    elif step.type == "error":
        yield _sse({"type": "error", "message": "故障诊断服务异常，请稍后重试"})


async def _stream_ticket_agent(req: UnifiedChatRequest):
    """
    流式执行工单管理 Agent（Agent④）。

    支持：工单查询（按编号）/ 由诊断结论创建工单。
    """
    try:
        ticket_id = _extract_ticket_id(req.message)
        if ticket_id:
            yield _sse({"type": "progress", "stage": "查询工单中..."})
            info = await _query_ticket_info(ticket_id)
            if info:
                yield _sse({"type": "token", "content": info})
            else:
                yield _sse({"type": "token", "content": f"未找到工单 {ticket_id}，请核对工单号。"})
            # 查到档案 → 高置信；查无此单 → 低置信
            yield _sse({"type": "meta", "answer_mode": "ticket",
                        "confidence": 0.95 if info else 0.5, "sources": []})
            return

        # 无工单号：视为创建报修工单
        yield _sse({"type": "progress", "stage": "创建工单中..."})
        ticket_result = await supervisor.run_ticket(
            req.session_id, req.customer_id,
            {"fault_description": req.message, "device_model": req.device_model,
             "customer_id": req.customer_id},
        )
        yield _sse({"type": "token", "content": ticket_result.reply})
        yield _sse({
            "type": "meta", "answer_mode": "ticket",
            "confidence": 0.9 if ticket_result.ticket_id else 0.3, "sources": [],
            "ticket_id": ticket_result.ticket_id or "",
        })

    except Exception as e:
        logger.error("unified_chat.ticket_stream_error", error=str(e), exc_info=True)
        yield _sse({"type": "error", "message": "工单服务异常，请稍后重试"})


async def _stream_after_sale_agent(req: UnifiedChatRequest):
    """
    流式执行售后协调 Agent（Agent⑤）。

    支持：查询保修期 / 查询配件库存 / 预约上门维修。
    """
    try:
        graph = supervisor.get_after_sale_graph()
        initial_state: dict = {
            "messages": [], "session_id": req.session_id,
            "customer_id": req.customer_id, "device_sn": req.device_sn,
            "message": req.message, "request_type": "",
            "warranty_info": None, "part_order": None,
            "appointment_time": None, "appointment_info": None,
            "stock_info": None, "reply": "", "service_completed": False,
        }

        final_state: dict = {}
        async for mode, data in graph.astream(
            initial_state, config=None, stream_mode=["updates", "values"],
        ):
            if mode == "updates":
                for node in data:
                    label = _PROGRESS_LABELS.get(node)
                    if label:
                        yield _sse({"type": "progress", "stage": label})
            elif mode == "values":
                final_state = data

        reply = _render_after_sale_result(final_state)
        yield _sse({"type": "token", "content": reply})
        yield _sse({
            "type": "meta", "answer_mode": "after_sale",
            "confidence": _after_sale_confidence(final_state), "sources": [],
            "request_type": final_state.get("request_type", ""),
        })

    except Exception as e:
        logger.error("unified_chat.after_sale_stream_error", error=str(e), exc_info=True)
        yield _sse({"type": "error", "message": "售后服务异常，请稍后重试"})


def _after_sale_confidence(final_state: dict) -> float:
    """售后 meta 置信度：按实际查询结果判定（查到档案高、未命中/异常低）。"""
    request_type = final_state.get("request_type", "")
    warranty = final_state.get("warranty_info")
    part_order = final_state.get("part_order")
    appointment = final_state.get("appointment_info")

    if request_type == "warranty":
        if warranty and warranty.get("status") in ("in_warranty", "out_of_warranty"):
            return 0.9
        return 0.4   # unknown / 缺少序列号 / 查询失败
    if request_type == "parts":
        stock_text = (part_order or {}).get("stock_info", "")
        if not stock_text or any(w in stock_text for w in ("未找到", "暂不可用", "请在问题中提供")):
            return 0.5
        return 0.9
    if request_type == "appointment":
        return 0.9 if appointment else 0.5
    return 0.5


async def _stream_pipeline_agent(req: UnifiedChatRequest, resume: bool = False):
    """
    流式执行故障报修全流程（pipeline）。

    resume=False：推送计划 → 诊断 → (need_ticket) 创建工单 → 售后。
    resume=True：恢复被 interrupt 暂停的诊断，完成后继续创建工单 → 售后。
    """
    try:
        if not resume:
            yield _sse(_PIPELINE_PLAN)
            yield _sse({"type": "progress", "stage": "启动故障诊断，开始分析故障现象..."})
            session_state.set_pipeline_after(req.session_id, True)

        # ── 阶段一：故障诊断（支持 resume）───────────────────
        report_payload: dict = {}
        diag_need_ticket = False
        diag_confidence = 0.0
        async for step in supervisor.stream_diagnosis(
            req.message, req.session_id,
            customer_id=req.customer_id, device_model=req.device_model,
            resume=resume, resume_answer=req.message if resume else "",
        ):
            if step.type == "interrupt":
                # 诊断中断：等待用户回答，pipeline 标记保持 True
                payload = step.payload or {}
                yield _sse({
                    "type": "guidance",
                    "message": payload.get("question", "请补充更多故障信息。"),
                    "action_label": "", "action_url": "",
                })
                return
            if step.type == "result":
                report_payload = (step.payload or {}).get("report") or {}
                diag_need_ticket = bool((step.payload or {}).get("need_ticket"))
                diag_confidence = float((step.payload or {}).get("confidence", 0.0))
            for event in _diagnosis_step_to_sse(step):
                yield event

        # ── 阶段二：need_ticket 则创建工单 ───────────────────
        if not diag_need_ticket:
            session_state.clear(req.session_id)
            yield _sse({
                "type": "meta", "answer_mode": "pipeline",
                "need_ticket": False, "agent_chain": ["diagnosis"],
            })
            return

        # ── 阶段三：创建工单 ────────────────────────────────
        yield _sse({"type": "progress", "stage": "基于诊断结论创建工单中..."})
        ticket_data = _build_ticket_data_from_supervisor(
            report_payload,
            {"user_input": req.message, "customer_id": req.customer_id,
             "device_model": req.device_model, "device_sn": req.device_sn},
        )
        ticket = await supervisor.run_ticket(req.session_id, req.customer_id, ticket_data)
        if ticket.ticket_id:
            session_state.set_ticket_id(req.session_id, ticket.ticket_id)
            yield _sse({"type": "token", "content": ticket.reply})

        # ── 阶段四：售后协调 ────────────────────────────────
        yield _sse({"type": "progress", "stage": "查询售后信息中..."})
        after = await supervisor.run_after_sale(
            req.session_id, req.customer_id,
            device_sn=req.device_sn, request_type="warranty", message=req.message,
        )
        yield _sse({"type": "token", "content": after.reply})

        session_state.clear(req.session_id)
        yield _sse({
            "type": "meta", "answer_mode": "pipeline",
            "confidence": diag_confidence, "need_ticket": True,
            "ticket_id": ticket.ticket_id or "",
            "agent_chain": ["diagnosis", "ticket", "after_sale"],
        })

    except Exception as e:
        logger.error("unified_chat.pipeline_stream_error", error=str(e), exc_info=True)
        yield _sse({"type": "error", "message": "故障报修全流程服务异常，请稍后重试"})


# ── 工具函数（chat.py 内部）───────────────────────────────────

_TICKET_ID_RE = re.compile(r"(TK-\d{8}-\d{6})")


def _extract_ticket_id(text: str) -> str:
    """从消息中提取工单号 TK-YYYYMMDD-XXXXXX，未命中返回空串。"""
    m = _TICKET_ID_RE.search(text or "")
    return m.group(1) if m else ""


async def _query_ticket_info(ticket_id: str) -> str:
    """查询工单信息（渲染状态）。DB 异常返回空串。"""
    try:
        from backend.agents.ticket.repo import get_ticket_record
        from backend.dependencies import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            record = await get_ticket_record(session, ticket_id)
            if record is None:
                return ""
            return (
                f"工单 **{record.ticket_id}** 当前状态：**{record.status}**。\n"
                f"故障描述：{record.fault_description or '无'}\n"
                f"设备：{record.device_model or '未知'}"
                f"（SN: {record.device_sn or '未知'}）"
                + (f"\n诊断结论：{record.diagnosis_result}" if record.diagnosis_result else "")
            )
    except Exception as e:
        logger.warning("unified_chat.query_ticket_db_error", error=str(e))
        return ""


def _build_ticket_data_from_supervisor(report: dict, context: dict) -> dict:
    """复用 supervisor._build_ticket_data，避免循环依赖。"""
    from backend.supervisor import _build_ticket_data
    return _build_ticket_data(report, context)


def _render_after_sale_result(state: dict) -> str:
    """渲染售后图最终状态为文本（与 supervisor._render_after_sale_result 一致）。"""
    request_type = state.get("request_type", "")
    warranty = state.get("warranty_info")
    part_order = state.get("part_order")
    appointment = state.get("appointment_info")

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
