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
from enum import Enum

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage

from backend.core.llm_factory import get_llm
from backend.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# Agent 类型枚举（对齐 Supervisor 状态机）
# ═══════════════════════════════════════════════════════════════

class AgentType(str, Enum):
    """Agent 类型枚举，与 Supervisor 路由表保持一致。"""
    KNOWLEDGE  = "knowledge"    # Agent② 产品知识问答
    DIAGNOSIS  = "diagnosis"    # Agent③ 故障诊断
    TICKET     = "ticket"       # Agent④ 工单管理
    AFTER_SALE = "after_sale"   # Agent⑤ 售后协调


class ExecutionMode(str, Enum):
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

# ── 各 Agent 节点进度提示 ──────────────────────────────────────
_PROGRESS_LABELS: dict[str, str] = {
    # 知识问答
    "retrieve":        "检索知识库中...",
    "rerank":          "精排文档中...",
    "generate_rag":    "生成回答中...",
    # 故障诊断
    "analyze_fault":   "分析故障现象中...",
    "match_diag_tree": "匹配诊断树中...",
    "generate_diag":   "生成诊断结论中...",
    # 工单
    "create_ticket":   "创建工单中...",
    "query_ticket":    "查询工单中...",
    # 售后
    "check_warranty":  "查询保修信息中...",
    "check_stock":     "查询配件库存中...",
    "book_appointment": "预约上门服务中...",
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
{{"label": "功能名", "reason": "一句话说明判断依据"}}

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

        if label not in _VALID_LABELS:
            logger.warning("unified_chat.llm_route_unknown_label", label=label, fallback="knowledge")
            label = "knowledge"

        logger.info("unified_chat.llm_route_result", label=label, reason=reason)

    except Exception as e:
        logger.warning("unified_chat.llm_route_failed", error=str(e), fallback="knowledge")
        label = "knowledge"
        reason = "路由判断异常，默认转入产品知识问答"

    return _RouteResult(
        label=label,
        agent_type=_LABEL_TO_AGENT[label],
        execution_mode=_LABEL_TO_MODE[label],
        confidence=0.85,
        reason=reason,
    )


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
            yield _sse(_PIPELINE_PLAN)
            # 推送完计划后，自动进入诊断阶段
            yield _sse({
                "type":    "progress",
                "stage":   "启动故障诊断，开始分析故障现象...",
            })
            async for event in _stream_diagnosis_agent(req):
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

    流程：query_classifier 二分类 → RAG 检索 → 精排 → LLM 生成
    """
    # TODO: 接入 Supervisor / Knowledge Agent Graph
    # from backend.supervisor import get_supervisor
    # supervisor = get_supervisor()
    # async for event in supervisor.run_knowledge(req.message, req.session_id):
    #     yield event

    # 占位实现：直接调用 LLM 做知识问答
    try:
        yield _sse({"type": "progress", "stage": "检索知识库中..."})

        llm = get_llm("knowledge", temperature=0)
        messages = [HumanMessage(content=(
            f"你是设备产品知识专家。请基于你的知识回答以下问题。\n"
            f"设备型号：{req.device_model or '通用'}\n"
            f"问题：{req.message}"
        ))]

        full_reply = ""
        async for chunk in llm.astream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                full_reply += chunk.content
                yield _sse({"type": "token", "content": chunk.content})

        yield _sse({
            "type":       "meta",
            "answer_mode": "direct",         # TODO: 接入 RAG 后改为 "rag"
            "confidence":  0.8,
            "sources":     [],
        })

    except Exception as e:
        logger.error("unified_chat.knowledge_stream_error", error=str(e), exc_info=True)
        yield _sse({"type": "error", "message": "知识问答服务异常，请稍后重试"})


async def _stream_diagnosis_agent(req: UnifiedChatRequest):
    """
    流式执行故障诊断 Agent（Agent③）。

    流程：提取故障信息 → 匹配诊断树 → LLM 推理根因 → 输出方案 + need_ticket 标记
    如果 need_ticket=True，在 meta 中附带诊断结论，供前端触发创建工单。
    """
    # TODO: 接入 Supervisor / Diagnosis Agent Graph
    # from backend.supervisor import get_supervisor
    # supervisor = get_supervisor()
    # async for event in supervisor.run_diagnosis(req.message, req.session_id):
    #     yield event

    try:
        yield _sse({"type": "progress", "stage": "分析故障现象中..."})

        llm = get_llm("diagnosis", temperature=0)
        prompt = (
            f"你是设备故障诊断专家。请分析以下故障并给出诊断结论。\n\n"
            f"设备型号：{req.device_model or '未指定'}\n"
            f"故障描述：{req.message}\n\n"
            f"请按以下格式输出：\n"
            f"1. 故障现象确认\n"
            f"2. 可能根因（按可能性排序）\n"
            f"3. 建议排查步骤\n"
            f"4. 解决方案\n"
            f"5. 是否需要创建报修工单（是/否）"
        )

        full_reply = ""
        async for chunk in llm.astream([HumanMessage(content=prompt)]):
            if hasattr(chunk, 'content') and chunk.content:
                full_reply += chunk.content
                yield _sse({"type": "token", "content": chunk.content})

        # 判断是否需要创建工单
        need_ticket = "是" in full_reply[-200:] and "是否需要创建报修工单" not in full_reply[-200:]

        yield _sse({
            "type":        "meta",
            "answer_mode": "diagnosis",
            "confidence":  0.8,
            "sources":     [],
            "need_ticket": need_ticket,
            "diagnosis_summary": full_reply[:500],
        })

    except Exception as e:
        logger.error("unified_chat.diagnosis_stream_error", error=str(e), exc_info=True)
        yield _sse({"type": "error", "message": "故障诊断服务异常，请稍后重试"})


async def _stream_ticket_agent(req: UnifiedChatRequest):
    """
    流式执行工单管理 Agent（Agent④）。

    支持：工单查询（按编号）/ 催单 / 由诊断结论创建工单。
    """
    # TODO: 接入 Supervisor / Ticket Agent Graph
    try:
        yield _sse({"type": "progress", "stage": "查询工单中..."})

        llm = get_llm("ticket", temperature=0)
        prompt = (
            f"你是工单管理助手。请根据用户需求处理工单相关事宜。\n\n"
            f"用户输入：{req.message}\n"
            f"客户 ID：{req.customer_id or '未提供'}\n\n"
            f"请判断用户意图（查工单/催工单/创建工单/其他），并给出相应回复。"
        )

        full_reply = ""
        async for chunk in llm.astream([HumanMessage(content=prompt)]):
            if hasattr(chunk, 'content') and chunk.content:
                full_reply += chunk.content
                yield _sse({"type": "token", "content": chunk.content})

        yield _sse({
            "type":        "meta",
            "answer_mode": "ticket",
            "confidence":  0.85,
            "sources":     [],
        })

    except Exception as e:
        logger.error("unified_chat.ticket_stream_error", error=str(e), exc_info=True)
        yield _sse({"type": "error", "message": "工单服务异常，请稍后重试"})


async def _stream_after_sale_agent(req: UnifiedChatRequest):
    """
    流式执行售后协调 Agent（Agent⑤）。

    支持：查询保修期 / 查询配件库存 / 预约上门维修。
    """
    # TODO: 接入 Supervisor / AfterSale Agent Graph
    try:
        yield _sse({"type": "progress", "stage": "查询售后信息中..."})

        llm = get_llm("ticket", temperature=0)   # 售后暂用 ticket 模型
        prompt = (
            f"你是售后协调助手。请根据用户需求处理售后相关事宜。\n\n"
            f"用户输入：{req.message}\n"
            f"客户 ID：{req.customer_id or '未提供'}\n"
            f"设备型号：{req.device_model or '未指定'}\n\n"
            f"可用工具：查询保修期、查询配件库存、预约上门维修。\n"
            f"请判断用户意图并给出相应回复。"
        )

        full_reply = ""
        async for chunk in llm.astream([HumanMessage(content=prompt)]):
            if hasattr(chunk, 'content') and chunk.content:
                full_reply += chunk.content
                yield _sse({"type": "token", "content": chunk.content})

        yield _sse({
            "type":        "meta",
            "answer_mode": "after_sale",
            "confidence":  0.85,
            "sources":     [],
        })

    except Exception as e:
        logger.error("unified_chat.after_sale_stream_error", error=str(e), exc_info=True)
        yield _sse({"type": "error", "message": "售后服务异常，请稍后重试"})
