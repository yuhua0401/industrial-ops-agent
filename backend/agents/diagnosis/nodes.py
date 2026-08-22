"""
nodes - 故障诊断 Agent（Agent③）的节点函数

融合说明：
- 保留项目原有入口语义（device_model / fault_description / collected_symptoms）；
- 完整流程迁移自 EduAgent 课件 06-试卷批改 Agent（exam/nodes.py）的
  「三轨并行 + HitL + 降级」架构：
    exam 客观题规则引擎   → 第一轨：诊断树故障码精确匹配（确定性，无 LLM）
    exam 简答题 LLM 评分   → 第二轨：现象模糊匹配（Jaccard 规则 + 置信度）
    exam 代码题 LLM 评估   → 第三轨：LLM 深度推理假设
    exam teacher_review(HitL) → 追问确认（interrupt，最多 2 轮）
    exam apply_teacher_decision → 合并用户补充信息后重新诊断
    exam publish_results      → 路由：need_ticket → Supervisor（交 ①号） / 结束
"""
import asyncio
import json
import re
from pathlib import Path

# 直接运行（python nodes.py）时项目根不在 sys.path，这里补上；作为包导入时无副作用。
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from backend.agents.diagnosis.state import DiagnosisState, DiagnosisReport
from backend.agents.diagnosis.prompts import (
    SYSTEM_PROMPT,
    REASON_PROMPT,
    DIAGNOSIS_REPORT_PROMPT,
    CLARIFY_PROMPT,
)
from backend.agents.diagnosis.diag_tree import DiagTree
from backend.agents.diagnosis.kb_client import KBClient
from backend.core.llm_factory import get_llm
from backend.core.logger import get_logger

logger = get_logger(__name__)

MAX_CLARIFY_TURNS = 2          # 追问最多 2 轮，超限建议转人工
FUZZY_THRESHOLD   = 0.5        # 现象模糊匹配置信度阈值（2 现象命中 1 个=0.5 仍进候选）
LLM_HYP_CONFIDENCE = 0.4       # 纯 LLM 推理（无树命中）时置信度上限
MAX_INPUT_CHARS   = 2000       # 用户输入长度上限（防超长文本拖垮 LLM 调用/注入）

# LLM 调用保护：单次超时 + 有限重试（对齐 backend.core.retry 的策略，但保留本模块
# 自身的 try/except 降级语义——失败由调用方走模板兜底/空列表/转人工，而非 retry 的
# AgentFallbackHandler 返回（其返回结构与诊断节点期望的 LLM 消息不同）。
LLM_TIMEOUT_SECONDS = 60.0
LLM_MAX_RETRIES     = 2
LLM_RETRY_DELAYS    = [1.0, 3.0]

# 诊断树数据文件：使用完整诊断树（245 节点，含手册维修实例自动转换节点）。
# 数据由 scripts/extract_diag_examples.py + scripts/convert_examples_to_tree.py 生成，
# 路径相对项目根定位（不依赖 cwd）。
_DIAG_TREE_DATA = Path(__file__).resolve().parents[3] / "data" / "diag_tree_full.yaml"

# 全局诊断树单例（⑤号维护的 YAML 数据文件）
_diag_tree: DiagTree | None = None


def _get_diag_tree() -> DiagTree:
    """懒加载诊断树（进程内单例）。"""
    global _diag_tree
    if _diag_tree is None:
        _diag_tree = DiagTree()
        if _DIAG_TREE_DATA.exists():
            _diag_tree.load_yaml(str(_DIAG_TREE_DATA))
        else:
            logger.warning("diag_tree.data_missing", path=str(_DIAG_TREE_DATA))
    return _diag_tree


def _get_message_content(msg) -> str:
    """统一获取消息文本内容（兼容 text 属性和 content 属性）。"""
    if hasattr(msg, "text") and not callable(getattr(msg, "text", None)):
        return msg.text
    if isinstance(msg.content, str):
        return msg.content
    return str(msg.content)


async def _ainvoke_llm(llm, messages) -> Any:
    """
    带超时 + 有限重试的 LLM 调用（项目约定：模型层 max_retries=0，重试在此统一处理）。

    全部尝试失败后抛出最后一次异常，由各调用方 try/except 走既有降级路径
    （LLM 轨→空列表 / 报告→模板兜底 / 追问→转人工），保持诊断节点降级语义不变。
    """
    last_exc: Exception | None = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            return await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=LLM_TIMEOUT_SECONDS,
            )
        except Exception as e:
            last_exc = e
            if attempt < LLM_MAX_RETRIES:
                logger.warning(
                    "llm.attempt_failed", attempt=attempt + 1,
                    max_retries=LLM_MAX_RETRIES, delay=LLM_RETRY_DELAYS[attempt],
                    error=str(e),
                )
                await asyncio.sleep(LLM_RETRY_DELAYS[attempt])
    raise last_exc


# ──────────────────────────────────────────────────────────────
# 节点1：parse_input — 输入解析（文本/故障码/图片）
# ──────────────────────────────────────────────────────────────

_FAULT_CODE_RE = re.compile(r"[A-Za-z]{1,4}[-]?\d{2,4}")   # 如 E001 / ERR-1024

# 常见句尾/句首口语词缀：剥离后便于与诊断树现象精确/子串匹配
# 注意：这些词本身不再列入停用词（剥离后剩余部分仍可能是有意义的词）
_PHENOMENON_STOPWORDS = {"设备", "机器", "出现", "报", "故障", "问题", "什么", "怎么"}
_PHENOMENON_AFFIXES = ("了", "过", "在", "的", "吗", "呢", "啊", "吧", "着", "一下", "出现")


def _strip_phenomenon_affixes(word: str) -> str:
    """去掉口语词缀（了/过/在/的/吗…），保留主干。例：'电机不转了' → '电机不转'。"""
    w = word
    changed = True
    while changed and w:
        changed = False
        for af in _PHENOMENON_AFFIXES:
            if w.endswith(af) and len(w) > len(af):
                w = w[: -len(af)]
                changed = True
                break
    return w


def _extract_fault_code(user_input: str, supplied: str | None) -> str | None:
    """故障码提取：优先使用入参，其次从文本正则提取。"""
    if supplied and supplied.strip():
        return supplied.strip().upper()
    m = _FAULT_CODE_RE.search(user_input or "")
    return m.group(0).upper() if m else None


def _extract_phenomena(user_input: str) -> list[str]:
    """
    现象关键词抽取：去除停用词后按标点/空格切分，剥离口语词缀，
    保留 2-6 字片段（词缀剥离后 <2 字的丢弃，避免噪声词）。

    增强：额外用「部件词 + 故障表现词」词典从原文中提取原子关键词
    （如 '主轴不转'、'乱字符'、'刀架越位'），与诊断树节点的现象词对齐，
    避免整句片段因与节点原子词无公共子串而漏匹配。
    """
    if not user_input:
        return []
    cleaned = re.sub(r"[，。！？、；：,.!?;:\s]+", " ", user_input)
    words = []
    for seg in cleaned.split(" "):
        seg = seg.strip()
        if not seg or seg in _PHENOMENON_STOPWORDS or len(seg) > 12:
            continue
        stem = _strip_phenomenon_affixes(seg)
        if len(stem) < 2 or stem in _PHENOMENON_STOPWORDS:
            continue
        words.append(stem)
    # 词典原子词补充（去重保序）
    atoms = _extract_atom_phenomena(user_input)
    for a in atoms:
        if a not in words:
            words.append(a)
    return words[:6]


# 部件词表（与 scripts/convert_examples_to_tree.py 保持一致，最长优先匹配）
_PART_WORDS = [
    "光栅尺", "滚珠丝杠", "机械手", "编码器", "继电器", "伺服单元", "电动机",
    "步进电动机", "主轴", "刀架", "刀库", "工作台", "显示屏", "丝杠",
    "导轨", "换刀", "卡盘", "润滑", "电源", "系统", "刀塔", "冷却",
]
# 故障表现词表
_FAULT_WORDS = [
    "乱字符", "不能起动", "不能移动", "不能控制", "不能运行", "不能输入",
    "不到位", "越位", "失步", "超差", "失灵", "抖动", "振动", "报警",
    "烧坏", "断线", "失控", "无显示", "噪声", "错位", "超程", "卡死",
    "松动", "漏油", "掉电", "停止", "失效", "异响", "不转", "不准",
    "不稳定", "不动作", "时转时不转", "不进给",
    "发热",
    "不显示",
    "转不动",
    "不能转",
    "不动作",
    "不工作",
    "失控",
    "超程",
    "爬行",
    "窜动",
    "跳动",
    "摆动",
    "堵转",
    "闷车",
    "闪断",
    "误动作",
    "不换刀",
    "卡刀",
    "掉刀",
    "撞刀",
    "过切",
    "欠切",
    "偏心",
]


def _extract_atom_phenomena(user_input: str) -> list[str]:
    """从原文中提取「部件词+故障词」原子关键词。"""
    p = "".join(re.findall(r"[\u4e00-\u9fff]+", user_input or ""))
    if not p:
        return []
    atoms = []
    for part in _PART_WORDS:
        idx = p.find(part)
        if idx == -1:
            continue
        tail = p[idx + len(part): idx + len(part) + 12]
        for fault in _FAULT_WORDS:
            if fault in tail:
                atoms.append(part + fault)
                break
    for fault in _FAULT_WORDS:
        if fault in p:
            atoms.append(fault)
    # 去重保序，最多 4 个
    seen, out = set(), []
    for a in atoms:
        if a not in seen and 2 <= len(a) <= 10:
            seen.add(a)
            out.append(a)
    return out[:4]


async def parse_input_node(state: DiagnosisState) -> dict:
    """
    解析用户输入：提取故障码 + 现象关键词；图片走降级链
    （LLM 视觉描述 → OCR → 置空，失败不阻断流程）。
    兼容项目原有字段：fault_description 作为 user_input 兜底，device_model 透传。
    """
    user_input = state.get("user_input") or state.get("fault_description") or ""
    # 输入长度上限：防超长文本拖垮 LLM 调用（超长截断并记录，不阻断流程）
    if len(user_input) > MAX_INPUT_CHARS:
        logger.warning("parse_input.input_truncated",
                       original_len=len(user_input), max_chars=MAX_INPUT_CHARS)
        user_input = user_input[:MAX_INPUT_CHARS]

    # 图片降级链：一期先置空 image_desc，后续接入视觉/OCR 后替换
    image_desc = None
    if state.get("image"):
        # TODO(⑤): 接入视觉模型 / PaddleOCR，产出 image_desc
        logger.warning("parse_input.image_not_implemented", image=state["image"])

    return {
        "fault_code": _extract_fault_code(user_input, state.get("fault_code")),
        "phenomena":  _extract_phenomena(user_input),
        "image_desc": image_desc,
        "user_input": user_input,
    }


# ──────────────────────────────────────────────────────────────
# 节点2：load_diag_tree — 加载诊断树 + 知识库检索（调③）
# ──────────────────────────────────────────────────────────────

async def load_diag_tree_node(state: DiagnosisState) -> dict:
    """
    加载诊断树（⑤号维护）并调用知识库 hybrid_retrieve 检索。
    知识库服务不可用时优雅降级：kb_hits 置空，流程继续走诊断树，不中断。
    """
    user_input  = state.get("user_input", "")
    fault_code  = state.get("fault_code") or ""
    device_model = state.get("device_model", "")
    query = f"{fault_code} {user_input}".strip() or device_model or "故障诊断"

    kb_hits = []
    try:
        kb_hits = await KBClient.search(
            query=query,
            device_model=device_model or None,
            top_k=10,
        )
        logger.info("load_diag_tree.kb_ok", hits=len(kb_hits))
    except Exception as e:
        logger.warning("load_diag_tree.kb_failed", error=str(e))

    return {"kb_hits": kb_hits}


# ──────────────────────────────────────────────────────────────
# 三轨并行：精确匹配 / 模糊匹配 / LLM 推理
# ──────────────────────────────────────────────────────────────

def _run_exact_track(tree: DiagTree, fault_code: str | None) -> dict | None:
    """第一轨：故障码精确匹配（确定性规则，类似 exam 客观题规则引擎）。"""
    node = tree.exact_match(fault_code)
    if node is None:
        return None
    return {"node_id": node.node_id, "name": node.name, "confidence": 0.95,
            "source": "exact", "data": node.to_dict()}


def _run_fuzzy_track(tree: DiagTree, phenomena: list[str]) -> list[dict]:
    """第二轨：现象模糊匹配（Jaccard 置信度，类似 exam 规则引擎的确定性分支）。"""
    matches = tree.fuzzy_match(phenomena, threshold=FUZZY_THRESHOLD)
    return [{"node_id": n.node_id, "name": n.name, "confidence": round(s, 4),
             "source": "fuzzy", "data": n.to_dict()} for n, s in matches]


async def _run_llm_track(state: DiagnosisState) -> list[dict]:
    """
    第三轨：LLM 深度推理。树未命中时生成假设 + 置信度。
    失败降级为空列表。
    """
    try:
        llm = get_llm("diagnosis", temperature=0.3)
        prompt = (
            "用户报修如下故障，请推断 2-3 个最可能的故障原因及置信度（0-1）。\n"
            f"设备型号：{state.get('device_model') or '未知'}\n"
            f"故障码：{state.get('fault_code') or '未知'}\n"
            f"现象：{'、'.join(state.get('phenomena', [])) or '未知'}\n"
            "输出 JSON：{\"hypotheses\": [{\"desc\": \"...\", \"confidence\": 0.0}]}\n"
            "直接输出 JSON，不要 Markdown 代码块。"
        )
        resp = await _ainvoke_llm(llm, [HumanMessage(content=prompt)])
        raw = _get_message_content(resp).strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        hyps = data.get("hypotheses", [])
        return [{"desc": h.get("desc", ""), "confidence": float(h.get("confidence", 0)),
                 "source": "llm"} for h in hyps][:3]
    except Exception as e:
        logger.warning("run_llm_track.failed", error=str(e))
        return []


async def run_diag_tracks_node(state: DiagnosisState) -> dict:
    """
    三轨并行诊断：任一轨失败不影响其他轨（asyncio.gather return_exceptions）。
    对齐 exam.run_three_tracks_node 的容错模式。
    """
    tree       = _get_diag_tree()
    fault_code = state.get("fault_code")
    phenomena  = state.get("phenomena", [])

    raw = await asyncio.gather(
        asyncio.to_thread(_run_exact_track, tree, fault_code),   # 轨1（同步包装）
        asyncio.to_thread(_run_fuzzy_track, tree, phenomena),    # 轨2（同步包装）
        _run_llm_track(state),                                   # 轨3（async）
        return_exceptions=True,
    )

    exact   = raw[0] if not isinstance(raw[0], Exception) else None
    fuzzy   = raw[1] if not isinstance(raw[1], Exception) else []
    llm_hyp = raw[2] if not isinstance(raw[2], Exception) else []

    for name, exc in zip(["exact", "fuzzy", "llm"], raw):
        if isinstance(exc, Exception):
            logger.error(f"diag_tracks.{name}_failed", error=str(exc))

    logger.info("diag_tracks.done", exact=exact is not None,
                fuzzy=len(fuzzy), llm=len(llm_hyp))
    return {"exact_match": exact, "fuzzy_matches": fuzzy, "llm_hypotheses": llm_hyp}


# ──────────────────────────────────────────────────────────────
# 节点4：assemble_context — 组装诊断证据上下文
# ──────────────────────────────────────────────────────────────

def _format_candidates(exact: dict | None, fuzzy: list[dict], hyps: list[dict]) -> str:
    """把三轨结果格式化为给 LLM 的上下文文本。"""
    lines = []
    if exact:
        d = exact["data"]
        lines.append(f"[精确命中] {exact['name']} (置信度 {exact['confidence']})")
        lines.append(f"  可能原因：{'；'.join(c['desc'] for c in d.get('causes', []))}")
        lines.append(f"  处理方案：{'；'.join(s['step'] for s in d.get('solutions', []))}")
    if fuzzy:
        for m in fuzzy[:3]:
            lines.append(f"[模糊命中] {m['name']} (置信度 {m['confidence']})")
    if hyps:
        for h in hyps:
            lines.append(f"[LLM假设] {h['desc']} (置信度 {h['confidence']})")
    if not lines:
        lines.append("（无诊断树命中，需 LLM 兜底推理）")
    return "\n".join(lines)


async def assemble_context_node(state: DiagnosisState) -> dict:
    """组装证据上下文（树节点 + kb_hits 前3条），供报告生成节点使用。"""
    candidates = _format_candidates(
        state.get("exact_match"), state.get("fuzzy_matches", []), state.get("llm_hypotheses", []))
    kb_text = "\n".join(
        f"- [{h.get('source', '知识库')}] {h.get('content', '')[:200]}"
        for h in state.get("kb_hits", [])[:3]
    ) or "（无知识库命中）"
    return {"evidence_context": f"【诊断树/推理】\n{candidates}\n\n【知识库】\n{kb_text}"}


# ──────────────────────────────────────────────────────────────
# 节点5：generate_report — LLM 合成诊断报告（结构化输出）
# ──────────────────────────────────────────────────────────────

async def generate_report_node(state: DiagnosisState) -> dict:
    """
    两步生成：先 Think 推理，再结构化输出诊断报告。
    LLM 失败/JSON 解析失败 → 模板兜底（直接用命中节点的 causes/solutions 渲染）。
    对齐 exam._review_one_subjective 的两步 + exam 的 JSON 降级模式。
    """
    has_tree_hit = state.get("exact_match") is not None or bool(state.get("fuzzy_matches"))
    reasoning_trace = ""
    try:
        think_prompt = REASON_PROMPT.format(
            user_input=state.get("user_input", ""),
            fault_code=state.get("fault_code") or "未知",
            phenomena="、".join(state.get("phenomena", [])),
            matched_nodes=state.get("evidence_context", ""),
            kb_hits="\n".join(h.get("content", "")[:200] for h in state.get("kb_hits", [])[:3]) or "（无）",
        )
        think_llm = get_llm("diagnosis", temperature=0)
        resp = await _ainvoke_llm(think_llm, [HumanMessage(content=think_prompt)])
        reasoning_trace = _get_message_content(resp).strip()
    except Exception as e:
        logger.warning("generate_report.think_failed", error=str(e))

    report: dict | None = None
    try:
        report_prompt = DIAGNOSIS_REPORT_PROMPT.format(
            user_input=state.get("user_input", ""),
            fault_code=state.get("fault_code") or "未知",
            phenomena="、".join(state.get("phenomena", [])),
            matched_nodes=state.get("evidence_context", ""),
            kb_hits="\n".join(h.get("content", "")[:200] for h in state.get("kb_hits", [])[:3]) or "（无）",
            reasoning_trace=reasoning_trace or "（推理失败）",
        )
        llm = get_llm("diagnosis", temperature=0)
        resp = await _ainvoke_llm(llm, [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=report_prompt),
        ])
        raw = _get_message_content(resp).strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        # Pydantic 校验（对齐 exam 的 structured_output 校验思路）
        validated = DiagnosisReport(**data)
        report = validated.model_dump()
        # 树命中时，LLM 不允许把置信度抬得比树匹配更高
        if has_tree_hit:
            report["confidence"] = min(report["confidence"], 0.95)
        else:
            report["confidence"] = min(report["confidence"], LLM_HYP_CONFIDENCE)
    except Exception as e:
        logger.warning("generate_report.llm_failed", error=str(e), fallback=True)
        report = _template_fallback(state, has_tree_hit)

    # fallback_used 仅在真正走了模板兜底时置 True（_template_fallback 会打 _template 标记）。
    # 注意：无树命中但 LLM 正常出报告不算降级，只是纯 LLM 推理（置信度已被 LLM_HYP_CONFIDENCE 压制）。
    return {"report": report, "fallback_used": report.get("_template") is not None}


def _template_fallback(state: DiagnosisState, has_tree_hit: bool) -> dict:
    """模板兜底：直接用命中节点数据渲染报告，标记 _template=True。"""
    exact = state.get("exact_match")
    fuzzy = state.get("fuzzy_matches", [])
    # 注意运算符优先级：exact 优先，其次 fuzzy 第一条；fuzzy 为空时只看 exact
    node_data = exact or (fuzzy[0] if fuzzy else None)
    if not node_data:
        return {
            "conclusion": "无法自动诊断，建议转人工处理。",
            "causes": [], "solutions": [],
            "need_ticket": True, "confidence": 0.0,
            "ticket_reason": "诊断信息不足或链路故障，需人工介入",
            "_template": True,
        }
    d = node_data["data"]
    report = {
        "conclusion": f"诊断结论：{node_data['name']}",
        "causes": d.get("causes", []),
        "solutions": d.get("solutions", []),
        "need_ticket": d.get("need_ticket", False),
        "confidence": node_data["confidence"] if has_tree_hit else LLM_HYP_CONFIDENCE,
        "ticket_reason": "诊断树标记需上门处理" if d.get("need_ticket") else "",
        "_template": True,
    }
    return report


# ──────────────────────────────────────────────────────────────
# 节点6：check_sufficiency — 判断是否需要追问
# ──────────────────────────────────────────────────────────────

def _pick_clarify_question(state: DiagnosisState) -> str | None:
    """从命中节点的 questions 里取未问过的第一个；无命中节点则返回 None（转人工）。"""
    exact = state.get("exact_match")
    fuzzy = state.get("fuzzy_matches", [])
    # 注意运算符优先级：exact 优先，其次 fuzzy 第一条；fuzzy 为空时只看 exact
    node_data = exact or (fuzzy[0] if fuzzy else None)
    if not node_data or state.get("turn", 0) >= MAX_CLARIFY_TURNS:
        return None
    asked = {state.get("clarify_question")}
    for q in node_data["data"].get("questions", []):
        if q not in asked:
            return q
    return None


async def check_sufficiency_node(state: DiagnosisState) -> dict:
    """
    判断诊断是否可信：
      - 树命中 + 置信度足够 → 直接出报告
      - 树命中但存在未确认的追问点 或 无命中 → 生成追问
      - 追问已满 MAX_CLARIFY_TURNS → 强制出报告/转人工
    """
    report = state.get("report") or {}
    confidence = report.get("confidence", 0.0)

    if confidence >= 0.7 and report.get("need_ticket") is not None:
        return {"clarify_question": None}

    question = None
    if confidence < 0.5 or not state.get("exact_match"):
        # 信息不足：优先用节点预设追问；没有则 LLM 生成
        question = _pick_clarify_question(state)
        if question is None and state.get("turn", 0) < MAX_CLARIFY_TURNS:
            question = await _llm_generate_question(state)
    return {"clarify_question": question}


async def _llm_generate_question(state: DiagnosisState) -> str | None:
    """LLM 生成追问问题（CLARIFY_PROMPT）。失败返回 None → 转人工。"""
    try:
        prompt = CLARIFY_PROMPT.format(
            user_input=state.get("user_input", ""),
            phenomena="、".join(state.get("phenomena", [])),
            candidates=state.get("evidence_context", "")[:500],
        )
        llm = get_llm("diagnosis", temperature=0.3)
        resp = await _ainvoke_llm(llm, [HumanMessage(content=prompt)])
        return _get_message_content(resp).strip()[:100]
    except Exception as e:
        logger.warning("generate_question.failed", error=str(e))
        return None


# ──────────────────────────────────────────────────────────────
# 节点7：ask_clarify — HitL 追问暂停点
# ──────────────────────────────────────────────────────────────

async def ask_clarify_node(state: DiagnosisState) -> dict:
    """
    Human-in-the-Loop 暂停点（对齐 exam.teacher_review_node 的 interrupt 模式）。
    图在此暂停，等待用户回答；恢复后 interrupt() 返回值即澄清答案。
    """
    question = state.get("clarify_question") or "请补充更多故障信息。"
    clarify_answer = interrupt({
        "question": question,
        "fault_code": state.get("fault_code"),
        "evidence": state.get("evidence_context", "")[:500],
        "message": "请回答上述问题以继续诊断。",
    })
    logger.info("ask_clarify.resumed", question=question, turn=state.get("turn", 0) + 1)
    return {"clarify_answer": clarify_answer}


# ──────────────────────────────────────────────────────────────
# 节点8：apply_clarify_answer — 合并用户补充信息，重新诊断
# ──────────────────────────────────────────────────────────────

async def apply_clarify_answer_node(state: DiagnosisState) -> dict:
    """
    合并用户对追问的回答：追加入口、更新 fault_code、turn+1。
    对齐 exam.apply_teacher_decision_node 的「合并外部输入」思路。
    """
    answer = state.get("clarify_answer") or {}
    new_fault = answer.get("fault_code") or state.get("fault_code")

    merged_input = state.get("user_input", "") + " " + str(answer.get("answer", ""))
    phenomena = _extract_phenomena(merged_input)

    return {
        "user_input": merged_input,
        "phenomena":  phenomena,
        "fault_code": new_fault,
        "turn":       state.get("turn", 0) + 1,
        "clarify_answer": None,       # 已消费
        "report": None,               # 清空旧报告，重新诊断
        "exact_match": None,
        "fuzzy_matches": [],
        "llm_hypotheses": [],
    }


# ──────────────────────────────────────────────────────────────
# 节点9：route_next — 结果路由
# ──────────────────────────────────────────────────────────────

async def route_next_node(state: DiagnosisState) -> dict:
    """
    最终路由：
      - need_ticket=True → 返回 Supervisor（由 ④ 调 ①号 create_ticket，⑤ 不直接调①）
      - 否则 → 返回用户，结束
    """
    report = state.get("report") or {}
    return {
        "need_ticket":      bool(report.get("need_ticket")),
        "finished":         True,
        "structured_output": report,
    }


# ──────────────────────────────────────────────────────────────
# 直接运行演示：python nodes.py（注入假 LLM / 假知识库，离线可跑）
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from langchain_core.messages import AIMessage

    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    class _FakeLLM:
        """离线替身：按 prompt 关键字返回固定内容，模拟 LLM 输出。"""

        async def ainvoke(self, messages):
            text = messages[-1].content
            if "hypotheses" in text:
                return AIMessage(content=json.dumps({
                    "hypotheses": [
                        {"desc": "负载过大或机械卡死", "confidence": 0.6},
                        {"desc": "供电电压波动", "confidence": 0.3},
                    ]
                }, ensure_ascii=False))
            if "JSON" in text:
                return AIMessage(content=json.dumps({
                    "conclusion": "电机过载保护触发，建议优先排查负载与供电",
                    "causes": [{"desc": "负载过大或机械卡死", "probability": "高"}],
                    "solutions": [{"step": "断电后手动盘车确认是否卡死", "need_skill": False}],
                    "need_ticket": False,
                    "confidence": 0.9,
                    "ticket_reason": "",
                }, ensure_ascii=False))
            if "追问" in text:
                return AIMessage(content="故障发生时设备是否处于满载状态？")
            return AIMessage(content="故障现象与诊断树 D001 节点吻合，根因大概率是负载过大。")

    async def _fake_search(*args, **kwargs):
        return [{"content": "电机过载排查手册：先检查负载，再测三相电压……", "source": "kb", "score": 0.9}]

    async def demo():
        global get_llm

        def _llm_factory(*args, **kwargs):
            return _FakeLLM()

        get_llm = _llm_factory
        KBClient.search = _fake_search

        state = {
            "session_id": "demo", "device_model": "CNC-1000", "fault_description": "",
            "user_input": "设备报E001，电机不转了", "fault_code": None,
            "phenomena": [], "turn": 0, "messages": [],
            "image": None, "image_desc": None, "kb_hits": [],
            "exact_match": None, "fuzzy_matches": [], "llm_hypotheses": [],
            "evidence_context": "", "report": None, "confidence": 0.0,
            "clarify_question": None, "clarify_answer": None,
            "need_ticket": False, "finished": False,
            "fallback_used": False, "structured_output": None,
        }
        print("=== nodes 节点串行演示（parse → load → tracks → context → report → sufficiency → route）===")

        state.update(await parse_input_node(state))
        print(f"1. parse_input       → fault_code={state['fault_code']}, phenomena={state['phenomena']}")

        state.update(await load_diag_tree_node(state))
        print(f"2. load_diag_tree    → kb_hits={len(state['kb_hits'])} 条")

        state.update(await run_diag_tracks_node(state))
        print(f"3. run_diag_tracks   → exact={state['exact_match']['node_id'] if state['exact_match'] else None}, "
              f"fuzzy={[m['node_id'] for m in state['fuzzy_matches']]}, llm={len(state['llm_hypotheses'])} 条")

        state.update(await assemble_context_node(state))
        print(f"4. assemble_context  → evidence_context {len(state['evidence_context'])} 字符")

        state.update(await generate_report_node(state))
        print(f"5. generate_report   → conclusion={state['report']['conclusion'][:25]}…, "
              f"confidence={state['report']['confidence']}, fallback={state.get('fallback_used')}")

        state.update(await check_sufficiency_node(state))
        print(f"6. check_sufficiency → clarify_question={state['clarify_question']}")

        state.update(await route_next_node(state))
        print(f"7. route_next        → need_ticket={state['need_ticket']}, finished={state['finished']}")

    asyncio.run(demo())
