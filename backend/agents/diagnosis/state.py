"""
state - 故障诊断 Agent（Agent③）的状态定义

融合说明：
- 保留项目原有字段（device_model / fault_description / collected_symptoms / current_step /
  diagnosis_result / resolved），兼容既有调用方语义；
- 新增完整诊断流程字段（迁移自 EduAgent 课件 06-试卷批改 Agent 的 State 模式）：
  输入解析、三轨匹配（精确/模糊/LLM）、证据组装、追问循环、路由结果。
"""
from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class DiagnosisStep(BaseModel):
    """单条排查步骤。"""
    step_no: int = Field(description="步骤序号")
    action: str = Field(description="操作描述，如'检查电源线是否插紧'")
    expected_result: str = Field(description="预期结果")
    confirmation_required: bool = Field(default=True, description="是否需要客户确认此步结果")


class DiagnosisResult(BaseModel):
    """故障诊断的结构化输出（项目原有语义，兼容保留）。"""
    possible_causes: list[str] = Field(description="可能原因列表，按概率从高到低排序")
    steps: list[DiagnosisStep] = Field(description="排查步骤")
    required_tools: list[str] = Field(default_factory=list, description="排查所需工具")
    severity: str = Field(default="low", description="严重程度：low / medium / high / critical")
    recommendation: str = Field(default="", description="建议：联系工程师 / 自行处理 / 预约上门")


class CauseItem(BaseModel):
    """单个可能原因（迁移版诊断报告）。"""
    desc:        str   # 原因描述
    probability: str   # 高/中/低


class SolutionItem(BaseModel):
    """单个处理方案（迁移版诊断报告）。"""
    step:       str
    need_skill: bool   # True=需专业人员


class DiagnosisReport(BaseModel):
    """诊断报告（LLM 结构化输出，对应【诊断结论/可能原因/处理方案/是否需要工单】）。"""
    conclusion:    str
    causes:        list[CauseItem]
    solutions:     list[SolutionItem]
    need_ticket:   bool
    confidence:    float        # [0,1]，证据不足时给低分
    ticket_reason: str = ""     # need_ticket=True 时的原因


class ClarifyAnswer(BaseModel):
    """追问确认输入（interrupt 恢复时传入）。"""
    answer:     str                # 用户对追问的回答
    fault_code: str | None = None   # 用户补充的故障码


class DiagnosisState(TypedDict):
    """故障诊断 Agent 的完整状态。"""

    # ── 请求上下文（项目原有）──────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    device_model: str
    fault_description: str               # 客户描述的故障现象
    collected_symptoms: list[str]        # 已采集的症状列表
    current_step: int                    # 当前排查到第几步
    diagnosis_result: dict | None     # DiagnosisResult.model_dump()（兼容保留）
    resolved: bool                       # 是否已解决

    # ── 请求上下文（迁移新增）──────────────────────────────
    user_input:  str
    fault_code:  str | None
    image:       str | None
    image_desc:  str | None

    # ── 解析结果 ──────────────────────────────────────────────
    phenomena:   list[str]
    kb_hits:     list[dict]      # 知识库检索结果（hybrid_retrieve）

    # ── 三轨匹配结果 ──────────────────────────────────────────
    exact_match:    dict | None   # 第一轨：故障码精确匹配
    fuzzy_matches:  list[dict]       # 第二轨：现象模糊匹配
    llm_hypotheses: list[dict]       # 第三轨：LLM 推理假设

    # ── 组装上下文 ────────────────────────────────────────────
    evidence_context: str

    # ── 诊断报告（迁移版）─────────────────────────────────────
    report:      dict | None
    confidence:  float

    # ── 追问 / 人工确认 ───────────────────────────────────────
    clarify_question: str | None
    clarify_answer:   dict | None
    turn:             int

    # ── 最终结果 ──────────────────────────────────────────────
    need_ticket: bool
    finished:    bool

    # ── 降级标记 ──────────────────────────────────────────────
    fallback_used: bool
    structured_output: dict | None


# ──────────────────────────────────────────────────────────────
# 直接运行演示：python state.py
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== DiagnosisStep / DiagnosisResult（兼容保留的项目原有语义）===")
    result = DiagnosisResult(
        possible_causes=["负载过大或卡死", "供电电压过低"],
        steps=[
            DiagnosisStep(step_no=1, action="断电后手动盘车", expected_result="电机可转动"),
            DiagnosisStep(step_no=2, action="万用表检测三相供电电压", expected_result="380V±10%"),
        ],
        required_tools=["万用表"],
        severity="medium",
        recommendation="自行处理",
    )
    print(result.model_dump())

    print("\n=== CauseItem / SolutionItem / DiagnosisReport（迁移版诊断报告）===")
    report = DiagnosisReport(
        conclusion="电机过载保护触发",
        causes=[CauseItem(desc="负载过大或卡死", probability="高")],
        solutions=[SolutionItem(step="断电后手动盘车", need_skill=False)],
        need_ticket=False,
        confidence=0.9,
        ticket_reason="",
    )
    print(report.model_dump())

    print("\n=== 校验失败示例（缺失必填字段）===")
    try:
        DiagnosisReport(conclusion="缺字段")  # 缺 causes / solutions / need_ticket / confidence
    except Exception as e:
        print(f"  DiagnosisReport(conclusion='缺字段') → 校验失败: {type(e).__name__}")
