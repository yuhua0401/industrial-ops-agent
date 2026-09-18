"""
diagnosis - 故障诊断 API

设备智能客服 — Agent③ 故障诊断接口（非流式）。
调用 build_diagnosis_graph()，支持 interrupt 追问与 resume 恢复。
"""
from fastapi import APIRouter, Depends, HTTPException, status
from langgraph.types import Command
from pydantic import BaseModel, Field

from backend.core.logger import get_logger
from backend.dependencies import get_current_user
from backend.supervisor import get_supervisor

router = APIRouter()
logger = get_logger(__name__)
supervisor = get_supervisor()


class DiagnosisRequest(BaseModel):
    """故障诊断请求。"""
    session_id:   str   = Field(..., description="会话 ID（作为诊断图 thread_id）")
    message:      str   = Field(..., min_length=1, max_length=2000, description="故障描述")
    customer_id:  str   = Field(default="", description="客户 ID（可选）")
    device_model: str   = Field(default="", description="设备型号（可选，辅助诊断）")
    fault_code:   str   = Field(default="", description="故障码（可选）")
    image:        str   = Field(default="", max_length=7_000_000,
                                description="故障图片 base64 dataURL（可选，触发视觉描述）")


class DiagnosisResponse(BaseModel):
    """故障诊断响应。"""
    status:            str        = Field(..., description="completed / needs_clarification")
    session_id:        str        = Field(..., description="会话 ID")
    report:            dict | None = Field(default=None, description="诊断报告（completed 时）")
    question:          str        = Field(default="", description="追问问题")
    need_ticket:       bool       = Field(default=False, description="是否需要创建工单")
    confidence:        float      = Field(default=0.0, description="诊断置信度")
    agent_chain:       list[str]  = Field(default_factory=list, description="经过的 Agent 链路")


@router.post("", response_model=DiagnosisResponse)
async def run_diagnosis(
    req: DiagnosisRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    故障诊断（非流式）。

    若同一 session_id 的诊断图停在 ask_clarify（get_state().next 非空），
    则本次 message 视为对追问的回答，自动 resume。
    """
    graph = supervisor.get_diagnosis_graph()
    config = {"configurable": {"thread_id": req.session_id}}

    # 判断是否需要 resume（线程停在 ask_clarify）
    try:
        snapshot = graph.get_state(config)
        need_resume = bool(snapshot.next)
    except Exception as e:
        logger.warning("diagnosis.api_get_state_error", error=str(e))
        need_resume = False

    if need_resume:
        try:
            resume_cmd = Command(
                resume={"answer": req.message, "fault_code": req.fault_code or None},
            )
            result = await graph.ainvoke(resume_cmd, config)
        except Exception as e:
            logger.error("diagnosis.api_resume_error", error=str(e), exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"code": "DIAGNOSIS_RESUME_ERROR", "message": str(e)},
            ) from e
    else:
        initial_state = supervisor._diagnosis_initial_state(
            req.message, req.session_id,
            customer_id=req.customer_id, device_model=req.device_model,
            fault_code=req.fault_code or None,
            image=req.image or None,
        )
        try:
            result = await graph.ainvoke(initial_state, config)
        except Exception as e:
            logger.error("diagnosis.api_error", error=str(e), exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"code": "DIAGNOSIS_ERROR", "message": str(e)},
            ) from e

    # 中断检测
    interrupts = result.get("__interrupt__")
    if interrupts:
        payload = (
            dict(interrupts[0].value)
            if hasattr(interrupts[0], "value")
            else dict(interrupts[0])
        )
        return DiagnosisResponse(
            status="needs_clarification",
            session_id=req.session_id,
            question=payload.get("question", "请补充更多故障信息。"),
            agent_chain=["diagnosis"],
        )

    report = result.get("structured_output") or result.get("report") or {}
    return DiagnosisResponse(
        status="completed",
        session_id=req.session_id,
        report=report,
        need_ticket=bool(result.get("need_ticket", report.get("need_ticket", False))),
        confidence=float(report.get("confidence", 0.0)),
        agent_chain=["diagnosis"],
    )
