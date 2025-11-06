import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import ValidationError

from app.models.analyze_upload_models import AnalyzeUploadRequest, AnalyzeUploadResponse
from app.services.fast_analysis_service import FastAnalysisService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analysis"])

try:
    fast_analysis_service = FastAnalysisService()
except Exception as exc:  # pragma: no cover - defensive
    logger.error("Failed to initialise FastAnalysisService: %s", exc)
    fast_analysis_service = None


@router.post("/upload", summary="다 끝나고 보내는 엔드포인트", response_model=AnalyzeUploadResponse)
async def analyze_session_with_upload(request: AnalyzeUploadRequest):
    if not fast_analysis_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="분석 서비스가 초기화되지 않았습니다.",
        )

    logger.info(
        "Received analysis request for session_id=%s user_id=%s",
        request.session_id,
        request.user_id,
    )

    try:
        llm_payload = await fast_analysis_service.generate_simple_status_report(
            conversation=request.conversation,
            session_id=request.session_id,
            user_id=request.user_id,
        )
        return AnalyzeUploadResponse.model_validate(llm_payload)
    except ValidationError as exc:
        logger.error("LLM 응답 구조가 유효하지 않습니다: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI 분석 응답이 예상과 다릅니다.",
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to process analysis upload", exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="분석 처리 중 오류가 발생했습니다.",
        ) from exc
