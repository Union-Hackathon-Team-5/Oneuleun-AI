import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, HttpUrl, Field

from app.analyze.service import AnalyzeService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analysis"])

try:
    analyze_service = AnalyzeService()
except Exception as exc:  # pragma: no cover - defensive
    logger.error("Failed to initialise AnalyzeService: %s", exc)
    analyze_service = None


class AnalyzeRequest(BaseModel):
    session_id: str = Field(..., description="상담 세션 ID")
    user_id: str = Field(..., description="노인 사용자 ID")
    conversation: str = Field(..., description="질문:응답 딕셔너리(JSON 문자열)")
    audio_url: HttpUrl = Field(..., description="S3 음성 데이터 URL")


@router.post("/", summary="상담 세션 분석")
async def analyze_session(request: AnalyzeRequest):
    if not analyze_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="분석 서비스가 초기화되지 않았습니다.",
        )

    try:
        shout_result = await analyze_service.detect_shout_from_url(str(request.audio_url))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    return {
        "success": True,
        "session_id": request.session_id,
        "user_id": request.user_id,
        "conversation": request.conversation,
        "audio_url": str(request.audio_url),
        "shout_detection": shout_result,
    }
