import logging
from fastapi import APIRouter, HTTPException, status

from app.context.request.ContextRequest import ContextRequest
from app.services.vision_service import VisionService, BASE_EMOTIONS, EXTENDED_EMOTIONS, WARNING_SIGNS

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/context",
    tags=["context"]
)

try:
    vision_service = VisionService()
except ValueError as exc:
    logger.error("Failed to initialize VisionService: %s", exc)
    vision_service = None


@router.post("/")
async def analyze_context(request: ContextRequest):
    if not vision_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="분석 서비스가 초기화되지 않았습니다. OPENAI_API_KEY를 확인해주세요."
        )

    try:
        analysis = await vision_service.analyze_emotion(request.photo_url)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    return {
        "success": True,
        "session_id": request.session_id,
        "user_id": request.user_id,
        "photo_url": request.photo_url,
        "analysis": analysis,
        "categories": {
            "base_emotions": BASE_EMOTIONS,
            "extended_emotions": EXTENDED_EMOTIONS,
            "warning_signs": WARNING_SIGNS,
        },
    }
