import logging
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form

from app.context.request.ContextRequest import ContextRequest
from app.context.services.vision_service import VisionService, EMOTION_LABELS
from app.analyze.s3_service import S3Uploader

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

try:
    s3_uploader = S3Uploader()
except ValueError as exc:
    logger.error("Failed to initialize S3Uploader: %s", exc)
    s3_uploader = None


@router.post("/")
async def analyze_context(request: ContextRequest):
    if not vision_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="분석 서비스가 초기화되지 않았습니다. OPENAI_API_KEY를 확인해주세요."
        )

    try:
        analysis = await vision_service.analyze_emotion(request.photo_url)
        analysis = dict(analysis)
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
        "emotion_labels": EMOTION_LABELS,
    }


@router.post("/upload", summary="노인 감정 분석 (이미지 업로드)")
async def analyze_context_upload(
    session_id: str = Form(...),
    user_id: str = Form(...),
    image_file: UploadFile = File(...),
):
    if not vision_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="감정 분석 서비스가 초기화되지 않았습니다. OPENAI_API_KEY를 확인해주세요."
        )

    if not s3_uploader:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="S3 업로드 설정이 완료되지 않았습니다. AWS 환경 변수를 확인해주세요."
        )

    if not image_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미지 파일명이 필요합니다."
        )

    content = await image_file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="빈 파일은 업로드할 수 없습니다."
        )

    try:
        key = s3_uploader.upload_image(
            content=content,
            session_id=session_id,
            filename=image_file.filename,
            content_type=image_file.content_type,
        )
        image_url = s3_uploader.build_public_url(key)
        analysis = await vision_service.analyze_emotion(image_url)
        analysis = dict(analysis)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unexpected error during context upload")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="이미지 업로드 중 오류가 발생했습니다."
        ) from exc

    return {
        "success": True,
        "session_id": session_id,
        "user_id": user_id,
        "photo_url": image_url,
        "analysis": analysis,
        "emotion_labels": EMOTION_LABELS,
    }

