import logging

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from pydantic import BaseModel, HttpUrl, Field

from app.analyze.service import AnalyzeService
from app.services.caregiver_service import CaregiverService
from app.models.caregiver_models import CaregiverFriendlyResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analysis"])

try:
    analyze_service = AnalyzeService()
except Exception as exc:  # pragma: no cover - defensive
    logger.error("Failed to initialise AnalyzeService: %s", exc)
    analyze_service = None

try:
    caregiver_service = CaregiverService()
except Exception as exc:
    logger.error("Failed to initialise CaregiverService: %s", exc)
    caregiver_service = None


class AnalyzeRequest(BaseModel):
    session_id: str = Field(..., description="상담 세션 ID")
    user_id: str = Field(..., description="노인 사용자 ID")
    conversation: str = Field(..., description="질문:응답 딕셔너리(JSON 문자열)")
    audio_url: HttpUrl = Field(..., description="S3 음성 데이터 URL")


@router.post("/", summary="상담 세션 분석 (S3 URL)")
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


@router.post("/upload", summary="영상 편지 종합 분석 (보호자 친화적)", response_model=CaregiverFriendlyResponse)
async def analyze_session_with_upload(
    session_id: str = Form(...),
    user_id: str = Form(...),
    conversation: str = Form(..., description="AI 질문과 노인 응답이 포함된 대화 내용"),
    audio_file: UploadFile = File(...),
):
    if not analyze_service or not caregiver_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="분석 서비스가 초기화되지 않았습니다.",
        )

    try:
        # 기존 오디오 분석
        audio_bytes = await audio_file.read()
        upload_result = await analyze_service.upload_and_analyze_audio(
            session_id=session_id,
            audio_bytes=audio_bytes,
            filename=audio_file.filename,
            content_type=audio_file.content_type,
        )
        
        # TODO: 세션 ID로 이미지 분석 결과 가져오기 (사용자가 구현 예정)
        # image_analysis = get_image_analysis_by_session_id(session_id)
        image_analysis = _get_dummy_image_analysis(session_id)
        
        # 보호자 친화적 종합 리포트 생성
        caregiver_report = await caregiver_service.generate_caregiver_friendly_report(
            conversation=conversation,
            image_analysis=image_analysis,
            audio_analysis=upload_result,
            session_id=session_id,
            user_id=user_id
        )
        
        return caregiver_report
        
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to process caregiver-friendly analysis")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="영상 편지 분석 처리 중 오류가 발생했습니다.",
        ) from exc


def _get_dummy_image_analysis(session_id: str) -> dict:
    """더미 이미지 분석 결과 생성 (세션 ID 기반)"""
    import random
    
    # 다양한 더미 데이터 패턴
    dummy_patterns = [
        {
            "emotion": ["기쁨"],
            "summary": "노인이 밝은 표정으로 미소를 짓고 있습니다.",
            "concerns": []
        },
        {
            "emotion": ["슬픔"],
            "summary": "노인이 손으로 얼굴을 가리고 있어 슬픔을 표현하고 있습니다.",
            "concerns": ["우울증 우려"]
        },
        {
            "emotion": ["외로움", "슬픔"],
            "summary": "노인이 혼자 앉아 있으며 외로운 표정을 짓고 있습니다.",
            "concerns": ["사회적 고립", "우울증 우려"]
        },
        {
            "emotion": ["무기력함"],
            "summary": "노인이 기운이 없어 보이며 무기력한 상태입니다.",
            "concerns": ["식사 거부 징후", "건강 상태 악화 우려"]
        },
        {
            "emotion": ["분노"],
            "summary": "노인이 화난 표정을 짓고 있습니다.",
            "concerns": ["스트레스 증가", "혈압 상승 우려"]
        },
        {
            "emotion": ["행복", "기쁨"],
            "summary": "노인이 매우 밝고 행복한 표정으로 웃고 있습니다.",
            "concerns": []
        }
    ]
    
    # 세션 ID 기반으로 일관된 더미 데이터 선택
    random.seed(hash(session_id) % 1000)
    selected_pattern = random.choice(dummy_patterns)
    
    return {
        "analysis": selected_pattern,
        "confidence": random.randint(75, 95),
        "confidence_level": "높음",
        "confidence_comment": "모델이 감정 분류에 대해 상당한 확신을 갖고 있습니다."
    }
