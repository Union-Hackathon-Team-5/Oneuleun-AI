import logging

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from pydantic import BaseModel, HttpUrl, Field

from app.analyze.service import AnalyzeService
from app.services.analysis_service import AnalysisService
from app.models.analysis_models import (
    ComprehensiveAnalysisResult, SummaryCard, AlertInfo
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analysis"])

try:
    analyze_service = AnalyzeService()
except Exception as exc:  # pragma: no cover - defensive
    logger.error("Failed to initialise AnalyzeService: %s", exc)
    analyze_service = None

try:
    analysis_service = AnalysisService()
except Exception as exc:
    logger.error("Failed to initialise AnalysisService: %s", exc)
    analysis_service = None


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


@router.post("/upload", summary="상담 세션 종합 분석 (파일 업로드 + 대화 분석)")
async def analyze_session_with_upload(
    session_id: str = Form(...),
    user_id: str = Form(...),
    conversation: str = Form(..., description="AI 질문과 노인 응답이 포함된 대화 내용"),
    audio_file: UploadFile = File(...),
):
    if not analyze_service or not analysis_service:
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
        
        # TODO: 세션 ID로 이미지 URL 가져오기 (사용자가 구현 예정)
        # image_url = get_image_url_by_session_id(session_id)
        image_url = f"placeholder_image_url_for_session_{session_id}"
        
        # 대화 기반 종합 분석 (병렬 처리)
        comprehensive_analysis = await analysis_service.analyze_video_letter_comprehensive(conversation)
        
        # 종합 결과 생성
        summary_card = _generate_summary_card(comprehensive_analysis)
        alert_info = _generate_alert_info(comprehensive_analysis)
        
        return {
            "success": True,
            "session_id": session_id,
            "user_id": user_id,
            "conversation": conversation,
            "image_url": image_url,
            "audio_analysis": upload_result,
            "comprehensive_analysis": comprehensive_analysis.dict(),
            "summary_card": summary_card.dict(),
            "alert_info": alert_info.dict(),
        }
        
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to process uploaded audio and conversation analysis")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="업로드 및 분석 처리 중 오류가 발생했습니다.",
        ) from exc


def _generate_summary_card(analysis: ComprehensiveAnalysisResult) -> SummaryCard:
    """📊 오늘의 상태 요약 카드 생성"""
    from app.models.analysis_models import EmotionScore
    
    return SummaryCard(
        status_emoji=analysis.comprehensive_summary.status_emoji,
        status_text=analysis.comprehensive_summary.status_text,
        emotion_scores=EmotionScore(
            positive=analysis.emotion_analysis.positive,
            anxiety=analysis.emotion_analysis.anxiety,
            depression=analysis.emotion_analysis.depression
        ),
        main_summary=analysis.comprehensive_summary.main_summary,
        overall_mood=analysis.emotion_analysis.overall_mood
    )


def _generate_alert_info(analysis: ComprehensiveAnalysisResult) -> AlertInfo:
    """🚨 알림 정보 생성"""
    comprehensive = analysis.comprehensive_summary
    risk = analysis.risk_analysis
    
    alert_needed = comprehensive.alert_needed
    requires_immediate = comprehensive.requires_immediate_attention
    
    if not alert_needed:
        return AlertInfo(
            alert_type="none",
            message="현재 특별한 주의사항이 없습니다.",
            priority="보통",
            detected_keywords=[],
            immediate_concerns=[],
            recommended_actions=["정기적인 모니터링 유지"],
            requires_immediate_attention=False
        )
    
    alert_type = "urgent" if requires_immediate else "attention"
    
    # 알림 메시지 생성
    risk_keywords = risk.detected_keywords
    
    if requires_immediate:
        message = f"🚨 긴급 알림 - 위험 키워드 감지: {', '.join(risk_keywords[:3])}"
    else:
        message = f"📊 주의 필요 - {comprehensive.main_summary}"
    
    return AlertInfo(
        alert_type=alert_type,
        message=message,
        priority=comprehensive.priority_level,
        detected_keywords=risk_keywords,
        immediate_concerns=comprehensive.key_concerns,
        recommended_actions=comprehensive.recommended_actions,
        requires_immediate_attention=requires_immediate
    )
