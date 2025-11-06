from typing import List, Dict, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class StatusOverview(BaseModel):
    """🎯 1순위: 한눈에 상태 파악"""
    alert_level: Literal["urgent", "caution", "normal"] = Field(..., description="알림 수준")
    alert_badge: str = Field(..., description="알림 뱃지 (🚨, ⚠️, 😊)")
    alert_title: str = Field(..., description="알림 제목")
    alert_subtitle: str = Field(..., description="알림 부제목")
    status_color: str = Field(..., description="상태 색상 (#FF4444, #FF8800, #44FF44)")


class TodaySummary(BaseModel):
    """🎯 2순위: 오늘 어머니 상태"""
    headline: str = Field(..., description="오늘 상태 한줄 요약")
    mood_score: int = Field(..., ge=0, le=100, description="기분 점수 (0-100)")
    mood_label: str = Field(..., description="기분 라벨")
    mood_emoji: str = Field(..., description="기분 이모지")
    energy_score: int = Field(..., ge=0, le=100, description="활력 점수")
    pain_score: int = Field(..., ge=0, le=100, description="통증 점수 (높을수록 아픔)")
    mother_voice: List[str] = Field(..., description="어머니 목소리 직접 인용")


class KeyConcern(BaseModel):
    """주요 걱정거리 개별 항목"""
    concern_id: int = Field(..., description="걱정거리 ID")
    type: Literal["건강", "안전", "정서", "생활"] = Field(..., description="걱정 유형")
    icon: str = Field(..., description="아이콘")
    severity: Literal["urgent", "caution", "normal"] = Field(..., description="심각도")
    title: str = Field(..., description="걱정거리 제목")
    description: str = Field(..., description="구체적 설명")
    detected_from: List[str] = Field(..., description="감지 출처")
    urgency_reason: str = Field(..., description="왜 긴급한지")


class UrgentAction(BaseModel):
    """긴급 조치"""
    action_id: int = Field(..., description="조치 ID")
    priority: Literal["최우선", "긴급", "중요"] = Field(..., description="우선순위")
    icon: str = Field(..., description="아이콘")
    title: str = Field(..., description="조치 제목")
    reason: str = Field(..., description="왜 필요한지")
    detail: str = Field(..., description="구체적 설명")
    deadline: str = Field(..., description="언제까지")
    estimated_time: str = Field(..., description="소요 시간")
    suggested_topics: Optional[List[str]] = Field(default=None, description="대화 예시")
    options: Optional[List[str]] = Field(default=None, description="선택 옵션들")
    booking_button: Optional[bool] = Field(default=False, description="예약 버튼 표시")


class ActionPlan(BaseModel):
    """🎯 4순위: 지금 무엇을 해야 하나"""
    urgent_actions: List[UrgentAction] = Field(..., description="긴급 조치들")
    this_week_actions: List[UrgentAction] = Field(..., description="이번 주 조치들")
    long_term_actions: List[UrgentAction] = Field(..., description="장기 조치들")


class ConversationTopic(BaseModel):
    """대화 주제 분석"""
    topic: str = Field(..., description="주제")
    summary: str = Field(..., description="요약")
    concern_level: Literal["urgent", "caution", "normal"] = Field(..., description="우려 수준")


class EmotionTimeline(BaseModel):
    """감정 타임라인"""
    timestamp: str = Field(..., description="시간")
    emotion: str = Field(..., description="감정")
    intensity: int = Field(..., ge=0, le=100, description="강도")
    trigger: str = Field(..., description="트리거")


class VideoHighlight(BaseModel):
    """영상 하이라이트"""
    timestamp: str = Field(..., description="시간")
    thumbnail_url: str = Field(..., description="썸네일 URL")
    emotion: str = Field(..., description="감정")
    caption: str = Field(..., description="캡션")
    importance: Literal["urgent", "high", "medium"] = Field(..., description="중요도")


class RiskIndicator(BaseModel):
    """위험 지표"""
    level: Literal["high", "medium", "low"] = Field(..., description="위험도")
    factors: List[str] = Field(..., description="위험 요소들")


class AudioAnalysis(BaseModel):
    """음성 분석"""
    voice_energy: str = Field(..., description="목소리 에너지")
    speaking_pace: str = Field(..., description="말하기 속도")
    tone_quality: str = Field(..., description="음성 품질")
    emotional_indicators: List[str] = Field(..., description="감정 지표들")


class DetailedAnalysis(BaseModel):
    """🎯 5순위: 상세 분석"""
    conversation_summary: Dict = Field(..., description="대화 요약")
    emotion_timeline: List[EmotionTimeline] = Field(..., description="감정 타임라인")
    risk_indicators: Dict[str, RiskIndicator] = Field(..., description="위험 지표들")
    video_highlights: List[VideoHighlight] = Field(..., description="영상 하이라이트")
    audio_analysis: AudioAnalysis = Field(..., description="음성 분석")


class TrendChange(BaseModel):
    """추세 변화"""
    metric: str = Field(..., description="지표명")
    direction: Literal["up", "down", "stable"] = Field(..., description="방향")
    change: int = Field(..., description="변화량")
    icon: str = Field(..., description="아이콘")
    comment: str = Field(..., description="설명")


class TrendAnalysis(BaseModel):
    """🎯 6순위: 추세 분석"""
    compared_to: str = Field(..., description="비교 기준")
    changes: List[TrendChange] = Field(..., description="변화들")
    alert_message: str = Field(..., description="알림 메시지")
    pattern: str = Field(..., description="패턴")
    disabled: Optional[bool] = Field(default=False, description="비활성화 여부")
    reason: Optional[str] = Field(default=None, description="비활성화 이유")


class QuickStat(BaseModel):
    """빠른 통계"""
    label: str = Field(..., description="라벨")
    value: str = Field(..., description="값")
    emoji: str = Field(..., description="이모지")
    color: str = Field(..., description="색상")


class CTAButton(BaseModel):
    """행동 유도 버튼"""
    text: str = Field(..., description="버튼 텍스트")
    icon: str = Field(..., description="아이콘")
    color: str = Field(..., description="색상")
    action: str = Field(..., description="액션")


class UIComponents(BaseModel):
    """🎯 UI 표시용"""
    header: Dict = Field(..., description="헤더 정보")
    quick_stats: List[QuickStat] = Field(..., description="빠른 통계")
    cta_buttons: List[CTAButton] = Field(..., description="행동 유도 버튼들")


class EvidenceVisualization(BaseModel):
    """근거 시각화 데이터"""
    emotion_keywords: List[str] = Field(default_factory=list, description="감지된 감정 키워드 목록")
    keyword_weights: Dict[str, float] = Field(default_factory=dict, description="키워드별 가중치")
    facial_expression_timeline: List[Dict] = Field(default_factory=list, description="표정 변화 타임라인")
    voice_energy_waveform: Optional[Dict] = Field(default=None, description="음성 에너지 파형 데이터")
    score_breakdown: Dict[str, Dict] = Field(default_factory=dict, description="점수별 세부 분석")
    calculation_method: str = Field(..., description="점수 계산 방법 설명")


class MedicalDisclaimer(BaseModel):
    """의료 책임 면책 조항"""
    disclaimer_text: str = Field(..., description="면책 조항 텍스트")
    is_recommendation_not_diagnosis: bool = Field(..., description="권고사항임을 명시")
    suggested_action: str = Field(..., description="의사 상담 권장 여부")


class CaregiverFriendlyResponse(BaseModel):
    """보호자 친화적 응답 모델"""
    success: bool = Field(..., description="성공 여부")
    session_id: str = Field(..., description="세션 ID")
    user_id: str = Field(..., description="사용자 ID")
    recorded_at: str = Field(..., description="녹화 시간")
    
    # 🎯 핵심 섹션들 (우선순위 순)
    status_overview: StatusOverview = Field(..., description="1순위: 상태 개요")
    today_summary: TodaySummary = Field(..., description="2순위: 오늘 요약")
    key_concerns: List[KeyConcern] = Field(..., description="3순위: 주요 걱정거리")
    action_plan: ActionPlan = Field(..., description="4순위: 행동 계획")
    detailed_analysis: DetailedAnalysis = Field(..., description="5순위: 상세 분석")
    trend_analysis: TrendAnalysis = Field(..., description="6순위: 추세 분석")
    ui_components: UIComponents = Field(..., description="UI 컴포넌트")
    
    # 🆕 신뢰성 개선 필드
    evidence_visualization: EvidenceVisualization = Field(..., description="근거 시각화 데이터")
    baseline_comparison: Optional[Dict] = Field(default=None, description="개인 baseline 비교 결과")
    medical_disclaimer: MedicalDisclaimer = Field(..., description="의료 책임 면책 조항")