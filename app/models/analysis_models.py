from typing import List, Dict, Literal, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime


class EmotionEvidence(BaseModel):
    """감정 점수 계산 근거"""
    positive_factors: List[str] = Field(default_factory=list, description="긍정 점수에 기여한 요인들")
    negative_factors: List[str] = Field(default_factory=list, description="부정 점수에 기여한 요인들")
    anxiety_factors: List[str] = Field(default_factory=list, description="불안 점수에 기여한 요인들")
    depression_factors: List[str] = Field(default_factory=list, description="우울 점수에 기여한 요인들")
    loneliness_factors: List[str] = Field(default_factory=list, description="외로움 점수에 기여한 요인들")
    detected_keywords: List[str] = Field(default_factory=list, description="감지된 감정 키워드들")
    facial_expression_notes: Optional[str] = Field(default=None, description="표정 분석 결과")
    voice_energy_level: Optional[str] = Field(default=None, description="음성 에너지 수준")


class EmotionAnalysis(BaseModel):
    """감정 상태 분석 결과"""
    positive: int = Field(..., ge=0, le=100, description="긍정 감정 점수")
    negative: int = Field(..., ge=0, le=100, description="부정 감정 점수")
    anxiety: int = Field(..., ge=0, le=100, description="불안 점수")
    depression: int = Field(..., ge=0, le=100, description="우울 점수")
    loneliness: int = Field(..., ge=0, le=100, description="외로움 점수")
    overall_mood: Literal["매우좋음", "좋음", "보통", "나쁨", "매우나쁨"] = Field(..., description="전반적 기분")
    emotional_summary: str = Field(..., description="감정 상태 한 문장 요약")
    evidence: Optional[EmotionEvidence] = Field(default=None, description="점수 계산 근거")


class ContentAnalysis(BaseModel):
    """대화 내용 분석 결과"""
    summary: str = Field(..., description="대화 내용 한 문장 요약")
    main_topics: List[str] = Field(default_factory=list, description="주요 언급 주제들")
    daily_activities: List[str] = Field(default_factory=list, description="일상 활동들")
    social_interactions: List[str] = Field(default_factory=list, description="사회적 상호작용")
    health_mentions: List[str] = Field(default_factory=list, description="건강 관련 언급")
    mood_indicators: List[str] = Field(default_factory=list, description="기분 지표들")


class RiskCategories(BaseModel):
    """위험 요소 카테고리별 분류"""
    health: List[str] = Field(default_factory=list, description="건강 관련 위험 요소")
    safety: List[str] = Field(default_factory=list, description="안전 관련 위험 요소")
    mental: List[str] = Field(default_factory=list, description="정신 건강 위험 요소")
    social: List[str] = Field(default_factory=list, description="사회적 위험 요소")


class RiskAnalysis(BaseModel):
    """위험 키워드 감지 결과"""
    risk_level: Literal["안전", "보통", "주의", "긴급"] = Field(..., description="위험도 수준")
    detected_keywords: List[str] = Field(default_factory=list, description="감지된 위험 키워드")
    risk_categories: RiskCategories = Field(default_factory=RiskCategories, description="위험 요소 분류")
    immediate_concerns: List[str] = Field(default_factory=list, description="즉시 확인 필요 사항")
    recommended_actions: List[str] = Field(default_factory=list, description="권장 조치 사항")


class BaselineComparison(BaseModel):
    """개인 baseline 비교 결과"""
    comparison_period: str = Field(..., description="비교 기간 (예: '지난 7일')")
    metric: str = Field(..., description="비교 지표명")
    current_value: float = Field(..., description="현재 값")
    baseline_average: float = Field(..., description="baseline 평균값")
    difference: float = Field(..., description="차이 (현재 - baseline)")
    difference_percentage: float = Field(..., description="차이 비율 (%)")
    is_significant_change: bool = Field(..., description="유의미한 변화 여부")
    explanation: str = Field(..., description="변화 설명")


class AnomalyAnalysis(BaseModel):
    """이상 패턴 감지 결과"""
    pattern_detected: bool = Field(..., description="이상 패턴 감지 여부")
    pattern_type: Literal["급격한하락", "지속적하락", "행동변화", "언어패턴변화", "없음"] = Field(..., description="패턴 유형")
    severity: Literal["심각", "보통", "경미"] = Field(..., description="심각도")
    trend_analysis: str = Field(..., description="패턴 분석 설명")
    comparison_notes: str = Field(..., description="과거 대비 변화 설명")
    alert_needed: bool = Field(..., description="알림 필요 여부")
    monitoring_recommendations: List[str] = Field(default_factory=list, description="모니터링 권장사항")
    baseline_comparisons: List[BaselineComparison] = Field(default_factory=list, description="baseline 비교 결과")


class EmotionScore(BaseModel):
    """감정 점수 요약"""
    positive: int = Field(..., ge=0, le=100)
    anxiety: int = Field(..., ge=0, le=100)
    depression: int = Field(..., ge=0, le=100)


class ComprehensiveSummary(BaseModel):
    """종합 분석 요약"""
    overall_status: str = Field(..., description="전반적 상태 (이모지 포함)")
    status_emoji: str = Field(..., description="상태 이모지")
    status_text: str = Field(..., description="상태 텍스트")
    alert_needed: bool = Field(..., description="알림 필요 여부")
    priority_level: Literal["안전", "보통", "주의", "긴급"] = Field(..., description="우선순위 수준")
    main_summary: str = Field(..., description="주요 요약")
    emotion_score: EmotionScore = Field(..., description="감정 점수 요약")
    key_concerns: List[str] = Field(default_factory=list, description="주요 우려사항")
    recommended_actions: List[str] = Field(default_factory=list, description="권장 조치")
    requires_immediate_attention: bool = Field(..., description="즉시 주의 필요 여부")


class ComprehensiveAnalysisResult(BaseModel):
    """종합 분석 전체 결과"""
    timestamp: str = Field(..., description="분석 시각")
    emotion_analysis: EmotionAnalysis = Field(..., description="감정 분석 결과")
    content_analysis: ContentAnalysis = Field(..., description="내용 분석 결과")
    risk_analysis: RiskAnalysis = Field(..., description="위험 분석 결과")
    anomaly_analysis: AnomalyAnalysis = Field(..., description="이상 패턴 분석 결과")
    comprehensive_summary: ComprehensiveSummary = Field(..., description="종합 요약")


class SummaryCard(BaseModel):
    """📊 오늘의 상태 요약 카드"""
    status_emoji: str = Field(..., description="상태 이모지")
    status_text: str = Field(..., description="상태 텍스트")
    emotion_scores: EmotionScore = Field(..., description="감정 점수들")
    main_summary: str = Field(..., description="주요 요약")
    overall_mood: str = Field(..., description="전반적 기분")


class AlertInfo(BaseModel):
    """🚨 알림 정보"""
    alert_type: Literal["none", "attention", "urgent"] = Field(..., description="알림 유형")
    message: str = Field(..., description="알림 메시지")
    priority: str = Field(..., description="우선순위")
    detected_keywords: List[str] = Field(default_factory=list, description="감지된 키워드")
    immediate_concerns: List[str] = Field(default_factory=list, description="즉시 우려사항")
    recommended_actions: List[str] = Field(default_factory=list, description="권장 조치")
    requires_immediate_attention: bool = Field(..., description="즉시 주의 필요")


class AnalysisSessionResponse(BaseModel):
    """영상 편지 종합 분석 응답"""
    success: bool = Field(..., description="성공 여부")
    session_id: str = Field(..., description="세션 ID")
    user_id: str = Field(..., description="사용자 ID")
    photo_url: str = Field(..., description="사진 URL")
    conversation: str = Field(..., description="대화 내용")
    image_emotion_analysis: Dict = Field(..., description="이미지 기반 감정 분석")
    comprehensive_analysis: ComprehensiveAnalysisResult = Field(..., description="종합 분석 결과")
    summary_card: SummaryCard = Field(..., description="상태 요약 카드")
    alert_info: AlertInfo = Field(..., description="알림 정보")
    emotion_labels: List[str] = Field(..., description="감정 라벨 목록")