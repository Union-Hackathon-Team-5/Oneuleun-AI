import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import httpx
from pydantic import ValidationError

from app.models.analysis_models import (
    EmotionAnalysis, ContentAnalysis, RiskAnalysis, AnomalyAnalysis,
    ComprehensiveAnalysisResult, ComprehensiveSummary, EmotionScore
)

logger = logging.getLogger(__name__)


class AnalysisService:
    """병렬 OpenAI API 호출을 통한 영상 편지 종합 분석 서비스"""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.api_key = api_key
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    async def _call_openai(self, prompt: str) -> str:
        """OpenAI API 호출 (JSON 형식 강제)"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "당신은 노인 복지 전문 AI 분석사입니다. 반드시 유효한 JSON 형식으로만 응답해주세요."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1000,
            "response_format": {"type": "json_object"}
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        try:
            client = await self._get_client()
            response = await client.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.error("OpenAI API call failed: %s", exc)
            raise
    
    async def analyze_emotion_state(self, conversation: str, image_analysis: Optional[Dict] = None) -> EmotionAnalysis:
        """감정 상태 분석 (대화 + 이미지 분석 종합)"""
        
        # 이미지 분석 데이터가 있으면 추가 컨텍스트로 활용
        image_context = ""
        if image_analysis and "analysis" in image_analysis:
            img_data = image_analysis["analysis"]
            emotions = ", ".join(img_data.get("emotion", []))
            summary = img_data.get("summary", "")
            concerns = img_data.get("concerns", [])
            
            image_context = f"""

이미지 분석 결과:
- 감정: {emotions}
- 표정 설명: {summary}
- 우려사항: {", ".join(concerns) if concerns else "없음"}
"""
        
        prompt = f"""
다음 독거노인과 AI의 대화를 분석하여 감정 상태를 파악해주세요.

대화 내용:
{conversation}
{image_context}

위 대화 내용과 이미지 분석 결과를 종합하여 다음 JSON 형식으로 응답해주세요:
{{
    "positive": <0-100 긍정 점수>,
    "negative": <0-100 부정 점수>,
    "anxiety": <0-100 불안 점수>,
    "depression": <0-100 우울 점수>,
    "loneliness": <0-100 외로움 점수>,
    "overall_mood": "<전반적 기분: 매우좋음/좋음/보통/나쁨/매우나쁨>",
    "emotional_summary": "<한 문장 감정 요약>"
}}
"""
        
        try:
            response = await self._call_openai(prompt)
            data = json.loads(response)
            return EmotionAnalysis.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.error("Failed to parse emotion analysis response: %s", exc)
            return EmotionAnalysis(
                positive=50, negative=50, anxiety=50, 
                depression=50, loneliness=50,
                overall_mood="보통",
                emotional_summary="분석 실패"
            )
    
    async def analyze_conversation_content(self, conversation: str) -> ContentAnalysis:
        """대화 내용 분석"""
        prompt = f"""
다음 독거노인과 AI의 대화 내용을 분석하여 주요 정보를 추출해주세요.

대화 내용:
{conversation}

다음 JSON 형식으로 응답해주세요:
{{
    "summary": "<한 문장 요약>",
    "main_topics": ["<주제1>", "<주제2>", ...],
    "daily_activities": ["<활동1>", "<활동2>", ...],
    "social_interactions": ["<사회활동1>", "<사회활동2>", ...],
    "health_mentions": ["<건강 관련 언급1>", "<건강 관련 언급2>", ...],
    "mood_indicators": ["<기분 지표1>", "<기분 지표2>", ...]
}}
"""
        
        try:
            response = await self._call_openai(prompt)
            data = json.loads(response)
            return ContentAnalysis.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.error("Failed to parse conversation analysis response: %s", exc)
            return ContentAnalysis(
                summary="분석 실패",
                main_topics=[],
                daily_activities=[],
                social_interactions=[],
                health_mentions=[],
                mood_indicators=[]
            )
    
    async def detect_risk_keywords(self, conversation: str, image_analysis: Optional[Dict] = None) -> RiskAnalysis:
        """위험 키워드 감지 (대화 + 이미지 분석 종합)"""
        
        # 이미지 분석에서 우려사항 추출
        image_context = ""
        if image_analysis and "analysis" in image_analysis:
            img_data = image_analysis["analysis"]
            concerns = img_data.get("concerns", [])
            emotions = img_data.get("emotion", [])
            
            if concerns or any(emotion in ["슬픔", "무기력함"] for emotion in emotions):
                image_context = f"""

이미지 분석에서 감지된 우려사항:
- 감정 상태: {", ".join(emotions)}
- 우려사항: {", ".join(concerns) if concerns else "없음"}
"""
        
        prompt = f"""
다음 독거노인과 AI의 대화에서 위험 신호나 주의가 필요한 키워드를 감지해주세요.

대화 내용:
{conversation}
{image_context}

위 대화 내용과 이미지 분석 결과를 종합하여 다음 JSON 형식으로 응답해주세요:
{{
    "risk_level": "<긴급/주의/보통/안전>",
    "detected_keywords": ["<위험키워드1>", "<위험키워드2>", ...],
    "risk_categories": {{
        "health": ["<건강 위험 요소>", ...],
        "safety": ["<안전 위험 요소>", ...],
        "mental": ["<정신 건강 위험 요소>", ...],
        "social": ["<사회적 위험 요소>", ...]
    }},
    "immediate_concerns": ["<즉시 확인 필요 사항>", ...],
    "recommended_actions": ["<권장 조치1>", "<권장 조치2>", ...]
}}

위험 키워드 예시: 넘어졌어요, 아파요, 밥을 못 먹었어요, 어지러워요, 숨이 차요, 혼자 무서워요 등
이미지에서 "우울증 우려", "자살 위험 의심" 등이 감지되면 반드시 mental 카테고리에 포함하세요.
"""
        
        try:
            response = await self._call_openai(prompt)
            data = json.loads(response)
            return RiskAnalysis.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.error("Failed to parse risk analysis response: %s", exc)
            from app.models.analysis_models import RiskCategories
            return RiskAnalysis(
                risk_level="보통",
                detected_keywords=[],
                risk_categories=RiskCategories(),
                immediate_concerns=[],
                recommended_actions=[]
            )
    
    async def detect_anomaly_patterns(self, conversation: str, historical_data: Optional[List[Dict]] = None) -> AnomalyAnalysis:
        """이상 패턴 감지 (과거 데이터와 비교)"""
        historical_context = ""
        if historical_data:
            recent_moods = [data.get("overall_mood", "보통") for data in historical_data[-7:]]  # 최근 7일
            historical_context = f"\n최근 일주일 기분 변화: {' -> '.join(recent_moods)}"
        
        prompt = f"""
다음 독거노인의 오늘 대화와 과거 데이터를 비교하여 이상 패턴을 감지해주세요.

오늘 대화:
{conversation}

{historical_context}

다음 JSON 형식으로 응답해주세요:
{{
    "pattern_detected": <true/false>,
    "pattern_type": "<급격한하락/지속적하락/행동변화/언어패턴변화/없음>",
    "severity": "<심각/보통/경미>",
    "trend_analysis": "<패턴 분석 설명>",
    "comparison_notes": "<과거 대비 변화 설명>",
    "alert_needed": <true/false>,
    "monitoring_recommendations": ["<모니터링 권장사항1>", ...]
}}
"""
        
        try:
            response = await self._call_openai(prompt)
            data = json.loads(response)
            return AnomalyAnalysis.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.error("Failed to parse anomaly analysis response: %s", exc)
            return AnomalyAnalysis(
                pattern_detected=False,
                pattern_type="없음",
                severity="경미",
                trend_analysis="분석 실패",
                comparison_notes="과거 데이터 부족",
                alert_needed=False,
                monitoring_recommendations=[]
            )
    
    async def analyze_video_letter_comprehensive(
        self, 
        conversation: str, 
        historical_data: Optional[List[Dict]] = None,
        image_analysis: Optional[Dict] = None
    ) -> ComprehensiveAnalysisResult:
        """영상 편지 종합 분석 (병렬 처리)"""
        
        # 모든 분석을 병렬로 실행 (이미지 분석 데이터 포함)
        emotion_task = self.analyze_emotion_state(conversation, image_analysis)
        content_task = self.analyze_conversation_content(conversation)
        risk_task = self.detect_risk_keywords(conversation, image_analysis)
        anomaly_task = self.detect_anomaly_patterns(conversation, historical_data)
        
        try:
            # 모든 분석 결과를 병렬로 기다림
            emotion_result, content_result, risk_result, anomaly_result = await asyncio.gather(
                emotion_task, content_task, risk_task, anomaly_task,
                return_exceptions=True
            )
            
            # 예외 처리
            if isinstance(emotion_result, Exception):
                logger.error("Emotion analysis failed: %s", emotion_result)
                emotion_result = EmotionAnalysis(
                    positive=50, negative=50, anxiety=50, depression=50, loneliness=50,
                    overall_mood="보통", emotional_summary="분석 실패"
                )
            
            if isinstance(content_result, Exception):
                logger.error("Content analysis failed: %s", content_result)
                content_result = ContentAnalysis(summary="분석 실패")
            
            if isinstance(risk_result, Exception):
                logger.error("Risk analysis failed: %s", risk_result)
                from app.models.analysis_models import RiskCategories
                risk_result = RiskAnalysis(
                    risk_level="보통", detected_keywords=[], 
                    risk_categories=RiskCategories()
                )
            
            if isinstance(anomaly_result, Exception):
                logger.error("Anomaly analysis failed: %s", anomaly_result)
                anomaly_result = AnomalyAnalysis(
                    pattern_detected=False, pattern_type="없음", severity="경미",
                    trend_analysis="분석 실패", comparison_notes="과거 데이터 부족",
                    alert_needed=False
                )
            
            # 종합 분석 결과 생성
            comprehensive_result = self._generate_comprehensive_summary(
                emotion_result, content_result, risk_result, anomaly_result
            )
            
            return ComprehensiveAnalysisResult(
                timestamp=datetime.now().isoformat(),
                emotion_analysis=emotion_result,
                content_analysis=content_result,
                risk_analysis=risk_result,
                anomaly_analysis=anomaly_result,
                comprehensive_summary=comprehensive_result
            )
            
        except Exception as exc:
            logger.error("Comprehensive analysis failed: %s", exc)
            raise RuntimeError("영상 편지 분석 중 오류가 발생했습니다.") from exc
    
    def _generate_comprehensive_summary(
        self, 
        emotion: EmotionAnalysis, 
        content: ContentAnalysis, 
        risk: RiskAnalysis, 
        anomaly: AnomalyAnalysis
    ) -> ComprehensiveSummary:
        """종합 분석 결과 요약 생성"""
        
        # 전반적 상태 판정
        overall_status = "😊 좋음"
        if risk.risk_level == "긴급" or anomaly.alert_needed:
            overall_status = "🚨 긴급"
        elif risk.risk_level == "주의" or emotion.overall_mood in ["나쁨", "매우나쁨"]:
            overall_status = "😟 주의"
        elif emotion.overall_mood == "보통":
            overall_status = "😐 보통"
        
        # 알림 여부 결정
        alert_needed = (
            risk.risk_level in ["긴급", "주의"] or 
            anomaly.alert_needed or
            emotion.overall_mood in ["나쁨", "매우나쁨"]
        )
        
        # 권장 조치 통합
        all_actions = []
        all_actions.extend(risk.recommended_actions)
        all_actions.extend(anomaly.monitoring_recommendations)
        
        if not all_actions:
            if overall_status == "😊 좋음":
                all_actions = ["현재 상태 양호, 정기 확인 유지"]
            else:
                all_actions = ["상태 변화 모니터링 필요"]
        
        return ComprehensiveSummary(
            overall_status=overall_status,
            status_emoji=overall_status.split()[0],
            status_text=overall_status.split()[1],
            alert_needed=alert_needed,
            priority_level=risk.risk_level,
            main_summary=content.summary,
            emotion_score=EmotionScore(
                positive=emotion.positive,
                anxiety=emotion.anxiety,
                depression=emotion.depression
            ),
            key_concerns=risk.immediate_concerns,
            recommended_actions=all_actions[:3],  # 최대 3개만
            requires_immediate_attention=risk.risk_level == "긴급"
        )
    
    async def close(self):
        """클라이언트 정리"""
        if self._client:
            await self._client.aclose()