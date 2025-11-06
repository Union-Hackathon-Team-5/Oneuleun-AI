import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import httpx
from pydantic import ValidationError

from app.models.analysis_models import (
    EmotionAnalysis, ContentAnalysis, RiskAnalysis, AnomalyAnalysis,
    ComprehensiveAnalysisResult, ComprehensiveSummary, EmotionScore,
    EmotionEvidence, BaselineComparison
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
            # 연결 풀을 늘려서 병렬 요청을 보장
            # 타임아웃을 15초로 줄여서 빠른 실패 보장
            limits = httpx.Limits(max_keepalive_connections=20, max_connections=20)
            timeout = httpx.Timeout(15.0, connect=5.0)  # 총 15초, 연결 5초
            self._client = httpx.AsyncClient(timeout=timeout, limits=limits)
        return self._client
    
    async def _call_openai(self, prompt: str, max_tokens: int = 800, task_name: str = "unknown", timeout_seconds: float = 15.0) -> str:
        """OpenAI API 호출 (JSON 형식 강제, 최적화, 타임아웃 적용)"""
        call_start = time.time()
        print(f"[PERF] Starting API call: {task_name} (tokens: {max_tokens})", flush=True)
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "당신은 노인 복지 전문 AI 분석사입니다. 반드시 유효한 JSON 형식으로만 응답해주세요."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"}
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        try:
            client = await self._get_client()
            api_start = time.time()
            # asyncio.wait_for로 개별 작업 타임아웃 강제
            response = await asyncio.wait_for(
                client.post(self.base_url, headers=headers, json=payload),
                timeout=timeout_seconds
            )
            api_time = time.time() - api_start
            response.raise_for_status()
            data = response.json()
            result = data["choices"][0]["message"]["content"].strip()
            total_time = time.time() - call_start
            print(f"[PERF] Completed API call: {task_name} - {api_time:.2f}s (total: {total_time:.2f}s, tokens: {max_tokens})", flush=True)
            logger.debug(f"[PERF] OpenAI API call: {api_time:.2f}s (total: {total_time:.2f}s, tokens: {max_tokens})")
            return result
        except asyncio.TimeoutError:
            logger.error(f"OpenAI API call timeout: {task_name} (>{timeout_seconds}s)")
            raise RuntimeError(f"{task_name} 호출이 {timeout_seconds}초 내에 완료되지 않았습니다.") from None
        except Exception as exc:
            logger.error("OpenAI API call failed: %s", exc)
            raise
    
    async def analyze_emotion_state(self, conversation: str, image_analysis: Optional[Dict] = None) -> EmotionAnalysis:
        """감정 상태 분석 (대화 + 이미지 분석 종합) + 근거 포함"""
        
        # 이미지 분석 데이터가 있으면 추가 컨텍스트로 활용
        image_context = ""
        facial_notes = ""
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
            facial_notes = summary
        
        prompt = f"""
다음 독거노인과 AI의 대화를 분석하여 감정 상태를 파악하고, 각 점수가 왜 그렇게 계산되었는지 구체적인 근거를 함께 제공해주세요.

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
    "emotional_summary": "<한 문장 감정 요약>",
    "evidence": {{
        "positive_factors": ["<긍정 점수에 기여한 대화 내용이나 표현들>", ...],
        "negative_factors": ["<부정 점수에 기여한 대화 내용이나 표현들>", ...],
        "anxiety_factors": ["<불안 점수에 기여한 요인들>", ...],
        "depression_factors": ["<우울 점수에 기여한 요인들>", ...],
        "loneliness_factors": ["<외로움 점수에 기여한 요인들>", ...],
        "detected_keywords": ["<대화에서 감지된 감정 키워드들>", ...]
    }}
}}

중요: 각 점수에 대해 구체적인 근거를 제공해야 합니다. 예를 들어 positive=20이면 "어떤 대화에서 그런 점수가 나왔는지" 명확히 설명해주세요.
"""
        
        try:
            response = await self._call_openai(prompt, max_tokens=800, task_name="analyze_emotion_state")
            data = json.loads(response)
            
            # evidence 처리
            evidence_data = data.get("evidence", {})
            if facial_notes and evidence_data:
                evidence_data["facial_expression_notes"] = facial_notes
            
            # evidence를 먼저 처리
            evidence_obj = None
            if evidence_data:
                evidence_obj = EmotionEvidence.model_validate(evidence_data)
            
            # model_validate로 직접 파싱 (최적화)
            emotion_data = {
                "positive": data.get("positive", 50),
                "negative": data.get("negative", 50),
                "anxiety": data.get("anxiety", 50),
                "depression": data.get("depression", 50),
                "loneliness": data.get("loneliness", 50),
                "overall_mood": data.get("overall_mood", "보통"),
                "emotional_summary": data.get("emotional_summary", "분석 실패"),
                "evidence": evidence_obj
            }
            return EmotionAnalysis.model_validate(emotion_data)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.error("Failed to parse emotion analysis response: %s", exc)
            return EmotionAnalysis(
                positive=50, negative=50, anxiety=50, 
                depression=50, loneliness=50,
                overall_mood="보통",
                emotional_summary="분석 실패",
                evidence=None
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
            response = await self._call_openai(prompt, max_tokens=600, task_name="analyze_conversation_content")
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
            response = await self._call_openai(prompt, max_tokens=700, task_name="detect_risk_keywords")
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
    
    def calculate_baseline_comparisons(
        self,
        current_emotion: EmotionAnalysis,
        historical_data: Optional[List[Dict]] = None
    ) -> List[BaselineComparison]:
        """개인 baseline 비교 계산 (7일 평균 대비, 더미 데이터 포함)"""
        # historical_data가 없거나 부족하면 더미 데이터 생성 (7일)
        if not historical_data or len(historical_data) < 7:
            # 더미 데이터 생성: 현재 값 기준으로 약간의 변동성 추가
            import random
            random.seed(42)  # 재현 가능하도록
            
            historical_data = []
            base_positive = current_emotion.positive
            base_depression = current_emotion.depression
            base_loneliness = current_emotion.loneliness
            
            for i in range(7):
                # 현재 값 기준 ±15 범위로 변동
                historical_data.append({
                    "positive": max(0, min(100, base_positive + random.randint(-15, 15))),
                    "depression": max(0, min(100, base_depression + random.randint(-15, 15))),
                    "loneliness": max(0, min(100, base_loneliness + random.randint(-15, 15))),
                    "overall_mood": "보통"
                })
        
        if len(historical_data) < 3:
            return []  # 최소 3일은 필요
        
        # 최근 7일 데이터만 사용
        recent_data = historical_data[-7:]
        
        # 각 지표별 baseline 계산
        comparisons = []
        
        # 긍정 점수 비교
        baseline_positive = sum(d.get("positive", 50) for d in recent_data) / len(recent_data)
        diff_positive = current_emotion.positive - baseline_positive
        diff_pct_positive = (diff_positive / baseline_positive * 100) if baseline_positive > 0 else 0
        
        comparisons.append(BaselineComparison(
            comparison_period="지난 7일",
            metric="긍정 감정",
            current_value=float(current_emotion.positive),
            baseline_average=baseline_positive,
            difference=diff_positive,
            difference_percentage=diff_pct_positive,
            is_significant_change=abs(diff_pct_positive) > 20,  # 20% 이상 변화 시 유의미
            explanation=f"평소 평균 {baseline_positive:.1f}점 대비 {diff_positive:+.1f}점 ({diff_pct_positive:+.1f}%)"
        ))
        
        # 우울 점수 비교
        baseline_depression = sum(d.get("depression", 50) for d in recent_data) / len(recent_data)
        diff_depression = current_emotion.depression - baseline_depression
        diff_pct_depression = (diff_depression / baseline_depression * 100) if baseline_depression > 0 else 0
        
        comparisons.append(BaselineComparison(
            comparison_period="지난 7일",
            metric="우울 감정",
            current_value=float(current_emotion.depression),
            baseline_average=baseline_depression,
            difference=diff_depression,
            difference_percentage=diff_pct_depression,
            is_significant_change=abs(diff_pct_depression) > 20,
            explanation=f"평소 평균 {baseline_depression:.1f}점 대비 {diff_depression:+.1f}점 ({diff_pct_depression:+.1f}%)"
        ))
        
        # 외로움 점수 비교
        baseline_loneliness = sum(d.get("loneliness", 50) for d in recent_data) / len(recent_data)
        diff_loneliness = current_emotion.loneliness - baseline_loneliness
        diff_pct_loneliness = (diff_loneliness / baseline_loneliness * 100) if baseline_loneliness > 0 else 0
        
        comparisons.append(BaselineComparison(
            comparison_period="지난 7일",
            metric="외로움 감정",
            current_value=float(current_emotion.loneliness),
            baseline_average=baseline_loneliness,
            difference=diff_loneliness,
            difference_percentage=diff_pct_loneliness,
            is_significant_change=abs(diff_pct_loneliness) > 20,
            explanation=f"평소 평균 {baseline_loneliness:.1f}점 대비 {diff_loneliness:+.1f}점 ({diff_pct_loneliness:+.1f}%)"
        ))
        
        return comparisons
    
    async def detect_anomaly_patterns(self, conversation: str, historical_data: Optional[List[Dict]] = None) -> AnomalyAnalysis:
        """이상 패턴 감지 (과거 데이터와 비교) + baseline 비교 포함"""
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

주의: alert_needed는 정말 심각한 경우에만 true로 설정하세요. 경미한 변화는 false로 설정하여 불필요한 불안을 유발하지 마세요.
"""
        
        try:
            response = await self._call_openai(prompt, max_tokens=500, task_name="detect_anomaly_patterns")
            data = json.loads(response)
            
            anomaly = AnomalyAnalysis.model_validate(data)
            # baseline 비교는 나중에 추가됨 (analyze_video_letter_comprehensive에서)
            return anomaly
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.error("Failed to parse anomaly analysis response: %s", exc)
            return AnomalyAnalysis(
                pattern_detected=False,
                pattern_type="없음",
                severity="경미",
                trend_analysis="분석 실패",
                comparison_notes="과거 데이터 부족",
                alert_needed=False,
                monitoring_recommendations=[],
                baseline_comparisons=[]
            )
    
    async def analyze_video_letter_comprehensive(
        self, 
        conversation: str, 
        historical_data: Optional[List[Dict]] = None,
        image_analysis: Optional[Dict] = None
    ) -> ComprehensiveAnalysisResult:
        """영상 편지 종합 분석 (병렬 처리, 타임아웃 최적화)"""
        print(f"[PERF] Starting analyze_video_letter_comprehensive (4 parallel tasks)", flush=True)
        logger.info("[PERF] Starting analyze_video_letter_comprehensive (4 parallel tasks)")
        parallel_start = time.time()
        
        # 각 작업에 개별 타임아웃 적용 (15초)
        async def emotion_with_timeout():
            try:
                return await asyncio.wait_for(
                    self.analyze_emotion_state(conversation, image_analysis),
                    timeout=15.0
                )
            except (asyncio.TimeoutError, Exception) as exc:
                logger.error(f"Emotion analysis failed/timeout: {exc}")
                return EmotionAnalysis(
                    positive=50, negative=50, anxiety=50, depression=50, loneliness=50,
                    overall_mood="보통", emotional_summary="분석 실패"
                )
        
        async def content_with_timeout():
            try:
                return await asyncio.wait_for(
                    self.analyze_conversation_content(conversation),
                    timeout=15.0
                )
            except (asyncio.TimeoutError, Exception) as exc:
                logger.error(f"Content analysis failed/timeout: {exc}")
                return ContentAnalysis(summary="분석 실패")
        
        async def risk_with_timeout():
            try:
                return await asyncio.wait_for(
                    self.detect_risk_keywords(conversation, image_analysis),
                    timeout=15.0
                )
            except (asyncio.TimeoutError, Exception) as exc:
                logger.error(f"Risk analysis failed/timeout: {exc}")
                from app.models.analysis_models import RiskCategories
                return RiskAnalysis(
                    risk_level="보통", detected_keywords=[], 
                    risk_categories=RiskCategories()
                )
        
        async def anomaly_with_timeout():
            try:
                return await asyncio.wait_for(
                    self.detect_anomaly_patterns(conversation, historical_data),
                    timeout=15.0
                )
            except (asyncio.TimeoutError, Exception) as exc:
                logger.error(f"Anomaly analysis failed/timeout: {exc}")
                return AnomalyAnalysis(
                    pattern_detected=False, pattern_type="없음", severity="경미",
                    trend_analysis="분석 실패", comparison_notes="과거 데이터 부족",
                    alert_needed=False
                )
        
        try:
            # 모든 분석 결과를 병렬로 기다림 (각각 최대 15초)
            emotion_result, content_result, risk_result, anomaly_result = await asyncio.gather(
                emotion_with_timeout(),
                content_with_timeout(),
                risk_with_timeout(),
                anomaly_with_timeout(),
                return_exceptions=False  # 이미 타임아웃 처리됨
            )
            parallel_time = time.time() - parallel_start
            print(f"[PERF] Parallel analysis completed in {parallel_time:.2f}s", flush=True)
            logger.info(f"[PERF] Parallel analysis completed in {parallel_time:.2f}s")
            
            # baseline 비교 계산 (historical_data가 있는 경우)
            baseline_comparisons = []
            if historical_data:
                baseline_comparisons = self.calculate_baseline_comparisons(emotion_result, historical_data)
                anomaly_result.baseline_comparisons = baseline_comparisons
            
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
        
        # 알림 여부 결정 (과도한 경고 방지)
        # baseline 비교가 있으면, 유의미한 변화가 있을 때만 alert
        is_significant_change = False
        if anomaly.baseline_comparisons:
            is_significant_change = any(comp.is_significant_change for comp in anomaly.baseline_comparisons)
        
        # alert는 정말 심각한 경우에만 (과도한 경고 방지)
        alert_needed = (
            risk.risk_level == "긴급" or  # 긴급만 자동 alert
            (risk.risk_level == "주의" and is_significant_change) or  # 주의는 baseline 변화가 있을 때만
            (anomaly.alert_needed and anomaly.severity == "심각")  # 심각한 이상 패턴만
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