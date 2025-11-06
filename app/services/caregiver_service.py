import asyncio
import json
import logging
import time
from typing import Dict, List, Optional
from datetime import datetime

from app.models.caregiver_models import (
    CaregiverFriendlyResponse, StatusOverview, TodaySummary, KeyConcern,
    ActionPlan, UrgentAction, DetailedAnalysis, TrendAnalysis, TrendChange,
    UIComponents, QuickStat, CTAButton, EmotionTimeline, VideoHighlight,
    RiskIndicator, AudioAnalysis, ConversationTopic, EvidenceVisualization,
    MedicalDisclaimer
)
from app.services.analysis_service import AnalysisService
from app.models.analysis_models import ComprehensiveAnalysisResult

logger = logging.getLogger(__name__)


class CaregiverService:
    """보호자 친화적 분석 결과 생성 서비스"""
    
    def __init__(self):
        self.analysis_service = AnalysisService()
    
    async def generate_caregiver_friendly_report(
        self,
        conversation: str,
        image_analysis: Dict,
        audio_analysis: Dict,
        session_id: str,
        user_id: str,
        historical_data: Optional[List[Dict]] = None
    ) -> CaregiverFriendlyResponse:
        """보호자 친화적 리포트 생성"""
        start_time = time.time()
        
        # 기존 기술적 분석 실행 (historical_data 포함)
        print(f"[PERF] Starting comprehensive_analysis", flush=True)
        logger.info("[PERF] Starting comprehensive_analysis")
        comp_start = time.time()
        comprehensive_analysis = await self.analysis_service.analyze_video_letter_comprehensive(
            conversation=conversation,
            image_analysis=image_analysis,
            historical_data=historical_data
        )
        comp_time = time.time() - comp_start
        print(f"[PERF] comprehensive_analysis completed in {comp_time:.2f}s", flush=True)
        logger.info(f"[PERF] comprehensive_analysis completed in {comp_time:.2f}s")
        
        # 감성적, 액션 중심 리포트로 변환
        print(f"[PERF] Starting _transform_to_caregiver_format", flush=True)
        logger.info("[PERF] Starting _transform_to_caregiver_format")
        transform_start = time.time()
        result = await self._transform_to_caregiver_format(
            comprehensive_analysis=comprehensive_analysis,
            conversation=conversation,
            image_analysis=image_analysis,
            audio_analysis=audio_analysis,
            session_id=session_id,
            user_id=user_id
        )
        transform_time = time.time() - transform_start
        total_time = time.time() - start_time
        print(f"[PERF] _transform_to_caregiver_format completed in {transform_time:.2f}s", flush=True)
        print(f"[PERF] Total time: {total_time:.2f}s", flush=True)
        logger.info(f"[PERF] _transform_to_caregiver_format completed in {transform_time:.2f}s")
        logger.info(f"[PERF] Total time: {total_time:.2f}s")
        
        return result
    
    async def _transform_to_caregiver_format(
        self,
        comprehensive_analysis: ComprehensiveAnalysisResult,
        conversation: str,
        image_analysis: Dict,
        audio_analysis: Dict,
        session_id: str,
        user_id: str
    ) -> CaregiverFriendlyResponse:
        """기술적 분석을 보호자 친화적 형태로 변환"""
        
        # 감성적 인사이트 생성 (병렬 처리, 타임아웃 적용)
        print(f"[PERF] Starting parallel LLM calls (4 tasks)", flush=True)
        logger.info("[PERF] Starting parallel LLM calls (4 tasks)")
        parallel_start = time.time()
        
        # 각 작업에 타임아웃 래퍼 적용 (15초)
        async def insights_with_timeout():
            try:
                return await asyncio.wait_for(
                    self._generate_emotional_insights(conversation, comprehensive_analysis),
                    timeout=15.0
                )
            except (asyncio.TimeoutError, Exception) as exc:
                logger.error(f"Emotional insights generation failed/timeout: {exc}")
                return {
                    "headline": "어머니 상태를 확인이 필요합니다",
                    "mood_description": "평소보다 기분이 좋지 않으신 것 같아요",
                    "energy_level": "활력이 부족해 보입니다",
                    "pain_level": "몸이 불편하신 것 같아요",
                    "emotional_state": "관심과 돌봄이 필요한 상태입니다"
                }
        
        async def action_plan_with_timeout():
            try:
                return await asyncio.wait_for(
                    self._generate_actionable_plan(comprehensive_analysis, conversation),
                    timeout=15.0
                )
            except (asyncio.TimeoutError, Exception) as exc:
                logger.error(f"Action plan generation failed/timeout: {exc}")
                return self._create_default_action_plan(comprehensive_analysis)
        
        async def mother_voice_with_timeout():
            try:
                return await asyncio.wait_for(
                    self._extract_mother_voice(conversation),
                    timeout=15.0
                )
            except (asyncio.TimeoutError, Exception) as exc:
                logger.error(f"Mother voice extraction failed/timeout: {exc}")
                return [
                    "💬 \"요즘 컨디션이 별로 좋지 않아요\"",
                    "💬 \"혼자 있는 시간이 많아서 외로워요\"",
                    "💬 \"몸이 예전 같지 않아서 걱정이에요\""
                ]
        
        async def concerns_with_timeout():
            try:
                return await asyncio.wait_for(
                    self._identify_key_concerns(comprehensive_analysis, conversation, image_analysis),
                    timeout=15.0
                )
            except (asyncio.TimeoutError, Exception) as exc:
                logger.error(f"Key concerns identification failed/timeout: {exc}")
                return self._create_default_concerns(comprehensive_analysis)
        
        emotional_insights, action_plan, mother_voice, key_concerns = await asyncio.gather(
            insights_with_timeout(),
            action_plan_with_timeout(),
            mother_voice_with_timeout(),
            concerns_with_timeout()
        )
        parallel_time = time.time() - parallel_start
        print(f"[PERF] Parallel LLM calls completed in {parallel_time:.2f}s", flush=True)
        logger.info(f"[PERF] Parallel LLM calls completed in {parallel_time:.2f}s")
        
        # 병렬 LLM 호출 이후 후처리 작업들 시간 측정
        post_process_start = time.time()
        
        # 1순위: 상태 개요 (key_concerns 생성 후에 결정하여 일관성 보장)
        status_overview = self._create_status_overview(comprehensive_analysis, key_concerns)
        
        # 2순위: 오늘 요약
        today_summary = self._create_today_summary(
            comprehensive_analysis, emotional_insights, mother_voice
        )
        
        # 3순위: 주요 걱정거리 (이미 생성됨)
        
        # 4순위: 행동 계획 (이미 생성됨)
        
        # 5순위: 상세 분석 (key_concerns와 일치시킴)
        detailed_analysis = self._create_detailed_analysis(
            comprehensive_analysis, conversation, audio_analysis, key_concerns
        )
        
        # Baseline 비교 데이터 생성 (추세 분석 전에 필요)
        baseline_comparison = self._create_baseline_comparison(comprehensive_analysis)
        
        # 6순위: 추세 분석 (baseline 비교 기반으로 활성화/비활성화)
        trend_analysis = self._create_trend_analysis(comprehensive_analysis, baseline_comparison)
        
        # UI 컴포넌트
        ui_components = self._create_ui_components(status_overview, comprehensive_analysis)
        
        # 근거 시각화 데이터 생성 (맥락 충돌 감지 포함)
        evidence_viz = self._create_evidence_visualization(
            comprehensive_analysis, conversation, audio_analysis, image_analysis, key_concerns
        )
        
        # 의료 책임 면책 조항 생성 (action_plan과 일치시킴)
        medical_disclaimer = self._create_medical_disclaimer(comprehensive_analysis, action_plan, key_concerns)
        
        post_process_time = time.time() - post_process_start
        print(f"[PERF] Post-processing (data transformation) completed in {post_process_time:.2f}s", flush=True)
        
        return CaregiverFriendlyResponse(
            success=True,
            session_id=session_id,
            user_id=user_id,
            recorded_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            status_overview=status_overview,
            today_summary=today_summary,
            key_concerns=key_concerns,
            action_plan=action_plan,
            detailed_analysis=detailed_analysis,
            trend_analysis=trend_analysis,
            ui_components=ui_components,
            evidence_visualization=evidence_viz,
            baseline_comparison=baseline_comparison,
            medical_disclaimer=medical_disclaimer
        )
    
    async def _generate_emotional_insights(
        self, 
        conversation: str, 
        analysis: ComprehensiveAnalysisResult
    ) -> Dict:
        """감성적 인사이트 생성"""
        prompt = f"""
다음 독거노인의 대화에서 보호자가 알아야 할 감성적 포인트를 추출해주세요.

대화 내용:
{conversation}

분석 결과:
- 감정: {analysis.emotion_analysis.overall_mood}
- 주요 우려: {analysis.comprehensive_summary.key_concerns}

다음 JSON 형식으로 응답해주세요:
{{
    "headline": "<어머니 상태를 한줄로 요약 (감성적으로)>",
    "mood_description": "<기분 상태를 인간적으로 설명>",
    "energy_level": "<활력 수준 설명>",
    "pain_level": "<통증 수준 설명>",
    "emotional_state": "<전반적 감정 상태>"
}}

예시:
- "어머니께서 많이 힘들어하십니다"
- "평소보다 많이 지쳐 보이세요"
- "외로움을 많이 느끼고 계신 것 같아요"
"""
        
        try:
            task_start = time.time()
            response = await self.analysis_service._call_openai(prompt, max_tokens=500, task_name="_generate_emotional_insights")
            task_time = time.time() - task_start
            print(f"[PERF] _generate_emotional_insights API call: {task_time:.2f}s", flush=True)
            logger.debug(f"[PERF] _generate_emotional_insights API call: {task_time:.2f}s")
            return json.loads(response)
        except Exception as exc:
            logger.error("Failed to generate emotional insights: %s", exc)
            return {
                "headline": "어머니 상태를 확인이 필요합니다",
                "mood_description": "평소보다 기분이 좋지 않으신 것 같아요",
                "energy_level": "활력이 부족해 보입니다",
                "pain_level": "몸이 불편하신 것 같아요",
                "emotional_state": "관심과 돌봄이 필요한 상태입니다"
            }
    
    async def _generate_actionable_plan(
        self, 
        analysis: ComprehensiveAnalysisResult, 
        conversation: str
    ) -> ActionPlan:
        """실행 가능한 행동 계획 생성 (과도한 경고 방지)"""
        
        # baseline 비교 정보 추가
        baseline_context = ""
        if analysis.anomaly_analysis.baseline_comparisons:
            significant = [c for c in analysis.anomaly_analysis.baseline_comparisons if c.is_significant_change]
            if significant:
                baseline_context = f"\n개인 baseline 비교: {len(significant)}개의 유의미한 변화 감지"
        
        # 프롬프트 간소화: 핵심만 포함
        key_concerns_str = ", ".join(analysis.comprehensive_summary.key_concerns[:3]) if analysis.comprehensive_summary.key_concerns else "없음"
        recommended_str = ", ".join(analysis.comprehensive_summary.recommended_actions[:3]) if analysis.comprehensive_summary.recommended_actions else "없음"
        
        prompt = f"""분석 결과 기반 행동 계획 생성 (간결하게):

대화 요약: {conversation[:300]}...
위험도: {analysis.comprehensive_summary.priority_level}
주요 우려: {key_concerns_str}
권장 조치: {recommended_str}
{baseline_context}

JSON 형식으로 응답 (각 카테고리 최대 3개):
{{
    "urgent_actions": [{{"action_id": 1, "priority": "최우선|긴급|중요", "icon": "📞", "title": "구체적 행동", "reason": "이유", "detail": "어머니 말씀 인용", "deadline": "언제까지", "estimated_time": "소요시간", "suggested_topics": ["대화예시1", "대화예시2"]}}],
    "this_week_actions": [...],
    "long_term_actions": [...]
}}

주의: urgent_actions는 긴급시 1-2개만. priority는 "최우선", "긴급", "중요" 중 하나만. 건강 관련은 "의료진 상담 권장" 표현 사용.
"""
        
        try:
            task_start = time.time()
            # max_tokens를 600으로 줄임 (실제로는 urgent 2개 + this_week 3개 + long_term 2개 정도면 충분)
            response = await self.analysis_service._call_openai(prompt, max_tokens=600, task_name="_generate_actionable_plan")
            task_time = time.time() - task_start
            print(f"[PERF] _generate_actionable_plan API call: {task_time:.2f}s", flush=True)
            logger.debug(f"[PERF] _generate_actionable_plan API call: {task_time:.2f}s")
            data = json.loads(response)
            
            # priority 필드 검증 및 기본값 처리
            valid_priorities = ["최우선", "긴급", "중요"]
            
            def normalize_action(action: Dict) -> Dict:
                """priority 필드 정규화"""
                if "priority" in action:
                    priority = action["priority"]
                    if priority not in valid_priorities:
                        # 유효하지 않은 priority를 기본값으로 변경
                        if priority in ["보통", "낮음", "normal", "low"]:
                            action["priority"] = "중요"
                        elif priority in ["높음", "high", "urgent"]:
                            action["priority"] = "긴급"
                        else:
                            action["priority"] = "중요"  # 기본값
                else:
                    action["priority"] = "중요"  # 기본값
                return action
            
            # Pydantic model_validate로 최적화 (priority 정규화 후)
            urgent_actions = [UrgentAction.model_validate(normalize_action(action)) for action in data.get("urgent_actions", [])]
            this_week_actions = [UrgentAction.model_validate(normalize_action(action)) for action in data.get("this_week_actions", [])]
            long_term_actions = [UrgentAction.model_validate(normalize_action(action)) for action in data.get("long_term_actions", [])]
            
            return ActionPlan(
                urgent_actions=urgent_actions,
                this_week_actions=this_week_actions,
                long_term_actions=long_term_actions
            )
        except Exception as exc:
            logger.error("Failed to generate action plan: %s", exc)
            return self._create_default_action_plan(analysis)
    
    async def _extract_mother_voice(self, conversation: str) -> List[str]:
        """어머니 목소리 직접 인용 추출"""
        prompt = f"""
다음 대화에서 독거노인(어머니)이 직접 하신 말씀 중에서 보호자가 들으면 마음이 아프거나 걱정이 될 만한 부분을 찾아서 직접 인용해주세요.

대화 내용:
{conversation}

다음 JSON 형식으로 응답해주세요:
{{
    "mother_voice": [
        "💬 \\"실제 어머니가 하신 말씀1\\"",
        "💬 \\"실제 어머니가 하신 말씀2\\"",
        "💬 \\"실제 어머니가 하신 말씀3\\"",
        "💬 \\"실제 어머니가 하신 말씀4\\""
    ]
}}

예시:
- "💬 \"요즘은 자꾸 피곤해서 뭘 해도 금방 지치는 느낌이에요\""
- "💬 \"며칠째 밥맛이 없고 씹는 것도 좀 힘들어서요\""
- "💬 \"계속 집에만 있다 보니까 사람 목소리가 그립네요\""
"""
        
        try:
            task_start = time.time()
            response = await self.analysis_service._call_openai(prompt, max_tokens=400, task_name="_extract_mother_voice")
            task_time = time.time() - task_start
            print(f"[PERF] _extract_mother_voice API call: {task_time:.2f}s", flush=True)
            logger.debug(f"[PERF] _extract_mother_voice API call: {task_time:.2f}s")
            data = json.loads(response)
            return data.get("mother_voice", [])
        except Exception as exc:
            logger.error("Failed to extract mother voice: %s", exc)
            return [
                "💬 \"요즘 컨디션이 별로 좋지 않아요\"",
                "💬 \"혼자 있는 시간이 많아서 외로워요\"",
                "💬 \"몸이 예전 같지 않아서 걱정이에요\""
            ]
    
    async def _identify_key_concerns(
        self, 
        analysis: ComprehensiveAnalysisResult, 
        conversation: str, 
        image_analysis: Dict
    ) -> List[KeyConcern]:
        """주요 걱정거리 식별 (가족 케어 조언 톤)"""
        # 위험 분석 정보 간소화
        risk_level = analysis.risk_analysis.risk_level
        risk_keywords = ", ".join(analysis.risk_analysis.detected_keywords[:5])
        image_concerns = ", ".join(image_analysis.get('analysis', {}).get('concerns', [])[:3])
        
        prompt = f"""주요 걱정거리 식별 (최대 5개):

대화 요약: {conversation[:300]}...
위험도: {risk_level}
위험 키워드: {risk_keywords}
이미지 우려: {image_concerns or "없음"}

JSON 형식 (간결하게):
{{
    "concerns": [
        {{
            "concern_id": 1,
            "type": "건강|안전|정서|생활",
            "icon": "🏥",
            "severity": "urgent|caution|normal",
            "title": "구체적 문제",
            "description": "가족 케어 조언 톤으로 간단히 설명",
            "detected_from": ["대화", "표정"],
            "urgency_reason": "왜 중요한지"
        }}
    ]
}}

주의: "의료진 상담 권장" 표현 사용. "즉시 조치 필요" 같은 표현 피하기.
"""
        
        try:
            task_start = time.time()
            # max_tokens를 500으로 줄임 (concerns는 보통 3-5개, 각각 100 tokens 정도면 충분)
            response = await self.analysis_service._call_openai(prompt, max_tokens=500, task_name="_identify_key_concerns")
            task_time = time.time() - task_start
            print(f"[PERF] _identify_key_concerns API call: {task_time:.2f}s", flush=True)
            logger.debug(f"[PERF] _identify_key_concerns API call: {task_time:.2f}s")
            data = json.loads(response)
            # Pydantic model_validate로 최적화
            return [KeyConcern.model_validate(concern) for concern in data.get("concerns", [])]
        except Exception as exc:
            logger.error("Failed to identify key concerns: %s", exc)
            return self._create_default_concerns(analysis)
    
    def _create_status_overview(self, analysis: ComprehensiveAnalysisResult, key_concerns: List[KeyConcern]) -> StatusOverview:
        """상태 개요 생성 (Alert level 일관성 보장: 최고 위험도 기준)"""
        # 최고 위험도 기준으로 단일화 (key_concerns의 최고 severity 우선)
        max_concern_severity = "normal"
        if key_concerns:
            # key_concerns에서 최고 severity 찾기
            severity_map = {"urgent": 3, "caution": 2, "normal": 1}
            max_severity_value = max(severity_map.get(concern.severity, 1) for concern in key_concerns)
            max_concern_severity = [k for k, v in severity_map.items() if v == max_severity_value][0]
        
        # baseline 비교를 고려하여 경고 강도 조절
        has_significant_change = False
        if analysis.anomaly_analysis.baseline_comparisons:
            has_significant_change = any(
                comp.is_significant_change 
                for comp in analysis.anomaly_analysis.baseline_comparisons
            )
        
        # Alert level 결정: 최고 위험도 기준으로 단일화
        # urgent가 하나라도 있으면 urgent
        if max_concern_severity == "urgent" or analysis.comprehensive_summary.priority_level == "긴급":
            return StatusOverview(
                alert_level="urgent",
                alert_badge="🚨",
                alert_title="즉시 확인 필요",
                alert_subtitle="어머니께서 도움이 필요하신 것 같습니다",
                status_color="#FF4444"
            )
        elif max_concern_severity == "caution" or (analysis.comprehensive_summary.priority_level == "주의" and has_significant_change):
            # 주의는 baseline 변화가 있을 때만 강조
            return StatusOverview(
                alert_level="caution",
                alert_badge="⚠️",
                alert_title="평소와 다른 점 확인",
                alert_subtitle="지난 7일 평균 대비 변화가 감지되었습니다",
                status_color="#FF8800"
            )
        elif analysis.comprehensive_summary.priority_level == "주의":
            # 주의이지만 baseline 변화가 없으면 경미하게 표시
            return StatusOverview(
                alert_level="normal",
                alert_badge="📋",
                alert_title="일반 확인 권장",
                alert_subtitle="정기적으로 상태를 확인해주세요",
                status_color="#FFAA00"
            )
        else:
            return StatusOverview(
                alert_level="normal",
                alert_badge="😊",
                alert_title="안정적인 상태",
                alert_subtitle="특별한 문제는 없어 보입니다",
                status_color="#44FF44"
            )
    
    def _create_today_summary(
        self, 
        analysis: ComprehensiveAnalysisResult, 
        emotional_insights: Dict,
        mother_voice: List[str]
    ) -> TodaySummary:
        """오늘 요약 생성 (Baseline 비교 포함)"""
        mood_score = analysis.emotion_analysis.positive
        
        # Baseline 비교 정보 추출
        baseline_info = ""
        if analysis.anomaly_analysis.baseline_comparisons:
            for comp in analysis.anomaly_analysis.baseline_comparisons:
                if comp.metric == "긍정 감정":
                    baseline_info = f" (평소 평균 {comp.baseline_average:.0f}점 대비 {comp.difference:+.0f}점)"
                    break
        
        if mood_score >= 70:
            mood_label = f"좋음{baseline_info}" if baseline_info else "좋음"
            mood_emoji = "😊"
        elif mood_score >= 50:
            mood_label = f"보통{baseline_info}" if baseline_info else "보통"
            mood_emoji = "😐"
        elif mood_score >= 30:
            mood_label = f"우울함{baseline_info}" if baseline_info else "우울함"
            mood_emoji = "😔"
        else:
            mood_label = f"매우 우울함{baseline_info}" if baseline_info else "매우 우울함"
            mood_emoji = "😢"
        
        # headline 개선: 긴급 근거 중심 (용어·톤 일관성)
        headline = emotional_insights.get("headline", "어머니 상태를 확인해보세요")
        
        # 긴급한 경우(urgent) 건강 관련 구체적 근거를 headline에 포함
        if analysis.comprehensive_summary.priority_level == "긴급":
            urgent_health_issues = []
            health_str = str(analysis.risk_analysis.risk_categories.health)
            safety_str = str(analysis.risk_analysis.risk_categories.safety)
            
            if "식사" in health_str or "밥" in health_str or "음식" in health_str:
                urgent_health_issues.append("식사량 감소")
            if "통증" in health_str or "아파" in health_str:
                urgent_health_issues.append("통증")
            if "낙상" in safety_str:
                urgent_health_issues.append("낙상 위험")
            
            if urgent_health_issues:
                headline = f"{', '.join(urgent_health_issues)}가 동반되어 즉시 확인이 필요합니다"
            else:
                headline = "즉시 확인이 필요한 상태입니다"
        
        return TodaySummary(
            headline=headline,
            mood_score=mood_score,
            mood_label=mood_label,
            mood_emoji=mood_emoji,
            energy_score=max(0, 100 - analysis.emotion_analysis.depression),
            pain_score=analysis.emotion_analysis.anxiety,
            mother_voice=mother_voice[:4]  # 최대 4개
        )
    
    def _create_detailed_analysis(
        self, 
        analysis: ComprehensiveAnalysisResult, 
        conversation: str,
        audio_analysis: Dict,
        key_concerns: List[KeyConcern]
    ) -> DetailedAnalysis:
        """상세 분석 생성"""
        # 대화 주제별 요약
        topics = []
        if "식사" in conversation or "밥" in conversation:
            topics.append(ConversationTopic(
                topic="식사",
                summary="식욕 관련 언급이 있습니다",
                concern_level="caution" if "안 먹" in conversation else "normal"
            ))
        
        # 감정 타임라인 (더미 데이터)
        emotion_timeline = [
            EmotionTimeline(
                timestamp="00:00:30",
                emotion="무기력",
                intensity=75,
                trigger="피곤하다는 말씀"
            )
        ]
        
        # 위험 지표 (R3: key_concerns의 최대 severity 기준)
        health_concerns = [c for c in key_concerns if c.type == "건강"]
        mental_concerns = [c for c in key_concerns if c.type == "정서"]
        
        # 최대 severity 찾기
        severity_map = {"urgent": 3, "caution": 2, "normal": 1}
        
        health_max_severity = "normal"
        if health_concerns:
            health_max_severity_value = max(severity_map.get(c.severity, 1) for c in health_concerns)
            health_max_severity = [k for k, v in severity_map.items() if v == health_max_severity_value][0]
        
        mental_max_severity = "normal"
        if mental_concerns:
            mental_max_severity_value = max(severity_map.get(c.severity, 1) for c in mental_concerns)
            mental_max_severity = [k for k, v in severity_map.items() if v == mental_max_severity_value][0]
        
        # severity를 level로 변환 (urgent/caution -> high, normal -> medium/low)
        health_level = "high" if health_max_severity == "urgent" else "medium" if health_max_severity == "caution" else "low"
        mental_level = "high" if mental_max_severity == "urgent" else "medium" if mental_max_severity == "caution" else "low"
        
        # 기존 분석 결과와 병합 (더 높은 레벨 우선)
        if analysis.comprehensive_summary.priority_level == "긴급" and health_level != "high":
            health_level = "high"
        if analysis.emotion_analysis.depression > 70 and mental_level != "high":
            mental_level = "high"
        
        risk_indicators = {
            "health_risk": RiskIndicator(
                level=health_level,
                factors=analysis.risk_analysis.risk_categories.health
            ),
            "mental_risk": RiskIndicator(
                level=mental_level,
                factors=analysis.risk_analysis.risk_categories.mental
            )
        }
        
        # 영상 하이라이트 (더미)
        video_highlights = [
            VideoHighlight(
                timestamp="00:01:30",
                thumbnail_url="placeholder_thumbnail.jpg",
                emotion="우울",
                caption="표정이 어두워 보입니다",
                importance="high"
            )
        ]
        
        # 음성 분석
        audio_analysis_obj = AudioAnalysis(
            voice_energy=audio_analysis.get("voice_energy", "보통"),
            speaking_pace=audio_analysis.get("speaking_pace", "보통"),
            tone_quality=audio_analysis.get("tone_quality", "보통"),
            emotional_indicators=audio_analysis.get("emotional_indicators", [])
        )
        
        return DetailedAnalysis(
            conversation_summary={
                "total_exchanges": len(conversation.split("\n")),
                "conversation_topics": [topic.dict() for topic in topics]
            },
            emotion_timeline=emotion_timeline,
            risk_indicators=risk_indicators,
            video_highlights=video_highlights,
            audio_analysis=audio_analysis_obj
        )
    
    def _create_trend_analysis(self, analysis: ComprehensiveAnalysisResult, baseline_comparison: Optional[Dict]) -> TrendAnalysis:
        """추세 분석 생성 (R5: 7일 미만이면 비활성화)"""
        # baseline_comparison이 없거나 데이터 부족 시 비활성화
        if not baseline_comparison or baseline_comparison.get("comparison_period", "").endswith("데이터 부족"):
            return TrendAnalysis(
                compared_to="지난 7일",
                changes=[],
                alert_message="7일 미만 데이터로 신뢰 낮음",
                pattern="데이터 부족",
                disabled=True,
                reason="7일 미만 데이터로 신뢰 낮음"
            )
        
        # baseline_comparison에서 추세 데이터 생성
        significant_changes = baseline_comparison.get("significant_changes", [])
        changes = []
        
        for change in significant_changes[:5]:  # 최대 5개
            direction = "down" if change.get("difference", 0) < 0 else "up" if change.get("difference", 0) > 0 else "stable"
            icon = "📉" if direction == "down" else "📈" if direction == "up" else "➡️"
            
            changes.append(TrendChange(
                metric=change.get("metric", ""),
                direction=direction,
                change=int(change.get("difference", 0)),
                icon=icon,
                comment=change.get("explanation", "")
            ))
        
        if not changes:
            # 유의미한 변화가 없으면 안정적 표시
            return TrendAnalysis(
                compared_to="지난 7일",
                changes=[],
                alert_message="지난 7일 대비 큰 변화 없음",
                pattern="안정적"
            )
        
        alert_message = f"⚠️ 지난 7일 대비 {len(changes)}개의 유의미한 변화가 감지되었습니다"
        pattern = "지속적 하락" if any(c.direction == "down" for c in changes) else "지속적 상승" if any(c.direction == "up" for c in changes) else "변동"
        
        return TrendAnalysis(
            compared_to="지난 7일",
            changes=changes,
            alert_message=alert_message,
            pattern=pattern
        )
    
    def _create_ui_components(
        self, 
        status: StatusOverview, 
        analysis: ComprehensiveAnalysisResult
    ) -> UIComponents:
        """UI 컴포넌트 생성"""
        quick_stats = [
            QuickStat(
                label="기분",
                value=f"{analysis.emotion_analysis.positive}/100",
                emoji="😢" if analysis.emotion_analysis.positive < 50 else "😊",
                color=status.status_color
            ),
            QuickStat(
                label="활력",
                value="낮음" if analysis.emotion_analysis.depression > 50 else "보통",
                emoji="😴",
                color="#FF8800"
            )
        ]
        
        cta_buttons = [
            CTAButton(
                text="지금 전화하기",
                icon="📞",
                color="#FF4444",
                action="call"
            ),
            CTAButton(
                text="영상 전체보기",
                icon="🎬",
                color="#4444FF",
                action="watch_video"
            )
        ]
        
        if analysis.comprehensive_summary.priority_level == "긴급":
            cta_buttons.append(CTAButton(
                text="병원 예약하기",
                icon="🏥",
                color="#FF8800",
                action="book_hospital"
            ))
        
        return UIComponents(
            header={
                "badge_color": status.status_color,
                "badge_text": status.alert_title.split()[0],
                "title": status.alert_subtitle,
                "subtitle": f"오늘 {datetime.now().strftime('%H:%M')} 촬영"
            },
            quick_stats=quick_stats,
            cta_buttons=cta_buttons
        )
    
    def _create_default_action_plan(self, analysis: ComprehensiveAnalysisResult) -> ActionPlan:
        """기본 행동 계획 생성"""
        urgent_actions = [
            UrgentAction(
                action_id=1,
                priority="최우선",
                icon="📞",
                title="어머니께 안부 전화 드리기",
                reason="현재 상태 확인이 필요합니다",
                detail="어머니가 연락을 기다리고 계실 수 있습니다",
                deadline="오늘 중",
                estimated_time="10-15분",
                suggested_topics=[
                    "엄마 안 바빠요. 어디 불편하신 데 없으세요?",
                    "식사는 잘 하세요? 제가 반찬 좀 가져다 드릴게요",
                    "이번 주말에 갈게요. 뭔가 필요한 거 있으세요?"
                ]
            )
        ]
        
        return ActionPlan(
            urgent_actions=urgent_actions,
            this_week_actions=[],
            long_term_actions=[]
        )
    
    def _create_default_concerns(self, analysis: ComprehensiveAnalysisResult) -> List[KeyConcern]:
        """기본 걱정거리 생성"""
        concerns = []
        
        if analysis.emotion_analysis.depression > 70:
            concerns.append(KeyConcern(
                concern_id=1,
                type="정서",
                icon="💔",
                severity="caution",
                title="우울한 기분",
                description="평소보다 기분이 많이 가라앉아 있으신 것 같습니다. 관심을 더 기울여주시면 좋을 것 같습니다",
                detected_from=["대화", "표정"],
                urgency_reason="우울감 악화 가능성"
            ))
        
        if analysis.emotion_analysis.loneliness > 70:
            concerns.append(KeyConcern(
                concern_id=2,
                type="정서",
                icon="👥",
                severity="caution",
                title="외로움",
                description="혼자 계시는 시간이 많아 외로워하시는 것 같습니다. 이번 주말에 시간 내어 방문해주시면 좋겠습니다",
                detected_from=["대화"],
                urgency_reason="사회적 고립 우려"
            ))
        
        return concerns
    
    def _create_evidence_visualization(
        self,
        analysis: ComprehensiveAnalysisResult,
        conversation: str,
        audio_analysis: Dict,
        image_analysis: Dict,
        key_concerns: List[KeyConcern]
    ) -> EvidenceVisualization:
        """근거 시각화 데이터 생성"""
        emotion_evidence = analysis.emotion_analysis.evidence
        
        # 감정 키워드 추출
        emotion_keywords = []
        keyword_weights = {}
        
        if emotion_evidence:
            all_keywords = emotion_evidence.detected_keywords
            emotion_keywords = all_keywords[:10]  # 최대 10개
            
            # 키워드별 가중치 개선: 빈도×강도×최근성 (R6)
            keyword_counts = {}
            for keyword in all_keywords:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
            
            # 감정 강도 매핑 (높을수록 중요)
            emotion_intensity = {
                "우울": 0.9, "슬픔": 0.8, "외로움": 0.7, "불안": 0.8, "분노": 0.7,
                "무기력": 0.6, "피곤": 0.5, "행복": 0.3, "기쁨": 0.3
            }
            
            # 가중치 계산: 빈도 × 강도 (최근성은 일단 생략)
            raw_weights = {}
            for keyword, count in keyword_counts.items():
                intensity = emotion_intensity.get(keyword, 0.5)
                raw_weights[keyword] = count * intensity
            
            # Softmax 정규화 (합 = 1)
            total_weight = sum(raw_weights.values())
            if total_weight > 0:
                for keyword, weight in raw_weights.items():
                    keyword_weights[keyword] = weight / total_weight
            else:
                for keyword in all_keywords:
                    keyword_weights[keyword] = 1.0 / len(all_keywords) if all_keywords else 0.0
        
        # 표정 변화 타임라인 (신뢰도 기준 필터링)
        facial_timeline = []
        confidence_threshold = 60  # 60% 미만이면 감정 미검출로 처리
        if image_analysis.get("analysis"):
            img_data = image_analysis["analysis"]
            emotions = img_data.get("emotion", [])
            confidence = image_analysis.get("confidence", 0)
            
            # confidence가 낮으면 감정 미검출 처리
            if confidence >= confidence_threshold and emotions:
                # 텍스트 기반 감정 추출 (대화 맥락)
                text_emotions = set()
                if emotion_evidence and emotion_evidence.detected_keywords:
                    for kw in emotion_evidence.detected_keywords:
                        if "우울" in kw or "슬픔" in kw:
                            text_emotions.add("우울")
                        elif "피곤" in kw or "무기력" in kw:
                            text_emotions.add("무기력")
                        elif "외로움" in kw:
                            text_emotions.add("외로움")
                        elif "분노" in kw or "화" in kw:
                            text_emotions.add("분노")
                
                for i, emotion in enumerate(emotions[:5]):  # 최대 5개
                    # 각 감정별 confidence 계산 (전체 confidence를 기반으로)
                    emotion_confidence = max(confidence_threshold, confidence - (i * 5))
                    
                    # 맥락 충돌 감지 (R4): 텍스트와 표정 불일치 체크
                    context_mismatch = False
                    if text_emotions and emotion not in text_emotions:
                        # 예: 표정은 분노인데 텍스트는 우울/피곤
                        if emotion == "분노" and ("우울" in text_emotions or "무기력" in text_emotions):
                            context_mismatch = True
                        elif emotion in ["기쁨", "행복"] and ("우울" in text_emotions or "외로움" in text_emotions):
                            context_mismatch = True
                    
                    reliability = "높음" if emotion_confidence >= 80 else "보통" if emotion_confidence >= 60 else "낮음"
                    if context_mismatch:
                        reliability = "보류"
                    
                    facial_timeline.append({
                        "timestamp": f"00:0{i*10}:00",
                        "emotion": emotion,
                        "confidence": emotion_confidence,
                        "reliability": reliability,
                        "note": "텍스트/음성 맥락과 불일치하여 검증 필요" if context_mismatch else None
                    })
            else:
                # confidence가 낮으면 감정 미검출로 표시
                facial_timeline.append({
                    "timestamp": "00:00:00",
                    "emotion": "감정 미검출",
                    "confidence": confidence,
                    "reliability": "낮음",
                    "note": f"표정 분석 확실도 {confidence}%로 신중한 해석이 필요합니다"
                })
        
        # 음성 에너지 파형 데이터 (간단한 휴리스틱)
        voice_waveform = None
        if audio_analysis.get("shout_detection"):
            shout_data = audio_analysis["shout_detection"]
            voice_waveform = {
                "average_energy": shout_data.get("average_energy", 0.5),
                "max_energy": shout_data.get("max_energy", 0.7),
                "detected_shout": shout_data.get("detected_shout", False),
                "energy_level": "높음" if shout_data.get("average_energy", 0.5) > 0.6 else "보통"
            }
        
        # 점수별 세부 분석
        score_breakdown = {}
        if emotion_evidence:
            score_breakdown = {
                "positive": {
                    "score": analysis.emotion_analysis.positive,
                    "factors": emotion_evidence.positive_factors[:3],
                    "explanation": f"긍정 점수는 {', '.join(emotion_evidence.positive_factors[:2])} 등의 요인으로 계산되었습니다" if emotion_evidence.positive_factors else "긍정적 표현이 감지되지 않았습니다"
                },
                "depression": {
                    "score": analysis.emotion_analysis.depression,
                    "factors": emotion_evidence.depression_factors[:3],
                    "explanation": f"우울 점수는 {', '.join(emotion_evidence.depression_factors[:2])} 등의 요인으로 계산되었습니다" if emotion_evidence.depression_factors else "우울 지표가 낮습니다"
                },
                "anxiety": {
                    "score": analysis.emotion_analysis.anxiety,
                    "factors": emotion_evidence.anxiety_factors[:3],
                    "explanation": f"불안 점수는 {', '.join(emotion_evidence.anxiety_factors[:2])} 등의 요인으로 계산되었습니다" if emotion_evidence.anxiety_factors else "불안 지표가 낮습니다"
                }
            }
        else:
            # evidence가 없으면 기본값
            score_breakdown = {
                "positive": {
                    "score": analysis.emotion_analysis.positive,
                    "factors": [],
                    "explanation": "대화 내용 분석 기반으로 계산되었습니다"
                },
                "depression": {
                    "score": analysis.emotion_analysis.depression,
                    "factors": [],
                    "explanation": "대화 내용 분석 기반으로 계산되었습니다"
                }
            }
        
        # 멀티모달 신뢰도 가중 평균 계산 (R4)
        text_confidence = 0.75  # 텍스트 분석 기본 신뢰도
        audio_confidence = 0.65 if audio_analysis.get("shout_detection") else 0.60
        face_confidence = image_analysis.get("confidence", 0) / 100.0 if image_analysis.get("analysis") else 0
        
        # 가중치: text 0.6, audio 0.25, face 0.15
        w_text, w_audio, w_face = 0.6, 0.25, 0.15
        overall_confidence = (w_text * text_confidence + w_audio * audio_confidence + w_face * face_confidence) * 100
        
        # 계산 방법 설명 + 멀티모달 확실도 표시
        calculation_method = f"감정 점수는 대화 내용(가중치 60%), 음성 톤(가중치 25%), 표정 분석(가중치 15%)을 종합하여 계산합니다. "
        calculation_method += f"전체 신뢰도: {overall_confidence:.1f}% (텍스트 {text_confidence*100:.0f}%, 음성 {audio_confidence*100:.0f}%, 표정 {face_confidence*100:.0f}%). "
        calculation_method += "각 점수는 0-100 범위이며, 여러 요인을 고려하여 결정됩니다."
        
        if emotion_evidence and emotion_evidence.facial_expression_notes:
            calculation_method += f" 표정 분석 결과: '{emotion_evidence.facial_expression_notes[:50]}...'"
        
        return EvidenceVisualization(
            emotion_keywords=emotion_keywords,
            keyword_weights=keyword_weights,
            facial_expression_timeline=facial_timeline,
            voice_energy_waveform=voice_waveform,
            score_breakdown=score_breakdown,
            calculation_method=calculation_method
        )
    
    def _create_baseline_comparison(self, analysis: ComprehensiveAnalysisResult) -> Optional[Dict]:
        """Baseline 비교 데이터 생성 (명확한 표현 필수 포함)"""
        baseline_comparisons = analysis.anomaly_analysis.baseline_comparisons
        
        # baseline_comparisons가 없어도 기본 정보는 제공
        if not baseline_comparisons:
            # 현재 값만이라도 표시
            return {
                "comparison_period": "지난 7일 (데이터 부족)",
                "current_values": {
                    "긍정 감정": analysis.emotion_analysis.positive,
                    "우울 감정": analysis.emotion_analysis.depression,
                    "외로움 감정": analysis.emotion_analysis.loneliness
                },
                "summary": "과거 데이터가 부족하여 개인 평균 비교는 어렵습니다. 현재 상태만 확인됩니다.",
                "note": "일주일 이상 데이터가 쌓이면 개인 평균 대비 변화를 확인할 수 있습니다."
            }
        
        # 모든 변화 포함 (유의미한 것과 아닌 것 구분)
        all_changes = []
        significant_changes = []
        
        for comp in baseline_comparisons:
            change_data = {
                "metric": comp.metric,
                "current": comp.current_value,
                "baseline": comp.baseline_average,
                "difference": comp.difference,
                "difference_pct": comp.difference_percentage,
                "is_significant": comp.is_significant_change,
                "explanation": f"평소 평균 {comp.baseline_average:.1f}점 → 오늘 {comp.current_value:.1f}점 ({comp.difference:+.1f}점, {comp.difference_percentage:+.1f}%)"
            }
            all_changes.append(change_data)
            if comp.is_significant_change:
                significant_changes.append(change_data)
        
        # 가장 중요한 변화 추출 (긍정 감정 또는 우울 감정)
        mood_change = None
        for comp in baseline_comparisons:
            if comp.metric in ["긍정 감정", "우울 감정"]:
                mood_change = f"평소 평균 {comp.baseline_average:.0f}점 → 오늘 {comp.current_value:.0f}점 ({comp.difference:+.0f}점 감소)" if comp.difference < 0 else f"평소 평균 {comp.baseline_average:.0f}점 → 오늘 {comp.current_value:.0f}점 ({comp.difference:+.0f}점 증가)"
                break
        
        summary = ""
        if significant_changes:
            summary = f"지난 7일 평균 대비 {len(significant_changes)}개의 유의미한 변화가 감지되었습니다. "
            if mood_change:
                summary += mood_change
        else:
            summary = "지난 7일 평균 대비 큰 변화가 없습니다."
            if mood_change:
                summary += f" ({mood_change})"
        
        return {
            "comparison_period": "지난 7일",
            "all_changes": all_changes,
            "significant_changes": significant_changes,
            "summary": summary,
            "mood_comparison": mood_change  # 가장 중요한 비교 정보
        }
    
    def _create_medical_disclaimer(
        self, 
        analysis: ComprehensiveAnalysisResult, 
        action_plan: ActionPlan,
        key_concerns: List[KeyConcern]
    ) -> MedicalDisclaimer:
        """의료 책임 면책 조항 생성 (R2: action_plan과 일치)"""
        # 고정 면책 조항
        disclaimer_text = (
            "본 분석 결과는 참고용 정보이며, 의료 진단이 아닙니다. "
            "우려가 지속되면 의료진과 상담하세요."
        )
        
        # action_plan의 urgent_actions에서 건강 관련 액션 확인
        health_urgent_actions = []
        health_keywords = ["의사", "병원", "상담", "진료", "약", "증상", "통증", "식사", "음식"]
        
        for action in action_plan.urgent_actions:
            if any(keyword in action.title or keyword in action.detail for keyword in health_keywords):
                health_urgent_actions.append(action)
        
        # key_concerns에서 건강 관련 urgent 확인
        health_urgent_concerns = [c for c in key_concerns if c.type == "건강" and c.severity == "urgent"]
        
        # suggested_action 생성 (R2: action_plan과 일치)
        if health_urgent_actions or health_urgent_concerns:
            # urgent 액션이 있으면 구체적으로 표시
            action_titles = [a.title for a in health_urgent_actions[:2]]
            if action_titles:
                suggested_action = f"이번 주 내 가벼운 진료 예약 권장 ({', '.join(action_titles[:2])})"
            else:
                concern_titles = [c.title for c in health_urgent_concerns[:2]]
                suggested_action = f"이번 주 내 가벼운 진료 예약 권장 ({', '.join(concern_titles[:2])})"
        else:
            # 일반적인 건강 관련 권장사항만 있는 경우
            has_health_mention = any(keyword in action.title for action in action_plan.this_week_actions for keyword in health_keywords)
            if has_health_mention:
                suggested_action = "건강 관련 우려사항이 있으니 의료진 상담을 권장합니다."
            else:
                suggested_action = "현재 건강 관련 권장사항은 없습니다."
        
        return MedicalDisclaimer(
            disclaimer_text=disclaimer_text,
            is_recommendation_not_diagnosis=True,
            suggested_action=suggested_action
        )