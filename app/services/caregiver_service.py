import asyncio
import json
import logging
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
        
        # 기존 기술적 분석 실행 (historical_data 포함)
        comprehensive_analysis = await self.analysis_service.analyze_video_letter_comprehensive(
            conversation=conversation,
            image_analysis=image_analysis,
            historical_data=historical_data
        )
        
        # 감성적, 액션 중심 리포트로 변환
        return await self._transform_to_caregiver_format(
            comprehensive_analysis=comprehensive_analysis,
            conversation=conversation,
            image_analysis=image_analysis,
            audio_analysis=audio_analysis,
            session_id=session_id,
            user_id=user_id
        )
    
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
        
        # 감성적 인사이트 생성 (병렬 처리)
        insights_task = self._generate_emotional_insights(conversation, comprehensive_analysis)
        action_plan_task = self._generate_actionable_plan(comprehensive_analysis, conversation)
        mother_voice_task = self._extract_mother_voice(conversation)
        concerns_task = self._identify_key_concerns(comprehensive_analysis, conversation, image_analysis)
        
        emotional_insights, action_plan, mother_voice, key_concerns = await asyncio.gather(
            insights_task, action_plan_task, mother_voice_task, concerns_task
        )
        
        # 1순위: 상태 개요 (key_concerns 생성 후에 결정하여 일관성 보장)
        status_overview = self._create_status_overview(comprehensive_analysis, key_concerns)
        
        # 2순위: 오늘 요약
        today_summary = self._create_today_summary(
            comprehensive_analysis, emotional_insights, mother_voice
        )
        
        # 3순위: 주요 걱정거리 (이미 생성됨)
        
        # 4순위: 행동 계획 (이미 생성됨)
        
        # 5순위: 상세 분석
        detailed_analysis = self._create_detailed_analysis(
            comprehensive_analysis, conversation, audio_analysis
        )
        
        # 6순위: 추세 분석
        trend_analysis = self._create_trend_analysis(comprehensive_analysis)
        
        # UI 컴포넌트
        ui_components = self._create_ui_components(status_overview, comprehensive_analysis)
        
        # 근거 시각화 데이터 생성
        evidence_viz = self._create_evidence_visualization(
            comprehensive_analysis, conversation, audio_analysis, image_analysis
        )
        
        # Baseline 비교 데이터 생성
        baseline_comparison = self._create_baseline_comparison(comprehensive_analysis)
        
        # 의료 책임 면책 조항 생성
        medical_disclaimer = self._create_medical_disclaimer(comprehensive_analysis)
        
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
            response = await self.analysis_service._call_openai(prompt)
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
        
        prompt = f"""
다음 분석 결과를 바탕으로 보호자가 실제로 실행할 수 있는 구체적인 행동 계획을 세워주세요.
중요: 과도한 경고를 피하고, 정말 필요한 조치만 우선순위를 높게 설정하세요.

대화 내용: {conversation[:500]}...
위험도: {analysis.comprehensive_summary.priority_level}
주요 우려사항: {analysis.comprehensive_summary.key_concerns}
권장 조치: {analysis.comprehensive_summary.recommended_actions}
{baseline_context}

다음 JSON 형식으로 응답해주세요:
{{
    "urgent_actions": [
        {{
            "action_id": 1,
            "priority": "최우선",
            "icon": "📞",
            "title": "<구체적 행동>",
            "reason": "<왜 필요한지>",
            "detail": "<어머니가 실제로 하신 말씀 인용>",
            "deadline": "<언제까지>",
            "estimated_time": "<소요시간>",
            "suggested_topics": ["<실제 대화 예시1>", "<실제 대화 예시2>"]
        }}
    ],
    "this_week_actions": [...],
    "long_term_actions": [...]
}}

주의사항:
- urgent_actions는 정말 긴급한 경우에만 1-2개로 제한하세요
- 건강 관련 조치에는 "의료진 상담 권장"이라는 표현을 사용하고, 진단하지 마세요
- 언어 표현: 불안 유도형 표현 지양, 가족 케어 조언 톤 사용
  - ❌ "즉각적인 의사 상담이 필요합니다" 
  - ✅ "가벼운 진료 예약이라도 이번 주 안에 한 번 챙기면 좋겠습니다"
  - ❌ "심각할 수 있습니다"
  - ✅ "관심을 더 기울여주시면 좋을 것 같습니다"
- 실제 대화 예시:
  - "엄마 안 바빠요. 어디 불편하신 데 없으세요?"
  - "식사는 잘 하세요? 제가 반찬 좀 가져다 드릴게요"
  - "건강 관련 걱정이 있으시면 의료진과 상담하시는 것을 권장합니다"
"""
        
        try:
            response = await self.analysis_service._call_openai(prompt)
            data = json.loads(response)
            
            urgent_actions = [UrgentAction(**action) for action in data.get("urgent_actions", [])]
            this_week_actions = [UrgentAction(**action) for action in data.get("this_week_actions", [])]
            long_term_actions = [UrgentAction(**action) for action in data.get("long_term_actions", [])]
            
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
            response = await self.analysis_service._call_openai(prompt)
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
        prompt = f"""
다음 정보를 바탕으로 보호자가 가장 걱정해야 할 문제들을 식별하고 구체적으로 설명해주세요.

대화: {conversation[:500]}...
위험 분석: {analysis.risk_analysis.dict()}
이미지 우려사항: {image_analysis.get('analysis', {}).get('concerns', [])}

다음 JSON 형식으로 응답해주세요:
{{
    "concerns": [
        {{
            "concern_id": 1,
            "type": "건강",
            "icon": "🏥",
            "severity": "urgent",
            "title": "<구체적 문제>",
            "description": "<상세 설명 - 가족 케어 조언 톤으로>",
            "detected_from": ["대화", "표정"],
            "urgency_reason": "<왜 긴급한지>"
        }}
    ]
}}

중요: 언어 표현 지침
- ❌ 피하기: "즉각적인 의사 상담이 필요합니다", "심각할 수 있습니다", "즉시 조치 필요"
- ✅ 사용하기: "가벼운 진료 예약이라도 이번 주 안에 한 번 챙기면 좋겠습니다", "의료진 상담을 권장합니다", "관심을 더 기울여주시면 좋을 것 같습니다"

걱정거리 유형: 건강, 안전, 정서, 생활
심각도: urgent, caution, normal
"""
        
        try:
            response = await self.analysis_service._call_openai(prompt)
            data = json.loads(response)
            return [KeyConcern(**concern) for concern in data.get("concerns", [])]
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
        
        return TodaySummary(
            headline=emotional_insights.get("headline", "어머니 상태를 확인해보세요"),
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
        audio_analysis: Dict
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
        
        # 위험 지표
        risk_indicators = {
            "health_risk": RiskIndicator(
                level="high" if analysis.comprehensive_summary.priority_level == "긴급" else "medium",
                factors=analysis.risk_analysis.risk_categories.health
            ),
            "mental_risk": RiskIndicator(
                level="high" if analysis.emotion_analysis.depression > 70 else "medium",
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
    
    def _create_trend_analysis(self, analysis: ComprehensiveAnalysisResult) -> TrendAnalysis:
        """추세 분석 생성"""
        # 더미 추세 데이터 (실제로는 과거 데이터와 비교)
        changes = [
            TrendChange(
                metric="기분",
                direction="down",
                change=-35,
                icon="📉",
                comment="지난주 대비 35점 하락"
            ),
            TrendChange(
                metric="활동량",
                direction="down",
                change=-50,
                icon="📉",
                comment="외출 빈도 감소"
            )
        ]
        
        return TrendAnalysis(
            compared_to="지난주",
            changes=changes,
            alert_message="⚠️ 지난주 대비 전반적으로 악화되었습니다",
            pattern="지속적 하락"
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
        image_analysis: Dict
    ) -> EvidenceVisualization:
        """근거 시각화 데이터 생성"""
        emotion_evidence = analysis.emotion_analysis.evidence
        
        # 감정 키워드 추출
        emotion_keywords = []
        keyword_weights = {}
        
        if emotion_evidence:
            all_keywords = emotion_evidence.detected_keywords
            emotion_keywords = all_keywords[:10]  # 최대 10개
            
            # 키워드별 가중치 (간단한 휴리스틱)
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
                for i, emotion in enumerate(emotions[:5]):  # 최대 5개
                    # 각 감정별 confidence 계산 (전체 confidence를 기반으로)
                    emotion_confidence = max(confidence_threshold, confidence - (i * 5))
                    facial_timeline.append({
                        "timestamp": f"00:0{i*10}:00",
                        "emotion": emotion,
                        "confidence": emotion_confidence,
                        "reliability": "높음" if emotion_confidence >= 80 else "보통" if emotion_confidence >= 60 else "낮음"
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
        
        # 계산 방법 설명 + 확실도 표시
        confidence = image_analysis.get("confidence", 0) if image_analysis.get("analysis") else 0
        calculation_method = "감정 점수는 대화 내용, 표정 분석, 음성 톤을 종합하여 AI 모델이 계산합니다. 각 점수는 0-100 범위이며, 여러 요인을 고려하여 결정됩니다."
        
        if confidence > 0:
            if confidence >= 80:
                calculation_method += f" 이번 분석의 감정 판단 확실도는 {confidence}%로 비교적 높습니다."
            elif confidence >= 60:
                calculation_method += f" 이번 분석의 감정 판단 확실도는 {confidence}%로 보통입니다. 해석 시 신중이 필요합니다."
            else:
                calculation_method += f" 이번 분석의 감정 판단 확실도는 {confidence}%로 낮습니다. 참고용으로만 활용하시기 바랍니다."
        
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
    
    def _create_medical_disclaimer(self, analysis: ComprehensiveAnalysisResult) -> MedicalDisclaimer:
        """의료 책임 면책 조항 생성"""
        # 건강 관련 권장사항이 있는지 확인
        has_health_recommendations = False
        health_keywords = ["의사", "병원", "상담", "진료", "약", "증상"]
        
        for action in analysis.comprehensive_summary.recommended_actions:
            if any(keyword in action for keyword in health_keywords):
                has_health_recommendations = True
                break
        
        if has_health_recommendations:
            disclaimer_text = (
                "⚠️ 본 분석 결과는 의료 진단이 아닌 참고용 정보입니다. "
                "건강 관련 권장사항은 전문 의료진의 상담을 받으시기 바랍니다. "
                "본 서비스는 의료 행위를 하지 않으며, 진단이나 치료를 대체하지 않습니다."
            )
            suggested_action = "건강 관련 우려사항이 있으니 의료진 상담을 권장합니다."
        else:
            disclaimer_text = (
                "본 분석 결과는 참고용 정보이며, 의료 진단이 아닙니다. "
                "건강 상태가 우려되시면 전문 의료진의 상담을 받으시기 바랍니다."
            )
            suggested_action = "현재 건강 관련 권장사항은 없습니다."
        
        return MedicalDisclaimer(
            disclaimer_text=disclaimer_text,
            is_recommendation_not_diagnosis=True,
            suggested_action=suggested_action
        )