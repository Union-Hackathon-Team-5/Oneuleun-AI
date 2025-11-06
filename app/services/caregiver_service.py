import asyncio
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

from app.models.caregiver_models import (
    CaregiverFriendlyResponse, StatusOverview, TodaySummary, KeyConcern,
    ActionPlan, UrgentAction, DetailedAnalysis, TrendAnalysis, TrendChange,
    UIComponents, QuickStat, CTAButton, EmotionTimeline, VideoHighlight,
    RiskIndicator, AudioAnalysis, ConversationTopic
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
        user_id: str
    ) -> CaregiverFriendlyResponse:
        """보호자 친화적 리포트 생성"""
        
        # 기존 기술적 분석 실행
        comprehensive_analysis = await self.analysis_service.analyze_video_letter_comprehensive(
            conversation=conversation,
            image_analysis=image_analysis
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
        
        # 1순위: 상태 개요
        status_overview = self._create_status_overview(comprehensive_analysis)
        
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
            ui_components=ui_components
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
        """실행 가능한 행동 계획 생성"""
        prompt = f"""
다음 분석 결과를 바탕으로 보호자가 실제로 실행할 수 있는 구체적인 행동 계획을 세워주세요.

대화 내용: {conversation[:500]}...
위험도: {analysis.comprehensive_summary.priority_level}
주요 우려사항: {analysis.comprehensive_summary.key_concerns}
권장 조치: {analysis.comprehensive_summary.recommended_actions}

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

실제 대화 예시:
- "엄마 안 바빠요. 어디 불편하신 데 없으세요?"
- "식사는 잘 하세요? 제가 반찬 좀 가져다 드릴게요"
- "이번 주말에 갈게요. 병원 같이 가실까요?"
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
        """주요 걱정거리 식별"""
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
            "description": "<상세 설명>",
            "detected_from": ["대화", "표정"],
            "urgency_reason": "<왜 긴급한지>"
        }}
    ]
}}

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
    
    def _create_status_overview(self, analysis: ComprehensiveAnalysisResult) -> StatusOverview:
        """상태 개요 생성"""
        risk_level = analysis.comprehensive_summary.priority_level
        
        if risk_level == "긴급":
            return StatusOverview(
                alert_level="urgent",
                alert_badge="🚨",
                alert_title="즉시 확인 필요",
                alert_subtitle="어머니께서 도움이 필요하신 것 같습니다",
                status_color="#FF4444"
            )
        elif risk_level == "주의":
            return StatusOverview(
                alert_level="caution",
                alert_badge="⚠️",
                alert_title="주의 깊게 살펴보세요",
                alert_subtitle="평소와 다른 점들이 보입니다",
                status_color="#FF8800"
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
        """오늘 요약 생성"""
        mood_score = analysis.emotion_analysis.positive
        
        if mood_score >= 70:
            mood_label = "좋음"
            mood_emoji = "😊"
        elif mood_score >= 50:
            mood_label = "보통"
            mood_emoji = "😐"
        elif mood_score >= 30:
            mood_label = "우울함"
            mood_emoji = "😔"
        else:
            mood_label = "매우 우울함"
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
                description="평소보다 기분이 많이 가라앉아 있으신 것 같습니다",
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
                description="혼자 계시는 시간이 많아 외로워하십니다",
                detected_from=["대화"],
                urgency_reason="사회적 고립 우려"
            ))
        
        return concerns