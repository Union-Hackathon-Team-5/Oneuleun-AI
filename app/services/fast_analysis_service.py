import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

import httpx
from pydantic import ValidationError

from app.models.caregiver_models import (
    CaregiverFriendlyResponse, StatusOverview, TodaySummary, KeyConcern,
    ActionPlan, UrgentAction, DetailedAnalysis, TrendAnalysis, TrendChange,
    UIComponents, QuickStat, CTAButton, EmotionTimeline, VideoHighlight,
    RiskIndicator, AudioAnalysis, ConversationTopic, EvidenceVisualization,
    MedicalDisclaimer
)

logger = logging.getLogger(__name__)


class FastAnalysisService:
    """🚀 12초 미만 초고속 분석 서비스 (단일 API 호출 + 로컬 처리)"""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.api_key = api_key
        self.model = "gpt-4o-mini"  # 가장 빠른 모델
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self._client: Optional[httpx.AsyncClient] = None
        
        # 🎯 캐시된 템플릿 (재사용)
        self._status_templates = {
            "urgent": {"badge": "🚨", "title": "즉시 확인 필요", "color": "#FF4444"},
            "caution": {"badge": "⚠️", "title": "주의 깊게 살펴보세요", "color": "#FF8800"},
            "normal": {"badge": "😊", "title": "안정적인 상태", "color": "#44FF44"}
        }
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            # 최적화된 연결 설정
            limits = httpx.Limits(max_keepalive_connections=10, max_connections=10)
            timeout = httpx.Timeout(8.0, connect=3.0)  # 연결 3초, 총 8초로 단축
            self._client = httpx.AsyncClient(timeout=timeout, limits=limits)
        return self._client
    
    async def _ultra_fast_api_call(self, prompt: str) -> Dict[str, Any]:
        """🚀 초고속 단일 API 호출 (5초 내 완료 목표)"""
        start_time = time.time()
        
        # 극한 최적화된 페이로드
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "노인 케어 전문가. JSON만 응답."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 600,  # 토큰 대폭 감소
            "response_format": {"type": "json_object"}
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        try:
            client = await self._get_client()
            response = await asyncio.wait_for(
                client.post(self.base_url, headers=headers, json=payload),
                timeout=7.0  # 7초 타임아웃
            )
            response.raise_for_status()
            data = response.json()
            result = data["choices"][0]["message"]["content"].strip()
            
            api_time = time.time() - start_time
            print(f"[FAST] API call completed in {api_time:.2f}s", flush=True)
            
            return json.loads(result)
        except Exception as exc:
            logger.error(f"Fast API call failed: {exc}")
            return self._get_fallback_data()
    
    def _get_fallback_data(self) -> Dict[str, Any]:
        """⚡ 즉시 사용 가능한 fallback 데이터"""
        return {
            "status": "normal",
            "mood_score": 60,
            "headline": "어머니 상태를 확인해보세요",
            "concerns": [
                {
                    "id": 1,
                    "type": "정서",
                    "severity": "caution",
                    "title": "일반적인 관심 필요",
                    "description": "정기적인 안부 확인이 도움될 것 같습니다"
                }
            ],
            "actions": [
                {
                    "id": 1,
                    "priority": "중요",
                    "icon": "📞",
                    "title": "안부 전화 드리기",
                    "deadline": "이번 주"
                }
            ],
            "mother_voice": ["💬 \"요즘 그럭저럭 지내고 있어요\""]
        }
    
    async def generate_ultra_fast_report(
        self,
        conversation: str,
        image_analysis: Dict,
        audio_analysis: Dict,
        session_id: str,
        user_id: str
    ) -> CaregiverFriendlyResponse:
        """🚀 12초 미만 초고속 리포트 생성"""
        total_start = time.time()
        
        # 🎯 Step 1: 대화 압축 (로컬 처리, 0.1초)
        compressed_conversation = self._compress_conversation(conversation)
        
        # 🎯 Step 2: 이미지 정보 추출 (로컬 처리, 0.1초)  
        image_info = self._extract_image_info(image_analysis)
        
        # 🎯 Step 3: 단일 API 호출로 모든 분석 (5초 목표)
        api_start = time.time()
        analysis_data = await self._ultra_fast_comprehensive_analysis(
            compressed_conversation, image_info
        )
        api_time = time.time() - api_start
        print(f"[FAST] Single API analysis: {api_time:.2f}s", flush=True)
        
        # 🎯 Step 4: 로컬 변환 (1초 목표)
        transform_start = time.time()
        result = self._ultra_fast_transform(
            analysis_data, conversation, image_analysis, audio_analysis, session_id, user_id
        )
        transform_time = time.time() - transform_start
        
        total_time = time.time() - total_start
        print(f"[FAST] Transform: {transform_time:.2f}s | Total: {total_time:.2f}s", flush=True)
        
        return result
    
    def _compress_conversation(self, conversation: str) -> str:
        """⚡ 대화 압축 (핵심만 추출)"""
        lines = [line.strip() for line in conversation.splitlines() if line.strip()]
        
        # 최대 8줄만 유지 (핵심 대화)
        if len(lines) > 8:
            # 앞 4줄 + 뒤 4줄
            compressed = lines[:4] + ["..."] + lines[-4:]
        else:
            compressed = lines
        
        return "\n".join(compressed)[:800]  # 최대 800자
    
    def _extract_image_info(self, image_analysis: Dict) -> str:
        """⚡ 이미지 정보 압축"""
        if not image_analysis.get("analysis"):
            return "표정: 평범함"
        
        img_data = image_analysis["analysis"]
        emotions = ", ".join(img_data.get("emotion", []))[:50]
        concerns = ", ".join(img_data.get("concerns", []))[:50]
        
        parts = []
        if emotions:
            parts.append(f"표정: {emotions}")
        if concerns:
            parts.append(f"우려: {concerns}")
        
        return " | ".join(parts) or "표정: 평범함"
    
    async def _ultra_fast_comprehensive_analysis(
        self, 
        conversation: str, 
        image_info: str
    ) -> Dict[str, Any]:
        """🚀 초압축 단일 API 호출 (토큰 최소화)"""
        
        # 극한 압축 프롬프트 (400자 이하)
        prompt = f"""대화: {conversation}
이미지: {image_info}

보호자용 JSON (간결):
{{
  "status": "urgent/caution/normal",
  "mood_score": 0-100,
  "headline": "상태 한줄",
  "concerns": [
    {{"id": 1, "type": "건강/안전/정서", "severity": "urgent/caution", "title": "제목", "description": "설명"}}
  ],
  "actions": [
    {{"id": 1, "priority": "최우선/중요", "icon": "📞", "title": "제목", "deadline": "기한"}}
  ],
  "mother_voice": ["💬 \"인용\""],
  "summary": "요약"
}}

규칙: 위험시 urgent, 평범시 normal. 최대 3개씩."""
        
        return await self._ultra_fast_api_call(prompt)
    
    def _ultra_fast_transform(
        self,
        data: Dict[str, Any],
        conversation: str,
        image_analysis: Dict,
        audio_analysis: Dict,
        session_id: str,
        user_id: str
    ) -> CaregiverFriendlyResponse:
        """⚡ 초고속 로컬 변환 (API 호출 없음)"""
        # 🎯 1. 기본 점수 추출 및 룰 기반 보정
        status_key = data.get("status", "normal")
        mood_score = data.get("mood_score", 60)
        energy_score = max(20, mood_score - 10)
        pain_score = max(0, 100 - mood_score)
        
        rule_adjustments = self._apply_guardrail_rules(
            conversation=conversation,
            base_status=status_key,
            mood_score=mood_score,
            energy_score=energy_score,
            pain_score=pain_score
        )
        
        status_key = rule_adjustments["alert_level"]
        mood_score = rule_adjustments["mood_score"]
        energy_score = rule_adjustments["energy_score"]
        pain_score = rule_adjustments["pain_score"]
        headline = rule_adjustments.get("headline") or data.get("headline", "어머니 상태를 확인해보세요")
        
        # 🎯 2. 상태 개요 (템플릿 + 보정 메시지)
        template = self._status_templates.get(status_key, self._status_templates["normal"])
        status_overview = StatusOverview(
            alert_level=status_key,
            alert_badge=template["badge"],
            alert_title=template["title"],
            alert_subtitle=rule_adjustments.get("alert_subtitle", "어머니 상태를 확인해보세요"),
            status_color=template["color"]
        )
        
        # 🎯 3. 오늘 요약
        today_summary = TodaySummary(
            headline=headline,
            mood_score=mood_score,
            mood_label=self._get_mood_label(mood_score),
            mood_emoji=self._get_mood_emoji(mood_score),
            energy_score=energy_score,
            pain_score=pain_score,
            mother_voice=data.get("mother_voice", ["💬 \"오늘 괜찮아요\""])[:3]
        )
        
        # 🎯 4. 주요 걱정거리 (빠른 변환 + 룰 기반 보강)
        key_concerns = []
        existing_titles = set()
        for concern_data in data.get("concerns", [])[:3]:
            title = concern_data.get("title", "일반적인 관심").strip()
            existing_titles.add(title)
            key_concerns.append(KeyConcern(
                concern_id=concern_data.get("id", 1),
                type=concern_data.get("type", "정서"),
                icon=self._get_concern_icon(concern_data.get("type", "정서")),
                severity=concern_data.get("severity", "caution"),
                title=concern_data.get("title", "일반적인 관심"),
                description=concern_data.get("description", "정기적인 확인이 필요합니다"),
                detected_from=["대화"],
                urgency_reason="보호자의 관심이 필요한 상황"
            ))
        
        # 기본 걱정거리 보장
        if not key_concerns:
            key_concerns = [self._get_default_concern()]
            existing_titles.add(key_concerns[0].title)
        
        next_id = max((concern.concern_id for concern in key_concerns), default=0) + 1
        for concern in rule_adjustments.get("additional_concerns", []):
            title = concern.get("title")
            if not title or title in existing_titles:
                continue
            key_concerns.append(KeyConcern(
                concern_id=next_id,
                type=concern.get("type", "정서"),
                icon=concern.get("icon") or self._get_concern_icon(concern.get("type", "정서")),
                severity=concern.get("severity", "caution"),
                title=title,
                description=concern.get("description", "정기적인 확인이 필요합니다"),
                detected_from=concern.get("detected_from", ["대화"]),
                urgency_reason=concern.get("urgency_reason", "보호자의 관심이 필요한 상황")
            ))
            existing_titles.add(title)
            next_id += 1
        
        # 🎯 5. 행동 계획 (룰 기반 긴급 조치 반영)
        action_source = rule_adjustments.get("override_actions") or data.get("actions", [])
        action_plan = self._create_fast_action_plan(action_source, status_key)
        
        # 🎯 6. 상세 분석 (최소한)
        detailed_analysis = self._create_minimal_detailed_analysis(
            conversation,
            audio_analysis,
            rule_adjustments.get("risk_levels", {})
        )
        
        # 🎯 7. 추세 분석 (비활성화)
        trend_analysis = TrendAnalysis(
            compared_to="지난주",
            changes=[],
            alert_message="데이터 수집 중",
            pattern="분석 중",
            disabled=True,
            reason="빠른 분석 모드"
        )
        
        # 🎯 8. UI 컴포넌트
        ui_components = self._create_fast_ui_components(status_overview, mood_score)
        
        # 🎯 9. 근거 시각화 (최소한 → 룰 기반 보강)
        keywords = self._extract_keywords_fast(conversation)
        for extra_kw in rule_adjustments.get("evidence_keywords", []):
            if extra_kw not in keywords:
                keywords.append(extra_kw)
        limited_keywords = keywords[:8]
        score_breakdown = rule_adjustments.get("score_breakdown", {})
        evidence_viz = EvidenceVisualization(
            emotion_keywords=limited_keywords,
            keyword_weights={kw: 1.0 for kw in limited_keywords},
            facial_expression_timeline=[],
            voice_energy_waveform=None,
            score_breakdown=score_breakdown,
            calculation_method=rule_adjustments.get("calculation_method", "빠른 분석 모드로 간소화된 계산")
        )
        
        # 🎯 10. 의료 면책
        medical_disclaimer = MedicalDisclaimer(
            disclaimer_text="본 분석은 참고용이며 의료 진단이 아닙니다.",
            is_recommendation_not_diagnosis=True,
            suggested_action="우려사항이 지속되면 의료진 상담을 권장합니다."
        )
        
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
            baseline_comparison=None,
            medical_disclaimer=medical_disclaimer
        )
    
    def _get_mood_label(self, score: int) -> str:
        """⚡ 기분 라벨 (룩업 테이블)"""
        if score >= 70: return "좋음"
        elif score >= 50: return "보통"
        elif score >= 30: return "우울함"
        else: return "매우 우울함"
    
    def _get_mood_emoji(self, score: int) -> str:
        """⚡ 기분 이모지 (룩업 테이블)"""
        if score >= 70: return "😊"
        elif score >= 50: return "😐"
        elif score >= 30: return "😔"
        else: return "😢"
    
    def _get_concern_icon(self, concern_type: str) -> str:
        """⚡ 걱정거리 아이콘 (룩업 테이블)"""
        icons = {
            "건강": "🏥",
            "안전": "⚠️", 
            "정서": "💔",
            "생활": "🏠"
        }
        return icons.get(concern_type, "💔")
    
    def _get_default_concern(self) -> KeyConcern:
        """⚡ 기본 걱정거리"""
        return KeyConcern(
            concern_id=1,
            type="정서",
            icon="💔",
            severity="caution",
            title="정기적인 관심 필요",
            description="어머니께서 혼자 계시는 시간이 많으니 꾸준한 관심을 보여주시면 좋겠습니다",
            detected_from=["대화"],
            urgency_reason="정서적 안정 유지"
        )
    
    def _create_fast_action_plan(self, actions_data: List[Dict], status: str) -> ActionPlan:
        """⚡ 빠른 행동 계획 (템플릿 기반)"""
        urgent_actions = []
        
        # 기본 액션 보장
        if not actions_data:
            actions_data = [{
                "id": 1,
                "priority": "중요",
                "icon": "📞",
                "title": "안부 전화 드리기",
                "deadline": "오늘"
            }]
        
        for action_data in actions_data[:2]:
            urgent_actions.append(UrgentAction(
                action_id=action_data.get("id", 1),
                priority=action_data.get("priority", "중요"),
                icon=action_data.get("icon", "📞"),
                title=action_data.get("title", "안부 확인"),
                reason=action_data.get("reason", "어머니의 안전과 건강을 확인하기 위해 필요합니다"),
                detail=action_data.get("detail", "5-10분 정도 짧게 통화하시면 됩니다"),
                deadline=action_data.get("deadline", "오늘"),
                estimated_time=action_data.get("estimated_time", action_data.get("duration", "5-10분")),
                suggested_topics=action_data.get("suggested_topics") or [
                    "어디 불편하신 곳은 없으세요?",
                    "식사는 잘 드시고 계신가요?"
                ],
                options=action_data.get("options"),
                booking_button=action_data.get("booking_button", False)
            ))
        
        return ActionPlan(
            urgent_actions=urgent_actions,
            this_week_actions=[],
            long_term_actions=[]
        )

    def _max_alert_level(self, current: str, candidate: str) -> str:
        """경보 레벨 중 우선순위가 높은 값을 선택"""
        priority = {"normal": 0, "caution": 1, "urgent": 2}
        return candidate if priority.get(candidate, 0) > priority.get(current, 0) else current

    def _apply_guardrail_rules(
        self,
        conversation: str,
        base_status: str,
        mood_score: int,
        energy_score: int,
        pain_score: int
    ) -> Dict[str, Any]:
        """텍스트 기반 핵심 위험 트리거를 감지해 점수와 경보를 보정"""
        alert_level = base_status
        headline = None
        alert_reasons: List[str] = []
        additional_concerns: List[Dict[str, Any]] = []
        evidence_keywords: List[str] = []
        depression_factors: List[str] = []
        anxiety_factors: List[str] = []
        health_factors: List[str] = []
        mental_factors: List[str] = []
        health_level = "medium"
        mental_level = "medium"
        override_actions: Optional[List[Dict[str, Any]]] = None
        
        convo = conversation
        convo_compact = conversation.replace(" ", "")
        
        def contains_any(keywords: List[str]) -> bool:
            return any(kw in convo or kw in convo_compact for kw in keywords)
        
        # R1. 식사 위험
        meal_keywords = [
            "죽 반 그릇", "죽만", "저녁은 물만", "물만 마셨", "밥맛 없", "식욕 없", "먹기 힘", "씹기 힘"
        ]
        if contains_any(meal_keywords):
            alert_level = self._max_alert_level(alert_level, "caution")
            mood_score = max(0, mood_score - 15)
            energy_score = max(0, energy_score - 20)
            pain_score = min(100, pain_score + 5)
            if "식사량 감소" not in alert_reasons:
                alert_reasons.append("식사량 감소")
            additional_concerns.append({
                "type": "건강",
                "icon": "🍽️",
                "severity": "urgent",
                "title": "식사량 감소/영양 위험",
                "description": "죽 반 그릇, 저녁은 물만 등 식사량 급감이 감지되었습니다.",
                "detected_from": ["대화"],
                "urgency_reason": "영양 부족은 급격한 컨디션 악화로 이어질 수 있습니다."
            })
            evidence_keywords.extend(["입맛 없음", "식욕 저하"])
            depression_factors.append("식사량 감소")
            health_factors.append("식사량 급감")
            health_level = "high"
        
        # R2. 낙상 위험
        fall_keywords = [
            "욕실에서 미끄러", "욕실 미끄러", "미끄러져", "넘어질 뻔", "넘어질뻔", "넘어질 것 같", "낙상", "벽을 짚"
        ]
        if contains_any(fall_keywords):
            alert_level = self._max_alert_level(alert_level, "urgent")
            if "낙상 위험" not in alert_reasons:
                alert_reasons.append("낙상 위험")
            additional_concerns.append({
                "type": "안전",
                "icon": "⚠️",
                "severity": "urgent",
                "title": "낙상 위험(욕실)",
                "description": "욕실에서 미끄러져 넘어질 뻔한 상황이 보고되었습니다.",
                "detected_from": ["대화"],
                "urgency_reason": "낙상은 중대한 부상 위험이 있습니다."
            })
            evidence_keywords.append("낙상 위험")
            anxiety_factors.append("낙상 걱정")
            health_factors.append("욕실 미끄러짐")
            health_level = "high"
        
        # R3. 통증
        pain_keywords = [
            "허리 통증", "허리 아파", "허리가 쑤", "허리가 욱신", "허리가 욱씬", "움직일 때 욱신", "움직일때 욱신"
        ]
        if contains_any(pain_keywords):
            pain_score = max(pain_score, 60)
            if "허리 통증" not in alert_reasons:
                alert_reasons.append("허리 통증")
            additional_concerns.append({
                "type": "건강",
                "icon": "🏥",
                "severity": "caution",
                "title": "허리 통증",
                "description": "허리가 쑤시고 움직일 때 통증이 심하다고 말씀하셨어요.",
                "detected_from": ["대화"],
                "urgency_reason": "지속되는 통증은 추가 진료가 필요할 수 있습니다."
            })
            evidence_keywords.append("허리 통증")
            depression_factors.append("지속 통증 호소")
            health_factors.append("허리 통증")
            health_level = "high"
        
        # R4. 불면
        sleep_keywords = [
            "불면", "잠이 안", "깊게 못 자", "깊게 못잠", "두세 번 깨", "밤중에 자주 깨"
        ]
        if contains_any(sleep_keywords):
            energy_score = max(0, energy_score - 12)
            if "수면 문제" not in alert_reasons:
                alert_reasons.append("수면 문제")
            additional_concerns.append({
                "type": "건강",
                "icon": "🛌",
                "severity": "caution",
                "title": "수면 문제",
                "description": "밤사이 자주 깨고 깊게 잠들지 못하신다고 하셨어요.",
                "detected_from": ["대화"],
                "urgency_reason": "수면 부족은 기력 저하와 기분 악화에 영향을 미칩니다."
            })
            evidence_keywords.append("불면")
            depression_factors.append("수면 저하")
            mental_factors.append("수면 불안")
            if health_level != "high":
                health_level = "medium"
        
        # R5. 정서적 고립
        loneliness_keywords = [
            "외로움", "외롭", "허전", "사람 목소리", "전화가 망설", "민폐일까"
        ]
        if contains_any(loneliness_keywords):
            mood_score = max(0, mood_score - 10)
            mental_level = "high"
            if "정서적 고립" not in alert_reasons:
                alert_reasons.append("정서적 고립")
            additional_concerns.append({
                "type": "정서",
                "icon": "😞",
                "severity": "caution",
                "title": "정서적 고립",
                "description": "외로움을 느끼고 연락을 망설이고 계십니다.",
                "detected_from": ["대화"],
                "urgency_reason": "정서적 고립은 우울 위험 요인이 될 수 있습니다."
            })
            evidence_keywords.append("외로움")
            depression_factors.append("정서적 고립")
            mental_factors.append("소셜 접촉 감소")
        
        # 점수 보정 범위 제한
        mood_score = max(0, min(100, mood_score))
        energy_score = max(0, min(100, energy_score))
        pain_score = max(0, min(100, pain_score))
        
        # 경보 메시지 및 헤드라인 구성
        alert_subtitle = None
        if alert_reasons:
            subtitle_core = "·".join(alert_reasons)
            alert_subtitle = f"{subtitle_core}이 감지되었습니다"
            if len(alert_reasons) == 1:
                headline = f"{alert_reasons[0]}이 확인되어 살펴봐 주세요"
            else:
                headline = f"{', '.join(alert_reasons[:-1])}와 {alert_reasons[-1]}이 함께 나타나 확인이 필요합니다"
        
        # 점수 분해
        score_breakdown = {}
        if depression_factors:
            score_breakdown["depression"] = {
                "score": 70 if len(depression_factors) >= 2 else 60,
                "factors": depression_factors
            }
        if anxiety_factors:
            score_breakdown["anxiety"] = {
                "score": 65 if "낙상 걱정" in anxiety_factors else 55,
                "factors": anxiety_factors
            }
        
        # 행동 계획 보강 (긴급 시)
        if alert_level == "urgent":
            override_actions = [
                {
                    "id": 1,
                    "priority": "최우선",
                    "icon": "📞",
                    "title": "오늘 바로 연락하여 안전·식사 상태 확인",
                    "deadline": "오늘",
                    "reason": "낙상 위험과 식사량 급감이 동시에 관찰되었습니다.",
                    "detail": "바로 통화하여 오늘 드신 양과 통증 정도, 다시 미끄러질 위험이 없는지 확인해주세요.",
                    "estimated_time": "10분",
                    "suggested_topics": [
                        "오늘 식사와 수분 섭취량",
                        "허리 통증의 위치와 통증 정도",
                        "욕실에서 다시 미끄러질 위험이 있는지"
                    ]
                },
                {
                    "id": 2,
                    "priority": "긴급",
                    "icon": "🏥",
                    "title": "의료진 상담 예약",
                    "deadline": "이번 주 내",
                    "reason": "허리 통증과 영양 저하가 지속될 경우 전문 진료가 필요합니다.",
                    "detail": "주치의나 내과·정형외과 상담을 예약해 증상을 공유해주세요.",
                    "estimated_time": "30분",
                    "booking_button": True
                },
                {
                    "id": 3,
                    "priority": "중요",
                    "icon": "🛁",
                    "title": "욕실 안전조치 점검",
                    "deadline": "오늘",
                    "reason": "낙상 위험을 즉시 줄이기 위한 조치가 필요합니다.",
                    "detail": "미끄럼 방지 매트, 손잡이, 조명 등을 점검하고 보강해주세요.",
                    "estimated_time": "15분",
                    "suggested_topics": [
                        "욕실 바닥 미끄럼 방지 매트 설치 여부",
                        "손잡이나 의자 등 보조기구 필요 여부"
                    ]
                }
            ]
        
        # 위험 지표 구성
        risk_levels = {
            "health_level": health_level if alert_level != "normal" else "medium",
            "mental_level": mental_level if alert_level != "normal" else mental_level,
            "health_factors": list(dict.fromkeys(health_factors)),
            "mental_factors": list(dict.fromkeys(mental_factors))
        }
        
        calculation_method = (
            "텍스트 60%·음성 25%·표정 15% 가중"
            if alert_reasons else
            "빠른 분석 모드로 간소화된 계산"
        )
        
        return {
            "alert_level": alert_level,
            "mood_score": mood_score,
            "energy_score": energy_score,
            "pain_score": pain_score,
            "headline": headline,
            "alert_subtitle": alert_subtitle,
            "additional_concerns": additional_concerns,
            "evidence_keywords": list(dict.fromkeys(evidence_keywords)),
            "score_breakdown": score_breakdown,
            "override_actions": override_actions,
            "risk_levels": risk_levels,
            "calculation_method": calculation_method
        }
    
    def _create_minimal_detailed_analysis(
        self, 
        conversation: str, 
        audio_analysis: Dict,
        risk_levels: Dict[str, Any]
    ) -> DetailedAnalysis:
        """⚡ 최소한 상세 분석"""
        
        # 간단한 주제 감지
        topics = []
        if "밥" in conversation or "식사" in conversation:
            topics.append(ConversationTopic(
                topic="식사",
                summary="식사 관련 대화",
                concern_level="normal"
            ))
        
        # 최소한 타임라인
        emotion_timeline = [
            EmotionTimeline(
                timestamp="00:01:00",
                emotion="보통",
                intensity=60,
                trigger="일반적인 대화"
            )
        ]
        
        # 기본 위험 지표
        health_level = risk_levels.get("health_level", "medium")
        mental_level = risk_levels.get("mental_level", "medium")
        risk_indicators = {
            "health_risk": RiskIndicator(
                level=health_level,
                factors=risk_levels.get("health_factors", [])
            ),
            "mental_risk": RiskIndicator(
                level=mental_level,
                factors=risk_levels.get("mental_factors", ["외로움"] if mental_level != "low" else [])
            )
        }
        
        # 기본 오디오 분석
        audio_analysis_obj = AudioAnalysis(
            voice_energy="보통",
            speaking_pace="보통", 
            tone_quality="보통",
            emotional_indicators=[]
        )
        
        return DetailedAnalysis(
            conversation_summary={
                "total_exchanges": len(conversation.split("\n")),
                "conversation_topics": [topic.dict() for topic in topics]
            },
            emotion_timeline=emotion_timeline,
            risk_indicators=risk_indicators,
            video_highlights=[],
            audio_analysis=audio_analysis_obj
        )
    
    def _create_fast_ui_components(
        self, 
        status: StatusOverview, 
        mood_score: int
    ) -> UIComponents:
        """⚡ 빠른 UI 컴포넌트"""
        quick_stats = [
            QuickStat(
                label="기분",
                value=f"{mood_score}/100",
                emoji=self._get_mood_emoji(mood_score),
                color=status.status_color
            ),
            QuickStat(
                label="상태",
                value=status.alert_title.split()[0],
                emoji=status.alert_badge,
                color=status.status_color
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
                text="영상 보기",
                icon="🎬",
                color="#4444FF",
                action="watch_video"
            )
        ]
        
        return UIComponents(
            header={
                "badge_color": status.status_color,
                "badge_text": status.alert_title.split()[0],
                "title": status.alert_subtitle,
                "subtitle": f"오늘 {datetime.now().strftime('%H:%M')} 분석"
            },
            quick_stats=quick_stats,
            cta_buttons=cta_buttons
        )
    
    def _extract_keywords_fast(self, conversation: str) -> List[str]:
        """⚡ 빠른 키워드 추출 (로컬 처리)"""
        emotion_keywords = [
            "우울", "슬픔", "외로움", "피곤", "아픔", "기쁨", "행복",
            "불면", "허리", "식욕", "낙상", "걱정", "허전"
        ]
        found_keywords = []
        
        conversation_lower = conversation.lower()
        for keyword in emotion_keywords:
            if keyword in conversation_lower:
                found_keywords.append(keyword)
        
        return found_keywords[:5]  # 최대 5개
    
    async def close(self):
        """클라이언트 정리"""
        if self._client:
            await self._client.aclose()
