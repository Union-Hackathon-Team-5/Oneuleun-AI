import asyncio
import contextlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from pydantic import ValidationError

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
        self._concern_type_aliases = {
            "신체": "건강",
            "신체건강": "건강",
            "의료": "건강",
            "통증": "건강",
            "health": "건강",
            "safety": "안전",
            "안전위험": "안전",
            "정신": "정서",
            "정신건강": "정서",
            "외로움": "정서",
            "고립": "정서",
            "생활환경": "생활",
            "일상": "생활",
            "환경": "생활",
        }
    
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
        comprehensive_analysis, fact_snapshot = await self.analysis_service.analyze_video_letter_comprehensive(
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
            fact_snapshot=fact_snapshot,
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
        fact_snapshot: Dict[str, Any],
        session_id: str,
        user_id: str
    ) -> CaregiverFriendlyResponse:
        """기술적 분석을 보호자 친화적 형태로 변환"""
        
        print(f"[PERF] Starting caregiver task race (bundle vs fallback)", flush=True)
        logger.info("[PERF] Starting caregiver task race (bundle vs fallback)")
        race_start = time.time()

        bundle_task = asyncio.create_task(
            self._generate_caregiver_bundle(
                conversation=conversation,
                comprehensive_analysis=comprehensive_analysis,
                image_analysis=image_analysis,
                fact_snapshot=fact_snapshot
            ),
            name="caregiver_bundle"
        )
        fallback_task = asyncio.create_task(
            self._run_legacy_caregiver_tasks(
                comprehensive_analysis=comprehensive_analysis,
                conversation=conversation,
                image_analysis=image_analysis
            ),
            name="caregiver_fallback"
        )

        emotional_insights: Dict[str, Any]
        action_plan: ActionPlan
        mother_voice: List[str]
        key_concerns: List[KeyConcern]

        done, pending = await asyncio.wait(
            {bundle_task, fallback_task},
            return_when=asyncio.FIRST_COMPLETED
        )

        bundle_result: Optional[Dict[str, Any]] = None
        fallback_result: Optional[Tuple[Dict[str, Any], ActionPlan, List[str], List[KeyConcern]]] = None

        if bundle_task in done:
            try:
                bundle_result = bundle_task.result()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Caregiver bundle execution error: %s", exc)
                bundle_result = None

        if fallback_task in done:
            try:
                fallback_result = fallback_task.result()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Fallback caregiver tasks error: %s", exc)
                fallback_result = None

        # Decide which result to use
        if bundle_result:
            # cancel fallback if still running
            if fallback_task not in done:
                fallback_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await fallback_task
            emotional_insights, action_plan, mother_voice, key_concerns = self._parse_bundle_result(
                bundle_result,
                comprehensive_analysis
            )
            winner = "bundle"
        else:
            if fallback_result is None:
                # Wait for fallback to finish if bundle failed
                fallback_result = await fallback_task
            else:
                if bundle_task not in done:
                    bundle_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await bundle_task
            emotional_insights, action_plan, mother_voice, key_concerns = fallback_result
            winner = "fallback"

        race_time = time.time() - race_start
        print(f"[PERF] Caregiver task race winner: {winner} in {race_time:.2f}s", flush=True)
        logger.info("[PERF] Caregiver task race winner: %s in %.2fs", winner, race_time)
        
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

    def _build_action_plan_from_dict(self, data: Dict) -> ActionPlan:
        """LLM이 생성한 딕셔너리를 ActionPlan 모델로 변환 (우선순위 정규화, 중복 제거 포함)"""
        valid_priorities = ["최우선", "긴급", "중요"]

        def normalize_action(action: Dict) -> Dict:
            if "priority" in action:
                priority = action["priority"]
                if priority not in valid_priorities:
                    if priority in ["보통", "낮음", "normal", "low"]:
                        action["priority"] = "중요"
                    elif priority in ["높음", "high", "urgent"]:
                        action["priority"] = "긴급"
                    else:
                        action["priority"] = "중요"
            else:
                action["priority"] = "중요"
            return action

        def deduplicate(actions: List[Dict]) -> List[Dict]:
            seen_titles = set()
            unique_actions = []
            for action in actions:
                title = action.get("title", "").strip()
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                unique_actions.append(action)
            return unique_actions

        urgent_actions_raw = deduplicate(data.get("urgent_actions", []))
        this_week_actions_raw = deduplicate(data.get("this_week_actions", []))
        long_term_actions_raw = deduplicate(data.get("long_term_actions", []))

        urgent_actions = [UrgentAction.model_validate(normalize_action(action)) for action in urgent_actions_raw]
        this_week_actions = [UrgentAction.model_validate(normalize_action(action)) for action in this_week_actions_raw]
        long_term_actions = [UrgentAction.model_validate(normalize_action(action)) for action in long_term_actions_raw]

        return ActionPlan(
            urgent_actions=urgent_actions,
            this_week_actions=this_week_actions,
            long_term_actions=long_term_actions
        )

    def _normalize_concern_entry(self, concern: Dict[str, Any], idx: int) -> Dict[str, Any]:
        concern = dict(concern)
        concern_type = str(concern.get("type", "")).strip()
        normalized = self._concern_type_aliases.get(concern_type, concern_type)
        if normalized not in {"건강", "안전", "정서", "생활"}:
            logger.warning("Unknown concern type '%s' at index %s, defaulting to '정서'", concern_type, idx)
            normalized = "정서"
        concern["type"] = normalized
        return concern

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
        
        # 프롬프트 최적화: 더 간결하게
        key_concerns_str = ", ".join(analysis.comprehensive_summary.key_concerns[:2]) if analysis.comprehensive_summary.key_concerns else "없음"
        recommended_str = ", ".join(analysis.comprehensive_summary.recommended_actions[:2]) if analysis.comprehensive_summary.recommended_actions else "없음"
        
        prompt = f"""행동 계획 생성 (간결, 필수만):

위험도: {analysis.comprehensive_summary.priority_level}
우려: {key_concerns_str}
조치: {recommended_str}

JSON 응답 (최소화):
{{
    "urgent_actions": [{{"action_id": 1, "priority": "최우선", "icon": "📞", "title": "제목", "reason": "이유", "detail": "말씀", "deadline": "오늘", "estimated_time": "10분", "suggested_topics": ["예시1", "예시2"]}}],
    "this_week_actions": [{{"action_id": 2, "priority": "중요", "icon": "📅", "title": "제목", "reason": "이유", "detail": "말씀", "deadline": "이번주", "estimated_time": "30분", "suggested_topics": ["예시"]}}],
    "long_term_actions": []
}}

규칙:
- urgent 최대 2개, this_week 최대 3개, long_term 최대 2개.
- 연락 방법은 반복하지 말고 다양하게 제시 (예: 전화, 음성 메시지, 영상 통화 예약, 방문 일정, 복지센터 프로그램 등).
- 각 액션은 구체적 이유와 실행 방법을 제공하고, 1-2문장으로 간결하게 작성.
- priority는 "최우선", "긴급", "중요"만.
"""
        
        try:
            task_start = time.time()
            # max_tokens를 500으로 더 줄임 (각 액션 필드를 더 간결하게 만들었으므로)
            response = await self.analysis_service._call_openai(prompt, max_tokens=500, task_name="_generate_actionable_plan")
            task_time = time.time() - task_start
            print(f"[PERF] _generate_actionable_plan API call: {task_time:.2f}s", flush=True)
            logger.debug(f"[PERF] _generate_actionable_plan API call: {task_time:.2f}s")
            data = json.loads(response)
            return self._build_action_plan_from_dict(data)
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

    async def _generate_caregiver_bundle(
        self,
        conversation: str,
        comprehensive_analysis: ComprehensiveAnalysisResult,
        image_analysis: Dict,
        fact_snapshot: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """감성 인사이트, 행동 계획, 주요 걱정거리, 어머니 목소리를 한 번의 호출로 생성"""
        summary = comprehensive_analysis.comprehensive_summary
        emotion = comprehensive_analysis.emotion_analysis
        risk = comprehensive_analysis.risk_analysis
        anomaly = comprehensive_analysis.anomaly_analysis

        image_concerns = []
        if image_analysis.get("analysis"):
            image_concerns = image_analysis["analysis"].get("concerns", [])
        image_concern_text = ", ".join(image_concerns) if image_concerns else "없음"

        trend_label = anomaly.pattern_type if anomaly.pattern_type != "없음" else anomaly.trend_analysis

        fact_json = json.dumps(fact_snapshot, ensure_ascii=False)

        convo_lines = [line.strip() for line in conversation.splitlines() if line.strip()]
        trimmed_conversation = "\n".join(convo_lines[-20:])  # 최신 발언 위주 20줄

        prompt = f"""
당신은 독거노인 케어 전문가입니다. 다음 정보를 바탕으로 보호자용 보고서 요소를 한 번에 생성하세요.

Facts 스냅샷:
{fact_json}

감정 점수: 긍정 {emotion.positive}/100, 불안 {emotion.anxiety}/100, 우울 {emotion.depression}/100, 외로움 {emotion.loneliness}/100
전반 기분: {emotion.overall_mood}
우선순위: {summary.priority_level}, 상태 요약: {summary.main_summary}
위험도: {risk.risk_level}, 즉시 우려: {', '.join(risk.immediate_concerns) or '없음'}
이미지 우려: {image_concern_text}
추세/패턴: {trend_label}

최근 대화 발췌:
{trimmed_conversation}

JSON 형식으로만 응답하세요:
{{
  "emotional_insights": {{
    "headline": "<감성 요약 제목>",
    "mood_description": "<기분 설명>",
    "energy_level": "<활력 설명>",
    "pain_level": "<신체 불편 설명>",
    "emotional_state": "<전반 감정 상태>"
  }},
  "action_plan": {{
    "urgent_actions": [
      {{
        "action_id": 1,
        "priority": "최우선",
        "icon": "📞",
        "title": "제목",
        "reason": "이유",
        "detail": "실행 방법",
        "deadline": "오늘",
        "estimated_time": "10분",
        "suggested_topics": ["예시"]
      }}
    ],
    "this_week_actions": [],
    "long_term_actions": []
  }},
  "mother_voice": ["💬 \"실제 인용\""],
  "key_concerns": [
    {{
      "concern_id": 1,
      "type": "건강|안전|정서|생활",
      "icon": "🏥",
      "severity": "urgent|caution|normal",
      "title": "걱정 요약",
      "description": "간결한 설명",
      "detected_from": ["대화", "표정"],
      "urgency_reason": "이 항목이 중요한 이유"
    }}
  ]
}}

규칙:
- action_plan 항목은 최대 6개(긴급 2, 이번 주 3, 장기 1) 이내이며, 연락 방법을 중복하지 말고 다양하게 제안하세요 (전화, 음성 메시지, 영상 통화, 방문 예약, 복지센터 프로그램 등).
- 우울/절망 징후가 강하면 concerns에 '우울증 우려', '자살 위험 의심' 등을 명시하고, action_plan에도 이에 대한 구체 조치를 포함하세요.
- mother_voice에는 실제 대화 인용을 그대로 사용하고, 없으면 facts.notable_quotes에서 골라주세요.
- key_concerns는 가장 중요한 3개까지, severity와 urgency_reason을 구체적으로 작성하세요.
"""

        bundle_schema = {
            "name": "caregiver_bundle",
            "schema": {
                "type": "object",
                "properties": {
                    "emotional_insights": {
                        "type": "object",
                        "properties": {
                            "headline": {"type": "string"},
                            "mood_description": {"type": "string"},
                            "energy_level": {"type": "string"},
                            "pain_level": {"type": "string"},
                            "emotional_state": {"type": "string"},
                        },
                        "required": ["headline", "mood_description", "energy_level", "pain_level", "emotional_state"],
                        "additionalProperties": False,
                    },
                    "action_plan": {
                        "type": "object",
                        "properties": {
                            "urgent_actions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "action_id": {"type": "integer"},
                                        "priority": {"type": "string", "enum": ["최우선", "긴급", "중요"]},
                                        "icon": {"type": "string"},
                                        "title": {"type": "string"},
                                        "reason": {"type": "string"},
                                        "detail": {"type": "string"},
                                        "deadline": {"type": "string"},
                                        "estimated_time": {"type": "string"},
                                        "suggested_topics": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "maxItems": 4
                                        }
                                    },
                                    "required": ["action_id", "priority", "icon", "title", "reason", "detail", "deadline", "estimated_time"],
                                    "additionalProperties": False
                                },
                                "maxItems": 2
                            },
                            "this_week_actions": {
                                "type": "array",
                                "items": {"$ref": "#/$defs/action_item"},
                                "maxItems": 3
                            },
                            "long_term_actions": {
                                "type": "array",
                                "items": {"$ref": "#/$defs/action_item"},
                                "maxItems": 2
                            }
                        },
                        "required": ["urgent_actions", "this_week_actions", "long_term_actions"],
                        "additionalProperties": False
                    },
                    "mother_voice": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 4
                    },
                    "key_concerns": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "concern_id": {"type": "integer"},
                                "type": {"type": "string", "enum": ["건강", "안전", "정서", "생활"]},
                                "icon": {"type": "string"},
                                "severity": {"type": "string", "enum": ["urgent", "caution", "normal"]},
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "detected_from": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "maxItems": 3
                                },
                                "urgency_reason": {"type": "string"}
                            },
                            "required": ["concern_id", "type", "icon", "severity", "title", "description", "detected_from", "urgency_reason"],
                            "additionalProperties": False
                        },
                        "maxItems": 3
                    }
                },
                "required": ["emotional_insights", "action_plan", "mother_voice", "key_concerns"],
                "additionalProperties": False,
                "$defs": {
                    "action_item": {
                        "type": "object",
                        "properties": {
                            "action_id": {"type": "integer"},
                            "priority": {"type": "string", "enum": ["최우선", "긴급", "중요"]},
                            "icon": {"type": "string"},
                            "title": {"type": "string"},
                            "reason": {"type": "string"},
                            "detail": {"type": "string"},
                            "deadline": {"type": "string"},
                            "estimated_time": {"type": "string"},
                            "suggested_topics": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 4
                            }
                        },
                        "required": ["action_id", "priority", "icon", "title", "reason", "detail", "deadline", "estimated_time"],
                        "additionalProperties": False
                    }
                }
            }
        }

        try:
            response = await self.analysis_service._call_openai(
                prompt,
                max_tokens=700,
                task_name="_generate_caregiver_bundle",
                timeout_seconds=8.0,
                temperature=0.25,
                response_format={"type": "json_schema", "json_schema": bundle_schema}
            )
            return json.loads(response)
        except Exception as exc:
            logger.error("Failed to generate caregiver bundle: %s", exc)
            return None
    
    async def _run_legacy_caregiver_tasks(
        self,
        comprehensive_analysis: ComprehensiveAnalysisResult,
        conversation: str,
        image_analysis: Dict
    ) -> Tuple[Dict[str, Any], ActionPlan, List[str], List[KeyConcern]]:
        """기존 4분할 LLM 호출을 병렬로 실행"""
        default_emotional = {
            "headline": "어머니 상태를 확인이 필요합니다",
            "mood_description": "평소보다 기분이 좋지 않으신 것 같아요",
            "energy_level": "활력이 부족해 보입니다",
            "pain_level": "몸이 불편하신 것 같아요",
            "emotional_state": "관심과 돌봄이 필요한 상태입니다"
        }
        default_action_plan = self._create_default_action_plan(comprehensive_analysis)
        default_mother_voice = [
            "💬 \"요즘 컨디션이 별로 좋지 않아요\"",
            "💬 \"혼자 있는 시간이 많아서 외로워요\"",
            "💬 \"몸이 예전 같지 않아서 걱정이에요\""
        ]
        default_concerns = self._create_default_concerns(comprehensive_analysis)

        async def safe_call(coro, timeout: float, label: str, default_value):
            try:
                return await asyncio.wait_for(coro, timeout)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("%s failed/timeout: %s", label, exc)
                return default_value

        tasks = [
            asyncio.create_task(
                safe_call(
                    self._generate_emotional_insights(conversation, comprehensive_analysis),
                    10.0,
                    "Emotional insights",
                    default_emotional
                ),
                name="legacy_emotional_insights"
            ),
            asyncio.create_task(
                safe_call(
                    self._generate_actionable_plan(comprehensive_analysis, conversation),
                    10.0,
                    "Action plan",
                    default_action_plan
                ),
                name="legacy_action_plan"
            ),
            asyncio.create_task(
                safe_call(
                    self._extract_mother_voice(conversation),
                    10.0,
                    "Mother voice",
                    default_mother_voice
                ),
                name="legacy_mother_voice"
            ),
            asyncio.create_task(
                safe_call(
                    self._identify_key_concerns(comprehensive_analysis, conversation, image_analysis),
                    10.0,
                    "Key concerns",
                    default_concerns
                ),
                name="legacy_key_concerns"
            ),
        ]

        results = await asyncio.gather(*tasks)
        emotional_insights = results[0]
        action_plan = results[1]
        mother_voice = results[2]
        key_concerns = results[3]
        return emotional_insights, action_plan, mother_voice, key_concerns

    def _parse_bundle_result(
        self,
        bundle: Dict[str, Any],
        comprehensive_analysis: ComprehensiveAnalysisResult
    ) -> Tuple[Dict[str, Any], ActionPlan, List[str], List[KeyConcern]]:
        emotional_insights = bundle.get("emotional_insights") or {}
        if not emotional_insights:
            emotional_insights = {
                "headline": "어머니 상태를 확인이 필요합니다",
                "mood_description": "평소보다 기분이 좋지 않으신 것 같아요",
                "energy_level": "활력이 부족해 보입니다",
                "pain_level": "몸이 불편하신 것 같아요",
                "emotional_state": "관심과 돌봄이 필요한 상태입니다"
            }

        action_plan_data = bundle.get("action_plan") or {}
        try:
            action_plan = self._build_action_plan_from_dict(action_plan_data)
        except Exception as exc:
            logger.error("Failed to parse bundled action plan: %s", exc)
            action_plan = self._create_default_action_plan(comprehensive_analysis)

        mother_voice = bundle.get("mother_voice") or []
        if not isinstance(mother_voice, list):
            mother_voice = []
        mother_voice = [str(item).strip() for item in mother_voice if str(item).strip()]
        if not mother_voice:
            mother_voice = [
                "💬 \"요즘 컨디션이 별로 좋지 않아요\"",
                "💬 \"혼자 있는 시간이 많아서 외로워요\"",
                "💬 \"몸이 예전 같지 않아서 걱정이에요\""
            ]

        key_concerns_data = bundle.get("key_concerns") or []
        parsed_concerns: List[KeyConcern] = []
        for idx, concern in enumerate(key_concerns_data, start=1):
            try:
                normalized = self._normalize_concern_entry(concern, idx)
                parsed_concerns.append(KeyConcern.model_validate(normalized))
            except ValidationError as exc:
                logger.error("Invalid concern item at %s: %s", idx, exc)
        key_concerns = parsed_concerns or self._create_default_concerns(comprehensive_analysis)

        return emotional_insights, action_plan, mother_voice, key_concerns
    
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
            # max_tokens를 600으로 증가 (JSON 파싱 에러 방지, concerns는 보통 3-5개)
            response = await self.analysis_service._call_openai(prompt, max_tokens=600, task_name="_identify_key_concerns")
            task_time = time.time() - task_start
            print(f"[PERF] _identify_key_concerns API call: {task_time:.2f}s", flush=True)
            logger.debug(f"[PERF] _identify_key_concerns API call: {task_time:.2f}s")
            
            # JSON 파싱 전에 응답 확인 및 정리
            response = response.strip()
            # JSON 파싱 에러 방지를 위한 처리
            if not response.startswith('{'):
                # JSON 시작 부분 찾기
                start_idx = response.find('{')
                if start_idx > 0:
                    response = response[start_idx:]
            # JSON 끝 부분 정리
            if not response.endswith('}'):
                # 마지막 닫는 중괄호 찾기
                last_idx = response.rfind('}')
                if last_idx > 0:
                    response = response[:last_idx+1]
            
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
            mood_score = analysis.emotion_analysis.positive
            alert_message = "지난 7일 대비 큰 변화 없음"
            if mood_score <= 40:
                alert_message = (
                    f"{alert_message} — 현재 기분 점수는 {mood_score}/100으로 낮지만 "
                    "지난 7일 동안 비슷한 수준이 유지되었습니다. 급격한 악화는 감지되지 않았습니다."
                )
            return TrendAnalysis(
                compared_to="지난 7일",
                changes=[],
                alert_message=alert_message,
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
        
        cta_buttons = []
        if analysis.comprehensive_summary.priority_level in ["긴급", "높음"]:
            cta_buttons.append(
                CTAButton(
                    text="지금 전화하기",
                    icon="📞",
                    color="#FF4444",
                    action="call"
                )
            )
        else:
            cta_buttons.append(
                CTAButton(
                    text="짧은 안부 전화하기",
                    icon="📞",
                    color="#FF6666",
                    action="call"
                )
            )

        cta_buttons.append(
            CTAButton(
                text="음성 메시지 보내기",
                icon="🎙️",
                color="#FF8800",
                action="send_voice_note"
            )
        )
        cta_buttons.append(
            CTAButton(
                text="영상 전체보기",
                icon="🎬",
                color="#4444FF",
                action="watch_video"
            )
        )
        
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
                title="어머니께 짧게 안부 전화 드리기",
                reason="오늘 기분과 몸 상태를 직접 확인하면 안심하실 수 있습니다",
                detail="5분 정도 통화하면서 현재 컨디션과 필요한 도움이 있는지 여쭤보세요",
                deadline="오늘 중",
                estimated_time="10분 이내",
                suggested_topics=[
                    "지금 어디 불편하신 곳은 없는지 편하게 말씀해주세요",
                    "식사나 약 챙기시는 데 도와드릴 일 있을까요?",
                    "이번 주 후반에 제가 들를 수 있는데 괜찮으세요?"
                ]
            ),
            UrgentAction(
                action_id=2,
                priority="긴급",
                icon="🎥",
                title="짧은 영상 통화 시간 잡기",
                reason="얼굴을 보고 이야기하면 정서적 안정에 도움이 됩니다",
                detail="늦지 않은 시간에 10분 정도 영상 통화를 제안해보세요",
                deadline="오늘 중",
                estimated_time="5-10분",
                suggested_topics=[
                    "얼굴 뵙고 바로 안부 여쭤보고 싶어요, 괜찮으세요?",
                    "요즘 집안에 바뀐 점이나 도움이 필요하신 게 있는지 봐드릴게요",
                    "다음에 함께 하고 싶은 활동 있으면 말씀해주세요"
                ]
            ),
        ]
        
        this_week_actions = [
            UrgentAction(
                action_id=3,
                priority="중요",
                icon="🎙️",
                title="음성 메시지로 응원 남기기",
                reason="통화가 어려운 날에도 꾸준한 정서적 연결을 유지할 수 있습니다",
                detail="오늘 들은 이야기나 응원의 말을 짧게 녹음해 보내주세요",
                deadline="이번 주",
                estimated_time="5분",
                suggested_topics=[
                    "오늘 있었던 기분 좋은 일이나 감사 인사를 전해주세요",
                    "어머니께서 좋아하시는 노래 한 소절을 불러드려도 좋아요"
                ]
            ),
            UrgentAction(
                action_id=4,
                priority="중요",
                icon="🚶",
                title="근처 복지센터 산책 프로그램 문의하기",
                reason="규칙적인 외출과 사회적 교류는 기분 회복에 큰 도움이 됩니다",
                detail="어머니와 함께 참여할 수 있는 가벼운 산책 또는 운동 프로그램을 알아보세요",
                deadline="이번 주",
                estimated_time="20분",
                suggested_topics=[
                    "날씨 좋은 날 같이 걸을 수 있는 프로그램이 있는지 확인해보세요",
                    "참여 시 필요한 준비물이나 일정도 함께 정리해주세요"
                ]
            )
        ]
        
        return ActionPlan(
            urgent_actions=urgent_actions,
            this_week_actions=this_week_actions,
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
