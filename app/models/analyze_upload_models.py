from typing import List

from pydantic import BaseModel, Field


class AnalyzeUploadRequest(BaseModel):
    session_id: str = Field(..., description="상담 세션 ID")
    user_id: str = Field(..., description="노인 사용자 ID")
    senior_name: str = Field(..., description="노인 이름")
    conversation: str = Field(..., description="AI 질문과 노인 응답이 포함된 대화 내용")
    audio_file_base64: Optional[str] = Field(
        default=None,
        description="Base64 인코딩된 음성 파일 (선택 사항)",
    )


class StatusSignal(BaseModel):
    health: str = Field(..., description="건강 상태 신호")
    emotion: str = Field(..., description="감정 상태 신호")
    daily_function: str = Field(..., description="일상 기능 상태 신호")
    summary: str = Field(..., description="주요 상태 요약")


class GraphDatum(BaseModel):
    day: str = Field(..., description="요일")
    mood: int = Field(..., description="해당 요일 기분 점수")


class WeeklyChange(BaseModel):
    mood: int = Field(..., description="주간 기분 변화율(%)")
    meal: int = Field(..., description="주간 식사량 변화율(%)")
    activity: int = Field(..., description="주간 활동량 변화율(%)")
    graph_dummy_data: List[GraphDatum] = Field(..., description="주간 기분 그래프용 더미 데이터")


class AICarePlan(BaseModel):
    today: str = Field(..., description="오늘 케어 플랜")
    this_week: str = Field(..., description="이번 주 케어 플랜")
    this_month: str = Field(..., description="이번 달 케어 플랜")
    this_year: str = Field(..., description="올해 케어 플랜")


class AnalyzeUploadResponse(BaseModel):
    status_signal: StatusSignal = Field(..., description="주요 상태 신호")
    key_phrases: List[str] = Field(..., description="핵심 문장 리스트")
    care_todo: List[str] = Field(..., description="케어 TODO 리스트")
    weekly_change: WeeklyChange = Field(..., description="주간 변화 요약")
    ai_care_plan: AICarePlan = Field(..., description="AI 케어 플랜 제안")
