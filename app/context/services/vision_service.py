import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

BASE_EMOTIONS: List[str] = ["기쁨", "슬픔", "분노", "놀람", "공포", "혐오", "중립"]
EXTENDED_EMOTIONS: List[str] = ["평온", "우울", "피로", "외로움", "불안", "긴장", "만족", "무기력"]
WARNING_SIGNS: List[str] = ["통증", "혼미", "호흡곤란", "장기침묵", "자책발언", "사회고립"]


class VisionService:
    """HTTP-based wrapper around the OpenAI Responses API for emotion classification."""

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.api_key = api_key
        self.model = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
        self.base_url = "https://api.openai.com/v1/responses"
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def analyze_emotion(self, image_url: str) -> Dict[str, Any]:
        if not image_url:
            raise ValueError("image_url is required for analysis")

        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "당신은 노인 복지 센터에서 사용하는 감정 분석 도우미입니다.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": self._build_prompt()},
                        {"type": "input_image", "image_url": image_url},
                    ],
                },
            ],
            "temperature": 0.1,
            "top_p": 0.8,
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

            raw_text = self._extract_text(data)
            logger.debug("OpenAI raw response: %s", raw_text)

            parsed = json.loads(raw_text)
            self._validate_response(parsed)
            return parsed
        except httpx.HTTPStatusError as exc:
            logger.error("OpenAI API returned %s: %s", exc.response.status_code, exc.response.text)
            raise RuntimeError("OpenAI 분석 요청이 거부되었습니다.") from exc
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode OpenAI response: %s", exc)
            raise ValueError("모델 응답을 해석하지 못했습니다. 다시 시도해주세요.") from exc
        except ValueError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Unexpected error while calling OpenAI: %s", exc)
            raise RuntimeError("OpenAI 분석 요청 중 오류가 발생했습니다.") from exc

    def _build_prompt(self) -> str:
        return (
            "다음 노인 분의 표정과 전반 상황을 살펴보고 감정을 분류해주세요.\n"
            "1) 기본 감정 7가지 중 하나를 선택하세요: "
            f"{', '.join(BASE_EMOTIONS)}\n"
            "2) 확장 정서 8가지 중 하나를 선택하세요: "
            f"{', '.join(EXTENDED_EMOTIONS)}\n"
            "3) 위험 징후 6가지에서 해당되는 항목이 있으면 모두 나열하세요: "
            f"{', '.join(WARNING_SIGNS)} (없다면 빈 배열)\n\n"
            "다음 JSON 형식으로만 답변하세요:\n"
            "{\n"
            '  "base_emotion": "<기본 감정 1개>",\n'
            '  "extended_emotion": "<확장 정서 1개>",\n'
            '  "warning_signs": ["<위험 징후 또는 빈 배열>"],\n'
            '  "confidence": <0~100 숫자>,\n'
            '  "summary": "<짧은 한 줄 설명>"\n'
            "}\n"
            "선택지는 반드시 위에서 제시한 단어만 사용하세요."
        )

    def _extract_text(self, response: Dict[str, Any]) -> str:
        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        chunks: List[str] = []
        for item in response.get("output", []):
            for block in item.get("content", []):
                if block.get("type") == "output_text" and block.get("text"):
                    chunks.append(block["text"])

        if not chunks:
            raise ValueError("모델에서 텍스트 응답을 받지 못했습니다.")

        return "".join(chunks).strip()

    def _validate_response(self, data: Dict[str, Any]) -> None:
        base = data.get("base_emotion")
        extended = data.get("extended_emotion")
        warnings = data.get("warning_signs", [])

        if base not in BASE_EMOTIONS:
            raise ValueError("기본 감정이 올바르지 않습니다.")
        if extended not in EXTENDED_EMOTIONS:
            raise ValueError("확장 정서가 올바르지 않습니다.")
        if not isinstance(warnings, list):
            raise ValueError("warning_signs 항목은 배열이어야 합니다.")

        for sign in warnings:
            if sign not in WARNING_SIGNS:
                raise ValueError(f"지원하지 않는 위험 징후: {sign}")
