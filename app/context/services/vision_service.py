import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

EMOTION_LABELS: List[str] = ["기쁨", "분노", "슬픔", "외로움", "무기력함", "행복"]

logger.info("Loaded emotion labels: %s", EMOTION_LABELS)
print(f"[ContextEmotionLabels] {EMOTION_LABELS}", flush=True)


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
        label_text = ", ".join(EMOTION_LABELS)
        return (
            "다음 노인 분의 표정과 전반 상황을 면밀히 살펴본 뒤 감정을 분석해주세요.\n"
            f"가능한 감정 라벨은 다음 목록에서만 선택하세요: {label_text}\n\n"
            "다음 JSON 형식으로만 답변하세요:\n"
            "{\n"
            '  "emotion": ["<감정 키워드 1개 또는 2개>"],\n'
            '  "summary": "<짧은 한 줄 설명>",\n'
            '  "concerns": ["<이미지 속에서 관찰되는 우려 사항 목록 (없으면 빈 배열)>"]\n'
            "}\n"
            "concerns 항목에는 노인의 상태나 환경에서 주의가 필요한 점이 있다면 최대한 구체적으로 작성하세요.\n"
            "특히 극심한 좌절·절망, 무기력함, 생명에 대한 위험 신호가 포착되면 각각 '우울증 우려', '자살 위험 의심'과 같은 표현을 포함하세요.\n"
            "추가로 식사 거부 징후, 낙상 위험, 안전하지 않은 생활 환경 등이 보이면 구체적으로 서술하세요.\n"
            "특별한 우려가 없으면 concerns 는 빈 배열 []을 넣으세요.\n"
            "emotion 배열에는 감정 라벨을 최대 2개까지 넣되, 첫 번째 요소가 가장 가능성이 높은 감정이 되도록 하세요."
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
        emotion_value = data.get("emotion")
        if isinstance(emotion_value, str):
            emotion_list = [emotion_value]
        elif isinstance(emotion_value, list):
            emotion_list = emotion_value
        else:
            raise ValueError("감정 레이블은 리스트 형식이어야 합니다.")

        if not emotion_list:
            raise ValueError("최소 하나의 감정 레이블이 필요합니다.")

        invalid_labels = [label for label in emotion_list if label not in EMOTION_LABELS]
        if invalid_labels:
            raise ValueError("감정 레이블이 허용된 목록에 없습니다.")

        data["emotion"] = emotion_list

        summary = data.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("요약 문장은 비어 있을 수 없습니다.")

        concerns = data.get("concerns")
        if concerns is None:
            data["concerns"] = []
        elif isinstance(concerns, str):
            data["concerns"] = [concerns] if concerns.strip() else []
        elif isinstance(concerns, list):
            normalized_concerns = []
            for item in concerns:
                if not isinstance(item, str):
                    raise ValueError("concerns 항목은 문자열이어야 합니다.")
                stripped = item.strip()
                if stripped:
                    normalized_concerns.append(stripped)
            data["concerns"] = normalized_concerns
        else:
            raise ValueError("concerns 항목은 리스트 또는 문자열이어야 합니다.")
