# Oneuleun-AI

간단한 해커톤 버전의 영상 상담 분석 API 백엔드입니다.  
S3 이미지 URL을 전달하면 OpenAI Vision 모델을 통해 감정을 분류합니다.

## 빠른 시작

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Docker 실행

```bash
# .env 파일에 OPENAI_API_KEY 값을 설정한 뒤
docker compose up --build
```

기본적으로 8000 번 포트가 열리며 `http://localhost:8000/context/` 로 접근할 수 있습니다.

## 환경 변수

- `OPENAI_API_KEY`: OpenAI Responses API 키 (필수)
- `OPENAI_VISION_MODEL`: 사용할 비전 모델 명칭 (선택, 기본값 `gpt-4o-mini`)

## 엔드포인트

- `POST /context/`  
  요청 바디: `session_id`, `user_id`, `photo_url`  
  응답: 기본 감정 7종, 확장 정서 8종, 위험 징후 6종 중 해당 항목과 신뢰도/요약을 포함한 JSON
- `POST /analyze/`  
  요청 바디: `session_id`, `user_id`, `conversation`(질문:응답 딕셔너리 JSON 문자열), `audio_url`  
  응답: 입력 정보와 함께 S3 음성 데이터를 다운로드하여 고함 여부(`shout_detection`)를 반환합니다.

### 예시 요청

```bash
curl -X POST "http://localhost:8000/context/" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session_123",
    "user_id": "elderly_001",
    "photo_url": "https://example-bucket.s3.ap-northeast-2.amazonaws.com/sessions/123/snapshot_001.jpg"
  }'
```

### 예시 응답

```json
{
  "success": true,
  "session_id": "session_123",
  "user_id": "elderly_001",
  "photo_url": "https://example-bucket.s3.ap-northeast-2.amazonaws.com/sessions/123/snapshot_001.jpg",
  "analysis": {
    "base_emotion": "기쁨",
    "extended_emotion": "만족",
    "warning_signs": [],
    "confidence": 82,
    "summary": "잔잔한 미소와 안정된 표정이 관찰됩니다."
  },
  "categories": {
    "base_emotions": ["기쁨", "슬픔", "분노", "놀람", "공포", "혐오", "중립"],
    "extended_emotions": ["평온", "우울", "피로", "외로움", "불안", "긴장", "만족", "무기력"],
    "warning_signs": ["통증", "혼미", "호흡곤란", "장기침묵", "자책발언", "사회고립"]
  }
}
```

### 고함 감지 요청 예시

```bash
curl -X POST "http://localhost:8000/analyze/" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session_123",
    "user_id": "elderly_001",
    "conversation": "{\"오늘 기분이 어떠세요?\": \"오늘은 조금 피곤해요.\"}",
    "audio_url": "https://example-bucket.s3.ap-northeast-2.amazonaws.com/audio/session_123.wav"
  }'
```

응답 예시:

```json
{
  "success": true,
  "session_id": "session_123",
  "user_id": "elderly_001",
  "conversation": "{\"오늘 기분이 어떠세요?\": \"오늘은 조금 피곤해요.\"}",
  "audio_url": "https://example-bucket.s3.ap-northeast-2.amazonaws.com/audio/session_123.wav",
  "shout_detection": {
    "present": false,
    "start_ms": null,
    "end_ms": null,
    "peak_dbfs": null,
    "confidence": null
  }
}
```
