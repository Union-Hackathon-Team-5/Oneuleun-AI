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

## 환경 변수

- `OPENAI_API_KEY`: OpenAI Responses API 키 (필수)
- `OPENAI_VISION_MODEL`: 사용할 비전 모델 명칭 (선택, 기본값 `gpt-4o-mini`)

## 엔드포인트

- `POST /context/`  
  요청 바디: `session_id`, `user_id`, `photo_url`  
  응답: 기본 감정 7종, 확장 정서 8종, 위험 징후 6종 중 해당 항목과 신뢰도/요약을 포함한 JSON

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
