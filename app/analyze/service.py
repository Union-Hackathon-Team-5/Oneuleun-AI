import logging
from dataclasses import asdict
from typing import Any, Dict, Optional

from app.analyze.audio_service import AudioFetcher
from app.analyze.shout_service import (
    load_audio_from_bytes,
    to_mono_16k,
    detect_shout,
    TARGET_SR,
)
from app.analyze.s3_service import S3Uploader

logger = logging.getLogger(__name__)


class AnalyzeService:
    def __init__(self):
        self.audio_fetcher = AudioFetcher()
        try:
            self.s3_uploader = S3Uploader()
        except ValueError as exc:
            logger.warning("S3Uploader initialisation failed: %s", exc)
            self.s3_uploader = None

    async def detect_shout_from_url(self, audio_url: str) -> Dict[str, Any]:
        logger.info("Fetching audio for shout detection: %s", audio_url)
        print(f"[ShoutDetection] Fetching audio: {audio_url}", flush=True)
        audio_bytes = await self.audio_fetcher.fetch(audio_url)
        samples, sr = load_audio_from_bytes(audio_bytes)
        pcm16 = to_mono_16k(samples, sr)
        shout_result = detect_shout(pcm16, TARGET_SR)
        logger.info("Shout detection result: %s", shout_result)
        print(f"[ShoutDetection] Result: {shout_result}", flush=True)
        return asdict(shout_result)

    async def upload_and_analyze_audio(
        self,
        *,
        session_id: str,
        audio_bytes: bytes,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        prefix: str = "oneuld/audio",
    ) -> Dict[str, Any]:
        if not self.s3_uploader:
            raise RuntimeError("S3 업로더가 초기화되지 않았습니다. AWS 환경 변수를 확인하세요.")

        key = self.s3_uploader.upload_audio(
            content=audio_bytes,
            session_id=session_id,
            filename=filename,
            content_type=content_type,
            prefix=prefix,
        )
        audio_url = self.s3_uploader.build_public_url(key)
        logger.info("Uploaded audio and generated public URL %s", audio_url)
        shout_result = await self.detect_shout_from_url(audio_url)
        return {
            "audio_url": audio_url,
            "shout_detection": shout_result,
        }
