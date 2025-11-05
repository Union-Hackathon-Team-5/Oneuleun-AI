import logging
from dataclasses import asdict
from typing import Any, Dict

from app.analyze.audio_service import AudioFetcher
from app.analyze.shout_service import (
    load_audio_from_bytes,
    to_mono_16k,
    detect_shout,
    TARGET_SR,
)

logger = logging.getLogger(__name__)


class AnalyzeService:
    def __init__(self):
        self.audio_fetcher = AudioFetcher()

    async def detect_shout_from_url(self, audio_url: str) -> Dict[str, Any]:
        logger.info("Fetching audio for shout detection: %s", audio_url)
        audio_bytes = await self.audio_fetcher.fetch(audio_url)
        samples, sr = load_audio_from_bytes(audio_bytes)
        pcm16 = to_mono_16k(samples, sr)
        shout_result = detect_shout(pcm16, TARGET_SR)
        logger.info("Shout detection result: %s", shout_result)
        return asdict(shout_result)
