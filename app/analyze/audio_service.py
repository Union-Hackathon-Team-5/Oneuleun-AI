import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class AudioFetcher:
    """
    Minimal HTTP downloader for audio assets stored on S3.
    Assumes the provided URL is publicly accessible or pre-signed.
    """

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def fetch(self, url: str) -> bytes:
        if not url:
            raise ValueError("audio url is required")

        client = await self._get_client()

        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as exc:
            logger.error("Failed to fetch audio from %s (%s)", url, exc.response.status_code)
            raise RuntimeError("음성 데이터를 다운로드하지 못했습니다.") from exc
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Unexpected error fetching audio from %s: %s", url, exc)
            raise RuntimeError("음성 데이터 처리 중 오류가 발생했습니다.") from exc
