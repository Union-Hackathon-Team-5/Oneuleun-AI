import logging
import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.context.router import router as context_router
from app.analyze.router import router as analyze_router

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: List[str] | None = None) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return default or []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _configure_cors(app_: FastAPI) -> None:
    origins_raw = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
    allow_credentials = _env_bool("CORS_ALLOW_CREDENTIALS", False)
    allow_methods = _env_list("CORS_ALLOW_METHODS") or ["*"]
    allow_headers = _env_list("CORS_ALLOW_HEADERS") or ["*"]
    expose_headers = _env_list("CORS_EXPOSE_HEADERS")

    if origins_raw == "*":
        if allow_credentials:
            logger.warning(
                "CORS_ALLOW_CREDENTIALS requested with wildcard origins; disabling credentials."
            )
            allow_credentials = False
        allow_origins = ["*"]
    else:
        allow_origins = _env_list("CORS_ALLOW_ORIGINS") or ["*"]

    logger.info(
        "Configuring CORS: origins=%s credentials=%s methods=%s headers=%s expose=%s",
        allow_origins,
        allow_credentials,
        allow_methods,
        allow_headers,
        expose_headers,
    )

    app_.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=allow_methods,
        allow_headers=allow_headers,
        expose_headers=expose_headers,
    )


app = FastAPI(title="Oneuleun AI API", version="0.2.0")

_configure_cors(app)

app.include_router(context_router)
app.include_router(analyze_router)


@app.get("/")
async def root():
    return {"service": "Oneuleun AI API", "status": "running"}
