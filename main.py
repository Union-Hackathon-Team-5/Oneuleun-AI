from fastapi import FastAPI

from app.context.router import router as context_router
from app.analyze.router import router as analyze_router

app = FastAPI(title="Oneuleun AI API", version="0.2.0")

app.include_router(context_router)
app.include_router(analyze_router)


@app.get("/")
async def root():
    return {"service": "Oneuleun AI API", "status": "running"}
