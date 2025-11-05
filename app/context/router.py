from fastapi import APIRouter
from app.context.request.ContextRequest import ContextRequest

router = APIRouter(
    prefix="/context",
    tags=["context"]
)



@router.post("/")
async def analyze_context(request: ContextRequest):
    return {
        "message": "Context created successfully",
        "session_id": request.session_id,
        "user_id": request.user_id,
        "photo_url": request.photo_url
    }