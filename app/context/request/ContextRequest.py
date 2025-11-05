from pydantic import BaseModel


class ContextRequest(BaseModel):
    session_id: str
    user_id: str
    photo_url: str
