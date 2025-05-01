from pydantic import BaseModel,Field
from datetime import date,datetime
from typing import Optional

class VideoRequest(BaseModel):
    youtube_url: str=Field(description="Youtube video link")


class VideoResponse(BaseModel):
    video_response:str=Field(description="The answer of the question")
class ChatRequest(BaseModel):
    youtube_url: str
    message: str
    chat_history:list[tuple[str, str]]=[]
class ChatResponse(BaseModel):
    response: str
