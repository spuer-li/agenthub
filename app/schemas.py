from pydantic import BaseModel, Field

class EchoRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=100)
    uppercase: bool = False

class EchoResponse(BaseModel):
    message: str
    length: int 

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)

class ChatResponse(BaseModel):
    answer: str