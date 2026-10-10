from fastapi import FastAPI, HTTPException
from app.schemas import EchoRequest, EchoResponse, ChatRequest, ChatResponse
from app.llm import chat
from openai import APIError

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(title="AgentHub")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/echo", response_model=EchoResponse)
def echo(req: EchoRequest):
    result = req.message.upper() if req.uppercase else req.message
    return EchoResponse(message=result, length=len(result))

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    logger.info("收到聊天请求，message_length=%d", len(req.message))

    try:
        answer = await chat(req.message)
        logger.info("模型调用成功")
        return ChatResponse(answer=answer)
    
    except APIError:
        logger.exception("模型服务返回 API 错误")    
        raise HTTPException(
            status_code=502,
            detail="大模型服务调用失败，请检查配置或稍后重试。",
        )

    except Exception:
        logger.exception("聊天接口发生未预期错误")
        raise HTTPException(
            status_code=500,
            detail="服务器内部错误，请稍后重试。",
        )