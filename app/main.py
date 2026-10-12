from fastapi import FastAPI, HTTPException
from app.schemas import EchoRequest, EchoResponse, ChatRequest, ChatResponse
from app.llm import chat
from openai import APIError

import logging
from uuid import uuid4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(title="AgentHub")

conservation_store: dict[str, list[dict[str, str]]] = {}

@app.get("/sessions")
def list_sessions():
    return {
        "sessions": [
            {
                "session_id": session_id,
                "message_count": len(history),
            }
            for session_id, history in conservation_store.items()
        ]
    }

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id not in conservation_store:
        raise HTTPException(
            status_code=404,
            detail="会话不存在。",
        )
    
    del conservation_store[session_id]

    logger.info("会话已删除，session_id=%s", session_id)

    return {
        "message": "会话删除成功",
        "session_id": session_id,
    }

MAX_HISTORY_MESSAGE = 20

SYSTEM_PROMPT = "你是一个友好、准确的AI助手，请使用中文回答用户"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/echo", response_model=EchoResponse)
def echo(req: EchoRequest):
    result = req.message.upper() if req.uppercase else req.message
    return EchoResponse(message=result, length=len(result))

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    session_id = req.session_id or str(uuid4())

    history = conservation_store.get(session_id, [])

    message = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": req.message}
    ]
    logger.info(
        "收到聊天请求，session_id=%s，message_length=%d",
        session_id,
        len(req.message),
    )

    try:
        answer = await chat(message)

        updated_history = [
            *history,
            {"role": "user", "content": req.message},
            {"role": "assistant", "content": answer}
        ]

        conservation_store[session_id] = updated_history[
            -MAX_HISTORY_MESSAGE:
        ]

        logger.info("模型调用成功，session_id=%s", session_id)
        return ChatResponse(answer=answer, session_id=session_id)
    
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