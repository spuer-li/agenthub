from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_echo():
    response = client.post(
        "/echo",
        json={
            "message": "hello",
            "uppercase": True,
        }
    )

    assert response.status_code == 200
    assert response.json() == {
        "message": "HELLO",
        "length": 5
    }

def test_echo_validation():
    response = client.post(
        "/echo",
        json={
            "message": "",
            "uppercase": False,
        },
    )

    assert response.status_code == 422

from unittest.mock import AsyncMock, patch

def test_chat():
    # 模拟大模型返回，不发送真实 API 请求
    with patch(
        "app.main.chat",
        new_callable=AsyncMock,
    ) as mock_chat:
        mock_chat.return_value = "你好，我是模拟模型。"

        response = client.post(
            "/chat",
            json={"message": "你好"},
        )

    assert response.status_code == 200

    data = response.json()
    assert data["answer"] == "你好，我是模拟模型。"
    assert "session_id" in data

    # 确认模型收到的消息中包含用户问题
    messages = mock_chat.call_args.args[0]
    assert messages[-1] == {
        "role": "user",
        "content": "你好",
    }

def test_chat_session_memory():
    session_id = "test-session-memory"

    with patch(
        "app.main.chat",
        new_callable=AsyncMock,
    ) as mock_chat:
        mock_chat.return_value = "好的，我记住了。"

        first_response = client.post(
            "/chat",
            json={
                "message": "我叫小明",
                "session_id": session_id,
            },
        )

        second_response = client.post(
            "/chat",
            json={
                "message": "我叫什么？",
                "session_id": session_id,
            },
        )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    # 第二次请求应包含第一次的聊天记录
    second_messages = mock_chat.call_args.args[0]

    assert any(
        msg["role"] == "user" and msg["content"] == "我叫小明"
        for msg in second_messages
    )
    assert second_messages[-1] == {
        "role": "user",
        "content": "我叫什么？",
    }