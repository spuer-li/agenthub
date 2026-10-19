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


def test_chat_calls_calculator():
    from unittest.mock import AsyncMock, patch

    from app.llm import chat

    # 模拟大模型第一次返回工具调用请求
    first_response = type(
        "Response",
        (),
        {
            "choices": [
                type(
                    "Choice",
                    (),
                    {
                        "message": type(
                            "Message",
                            (),
                            {
                                "content": None,
                                "tool_calls": [
                                    type(
                                        "ToolCall",
                                        (),
                                        {
                                            "id": "call_test_123",
                                            "function": type(
                                                "Function",
                                                (),
                                                {
                                                    "name": "calculator",
                                                    "arguments": (
                                                        '{"expression": "125 * 36"}'
                                                    ),
                                                },
                                            )(),
                                        },
                                    )()
                                ],
                                "model_dump": lambda self, **kwargs: {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call_test_123",
                                            "type": "function",
                                            "function": {
                                                "name": "calculator",
                                                "arguments": (
                                                    '{"expression": "125 * 36"}'
                                                ),
                                            },
                                        }
                                    ],
                                },
                            },
                        )(),
                    },
                )()
            ]
        },
    )()

    # 模拟大模型收到计算结果后生成最终答案
    second_response = type(
        "Response",
        (),
        {
            "choices": [
                type(
                    "Choice",
                    (),
                    {
                        "message": type(
                            "Message",
                            (),
                            {
                                "content": "结果是 4500",
                                "tool_calls": None,
                            },
                        )()
                    },
                )()
            ]
        },
    )()

    with patch(
        "app.llm.client.chat.completions.create",
        new_callable=AsyncMock,
        side_effect=[first_response, second_response],
    ) as mock_create:
        result = __import__(
            "asyncio"
        ).run(chat([{"role": "user", "content": "125乘以36"}]))

    assert result == "结果是 4500"
    assert mock_create.await_count == 2

    # 第二次请求应包含工具执行结果
    second_messages = mock_create.await_args_list[1].kwargs["messages"]

    assert any(
        msg.get("role") == "tool"
        and msg.get("tool_call_id") == "call_test_123"
        and msg.get("content") == "4500"
        for msg in second_messages
    )

def test_text_stats():
    from app.tools import execute_tool
    import json

    result = execute_tool(
        "text_stats",
        {"text": "Hello World"}
    )

    data = json.loads(result)

    assert data["character_count"] == 11
    assert data["non_whitespace_count"] == 10
    assert data["word_count"] == 2

def test_execute_tool_rejects_non_dict_arguments():
    from app.tools import execute_tool
    import pytest

    with pytest.raises(ValueError, match="工具参数必须是字典"):
        execute_tool("calculator", None)

def test_execute_tool_rejects_unknown_tool():
    from app.tools import execute_tool
    import pytest

    with pytest.raises(ValueError, match="不支持的工具"):
        execute_tool("unknown_tool", {})

def test_add_numbers_integers():
    from app.tools import execute_tool

    result = execute_tool("add_numbers", {"a": 12, "b": 30})

    assert result == "42"

def test_add_numbers_float():
    from app.tools import execute_tool

    result = execute_tool("add_numbers", {"a": 1.5, "b": 2.25})

    assert result == "3.75"

def test_add_numbers_reject_boolean():

    import pytest
    from app.tools import execute_tool

    with pytest.raises(ValueError):
        execute_tool("add_numbers", {"a": True, "b": 2})

def test_repeat_text_success():
    from app.tools import execute_tool

    result = execute_tool("repeat_text", {"text": "Hi", "times": 3})

    assert result == "HiHiHi"

def test_repeat_text_reject_float():
    import pytest
    from app.tools import execute_tool

    with pytest.raises(ValueError):
        execute_tool("repeat_text", {"text": "Hi", "times": 2.5})

def test_repeat_text_reject_boolean():
    import pytest
    from app.tools import execute_tool

    with pytest.raises(ValueError):
        execute_tool("repeat_text", {"text": "Hi", "Times": True})

def test_format_text_uppercase():
    from app.tools import execute_tool

    result = execute_tool("format_text", {"text": "hello", "uppercase": True})

    assert result == "HELLO"

def test_format_text_preserves_case():
    from app.tools import execute_tool

    result = execute_tool("format_text", {"text": "hello", "uppercase": False})

    assert result == "hello"

def test_format_text_reject_integer_as_boolean():
    import pytest
    from app.tools import execute_tool

    with pytest.raises(ValueError):
        execute_tool("format_text", {"text": "hello", "uppercase": 1})