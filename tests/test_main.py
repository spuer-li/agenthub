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



import asyncio
import pytest
from openai import APIError
from app import llm


def test_chat_reraises_api_error(monkeypatch):
    async def mock_create(**kwargs):
        raise APIError(
            message="模拟模型服务故障",
            request=None,
            body=None,
        )

    monkeypatch.setattr(
        llm.client.chat.completions,
        "create",
        mock_create,
    )

    with pytest.raises(APIError, match="模拟模型服务故障"):
        asyncio.run(
            llm.chat(
                [{"role": "user", "content": "你好"}]
            )
        )


def test_chat_endpoint_returns_502_on_api_error(monkeypatch):
    from fastapi.testclient import TestClient
    from openai import APIError
    import app.main as main_module

    async def mock_chat_async(messages):
        raise APIError(
            message="模拟模型服务故障",
            request=None,
            body=None,
        )

    monkeypatch.setattr(main_module, "chat", mock_chat_async)

    client = TestClient(main_module.app)
    response = client.post(
        "/chat",
        json={"message": "你好"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "大模型服务调用失败，请检查配置或稍后重试。"
    )

def test_execute_tool_calculator_division_by_zero():
    import pytest
    from app.tools import execute_tool

    with pytest.raises(ValueError, match="除数不能为零"):
        execute_tool(
            "calculator",
            {"expression": "10 / 0"}
        )


def test_chat_handles_tool_execution_error(monkeypatch):
    import asyncio
    import json
    from types import SimpleNamespace
    from app import llm

    calls = {"count": 0}
    captured_messages = []

    async def mock_create(**kwargs):
        calls["count"] += 1
        captured_messages.append(kwargs["messages"].copy())

        if calls["count"] == 1:
            tool_call = SimpleNamespace(
                id = "test_call_1",
                function=SimpleNamespace(
                    name="calculator",
                    arguments=json.dumps(
                        {"expression": "10 / 0"}
                    )
                )
            )
            message = SimpleNamespace(
                content=None,
                tool_calls=[tool_call],
                model_dump=lambda **kwargs: {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "test_call_1",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": json.dumps(
                                    {"expression": "10 / 0"}
                                )
                            }
                        }
                    ]
                }
            )
        else:
            message = SimpleNamespace(
                content = "计算失败，因为除数不能为零",
                tool_calls=None,
                model_dump=lambda **kwargs: {
                    "role": "assistant",
                    "content": "计算失败，因为除数不能为零"
                }
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )

    monkeypatch.setattr(
        llm.client.chat.completions,
        "create",
        mock_create
    )

    answer = asyncio.run(
        llm.chat(
            [{"role": "user", "content": "计算 10 / 0"}]
        )
    )

    assert answer == "计算失败，因为除数不能为零"
    assert calls["count"] == 2

    tool_messages = [
        message
        for message in captured_messages[1]
        if message.get("role") == "tool"
    ]
    assert len(tool_messages) == 1
    assert "除数不能为零" in tool_messages[0]["content"]


def test_chat_handles_invalid_tool_arguments(monkeypatch, caplog):
    import asyncio
    import logging
    from types import SimpleNamespace
    from app import llm

    calls = {"count": 0}

    async def mock_create(**kwargs):
        calls["count"] += 1

        if calls["count"] == 1:
            message = SimpleNamespace(
                content=None,
                tool_calls=[
                    SimpleNamespace(
                        id="invalid_json_call",
                        function=SimpleNamespace(
                            name="calculator",
                            arguments='{"expression": ',
                        ),
                    )
                ],
                model_dump=lambda **kwargs: {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "invalid_json_call",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression": ',
                            },
                        }
                    ],
                },
            )
        else:
            message = SimpleNamespace(
                content="工具参数格式有误，无法完成计算。",
                tool_calls=None,
                model_dump=lambda **kwargs: {
                    "role": "assistant",
                    "content": "工具参数格式有误，无法完成计算。",
                },
            )

        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )

    monkeypatch.setattr(
        llm.client.chat.completions,
        "create",
        mock_create,
    )

    with caplog.at_level(logging.WARNING, logger="app.llm"):
        answer = asyncio.run(
            llm.chat(
                [{"role": "user", "content": "请计算 10 / 2"}]
            )
        )

    assert calls["count"] == 2
    assert "工具参数格式有误" in answer
    assert "Agent 工具执行失败" in caplog.text
    assert "参数解析失败" in caplog.text

def test_chat_handles_multiple_tool_calls(monkeypatch):
    import asyncio
    import json
    from types import SimpleNamespace
    from app import llm

    calls = {"count": 0}
    captured_message = []

    async def mock_create(**kwargs):
        calls["count"] += 1 
        captured_message.append(kwargs["messages"].copy())

        if calls["count"] == 1:
            tool_calls = [
                SimpleNamespace(
                    id="calc_call",
                    function=SimpleNamespace(
                        name="calculator",
                        arguments=json.dumps(
                            {"expression": "125 * 36"}
                        )
                    )
                ),
                SimpleNamespace(
                    id="format_call",
                    function=SimpleNamespace(
                        name="format_text",
                        arguments=json.dumps(
                            {"text": "hello", "uppercase": True}
                        )
                    )
                )
            ]

            message = SimpleNamespace(
                content=None,
                tool_calls=tool_calls,
                model_dump=lambda **kwargs: {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "calc_call",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": json.dumps(
                                    {"expression": "125 * 36"}
                                )
                            }
                        },
                        {
                            "id": "format_call",
                            "type": "function",
                            "function": {
                                "name": "format_text",
                                "arguments": json.dumps(
                                    {"text": "hello", "uppercase": True}
                                )
                            }
                        }
                    ]
                }
            )
        else:
            message = SimpleNamespace(
                content="计算结果为 4500，转换后的文本为 HELLO。",
                tool_calls=None,
                model_dump=lambda **kwargs: {
                    "role": "assistant",
                    "content": "计算结果为 4500，转换后的文本为 HELLO。"
                }
            )

        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )

    monkeypatch.setattr(
        llm.client.chat.completions,
        "create",
        mock_create
    )

    answer = asyncio.run(
        llm.chat(
            [{"role": "user", "content": "计算 125 * 36，并把 hello 转为大写"}]
        )
    )

    assert calls["count"] == 2
    assert "4500" in answer
    assert "HELLO" in answer

    tool_messages = [
        message
        for message in captured_message[1]
        if message.get("role") == "tool"
    ]

    assert len(tool_messages) == 2

    results_by_id = {
        message["tool_call_id"]: message["content"]
        for message in tool_messages
    }

    assert results_by_id["calc_call"] == "4500"
    assert results_by_id["format_call"] == "HELLO"

def test_chat_stops_after_five_rounds(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from app import llm

    calls = {"count": 0}

    async def mock_create(**kwargs):
        calls["count"] += 1

        message = SimpleNamespace(
            content=None,
            tool_calls=[],
            model_dump=lambda **kwargs: {
                "role": "assistant",
                "content": None,
                "tool_calls": []
            }
        )

        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )

    monkeypatch.setattr(
        llm.client.chat.completions,
        "create",
        mock_create
    )

    answer = asyncio.run(
        llm.chat([{"role": "user", "content": "测试"}])
    )

    assert calls["count"] == 1
    assert answer == ""

def test_chat_stops_after_five_tool_rounds(monkeypatch):
    import asyncio
    import json
    from types import SimpleNamespace
    from app import llm

    calls = {"count": 0}

    async def mock_create(**kwargs):
        calls["count"] += 1

        tool_call = SimpleNamespace(
            id=f"call_{calls['count']}",
            function=SimpleNamespace(
                name="calculator",
                arguments=json.dumps({"expression": "1 + 1"})
            )
        )

        message = SimpleNamespace(
            content=None,
            tool_calls=[tool_call],
            model_dump=lambda **kwargs: {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{calls['count']}",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": json.dumps(
                                {"expression": "1 + 1"}
                            )
                        }
                    }
                ]
            }
        )

        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )

    monkeypatch.setattr(
        llm.client.chat.completions,
        "create",
        mock_create
    )

    answer = asyncio.run(
        llm.chat([{"role": "user", "content": "一直计算1+1"}])
    )

    assert calls["count"] == 5
    assert answer == "工具调用次数过多，请简化问题后重试。"

def test_chat_continues_after_one_tool_fails(monkeypatch):
    import json
    import asyncio
    from types import SimpleNamespace
    from app import llm

    calls = {"count": 0}
    captured_messages = []

    async def mock_create(**kwargs):
        calls["count"] += 1
        captured_messages.append(kwargs["messages"].copy())

        if calls["count"] == 1:
            tool_calls = [
                SimpleNamespace(
                    id="calc_call",
                    function=SimpleNamespace(
                        name="calculator",
                        arguments=json.dumps(
                            {"expression": "1 / 0"}
                        )
                    )
                ),
                SimpleNamespace(
                    id="format_call",
                    function=SimpleNamespace(
                        name="format_text",
                        arguments=json.dumps(
                            {"text": "hello", "uppercase": True}
                        )
                    )
                )
            ]

            message = SimpleNamespace(
                content=None,
                tool_calls=tool_calls,
                model_dump=lambda **kwargs: {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "calc_call",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": json.dumps(
                                    {"expression": "1 / 0"}
                                )
                            }
                        },
                        {
                            "id": "format_call",
                            "type": "function",
                            "function": {
                                "name": "format_text",
                                "arguments": json.dumps(
                                    {"text": "hello", "uppercase": True}
                                )
                            }
                        }
                    ]
                }
            )
        else:
            message = SimpleNamespace(
                content="计算工具执行失败，但文本转换成功，结果是 HELLO。",
                tool_calls=None,
                model_dump=lambda **kwargs: {
                    "role": "assistant",
                    "content": "计算工具执行失败，但文本转换成功，结果是 HELLO。"
                }
            )

        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )

    monkeypatch.setattr(
        llm.client.chat.completions,
        "create",
        mock_create
    )

    answer = asyncio.run(
        llm.chat(
            [{"role": "user", "content": "计算 1 / 0，并把 hello 转为大写"}]
        )
    )

    assert calls["count"] == 2
    assert "HELLO" in answer

    tool_message = [
        message
        for message in captured_messages[1]
        if message.get("role") == "tool"
    ]

    assert len(tool_message) == 2

    results_by_id = {
        message["tool_call_id"]: message["content"]
        for message in tool_message
    }

    assert "工具执行失败" in results_by_id["calc_call"]
    assert "除数不能为零" in results_by_id["calc_call"]
    assert results_by_id["format_call"] == "HELLO"

def test_chat_does_not_modify_original_messages(monkeypatch):
    import asyncio
    import json
    from types import SimpleNamespace
    from app import llm

    original_message = [
        {"role": "user", "content": "计算2 + 3"}
    ]

    origianl_snapshot = [
        message.copy() for message in original_message
    ]

    calls = {"count": 0}

    async def mock_create(**kwargs):
        calls["count"] += 1

        if calls["count"] == 1:
            tool_call = SimpleNamespace(
                id="calc_call",
                function=SimpleNamespace(
                    name="calculator",
                    arguments=json.dumps(
                        {"expression": "2 + 3"}
                    )
                )
            )

            message = SimpleNamespace(
                content=None,
                tool_calls=[tool_call],
                model_dump=lambda **kwargs: {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "calc_call",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": json.dumps(
                                    {"expression": "2 + 3"}
                                )
                            }
                        }
                    ]
                }
            )
        else:
            message = SimpleNamespace(
                content="计算结果是5",
                tool_calls=None
            )

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=message
                )
            ]
        )

    monkeypatch.setattr(
        llm.client.chat.completions,
        "create",
        mock_create
    )

    answer = asyncio.run(llm.chat(original_message))

    assert answer == "计算结果是5"
    assert calls["count"] == 2
    assert original_message == origianl_snapshot

def test_chat_handles_tool_call_without_id(monkeypatch):
    import asyncio
    import json
    from types import SimpleNamespace
    from app import llm

    calls = {"count": 0}

    async def mock_create(**kwargs):
        calls["count"] += 1

        tool_call=SimpleNamespace(
            id=None,
            function=SimpleNamespace(
                name="calculator",
                arguments=json.dumps(
                    {"expression": "2 + 3"}
                )
            )
        )

        message=SimpleNamespace(
            content=None,
            tool_calls=[tool_call],
            model_dump=lambda **kwargs: {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": None,
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": json.dumps(
                                {"expression": "2 + 3"}
                            )
                        }
                    }
                ]
            }
        )

        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )

    monkeypatch.setattr(
        llm.client.chat.completions,
        "create",
        mock_create
    )

    answer = asyncio.run(llm.chat(
            [{"role": "user", "content": "计算 2 + 3"}]
        ))

    assert calls["count"] == 1
    assert answer == "工具调用 ID 缺失，无法继续执行。"

def test_chat_handles_missing_tool_function(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from app import llm

    calls = {"count": 0}

    async def mock_create(**kwargs):
        calls["count"] += 1

        if calls["count"] == 1:
            message=SimpleNamespace(
                content=None,
                tool_calls=[
                    SimpleNamespace(
                        id="missing_function_call",
                        function=None
                    )
                ],
                model_dump=lambda **kwargs: {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "missiong_function_call",
                            "type": "function",
                            "function": None
                        }
                    ]
                }
            )
        else:
            message=SimpleNamespace(
                content="工具调用结构异常。",
                tool_calls=None,
                model_dump=lambda **kwargs: {
                    "role": "assistant",
                    "content": "工具调用结构异常。",
                },
            )

        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )

    monkeypatch.setattr(
        llm.client.chat.completions,
        "create",
        mock_create,
    )

    answer = asyncio.run(
        llm.chat(
            [{"role": "user", "content": "请计算 2 + 3"}]
        )
    )

    assert answer == "工具调用结构异常，无法继续执行。"
    assert calls["count"] == 1

def test_chat_handles_missing_tool_name(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from app import llm

    async def mock_create(**kwargs):
        tool_call = SimpleNamespace(
            id="missing_name_call",
            function=SimpleNamespace(
                name=None,
                arguments='{"expression": "2 + 3"}',
            ),
        )
        message = SimpleNamespace(
            content=None,
            tool_calls=[tool_call],
            model_dump=lambda **kwargs: {},
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )

    monkeypatch.setattr(
        llm.client.chat.completions,
        "create",
        mock_create,
    )

    answer = asyncio.run(llm.chat([
        {"role": "user", "content": "计算 2 + 3"}
    ]))

    assert answer == "工具调用名称缺失，无法继续执行。"

def test_chat_propagates_unexpected_tool_error(monkeypatch):
    import asyncio
    import json
    import pytest
    from types import SimpleNamespace
    from app import llm

    async def mock_create(**kwargs):
        tool_call=SimpleNamespace(
            id="runtime_error_call",
            function=SimpleNamespace(
                name="calculator",
                arguments=json.dumps({"expression": "1 + 1"})
            )
        )

        message=SimpleNamespace(
            content=None,
            tool_calls=[tool_call],
            model_dump=lambda **kwargs: {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "runtime_error_call",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": json.dumps(
                                {"expression": "1 + 1"}
                            )
                        }
                    }
                ]
            }
        )

        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )

    def mock_execute_tool(tool_name, arguments):
        raise RuntimeError("模拟工具内部异常") 

    monkeypatch.setattr(
        llm.client.chat.completions,
        "create",
        mock_create
    )

    monkeypatch.setattr(
        llm,
        "execute_tool",
        mock_execute_tool
    )

    with pytest.raises(RuntimeError, match="模拟工具内部异常"):
        asyncio.run(
            llm.chat(
                [{"role": "user", "content": "1 + 1"}]
            )
        )

