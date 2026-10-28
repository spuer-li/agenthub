import json
import logging
import time

from openai import AsyncOpenAI, APIError

from app.config import settings
from app.tools import TOOL_REGISTRY, execute_tool

logger = logging.getLogger(__name__)

client = AsyncOpenAI(
    api_key=settings.dashscope_api_key,
    base_url=settings.dashscope_base_url,
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": config["description"],
            "parameters": {
                "type": "object",
                "properties": {
                    parameter_name: {
                        "type": parameter_config["type"],
                        "description": parameter_config["description"],
                    }
                    for parameter_name, parameter_config in config["parameters"].items()
                },
                "required": [
                    parameter_name
                    for parameter_name, parameter_config in config["parameters"].items()
                    if parameter_config.get("required", True)
                    ],
                "additionalProperties": False,
            },
        },
    }
    for tool_name, config in TOOL_REGISTRY.items()
]


async def chat(messages: list[dict]) -> str:
    messages = [message.copy() for message in messages]

    for _ in range(5):
        try:
            response = await client.chat.completions.create(
                model=settings.dashscope_model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        except APIError:
            logging.exception("大模型 API 调用失败")
            raise

        message = response.choices[0].message
        tool_calls = message.tool_calls

        if not tool_calls:
            return message.content or ""

        for tool_call in tool_calls:
            if not tool_call.id:
                logger.error("模型返回的工具调用缺少 ID")
                return "工具调用 ID 缺失，无法继续执行。"

        messages.append(message.model_dump(exclude_none=True))

        for tool_call in tool_calls:
            start_time = time.perf_counter()
            arguments = "参数解析失败"

            if not getattr(tool_call, "function", None):
                logger.error(
                    "模型调用的返回缺少function对象：tool_call_id=%s",
                    getattr(tool_call, "id", None),
                )
                return "工具调用结构异常，无法继续执行。"

            tool_name = getattr(tool_call.function, "name", None)

            if not isinstance(tool_name, str) or not tool_name.strip():
                logger.error(
                    "模型返回的工具调用缺少有效名称：tool_call_id=%s",
                    tool_call.id,
                )
                return "工具调用名称缺失，无法继续执行。"
            try:
                raw_arguments = getattr(tool_call.function, "arguments", None)

                if not isinstance(raw_arguments, str) or not raw_arguments.strip():
                    raise ValueError("工具参数必须是非空 JSON 字符串")

                arguments = json.loads(raw_arguments)

                if not isinstance(arguments, dict):
                    raise ValueError("工具参数必须是 JSON 对象")

                if tool_name not in TOOL_REGISTRY:
                    raise ValueError(f"不支持的工具：{tool_name}")

                result = execute_tool(tool_name, arguments)

                elapsed = time.perf_counter() - start_time

                logger.info(
                    "Agent 工具执行成功：tool=%s, arguments=%s,耗时=%.3f 秒",
                    tool_name,
                    arguments,
                    elapsed
                )

            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                elapsed = time.perf_counter() - start_time

                logger.warning(
                    "Agent 工具执行失败：tool=%s, arguments=%s, 耗时=%.3f 秒，原因=%s",
                    tool_name,
                    arguments,
                    elapsed,
                    exc,
                )
                result = f"工具执行失败：{exc}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                }
            )

    return "工具调用次数过多，请简化问题后重试。"
