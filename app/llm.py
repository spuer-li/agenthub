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

        messages.append(message.model_dump(exclude_none=True))

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            start_time = time.perf_counter()
            arguments = "参数解析失败"

            try:
                arguments = json.loads(
                    tool_call.function.arguments
                )

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
