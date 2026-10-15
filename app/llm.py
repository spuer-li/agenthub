
import json
import logging

from openai import AsyncOpenAI

from app.config import settings
from app.tools import TOOL_REGISTRY, execute_tool, text_stats

logger = logging.getLogger(__name__)

client = AsyncOpenAI(
    api_key=settings.dashscope_api_key,
    base_url=settings.dashscope_base_url,
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学四则运算。当用户需要计算时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如 125 * 36",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    
    {
        "type": "function",
        "function": {
            "name": "text_stats",
            "description": (
                "统计一段文本的字符总数、非空白字符数和单词数。"
                "当用户要求统计文本长度或单词数量时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "需要统计的文本内容",
                    }
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    },

]


async def chat(messages: list[dict]) -> str:
    for _ in range(5):
        response = await client.chat.completions.create(
            model=settings.dashscope_model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message
        tool_calls = message.tool_calls

        if not tool_calls:
            return message.content or ""

        messages.append(message.model_dump(exclude_none=True))

        for tool_call in tool_calls:
            tool_name = tool_call.function.name

            try:
                arguments = json.loads(
                    tool_call.function.arguments
                )

                if not isinstance(arguments, dict):
                    raise ValueError("工具参数必须是 JSON 对象")

                if tool_name not in TOOL_REGISTRY:
                    raise ValueError(f"不支持的工具：{tool_name}")

                result = execute_tool(tool_name, arguments)

                logger.info(
                    "Agent 工具执行成功：tool=%s",
                    tool_name,
                )

            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                logger.warning(
                    "Agent 工具执行失败：tool=%s，原因=%s",
                    tool_name,
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
