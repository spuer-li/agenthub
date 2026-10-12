import json
import logging

from openai import AsyncOpenAI

from app.config import settings
from app.tools import calculator

logger = logging.getLogger(__name__)

client = AsyncOpenAI(
    api_key=settings.dashscope_api_key,
    base_url=settings.dashscope_base_url,
)


# 向模型声明：当前可以使用哪些工具
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
    }
]


async def chat(messages: list[dict]) -> str:
    # 最多执行 5 轮模型请求，防止工具调用无限循环
    for _ in range(5):
        response = await client.chat.completions.create(
            model=settings.dashscope_model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message
        tool_calls = message.tool_calls

        # 模型没有请求调用工具，直接返回最终回答
        if not tool_calls:
            return message.content or ""

        # 把模型生成的工具调用请求加入对话
        messages.append(
            message.model_dump(exclude_none=True)
        )

        # Python 根据模型的请求执行工具
        for tool_call in tool_calls:
            tool_name = tool_call.function.name

            try:
                arguments = json.loads(
                    tool_call.function.arguments
                )

                if tool_name == "calculator":
                    logger.info(
                        "Agent 工具调用：工具名称=%s，参数=%s",
                        tool_name,
                        arguments,
                    )
                    expression = arguments.get("expression")

                    if not isinstance(expression, str):
                        raise ValueError("expression 必须是字符串")

                    result = calculator(expression)
                else:
                    result = f"错误：不支持的工具 {tool_name}"

            except (ValueError, TypeError, json.JSONDecodeError):
                result = "工具调用失败：参数无效，请检查表达式。"

            # 将工具执行结果交还给模型
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                }
            )

    return "工具调用次数过多，请简化问题后重试。"
