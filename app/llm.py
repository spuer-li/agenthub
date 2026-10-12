from openai import AsyncOpenAI

from app.config import settings

client = AsyncOpenAI(
    api_key=settings.dashscope_api_key,
    base_url=settings.dashscope_base_url
)

async def chat(messages: list[dict[str, str]]) -> str:
    response = await client.chat.completions.create(
        model=settings.dashscope_model,
        messages=messages
    )

    content = response.choices[0].message.content
    return content or ""