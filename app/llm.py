from openai import AsyncOpenAI

from app.config import settings

client = AsyncOpenAI(
    api_key=settings.dashscope_api_key,
    base_url=settings.dashscope_base_url
)

async def chat(prompt: str) -> str:
    response = await client.chat.completions.create(
        model=settings.dashscope_model,
        messages=[
            {
                "role": "system",
                "content": "你是一个友好、准确的 AI 助手，请使用中文回答用户。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )

    content = response.choices[0].message.content
    return content or ""