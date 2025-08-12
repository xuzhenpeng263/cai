from __future__ import annotations

import os

from cai import Agent, Runner, RunConfig, function_tool
from cai.sdk.agents.models import ZhipuAIProvider

# Set your ZhipuAI API key
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY") or "your-api-key"

if not ZHIPUAI_API_KEY or ZHIPUAI_API_KEY == "your-api-key":
    raise ValueError(
        "Please set ZHIPUAI_API_KEY via env var or code."
    )


# Create a ZhipuAI provider
zhipuai_provider = ZhipuAIProvider(api_key=ZHIPUAI_API_KEY)


@function_tool
def get_weather(city: str):
    """Get the weather for a city"""
    print(f"[debug] getting weather for {city}")
    return f"The weather in {city} is sunny."


async def main():
    # Create an agent with ZhipuAI provider
    agent = Agent(
        name="Assistant", 
        instructions="You are a helpful assistant.", 
        tools=[get_weather]
    )

    # Run with ZhipuAI provider
    result = await Runner.run(
        agent,
        "What's the weather in Tokyo?",
        run_config=RunConfig(model_provider=zhipuai_provider, model="glm-4.5"),
    )
    print(result.final_output)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())