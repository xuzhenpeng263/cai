from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel,Agent,Runner
from agents.model_settings import ModelSettings
from agents import set_default_openai_client, set_tracing_disabled
import os

external_client = AsyncOpenAI(
    base_url = 'https://api.anthropic.com/v1',
    api_key=os.getenv('ANTHROPIC_API_KEY', None)
)

set_default_openai_client(external_client)
set_tracing_disabled(True)

agent = Agent(
    name="Assistant", 
    instructions="You are a helpful assistant",
    model=OpenAIChatCompletionsModel(
        model="claude-3-5-sonnet-20240620",
        openai_client=external_client,
    )
)

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)
