from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel,Agent,Runner
from agents.model_settings import ModelSettings
from agents import set_default_openai_client, set_tracing_disabled

external_client = AsyncOpenAI(
    base_url = 'http://localhost:8000/v1',
    api_key='ollama', # required, but unused
)

set_default_openai_client(external_client)
set_tracing_disabled(True)

agent = Agent(
    name="Assistant", 
    instructions="You are a helpful assistant",
    model=OpenAIChatCompletionsModel(
        model="qwen2.5:14b",
        openai_client=external_client,
    )
    )

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.