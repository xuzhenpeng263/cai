from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel,Agent,Runner
from agents.model_settings import ModelSettings
from agents import set_default_openai_client, set_tracing_disabled

external_client = AsyncOpenAI(
    base_url = 'http://localhost:4000',
    api_key="cai")

set_default_openai_client(external_client)
set_tracing_disabled(True)

# llm_model="qwen2.5:14b"
# llm_model="claude-3-7"
llm_model="gpt-4o"

# For Qwen models, we need to skip system instructions as they're not supported
instructions = None if "qwen" in llm_model.lower() else "You are a helpful assistant"

agent = Agent(
    name="Assistant", 
    instructions=instructions,
    model=OpenAIChatCompletionsModel(
        model=llm_model,
        openai_client=external_client,
    )
)


result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.