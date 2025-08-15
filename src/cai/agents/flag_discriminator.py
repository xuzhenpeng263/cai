"""
CTF Flag Discriminator Agent with test
"""
import os
from cai.sdk.agents import Agent, OpenAIChatCompletionsModel, handoff
from openai import AsyncOpenAI
from cai.agents.one_tool import one_tool_agent

model = os.getenv('CAI_MODEL', "alias0")

# Create OpenAI client with fallback API key to prevent initialization errors
# The actual API key should be set in environment variables or .env file
api_key = os.getenv('OPENAI_API_KEY', 'sk-placeholder-key-for-local-models')

flag_discriminator = Agent(
    name="Flag discriminator",
    description="Agent focused on extracting the flag from the output",
    instructions="""指令：
                1. 您是一个专门从给定输出中提取标志的代理。
                2. 仅返回标志，不要返回其他内容。
                3. 标志可以是任何格式（不一定是CTF{...}）。
                4. 如果您没有找到标志，请调用`ctf_agent`继续调查。
                """,
    model=OpenAIChatCompletionsModel(
        model="alias0" if os.getenv('CAI_MODEL') == "o3-mini" else model,
        openai_client=AsyncOpenAI(api_key=api_key),
    ),
    handoffs=[
        handoff(
            agent=one_tool_agent,
            tool_name_override="ctf_agent",
            tool_description_override="如果没有找到标志，调用CTF代理继续调查"
        )
    ]
)

# Transfer Function
def transfer_to_flag_discriminator(**kwargs):  # pylint: disable=W0613
    """Transfer flag discriminator.
    Accepts any keyword arguments but ignores them."""
    return flag_discriminator
