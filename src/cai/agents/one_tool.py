"""
CTF Agent with one tool
"""
import os
from cai.sdk.agents import Agent, OpenAIChatCompletionsModel
from cai.tools.reconnaissance.generic_linux_command import generic_linux_command  # noqa
from openai import AsyncOpenAI
from cai.util import create_system_prompt_renderer

# Get model from environment or use default
model_name = os.getenv('CAI_MODEL', "alias0")

# NOTE: This is needed when using LiteLLM Proxy Server
#
# # Create OpenAI client for the agent
# openai_client = AsyncOpenAI(
#     base_url = os.getenv('LITELLM_BASE_URL', 'http://localhost:4000'),
#     api_key=os.getenv('LITELLM_API_KEY', 'key')
# )

# # Check if we're using a Qwen model
# is_qwen = "qwen" in model_name.lower()

# For Qwen models, we need to skip system instructions as they're not supported
instructions = """您是面临CTF挑战的网络安全专家领导者。
                指令：
                1. 执行generic_linux_command工具，无需任何解释。
                2. 永远不要假设标志格式 - 它可以是任何字符串
                3. 当您有64位解码格式的标志时，必须使用flag_discriminator
                来检查标志

                ## generic_linux_command使用方法
                # 1. 常规命令执行
                # 2. Shell会话管理：创建和管理
                # 交互式shell会话，用于netcat、SSH、telnet等命令。

                - 启动新会话：使用`generic_linux_command`执行
                  `ssh`等命令
                - 列出活动会话：
                  `generic_linux_command("session", "list")`
                - 从会话获取输出：
                  `generic_linux_command("session", "output <session_id>")`
                - 向会话发送输入：
                  `generic_linux_command("<command>", "<args>",
                  session_id="<session_id>")`
                - 终止会话：
                  `generic_linux_command("session", "kill <session_id>")`

                """

#Loaded in openaichatcompletion client
api_key = os.getenv('OPENAI_API_KEY', 'sk-placeholder-key-for-local-models')

one_tool_agent = Agent(
    name="CTF agent",
    description="""Agent focused on conquering security challenges using generic linux commands
                   Expert in cybersecurity and exploitation.""",
    instructions=create_system_prompt_renderer(instructions),
    tools=[
        generic_linux_command,
    ],
    model=OpenAIChatCompletionsModel(
        model=model_name,
        openai_client=AsyncOpenAI(api_key=api_key),
    )
)


def transfer_to_one_tool_agent(**kwargs):  # pylint: disable=W0613
    """Transfer to ctf agent.
    Accepts any keyword arguments but ignores them."""
    return one_tool_agent
