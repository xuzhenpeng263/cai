"""Red Team Base Agent"""
import os
from dotenv import load_dotenv
from cai.sdk.agents import Agent, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from cai.util import load_prompt_template, create_system_prompt_renderer
from cai.tools.command_and_control.sshpass import (  # pylint: disable=import-error # noqa: E501
    run_ssh_command_with_credentials
)

from cai.tools.reconnaissance.generic_linux_command import (  # pylint: disable=import-error # noqa: E501
    generic_linux_command
)
from cai.tools.web.google_search import (  # pylint: disable=import-error # noqa: E501
    google_search, google_dork_search
)

from cai.tools.reconnaissance.exec_code import (  # pylint: disable=import-error # noqa: E501
    execute_code
)

from cai.tools.reconnaissance.shodan import (  # pylint: disable=import-error # noqa: E501
    shodan_search,
    shodan_host_info
)

load_dotenv()
# Prompts
bug_bounter_system_prompt = load_prompt_template("prompts/system_bug_bounter.md")
# Define tools list based on available API keys
tools = [
    generic_linux_command,
    execute_code,
    shodan_search,
    shodan_host_info
]

if os.getenv('GOOGLE_SEARCH_API_KEY') and os.getenv('GOOGLE_SEARCH_CX'):
    tools.append(google_search)
    tools.append(google_dork_search)

bug_bounter_agent = Agent(
    name="Bug Bounter",
    instructions=create_system_prompt_renderer(bug_bounter_system_prompt),
    description="""Agent that specializes in bug bounty hunting and vulnerability discovery.
                   Expert in web security, API testing, and responsible disclosure.""",
    tools=tools,
    model=OpenAIChatCompletionsModel(
        model=os.getenv('CAI_MODEL', "alias0"),
        openai_client=AsyncOpenAI(),
    )
   
)