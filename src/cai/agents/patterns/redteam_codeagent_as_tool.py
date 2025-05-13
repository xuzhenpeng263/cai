"""
Implementation of a Red Team Agent with Code Execution Agent as Tool Pattern

This module establishes a specialized pattern where a Code Execution Agent
serves as a tool for the Red Team Agent. This allows the Red Team Agent to
delegate code execution tasks to a dedicated agent, enhancing security
assessment capabilities through specialized code execution.
"""
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

from cai.sdk.agents import Agent, OpenAIChatCompletionsModel
from cai.agents.red_teamer import redteam_agent
from cai.tools.reconnaissance.exec_code import execute_code
from cai.util import load_prompt_template


# Load environment variables
load_dotenv()
model_name = os.getenv("CAI_MODEL", "qwen2.5:14b")

# Create a specialized code execution agent with only the execute_code tool
code_execution_agent = Agent(
    name="Code Execution Agent",
    description="Specialized agent for executing code in security assessments",
    instructions=(
        "You are a specialized code execution agent that helps with security "
        "assessments. Execute the task and then transfer to red team"
    ),
    tools=[execute_code],
    handoffs=[],
    model=OpenAIChatCompletionsModel(
        model=model_name,
        openai_client=AsyncOpenAI(),
    ),
)

# Create a handoff for the red team agent
from cai.sdk.agents import handoff

redteam_handoff = handoff(
    agent=redteam_agent,
    tool_description_override="Transfer to Red Team Agent for security assessment and exploitation tasks"
)

# Add the handoff to the code execution agent
code_execution_agent.handoffs.append(redteam_handoff)

# Register the code execution agent as a tool for the red team agent
redteam_agent.tools.append(
    code_execution_agent.as_tool(
        tool_name="code_execution",
        tool_description=(
            "Use this tool when you need to execute code"
        ),
    )
)
if execute_code in redteam_agent.tools:
    redteam_agent.tools.remove(execute_code)
# Export the enhanced red team agent as the pattern
redteam_with_code_agent_pattern = redteam_agent
redteam_with_code_agent_pattern.pattern = "agent_as_tool"
