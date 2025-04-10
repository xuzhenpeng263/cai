import pytest
from tests.fake_model import FakeModel
from tests.test_responses import (
    get_text_message,
    get_function_tool_call,
    get_function_tool,
)
from cai.sdk.agents import Runner
from cai.agents.one_tool import transfer_to_one_tool_agent  
from cai.agents.one_tool import one_tool_agent  

@pytest.mark.asyncio
async def test_ctf_agent_instructions_and_configuration():
    agent = transfer_to_one_tool_agent()
    
    # Check that the agent has the generic_linux_command tool
    assert any(tool.name== "generic_linux_command" for tool in agent.tools)

    # Optionally, you can check the agent's instructions and configuration
    assert agent.instructions is not None
    assert agent.name == "CTF agent"

@pytest.mark.asyncio
async def test_ctf_agent_executes_linux_command():
    model = FakeModel()
    agent = transfer_to_one_tool_agent()
    agent.model = model
    model.add_multiple_turn_outputs(
        [
            [
                get_text_message("executing comando..."),
                get_function_tool_call("generic_linux_command", '{"command": "ls"}')
            ],
            [
                get_text_message("result of the command: flag{12345}")
            ]
        ]
    )

    result = await Runner.run(agent, input="List files")

    assert result.final_output == "result of the command: flag{12345}"
    assert len(result.raw_responses) == 2

    assert any("generic_linux_command" in str(item) for item in result.to_input_list())

