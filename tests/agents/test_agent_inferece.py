import os
import pytest
import pytest
from cai.sdk.agents import Runner
from cai.agents import get_agent_by_name
from cai.agents.bug_bounter import bug_bounter_agent
from cai.agents.flag_discriminator import flag_discriminator
from cai.agents.mail import dns_smtp_agent
from cai.agents.one_tool import one_tool_agent
from cai.agents.red_teamer import redteam_agent
from cai.agents.thought import thought_agent

@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_blue_team_agent_inference():
    """
    Non-streaming inference test for the blueteam_agent.
    """
    prompt = "Monitor login attempts for suspicious activity what we can do?"
    result = await Runner.run(get_agent_by_name("blueteam_agent"), prompt)
    final_output = result.final_output or ""
    assert final_output, "Expected non-empty final output"
    assert "login" in final_output.lower(), f"Expected 'login' in output, got: {final_output}"

@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_bug_bounter_agent_inference():
    """
    Non-streaming inference test for the bug_bounter_agent.
    """
    prompt = "Find vulnerabilities in web application sample.com"
    result = await Runner.run(get_agent_by_name("bug_bounter_agent"), prompt)
    final_output = result.final_output or ""
    assert final_output, "Expected non-empty final output"
    assert "sample.com" in final_output.lower(), f"Expected 'sample.com' in output, got: {final_output}"

@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_dns_smtp_agent_inference():
    """
    Non-streaming inference test for the dns_smtp_agent.
    """
    prompt = "check DKIM record for example.com"
    result = await Runner.run(get_agent_by_name("dns_smtp_agent"), prompt)
    final_output = result.final_output or ""
    assert final_output, "Expected non-empty final output"
    assert "dkim" in final_output.lower(), f"Expected 'dkim' in output, got: {final_output}"

@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_one_tool_agent_inference():
    """
    Non-streaming inference test for the one_tool_agent.
    """
    prompt = "use nmap to scan 192.168.2.0/24"
    # Run the agent in non-streaming mode
    result = await Runner.run(get_agent_by_name("one_tool_agent"), prompt)
    final_output = result.final_output or ""
    assert final_output, "Expected non-empty final output"
    assert "generic_linux_command" in final_output.lower(), f"Expected 'generic_linux_command' in output, got: {final_output}"
    assert "nmap" in final_output.lower(), f"Expected 'nmap' in output, got: {final_output}"

@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_red_team_agent_inference():
    """
    Non-streaming inference test for the redteam_agent.
    """
    prompt = "perform penetration test on example.com ssh service"
    result = await Runner.run(get_agent_by_name("redteam_agent"), prompt)
    final_output = result.final_output or ""
    assert final_output, "Expected non-empty final output"
    assert "example.com" in final_output.lower(), f"Expected 'example.com' in output, got: {final_output}"
