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
    Verifies that Runner.run returns the final output containing the expected text.
    """
    prompt = "monitor login attempts for suspicious activity"
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
    prompt = "find vulnerabilities in web application sample.com"
    result = await Runner.run(get_agent_by_name("bug_bounter_agent"), prompt)
    final_output = result.final_output or ""
    assert final_output, "Expected non-empty final output"
    assert "sample.com" in final_output.lower(), f"Expected 'sample.com' in output, got: {final_output}"

@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_flag_discriminator_inference():
    """
    Non-streaming inference test for the flag_discriminator.
    """
    prompt = "is 'flag{test_flag}' a valid CTF flag?"
    result = await Runner.run(get_agent_by_name("flag_discriminator"), prompt)
    final_output = result.final_output or ""
    assert final_output, "Expected non-empty final output"
    assert "valid" in final_output.lower(), f"Expected 'valid' in output, got: {final_output}"

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
    Verifies that Runner.run returns the final output containing the expected text.
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
    assert "ssh" in final_output.lower(), f"Expected 'ssh' in output, got: {final_output}"

@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_thought_agent_inference():
    """
    Non-streaming inference test for the thought_agent.
    """
    prompt = "what are the next steps for network security assessment?"
    result = await Runner.run(get_agent_by_name("thought_agent"), prompt)
    final_output = result.final_output or ""
    assert final_output, "Expected non-empty final output"
    assert (
        "plan" in final_output.lower() or "analysis" in final_output.lower()
    ), f"Expected 'plan' or 'analysis' in output, got: {final_output}"