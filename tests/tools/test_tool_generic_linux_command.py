import os
import pytest
import json
from unittest.mock import MagicMock

# Set test environment variables to avoid OpenAI client initialization errors
os.environ["OPENAI_API_KEY"] = "test_key_for_ci_environment"

from cai.tools.reconnaissance.generic_linux_command import generic_linux_command  

async def test_generic_linux_command_regular_commands():
    """Test the execution of a regular command using the generic Linux command tool."""
    mock_ctx = MagicMock()  # Create a mock context for the command execution
    params = {
        "command": "echo",  # Command to be executed
        "args": "'hello'"   # Arguments for the command
    }

    # Invoke the tool with the specified parameters and await the result
    result = await generic_linux_command.on_invoke_tool(mock_ctx, json.dumps(params))

    # Assert that the result matches the expected output
    assert result.replace("\n", "") == 'hello'  

async def test_generic_linux_command_ls():
    """Test the execution of the 'ls' command using the generic Linux command tool."""
    mock_ctx = MagicMock()  # Create a mock context for the command execution
    params = {
        "command": "ls",  # Command to be executed
        "args": "-l"      # Arguments for the command
    }

    # Invoke the tool with the specified parameters and await the result
    result = await generic_linux_command.on_invoke_tool(mock_ctx, json.dumps(params))

    # Assert that the output contains 'total', which is typical for 'ls -l'
    assert "total" in result  

async def test_generic_linux_command_invalid_command():
    """Test the handling of an invalid command using the generic Linux command tool."""
    mock_ctx = MagicMock()  # Create a mock context for the command execution
    params = {
        "command": "invalid_command",  # Invalid command to be executed
        "args": ""                      # No arguments for the command
    }

    # Invoke the tool with the specified parameters and await the result
    result = await generic_linux_command.on_invoke_tool(mock_ctx, json.dumps(params))

    # Assert that the result indicates the command was not found
    assert "not found" in result 