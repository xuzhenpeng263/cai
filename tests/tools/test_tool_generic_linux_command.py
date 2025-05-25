import os
import pytest
import json
from unittest.mock import MagicMock

# Set test environment variables to avoid OpenAI client initialization errors
os.environ["OPENAI_API_KEY"] = "test_key_for_ci_environment"

from cai.tools.reconnaissance.generic_linux_command import generic_linux_command


def test_generic_linux_command_echo():
    """Test the execution of echo command using generic_linux_command."""
    result = generic_linux_command(command="echo 'hello'")
    assert result.strip() == 'hello'


def test_generic_linux_command_ls():
    """Test the execution of ls command using generic_linux_command."""
    result = generic_linux_command(command="ls -l")
    # Check that the output contains typical ls -l indicators
    assert "total" in result or "drwx" in result or "-rw" in result


def test_generic_linux_command_invalid_command():
    """Test handling of invalid command using generic_linux_command."""
    result = generic_linux_command(command="invalid_command_xyz123")
    # Check for common error indicators
    assert ("not found" in result.lower() or 
            "command not found" in result.lower() or
            "no such file" in result.lower())


def test_generic_linux_command_empty_command():
    """Test handling of empty command using generic_linux_command."""
    result = generic_linux_command(command="")
    assert "Error: No command provided" in result


def test_generic_linux_command_session_list():
    """Test session list functionality using generic_linux_command."""
    result = generic_linux_command(command="session list")
    assert "No active sessions" in result or "Active sessions:" in result


def test_generic_linux_command_env_info():
    """Test environment info functionality using generic_linux_command."""
    result = generic_linux_command(command="env info")
    assert "Current Environment:" in result
    assert "CTF Environment:" in result
    assert "Container:" in result
    assert "SSH:" in result
    assert "Workspace:" in result


def test_generic_linux_command_interactive_flag():
    """Test interactive flag functionality using generic_linux_command."""
    # Test with interactive=True but a simple command
    result = generic_linux_command(command="echo 'test'", interactive=True)
    # Should still work, just might have different session handling
    assert "test" in result


def test_generic_linux_command_with_session_id():
    """Test session_id parameter using generic_linux_command."""
    # Test with a non-existent session_id
    result = generic_linux_command(command="echo 'test'", 
                                   session_id="nonexistent123")
    # Should handle gracefully - either execute or give session error
    assert isinstance(result, str)