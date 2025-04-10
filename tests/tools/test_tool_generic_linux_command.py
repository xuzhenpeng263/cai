import pytest
import json
import asyncio
from unittest.mock import MagicMock
from unittest.mock import patch
from cai.tools.reconnaissance.generic_linux_command import generic_linux_command  


async def test_generic_linux_command_regular_commands():
    mock_ctx = MagicMock()  
    params = {
        "command": "echo",
        "args": "'hello'"
    }

    result = await generic_linux_command.on_invoke_tool(mock_ctx, json.dumps(params))

    assert result.replace("\n", "") == 'hello'  
