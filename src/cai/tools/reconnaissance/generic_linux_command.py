"""
This is used to create a generic linux command.
"""
import os
import time
import uuid
import subprocess
import sys
from cai.tools.common import (run_command,
                              list_shell_sessions,
                              get_session_output,
                              terminate_session)  # pylint: disable=import-error # noqa E501
from cai.sdk.agents import function_tool
from wasabi import color  # pylint: disable=import-error


@function_tool
def generic_linux_command(command: str = "",
                          args: str = "", ctf=None,
                          async_mode: bool = False,
                          session_id: str = None) -> str:
    """
    Execute Linux commands with support for interactive sessions.

    Two modes:
    1. Regular command execution:
       - For normal Linux commands
       - Example: generic_linux_command("ls", "-la")

    2. Session management:
       - For interactive commands(ssh)
       - Workflow:
         - Start: generic_linux_command("ssh", "user@host")
           → Returns session ID
         - List: generic_linux_command("session", "list")
         - Get output: generic_linux_command("session", "output <id>")
         - Send input: Use session_id parameter
         - End: generic_linux_command("session", "kill <id>")

    Args:
        command: Command name
        args: Command arguments
        ctf: CTF environment object
        async_mode: Force async session
        session_id: Existing session ID

    Returns:
        Command output, session ID, or status message
    """
    # Special commands for session management
    if command == "session":
        if args == "list":
            sessions = list_shell_sessions()
            if not sessions:
                return "No active sessions"

            result = "Active sessions:\n"
            for session in sessions:
                result += (f"ID: {session['session_id']} | "
                           f"Command: {session['command']} | "
                           f"Last activity: {session['last_activity']}\n")
            return result

        if args.startswith("output "):
            session_id = args.split(" ")[1]
            return get_session_output(session_id)

        if args.startswith("kill "):
            session_id = args.split(" ")[1]
            return terminate_session(session_id)

        return """Unknown session command.
        Available: list, output <id>, kill <id>"""

    # Regular command execution
    full_command = f'{command} {args}'.strip()

    # Detect if this should be an async command
    if not async_mode and not session_id:
        async_commands = ['ssh', 'python -m http.server']
        async_mode = any(cmd in full_command for cmd in async_commands)

    # For SSH sessions or async commands, use different timeout
    if session_id:
        timeout = 10
    else:
        timeout = 100
        
    # Check if streaming should be enabled
    stream = os.getenv('CAI_STREAM', 'false').lower() == 'true'
    
    # Generate a call_id for streaming if needed
    # Only use call_id for streaming mode to prevent duplicate panels
    # When CAI_STREAM=true, we want the Tool Execution panel but not the Tool Output panel
    call_id = str(uuid.uuid4())[:8] if stream else None
    
    # In streaming mode, prevent displaying both panels by forcing a call_id
    # This tricks cli_print_tool_output into thinking it's being called for a streaming update
    if stream and not call_id:
        call_id = str(uuid.uuid4())[:8]

    # Run the command with the appropriate parameters - pass the actual tool name!
    result = run_command(full_command, ctf=ctf,
                       async_mode=async_mode, session_id=session_id,
                       timeout=timeout, stream=stream, call_id=call_id,
                       tool_name="generic_linux_command")
    
    # For better output formatting, if we're in streaming mode, we can modify the output
    # to include any additional information needed while still preventing the duplicate panel
    return result
