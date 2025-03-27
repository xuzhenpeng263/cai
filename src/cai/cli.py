"""
This module provides a CLI interface for testing and
interacting with CAI agents.

Environment Variables
---------------------
    Required:
        N/A

    Optional:
        CTF_NAME: Name of the CTF challenge to
            run (e.g. "picoctf_static_flag")
        CTF_CHALLENGE: Specific challenge name
            within the CTF to test
        CTF_SUBNET: Network subnet for the CTF
            container (default: "192.168.2.0/24")
        CTF_IP: IP address for the CTF
            container (default: "192.168.2.100")
        CTF_INSIDE: Whether to conquer the CTF from
            within container (default: "true")

        CAI_MODEL: Model to use for agents
            (default: "qwen2.5:14b")
        CAI_DEBUG: Set debug output level (default: "1")
            - 0: Only tool outputs
            - 1: Verbose debug output
            - 2: CLI debug output
        CAI_BRIEF: Enable/disable brief output mode (default: "false")
        CAI_MAX_TURNS: Maximum number of turns for
            agent interactions (default: "inf")
        CAI_TRACING: Enable/disable OpenTelemetry tracing
            (default: "true"). When enabled, traces execution
            flow and agent interactions for debugging and analysis.
        CAI_AGENT_TYPE: Specify the agents to use it could take
            the value of (default: "one_tool_agent"). Use "/agent"
            command in CLI to list all available agents.
        CAI_STATE: Enable/disable stateful mode (default: "false").
            When enabled, the agent will use a state agent to keep
            track of the state of the network and the flags found.
        CAI_MEMORY: Enable/disable memory mode (default: "false")
            - episodic: use episodic memory
            - semantic: use semantic memory
            - all: use both episodic and semantic memorys
        CAI_MEMORY_ONLINE: Enable/disable online memory mode
            (default: "false")
        CAI_MEMORY_OFFLINE: Enable/disable offline memory
            (default: "false")
        CAI_ENV_CONTEXT: Add enviroment context, dirs and
            current env available (default: "true")
        CAI_MEMORY_ONLINE_INTERVAL: Number of turns between
            online memory updates (default: "5")
        CAI_PRICE_LIMIT: Price limit for the conversation in dollars
            (default: "1")
        CAI_SUPPORT_MODEL: Model to use for the support agent
            (default: "o3-mini")
        CAI_SUPPORT_INTERVAL: Number of turns between support agent
            executions (default: "5")

    Extensions (only applicable if the right extension is installed):

        "report"
            CAI_REPORT: Enable/disable reporter mode. Possible values:
                - ctf (default): do a report from a ctf resolution
                - nis2: do a report for nis2
                - pentesting: do a report from a pentesting

Usage Examples:

    # Run against a CTF
    CTF_NAME="kiddoctf" CTF_CHALLENGE="02 linux ii" \
        CAI_AGENT_TYPE="one_tool_agent" CAI_MODEL="qwen2.5:14b" \
        CAI_TRACING="false" python3 cai/cli.py

    # Run a harder CTF
    CTF_NAME="hackableii" CAI_AGENT_TYPE="redteam_agent" \
        CTF_INSIDE="False" CAI_MODEL="deepseek/deepseek-chat" \
        CAI_TRACING="false" python3 cai/cli.py

    # Run without a target in human-in-the-loop mode, generating a report
    CAI_TRACING=False CAI_REPORT=pentesting CAI_MODEL="gpt-4o" \
        python3 cai/cli.py

    # Run with online episodic memory
    #   registers memory every 5 turns:
    #   limits the cost to 5 dollars
    CTF_NAME="hackableII" CAI_MEMORY="episodic" \
        CAI_MODEL="o3-mini" CAI_MEMORY_ONLINE="True" \
        CTF_INSIDE="False" CTF_HINTS="False"  \
        CAI_PRICE_LIMIT="5" python3 cai/cli.py

    # Run with custom long_term_memory interval
    # Executes memory long_term_memory every 3 turns:
    CTF_NAME="hackableII" CAI_MEMORY="episodic" \
        CAI_MODEL="o3-mini" CAI_MEMORY_ONLINE_INTERVAL="3" \
        CAI_MEMORY_ONLINE="False" CTF_INSIDE="False" \
        CTF_HINTS="False" python3 cai/cli.py
"""

import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from cai.sdk.agents import OpenAIChatCompletionsModel, Agent, Runner
from cai.sdk.agents import set_default_openai_client, set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent
from rich.console import Console
import asyncio

# Import modules from cai.repl
from cai.repl.commands import FuzzyCommandCompleter, handle_command as commands_handle_command
from cai.repl.ui.keybindings import create_key_bindings
from cai.repl.ui.logging import setup_session_logging
from cai.repl.ui.banner import display_banner
from cai.repl.ui.prompt import get_user_input
from cai.repl.ui.toolbar import get_toolbar_with_refresh

# Import agents-related functions
from cai.agents import get_agent_by_name

# Load environment variables from .env file
load_dotenv()

external_client = AsyncOpenAI(
    base_url = os.getenv('LITELLM_BASE_URL', 'http://localhost:4000'),
    api_key=os.getenv('LITELLM_API_KEY', 'key'))

set_default_openai_client(external_client)
set_tracing_disabled(True)

# # llm_model=os.getenv('LLM_MODEL', 'gpt-4o-mini')
# # llm_model=os.getenv('LLM_MODEL', 'claude-3-7')
# llm_model=os.getenv('LLM_MODEL', 'qwen2.5:14b')


# # For Qwen models, we need to skip system instructions as they're not supported
# instructions = None if "qwen" in llm_model.lower() else "You are a helpful assistant"

# agent = Agent(
#     name="Assistant", 
#     instructions=instructions,
#     model=OpenAIChatCompletionsModel(
#         model=llm_model,
#         openai_client=external_client,
#     )
# )

def run_cai_cli(starting_agent, context_variables=None, stream=False, max_turns=float('inf')):
    """
    Run a simple interactive CLI loop for CAI.

    Args:
        starting_agent: The initial agent to use for the conversation
        context_variables: Optional dictionary of context variables to initialize the session
        stream: Boolean flag to enable/disable streaming responses (default: False)
        max_turns: Maximum number of interaction turns before terminating (default: infinity)

    Returns:
        None
    """
    agent = starting_agent
    turn_count = 0

    console = Console()

    # Initialize command completer and key bindings
    command_completer = FuzzyCommandCompleter()
    current_text = ['']
    kb = create_key_bindings(current_text)

    # Setup session logging
    history_file = setup_session_logging()

    # Display banner
    display_banner(console)

    while turn_count < max_turns:
        try:
            # Get user input with command completion and history
            user_input = get_user_input(
                command_completer,
                kb,
                history_file,
                get_toolbar_with_refresh,
                current_text
            )

            # Handle special commands
            if user_input.startswith('/') or user_input.startswith('$'):
                parts = user_input.strip().split()
                command = parts[0]
                args = parts[1:] if len(parts) > 1 else None

                # Process the command with the handler
                if commands_handle_command(command, args):
                    continue  # Command was handled, continue to next iteration

                # If command wasn't recognized, show error
                console.print(f"[red]Unknown command: {command}[/red]")
                continue

            # Process the conversation with the agent
            if stream:
                # Use streamed response
                print("Agent: ", end="", flush=True)

                async def process_streamed_response():
                    try:
                        result = Runner.run_streamed(agent, user_input)
                        async for event in result.stream_events():
                            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                                print(event.data.delta, end="", flush=True)
                        print()  # Add a newline at the end
                        return result
                    except Exception as e:
                        print()  # Add a newline after any partial output
                        import traceback
                        tb = traceback.format_exc()
                        print(f"\n[Error occurred during streaming: {str(e)}]\nLocation: {tb}")
                        return None

                asyncio.run(process_streamed_response())
            else:
                # Use non-streamed response
                console.print("[dim]Thinking...[/dim]")
                response = asyncio.run(Runner.run(agent, user_input))
                console.print(f"Agent: {response.final_output}")
            turn_count += 1
        except KeyboardInterrupt:
            break
        except Exception as e:
            import traceback
            import sys
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_info = traceback.extract_tb(exc_traceback)
            filename, line, func, text = tb_info[-1]
            console.print(f"[bold red]Error: {str(e)}[/bold red]")
            console.print(f"[bold red]Traceback: {tb_info}[/bold red]")


def main():
    # Get agent type from environment variables or use default
    agent_type = os.getenv('CAI_AGENT_TYPE', "one_tool_agent")

    llm_model=os.getenv('LLM_MODEL', 'qwen2.5:14b')
    # llm_model=os.getenv('LLM_MODEL', 'gpt-4o-mini')

    # For Qwen models, we need to skip system instructions as they're not supported
    instructions = None if "qwen" in llm_model.lower() else "You are a helpful assistant"

    agent = Agent(
        name="Assistant", 
        instructions=instructions,
        model=OpenAIChatCompletionsModel(
            model=llm_model,
            openai_client=external_client,
        )
    )

    # Get the agent instance by name
    agent = get_agent_by_name(agent_type)

    # Enable streaming by default, unless specifically disabled
    stream = os.getenv('CAI_STREAM', 'false').lower() != 'false'

    # Run the CLI with the selected agent
    run_cai_cli(agent, stream=stream)

if __name__ == "__main__":
    main()