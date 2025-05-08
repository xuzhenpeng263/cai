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
        CAI_STREAM: Enable/disable streaming output in rich panel
            (default: "true")

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
import sys
import time
from dotenv import load_dotenv
from openai import AsyncOpenAI
from cai.sdk.agents import OpenAIChatCompletionsModel, Agent, Runner, AsyncOpenAI
from cai.sdk.agents import set_default_openai_client, set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent
from rich.console import Console
import asyncio
from cai.util import fix_litellm_transcription_annotations, color, calculate_model_cost
from cai.util import create_agent_streaming_context, update_agent_streaming_content, finish_agent_streaming
from cai.sdk.agents.run_to_jsonl import get_session_recorder
from cai.util import start_idle_timer, stop_idle_timer, start_active_timer, stop_active_timer

# Import modules from cai.repl
from cai.repl.commands import FuzzyCommandCompleter, handle_command as commands_handle_command
from cai.repl.ui.keybindings import create_key_bindings
from cai.repl.ui.logging import setup_session_logging
from cai.repl.ui.banner import display_banner, display_quick_guide
from cai.repl.ui.prompt import get_user_input
from cai.repl.ui.toolbar import get_toolbar_with_refresh

# Import agents-related functions
from cai.agents import get_agent_by_name

# Import global message history from the OpenAI chat completions model
# to preserve conversation context between turns.
from cai.sdk.agents.models.openai_chatcompletions import (
    message_history,
    add_to_message_history,
)
from cai.sdk.agents.items import ToolCallOutputItem
from cai.sdk.agents.stream_events import RunItemStreamEvent

# Load environment variables from .env file
load_dotenv()

# NOTE: This is needed when using LiteLLM Proxy Server
#
# external_client = AsyncOpenAI(
#     base_url = os.getenv('LITELLM_BASE_URL', 'http://localhost:4000'),
#     api_key=os.getenv('LITELLM_API_KEY', 'key'))
#
# set_default_openai_client(external_client)

# Global variables for timing tracking
global START_TIME
START_TIME = time.time() 

set_tracing_disabled(True)

# llm_model=os.getenv('LLM_MODEL', 'gpt-4o-mini')
# # llm_model=os.getenv('LLM_MODEL', 'claude-3-7')
llm_model=os.getenv('LLM_MODEL', 'qwen2.5:14b')


# For Qwen models, we need to skip system instructions as they're not supported
instructions = None if "qwen" in llm_model.lower() else "You are a helpful assistant"

agent = Agent(
    name="Assistant", 
    instructions=instructions,
    model=OpenAIChatCompletionsModel(
        model=llm_model,
        openai_client=AsyncOpenAI()  # original OpenAI servers
        # openai_client = external_client  # LiteLLM Proxy Server
    )
)

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
    ACTIVE_TIME = 0
    idle_time = 0
    console = Console()
    last_model = os.getenv('CAI_MODEL', 'qwen2.5:14b')
    last_agent_type = os.getenv('CAI_AGENT_TYPE', 'one_tool_agent') 
    # Initialize command completer and key bindings
    command_completer = FuzzyCommandCompleter()
    current_text = ['']
    kb = create_key_bindings(current_text)

    # Setup session logging
    history_file = setup_session_logging()
    
    # Initialize session logger and display the filename
    session_logger = get_session_recorder()
    # Display banner
    display_banner(console)
    display_quick_guide(console)
    # Function to get the short name of the agent for display
    def get_agent_short_name(agent):
        if hasattr(agent, 'name'):
            # Return the full agent name instead of just the first word
            return agent.name
        return "Agent"
    
    # Prevent the model from using its own rich streaming to avoid conflicts
    # and suppress final output message to avoid duplicates
    if hasattr(agent, 'model'):
        if hasattr(agent.model, 'disable_rich_streaming'):
            agent.model.disable_rich_streaming = False  # Now True as the model handles streaming
        if hasattr(agent.model, 'suppress_final_output'):
            agent.model.suppress_final_output = True
            
        # Set the agent name in the model for proper display in streaming panel
        if hasattr(agent.model, 'set_agent_name'):
            agent.model.set_agent_name(get_agent_short_name(agent))

    while turn_count < max_turns:
        try:
            # Start measuring user idle time
            start_idle_timer()
            
            idle_start_time = time.time()
            
            # Check if model has changed and update if needed
            current_model = os.getenv('CAI_MODEL', 'qwen2.5:14b')
            if current_model != last_model and hasattr(agent, 'model'):
                # Update the model in the agent
                if hasattr(agent.model, 'model'):
                    agent.model.model = current_model
                    last_model = current_model
            
            # Check if agent type has changed and recreate agent if needed
            current_agent_type = os.getenv('CAI_AGENT_TYPE', 'one_tool_agent')
            if current_agent_type != last_agent_type:
                try:
                    # Import is already at the top level
                    agent = get_agent_by_name(current_agent_type)
                    last_agent_type = current_agent_type
                    
                    # Configure the new agent's model flags
                    if hasattr(agent, 'model'):
                        if hasattr(agent.model, 'disable_rich_streaming'):
                            agent.model.disable_rich_streaming = False  # Now False to let model handle streaming
                        if hasattr(agent.model, 'suppress_final_output'):
                            agent.model.suppress_final_output = True
                            
                        # Apply current model to the new agent
                        if hasattr(agent.model, 'model'):
                            agent.model.model = current_model
                            
                        # Set agent name in the model for streaming display
                        if hasattr(agent.model, 'set_agent_name'):
                            agent.model.set_agent_name(get_agent_short_name(agent))
                except Exception as e:
                    console.print(f"[red]Error switching agent: {str(e)}[/red]")

            # Get user input with command completion and history
            user_input = get_user_input(
                command_completer,
                kb,
                history_file,
                get_toolbar_with_refresh,
                current_text
            )
            idle_time += time.time() - idle_start_time
            
            # Stop measuring user idle time and start measuring active time
            stop_idle_timer()
            start_active_timer()
           
        except KeyboardInterrupt:
            def format_time(seconds):
                mins, secs = divmod(int(seconds), 60)
                hours, mins = divmod(mins, 60)
                return f"{hours:02d}:{mins:02d}:{secs:02d}"

            Total = time.time() - START_TIME
            idle_time += time.time() - idle_start_time
            try:
                # Get more accurate active and idle time measurements from the timer functions
                from cai.util import get_active_time_seconds, get_idle_time_seconds, COST_TRACKER
                
                # Use the precise measurements from our timers
                active_time_seconds = get_active_time_seconds()
                idle_time_seconds = get_idle_time_seconds()
                
                # Format for display
                active_time_formatted = format_time(active_time_seconds)
                idle_time_formatted = format_time(idle_time_seconds)

                # Get session cost from the global cost tracker
                session_cost = COST_TRACKER.session_total_cost

                metrics = {
                    "session_time": format_time(Total),
                    "active_time": active_time_formatted,
                    "idle_time": idle_time_formatted,
                    "llm_time": format_time(active_time_seconds),  # Using active time as LLM time
                    "llm_percentage": round((active_time_seconds / Total) * 100, 1) if Total > 0 else 0.0,
                    "session_cost": f"${session_cost:.6f}"  # Add formatted session cost
                }
                logging_path = session_logger.filename if hasattr(session_logger, 'filename') else None

                content = []
                content.append(f"Session Time: {metrics['session_time']}")
                content.append(f"Active Time: {metrics['active_time']} ({metrics['llm_percentage']}%)")
                content.append(f"Idle Time: {metrics['idle_time']}")
                content.append(f"Total Session Cost: {metrics['session_cost']}")  # Add cost to display
                if logging_path:
                    content.append(f"Log available at: {logging_path}")
                
                def print_session_summary(console, metrics, logging_path=None):
                    """
                    Print a session summary panel using Rich.
                    """
                    from rich.panel import Panel
                    from rich.text import Text
                    from rich.box import ROUNDED
                    from rich.console import Group

                    # Create Rich Text objects for each line
                    text_content = []
                    for i, line in enumerate(content):
                        if "Total Session Cost" in line:
                            # Format cost line with special styling
                            cost_text = Text()
                            parts = line.split(":")
                            cost_text.append(parts[0] + ":", style="bold")
                            cost_text.append(parts[1], style="bold green")
                            text_content.append(cost_text)
                        else:
                            text_content.append(Text(line))

                    time_panel = Panel(
                        Group(*text_content),
                        border_style="blue",
                        box=ROUNDED,
                        padding=(0, 1),
                        title="[bold]Session Summary[/bold]",
                        title_align="left"
                    )
                    console.print(time_panel)

                print_session_summary(console, metrics, logging_path)
                
                # Log session end
                if session_logger:
                    session_logger.log_session_end()
                    
                # Prevent duplicate cost display from the COST_TRACKER exit handler
                os.environ["CAI_COST_DISPLAYED"] = "true"
                    
            except Exception:
                pass
            break
      
        try:
            # Handle special commands
            if user_input.startswith('/') or user_input.startswith('$'):
                parts = user_input.strip().split()
                command = parts[0]
                args = parts[1:] if len(parts) > 1 else None

                # Process the command with the handler
                if commands_handle_command(command, args):
                    continue  # Command was handled, continue to next iteration

                # If command wasn't recognized, show error (skip for /shell or /s)
                if command not in ("/shell", "/s"):
                    console.print(f"[red]Unknown command: {command}[/red]")
                continue
            from rich.text import Text
            log_text = Text(
                f"Log file: {session_logger.filename}",
                style="yellow on black",
            )
            console.print(log_text)

            # Build conversation context from previous turns to give the
            # model short-term memory. We only keep messages that have plain
            # text content and ignore tool call entries to prevent schema
            # mismatches when converting to OpenAI chat format.
            history_context = []
            for msg in message_history:
                role = msg.get("role")
                content = msg.get("content")
                tool_calls = msg.get("tool_calls")

                if role == "user":
                    history_context.append({"role": "user", "content": content or ""})
                elif role == "system":
                    history_context.append({"role": "system", "content": content or ""})
                elif role == "assistant":
                    if tool_calls:
                        history_context.append(
                            {
                                "role": "assistant",
                                "content": content,  # Can be None
                                "tool_calls": tool_calls,
                            }
                        )
                    elif content is not None:
                        history_context.append({"role": "assistant", "content": content})
                    elif content is None and not tool_calls: # Explicitly handle empty assistant message
                         history_context.append({"role": "assistant", "content": None})
                elif role == "tool":
                    history_context.append(
                        {
                            "role": "tool",
                            "tool_call_id": msg.get("tool_call_id"),
                            "content": msg.get("content"), # Tool output
                        }
                    )

            # Append the current user input as the last message in the list.
            conversation_input: list | str
            if history_context:
                history_context.append({"role": "user", "content": user_input})
                conversation_input = history_context
            else:
                conversation_input = user_input

            # Process the conversation with the agent.
            if stream:
                async def process_streamed_response():
                    try:
                        result = Runner.run_streamed(agent, conversation_input)

                        # Consume events so the async generator is executed.
                        async for event in result.stream_events():
                            if isinstance(event, RunItemStreamEvent) and event.name == "tool_output":
                                # Ensure item is a ToolCallOutputItem before accessing attributes
                                if isinstance(event.item, ToolCallOutputItem):
                                    tool_msg = {
                                        "role": "tool",
                                        "tool_call_id": event.item.raw_item["call_id"], # Changed to dictionary access
                                        "content": event.item.output,
                                    }
                                    add_to_message_history(tool_msg)
                            # pass # Original logic was just pass

                        return result
                    except Exception as e:
                        import traceback
                        tb = traceback.format_exc()
                        print(
                            f"\n[Error occurred during streaming: {str(e)}]"
                            f"\nLocation: {tb}"
                        )
                        return None

                asyncio.run(process_streamed_response())
            else:
                # Use non-streamed response
                response = asyncio.run(Runner.run(agent, conversation_input))
                for item in response.new_items:
                    if isinstance(item, ToolCallOutputItem):
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": item.raw_item["call_id"],  # Usar formato consistente con streaming
                            "content": item.output,
                        }
                        add_to_message_history(tool_msg)
            turn_count += 1
            
            # Stop measuring active time and start measuring idle time again
            stop_active_timer()
            start_idle_timer()

        except KeyboardInterrupt:
            # No need to clean up streaming context as model handles it
            pass
        except Exception as e:
            import traceback
            import sys
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_info = traceback.extract_tb(exc_traceback)
            filename, line, func, text = tb_info[-1]
            console.print(f"[bold red]Error: {str(e)}[/bold red]")
            console.print(f"[bold red]Traceback: {tb_info}[/bold red]")
            
            # Make sure we switch back to idle mode even if there's an error
            stop_active_timer()
            start_idle_timer()

def main():
    # Apply litellm patch to fix the __annotations__ error
    patch_applied = fix_litellm_transcription_annotations()
    if not patch_applied:
        print(color("Something went wrong patching LiteLLM fix_litellm_transcription_annotations", color="red"))

    # Get agent type from environment variables or use default
    agent_type = os.getenv('CAI_AGENT_TYPE', "one_tool_agent")

    # Get the agent instance by name
    agent = get_agent_by_name(agent_type)
    
    # Configure model flags to work well with CLI
    if hasattr(agent, 'model'):
        # Disable rich streaming in the model to avoid conflicts
        if hasattr(agent.model, 'disable_rich_streaming'):
            agent.model.disable_rich_streaming = True
        # Suppress final output to avoid duplicates
        if hasattr(agent.model, 'suppress_final_output'):
            agent.model.suppress_final_output = True

    # Enable streaming by default, unless specifically disabled
    stream = os.getenv('CAI_STREAM', 'true').lower() != 'false'

    # Run the CLI with the selected agent
    run_cai_cli(agent, stream=stream)

if __name__ == "__main__":
    main()