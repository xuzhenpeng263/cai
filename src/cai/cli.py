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

# Import modules from cai.repl
from cai.repl.commands import FuzzyCommandCompleter, handle_command as commands_handle_command
from cai.repl.ui.keybindings import create_key_bindings
from cai.repl.ui.logging import setup_session_logging
from cai.repl.ui.banner import display_banner, display_quick_guide
from cai.repl.ui.prompt import get_user_input
from cai.repl.ui.toolbar import get_toolbar_with_refresh

# Import agents-related functions
from cai.agents import get_agent_by_name

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

    # Initialize command completer and key bindings
    command_completer = FuzzyCommandCompleter()
    current_text = ['']
    kb = create_key_bindings(current_text)

    # Setup session logging
    history_file = setup_session_logging()

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
            agent.model.disable_rich_streaming = True
        if hasattr(agent.model, 'suppress_final_output'):
            agent.model.suppress_final_output = True

    # Track streaming context to ensure proper cleanup
    current_streaming_context = None

    while turn_count < max_turns:
        try:
            idle_start_time = time.time()
            # Get user input with command completion and history
            user_input = get_user_input(
                command_completer,
                kb,
                history_file,
                get_toolbar_with_refresh,
                current_text
            )
            idle_time += time.time() - idle_start_time
           
        except KeyboardInterrupt:
            def format_time(seconds):
                mins, secs = divmod(int(seconds), 60)
                hours, mins = divmod(mins, 60)
                return f"{hours:02d}:{mins:02d}:{secs:02d}"

            Total = time.time() - START_TIME
            idle_time += time.time() - idle_start_time
            try:
                active_time = Total - idle_time

                metrics = {
                    "session_time": format_time(Total),
                    "active_time": format_time(active_time),
                    "idle_time": format_time(idle_time),
                    "llm_time": "0.0s",  # Placeholder, update if available
                    "llm_percentage": 0.0,  # Placeholder, update if available
                }
                logging_path = None  # Set this if you have a log file path

                content = []
                content.append(f"Session Time: {metrics['session_time']}")
                content.append(f"Active Time: {metrics['active_time']}")
                content.append(f"Idle Time: {metrics['idle_time']}")
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

                    time_panel = Panel(
                        Group(*[Text(line) for line in content]),
                        border_style="blue",
                        box=ROUNDED,
                        padding=(0, 1),
                        title="[bold]Session Summary[/bold]",
                        title_align="left"
                    )
                    console.print(time_panel)

                print_session_summary(console, metrics, logging_path)
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

                # If command wasn't recognized, show error
                console.print(f"[red]Unknown command: {command}[/red]")
                continue

            # Process the conversation with the agent
            if stream:
                # For classic fallback when streaming fails
                print_fallback = False
                
                async def process_streamed_response():
                    nonlocal current_streaming_context, print_fallback
                    
                    try:
                        # Get the model from the agent for display purposes
                        model_name = None
                        if hasattr(agent, 'model') and hasattr(agent.model, 'model'):
                            model_name = str(agent.model.model)
                        
                        # Set the agent name in the model if available (for proper display in streaming panel)
                        if hasattr(agent, 'model'):
                            agent.model.agent_name = get_agent_short_name(agent)
                        
                        # Make sure any previous streaming context is cleaned up
                        if current_streaming_context is not None:
                            try:
                                current_streaming_context["live"].stop()
                            except Exception:
                                pass  # Ignore errors on cleanup
                            current_streaming_context = None
                        
                        try:
                            # Create a new streaming context
                            current_streaming_context = create_agent_streaming_context(
                                agent_name=get_agent_short_name(agent),
                                counter=turn_count + 1,  # 1-indexed for display
                                model=model_name
                            )
                        except Exception as e:
                            # If rich display fails, fall back to classic print mode
                            print(f"Agent: ", end="", flush=True)
                            print_fallback = True
                            import traceback
                            print(f"[Warning: Falling back to simple streaming: {str(e)}]", file=sys.stderr)
                        
                        # Run the agent with streaming
                        result = Runner.run_streamed(agent, user_input)
                        
                        # List to collect all deltas for computing final token counts
                        collected_text = []
                        
                        # Process stream events
                        async for event in result.stream_events():
                            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                                collected_text.append(event.data.delta)
                                # If using streaming context, update the panel
                                if current_streaming_context is not None:
                                    update_agent_streaming_content(current_streaming_context, event.data.delta)
                                # Otherwise, print to console directly
                                elif print_fallback:
                                    print(event.data.delta, end="", flush=True)
                        
                        # Finish the streaming context if it exists
                        if current_streaming_context is not None:
                            # Get token stats for the final display
                            token_stats = None
                            
                            # Try to get token stats from the model
                            if hasattr(agent, 'model'):
                                # Get the actual input/output token counts from the model when available
                                model = agent.model
                                
                                # Calculate a more accurate output token estimate using tiktoken if available
                                output_text = "".join(collected_text)
                                output_tokens = len(output_text) // 4  # Fallback rough estimate
                                
                                try:
                                    import tiktoken
                                    encoding = tiktoken.get_encoding("cl100k_base")
                                    output_tokens = len(encoding.encode(output_text))
                                except Exception:
                                    # Fallback to rough estimate if tiktoken fails
                                    pass
                                
                                # Store current input tokens to calculate difference next time
                                if not hasattr(model, 'previous_input_tokens'):
                                    model.previous_input_tokens = 0
                                
                                # Get the available token counts from the model, or use reasonable defaults
                                interaction_input = getattr(model, 'total_input_tokens', 0) - model.previous_input_tokens
                                if interaction_input <= 0:
                                    interaction_input = output_tokens * 2  # Rough estimate based on output
                                
                                # Update previous tokens for next calculation
                                model.previous_input_tokens = getattr(model, 'total_input_tokens', 0)
                                
                                token_stats = {
                                    "interaction_input_tokens": interaction_input,
                                    "interaction_output_tokens": output_tokens,
                                    "interaction_reasoning_tokens": 0,
                                    "total_input_tokens": getattr(model, 'total_input_tokens', interaction_input),
                                    "total_output_tokens": getattr(model, 'total_output_tokens', output_tokens),
                                    "total_reasoning_tokens": getattr(model, 'total_reasoning_tokens', 0),
                                    "interaction_cost": calculate_model_cost(str(model), interaction_input, output_tokens),
                                    "total_cost": calculate_model_cost(str(model), getattr(model, 'total_input_tokens', interaction_input), getattr(model, 'total_output_tokens', output_tokens))
                                }
                            
                            finish_agent_streaming(current_streaming_context, token_stats)
                            current_streaming_context = None
                        elif print_fallback:
                            # Add a newline at the end of classic streaming
                            print("\n")
                            
                        return result
                    except Exception as e:
                        # In case of errors, ensure streaming context is cleaned up
                        if current_streaming_context is not None:
                            try:
                                current_streaming_context["live"].stop()
                            except Exception:
                                pass
                            current_streaming_context = None
                        
                        if print_fallback:
                            print()  # Add a newline after any partial output
                            
                        import traceback
                        tb = traceback.format_exc()
                        print(f"\n[Error occurred during streaming: {str(e)}]\nLocation: {tb}")
                        return None

                asyncio.run(process_streamed_response())
            else:
                # Use non-streamed response
                response = asyncio.run(Runner.run(agent, user_input))
                #console.print(f"Agent: {response.final_output}") # NOTE: this line is commented to avoid duplicate output
            turn_count += 1

        except KeyboardInterrupt:
            if stream:
                # Ensure streaming context is cleaned up on keyboard interrupt
                if current_streaming_context is not None:
                    try:
                        current_streaming_context["live"].stop()
                    except Exception:
                        pass
                    current_streaming_context = None
        except Exception as e:
            # Ensure streaming context is cleaned up on any exception
            if current_streaming_context is not None:
                try:
                    current_streaming_context["live"].stop()
                except Exception:
                    pass
                current_streaming_context = None
            
            import traceback
            import sys
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_info = traceback.extract_tb(exc_traceback)
            filename, line, func, text = tb_info[-1]
            console.print(f"[bold red]Error: {str(e)}[/bold red]")
            console.print(f"[bold red]Traceback: {tb_info}[/bold red]")

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