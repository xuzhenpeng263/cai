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

# Load environment variables from .env file
load_dotenv()

external_client = AsyncOpenAI(
    base_url = os.getenv('LITELLM_BASE_URL', 'http://localhost:4000'),
    api_key=os.getenv('LITELLM_API_KEY', 'key'))

set_default_openai_client(external_client)
set_tracing_disabled(True)

# llm_model=os.getenv('LLM_MODEL', 'gpt-4o-mini')
# llm_model=os.getenv('LLM_MODEL', 'claude-3-7')
llm_model=os.getenv('LLM_MODEL', 'qwen2.5:14b')


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
                        print(f"\n[Error occurred during streaming: {str(e)}]")
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
            console.print(f"[bold red]Error: {str(e)}[/bold red]")


if __name__ == "__main__":
    run_cai_cli(agent, stream=True)