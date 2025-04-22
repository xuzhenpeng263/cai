"""
Util model for CAI
"""
import os
import sys
import importlib.resources
import pathlib
from rich.console import Console
from rich.tree import Tree
from mako.template import Template  # pylint: disable=import-error
from wasabi import color
from rich.text import Text  # pylint: disable=import-error
from rich.panel import Panel  # pylint: disable=import-error
from rich.box import ROUNDED  # pylint: disable=import-error
from rich.theme import Theme  # pylint: disable=import-error
from rich.traceback import install  # pylint: disable=import-error
from rich.pretty import install as install_pretty  # pylint: disable=import-error # noqa: 501
from datetime import datetime

theme = Theme({
    # Primary colors - Material Design inspired
    "timestamp": "#00BCD4",  # Cyan 500
    "agent": "#4CAF50",      # Green 500
    "arrow": "#FFFFFF",      # White
    "content": "#ECEFF1",    # Blue Grey 50
    "tool": "#F44336",       # Red 500

    # Secondary colors
    "cost": "#009688",        # Teal 500
    "args_str": "#FFC107",  # Amber 500

    # UI elements
    "border": "#2196F3",      # Blue 500
    "border_state": "#FFD700",      # Yellow (Gold), complementary to Blue 500
    "model": "#673AB7",       # Deep Purple 500
    "dim": "#9E9E9E",         # Grey 500
    "current_token_count": "#E0E0E0",  # Grey 300 - Light grey
    "total_token_count": "#757575",    # Grey 600 - Medium grey
    "context_tokens": "#0A0A0A",       # Nearly black - Very high contrast

    # Status indicators
    "success": "#4CAF50",     # Green 500
    "warning": "#FF9800",     # Orange 500
    "error": "#F44336"        # Red 500
})

console = Console(theme=theme)
install()
install_pretty()


def get_ollama_api_base():
    """Get the Ollama API base URL from environment variable or default to localhost:8000."""
    return os.environ.get("OLLAMA_API_BASE", "http://localhost:8000/v1")

def load_prompt_template(template_path):
    """
    Load a prompt template from the package resources.
    
    Args:
        template_path: Path to the template file relative to the cai package,
                      e.g., "prompts/system_bug_bounter.md"
    
    Returns:
        The rendered template as a string
    """
    try:
        # Get the template file from package resources
        template_path_parts = template_path.split('/')
        package_path = ['cai'] + template_path_parts[:-1]
        package = '.'.join(package_path)
        filename = template_path_parts[-1]
        
        # Read the content from the package resources
        # Handle different importlib.resources APIs between Python versions
        try:
            # Python 3.9+ API
            template_content = importlib.resources.read_text(package, filename)
        except (TypeError, AttributeError):
            # Fallback for Python 3.8 and earlier
            with importlib.resources.path(package, filename) as path:
                template_content = pathlib.Path(path).read_text(encoding='utf-8')
        
        # Render the template
        return Template(template_content).render()
    except Exception as e:
        raise ValueError(f"Failed to load template '{template_path}': {str(e)}")

def visualize_agent_graph(start_agent):
    """
    Visualize agent graph showing all bidirectional connections between agents.
    Uses Rich library for pretty printing.
    """
    console = Console()  # pylint: disable=redefined-outer-name
    if start_agent is None:
        console.print("[red]No agent provided to visualize.[/red]")
        return

    tree = Tree(
        f"🤖 {
            start_agent.name} (Current Agent)",
        guide_style="bold blue")

    # Track visited agents and their nodes to handle cross-connections
    visited = {}
    agent_nodes = {}
    agent_positions = {}  # Track positions in tree
    position_counter = 0  # Counter for tracking positions

    def add_agent_node(agent, parent=None, is_transfer=False):  # pylint: disable=too-many-branches # noqa: E501
        """Add agent node and track for cross-connections"""
        nonlocal position_counter

        if agent is None:
            return None

        # Create or get existing node for this agent
        if id(agent) in visited:
            if is_transfer:
                # Add reference with position for repeated agents
                original_pos = agent_positions[id(agent)]
                parent.add(
                    f"[cyan]↩ Return to {
                        agent.name} (Top Level Agent #{original_pos})[/cyan]")
            return agent_nodes[id(agent)]

        visited[id(agent)] = True
        position_counter += 1
        agent_positions[id(agent)] = position_counter

        # Create node for current agent
        if is_transfer:
            node = parent
        else:
            node = parent.add(
                f"[green]{agent.name} (#{position_counter})[/green]") if parent else tree  # noqa: E501 pylint: disable=line-too-long
        agent_nodes[id(agent)] = node

        # Add tools as children
        tools_node = node.add("[yellow]Tools[/yellow]")
        for fn in getattr(agent, "functions", []):
            if callable(fn):
                fn_name = getattr(fn, "__name__", "")
                if ("handoff" not in fn_name.lower() and
                        not fn_name.startswith("transfer_to")):
                    tools_node.add(f"[blue]{fn_name}[/blue]")

        # Add Handoffs section
        transfers_node = node.add("[magenta]Handoffs[/magenta]")

        # Process handoff functions
        for fn in getattr(agent, "functions", []):  # pylint: disable=too-many-nested-blocks # noqa: E501
            if callable(fn):
                fn_name = getattr(fn, "__name__", "")
                if ("handoff" in fn_name.lower() or
                        fn_name.startswith("transfer_to")):
                    try:
                        next_agent = fn()
                        if next_agent:
                            # Show bidirectional connection
                            transfer = transfers_node.add(
                                f"🤖 {next_agent.name}")  # noqa: E501
                            add_agent_node(next_agent, transfer, True)
                    except Exception:  # nosec: B112 # pylint: disable=broad-exception-caught # noqa: E501
                        continue
        return node
    # Start recursive traversal from root agent
    add_agent_node(start_agent)
    console.print(tree)

def fix_litellm_transcription_annotations():
    """
    Apply a monkey patch to fix the TranscriptionCreateParams.__annotations__ issue in LiteLLM.
    
    This is a temporary fix until the issue is fixed in the LiteLLM library itself.
    """
    try:
        import litellm.litellm_core_utils.model_param_helper as model_param_helper
        
        # Override the problematic method to avoid the error
        original_get_transcription_kwargs = model_param_helper.ModelParamHelper._get_litellm_supported_transcription_kwargs
        
        def safe_get_transcription_kwargs():
            """A safer version that doesn't rely on __annotations__."""
            return set(["file", "model", "language", "prompt", "response_format", 
                       "temperature", "api_base", "api_key", "api_version", 
                       "timeout", "custom_llm_provider"])
        
        # Apply the monkey patch
        model_param_helper.ModelParamHelper._get_litellm_supported_transcription_kwargs = safe_get_transcription_kwargs        
        return True
    except (ImportError, AttributeError):
        # If the import fails or the attribute doesn't exist, the patch couldn't be applied
        return False

def fix_message_list(messages):  # pylint: disable=R0914,R0915,R0912
    """
    Sanitizes the message list passed as a parameter to align with the
    OpenAI API message format.

    Adjusts the message list to comply with the following rules:
        1. A tool call id appears no more than twice.
        2. Each tool call id appears as a pair, and both messages
            must have content.
        3. If a tool call id appears alone (without a pair), it is removed.
        4. There cannot be empty messages.

    Args:
        messages (List[dict]): List of message dictionaries containing
                            role, content, and optionally tool_calls or
                            tool_call_id fields.

    Returns:
        List[dict]: Sanitized list of messages with invalid tool calls
                   and empty messages removed.
    """
    # Step 1: Filter and discard empty messages (considered empty if 'content'
    # is None or only whitespace)
    cleaned_messages = []
    for msg in messages:
        content = msg.get("content")
        if content is not None and content.strip():
            cleaned_messages.append(msg)
    messages = cleaned_messages
    # Step 2: Collect tool call id occurrences.
    # In assistant messages, iterate through 'tool_calls' list.
    # In 'tool' type messages, use the 'tool_call_id' key.
    tool_calls_occurrences = {}
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and isinstance(
                msg.get("tool_calls"), list):
            for j, tool_call in enumerate(msg["tool_calls"]):
                tc_id = tool_call.get("id")
                if tc_id:
                    tool_calls_occurrences.setdefault(
                        tc_id, []).append((i, "assistant", j))
        elif msg.get("role") == "tool" and msg.get("tool_call_id"):
            tc_id = msg["tool_call_id"]
            tool_calls_occurrences.setdefault(
                tc_id, []).append((i, "tool", None))

    # Step 3: Mark indices in the message list to remove.
    # Maps message index (assistant) to set of indices (in tool_calls) to
    # delete, or directly marks message indices (tool) to delete.
    to_remove = {}
    for tc_id, occurrences in tool_calls_occurrences.items():
        if len(occurrences) > 2:
            # More than one assistant and tool message pair - trim down
            # by picking first pairing and removing the rest
            assistant_items = [
                occ for occ in occurrences if occ[1] == "assistant"]
            tool_items = [occ for occ in occurrences if occ[1] == "tool"]

            if assistant_items and tool_items:
                valid_assistant = assistant_items[0]
                valid_tool = tool_items[0]
                for item in occurrences:
                    if item != valid_assistant and item != valid_tool:
                        if item[1] == "assistant":
                            # If assistant message, mark specific tool_call index
                            to_remove.setdefault(item[0], set()).add(item[2])
                        else:
                            # If tool message, mark whole message
                            to_remove[item[0]] = None
            else:
                # Only one type of message, no complete pairs - remove them all
                for item in occurrences:
                    if item[1] == "assistant":
                        to_remove.setdefault(item[0], set()).add(item[2])
                    else:
                        to_remove[item[0]] = None
        elif len(occurrences) == 1:
            # Incomplete pair (only tool call without tool result or vice versa)
            item = occurrences[0]
            if item[1] == "assistant":
                to_remove.setdefault(item[0], set()).add(item[2])
            else:
                to_remove[item[0]] = None

    # Step 4: Apply the removals and reconstruct the message list
    sanitized_messages = []
    for i, msg in enumerate(messages):
        if i in to_remove and to_remove[i] is None:
            # Skip entirely removed messages
            continue

        # For assistant messages, remove marked tool_calls
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            new_tool_calls = []
            for j, tc in enumerate(msg["tool_calls"]):
                if i not in to_remove or j not in to_remove[i]:
                    new_tool_calls.append(tc)
            msg["tool_calls"] = new_tool_calls
            # If after modification message has no content and no tool_calls,
            # skip it
            if not (msg.get("content", "").strip() or
                   not msg.get("tool_calls")):
                continue

        sanitized_messages.append(msg)

    return sanitized_messages

def cli_print_tool_call(tool_name="", args="", output="", prefix="  "):
    """Print a tool call with pretty formatting"""
    if not tool_name:
        return

    print(f"\n{prefix}{color('Tool Call:', fg='cyan')}")
    print(f"{prefix}{color('Name:', fg='cyan')} {tool_name}")
    if args:
        print(f"{prefix}{color('Args:', fg='cyan')} {args}")
    if output:
        print(f"{prefix}{color('Output:', fg='cyan')} {output}")

def get_model_input_tokens(model):
    """
    Get the number of input tokens for
    max context window capacity for a given model.
    """
    model_tokens = {
        "gpt": 128000,
        "o1": 200000,
        "claude": 200000,
        "qwen2.5": 32000,  # https://ollama.com/library/qwen2.5, 128K input, 8K output  # noqa: E501  # pylint: disable=C0301
        "llama3.1": 32000,  # https://ollama.com/library/llama3.1, 128K input  # noqa: E501  # pylint: disable=C0301
        "deepseek": 128000  # https://api-docs.deepseek.com/quick_start/pricing  # noqa: E501  # pylint: disable=C0301
    }
    for model_type, tokens in model_tokens.items():
        if model_type in model:
            return tokens
    return model_tokens["gpt"]

def _create_token_display(  # pylint: disable=too-many-arguments,too-many-locals,too-many-statements,too-many-branches # noqa: E501
    interaction_input_tokens,
    interaction_output_tokens,  # noqa: E501, pylint: disable=R0913
    interaction_reasoning_tokens,
    total_input_tokens,
    total_output_tokens,
    total_reasoning_tokens,
    model,
    interaction_cost=None,
    total_cost=None
) -> Text:  # noqa: E501
    """
    Create a Text object displaying token usage information
    with enhanced formatting.
    """
    # print(f"\nDEBUG _create_token_display: Received costs - Interaction: {interaction_cost}, Total: {total_cost}")
    
    tokens_text = Text(justify="left")

    # Create a more compact, horizontal display
    tokens_text.append(" ", style="bold")  # Small padding
    
    # Current interaction tokens
    tokens_text.append("Current: ", style="bold")
    tokens_text.append(f"I:{interaction_input_tokens} ", style="green")
    tokens_text.append(f"O:{interaction_output_tokens} ", style="red")
    tokens_text.append(f"R:{interaction_reasoning_tokens} ", style="yellow")
    
    # Current cost - only calculate if not provided
    if interaction_cost is None:
        interaction_cost = calculate_model_cost(model, interaction_input_tokens, interaction_output_tokens)
    # Ensure interaction_cost is a float
    try:
        current_cost = float(interaction_cost) if interaction_cost is not None else 0.0
    except (ValueError, TypeError):
        current_cost = 0.0

    tokens_text.append(f"(${current_cost:.4f}) ", style="bold")
    
    # Separator
    tokens_text.append("| ", style="dim")
    
    # Total tokens
    tokens_text.append("Total: ", style="bold")
    tokens_text.append(f"I:{total_input_tokens} ", style="green")
    tokens_text.append(f"O:{total_output_tokens} ", style="red")
    tokens_text.append(f"R:{total_reasoning_tokens} ", style="yellow")
    
    # Total cost - only calculate if not provided
    if total_cost is None:
        total_cost = calculate_model_cost(model, total_input_tokens, total_output_tokens)
    # Ensure total_cost is a float
    try:
        total_cost_value = float(total_cost) if total_cost is not None else 0.0
    except (ValueError, TypeError):
        total_cost_value = 0.0
        
    tokens_text.append(f"(${total_cost_value:.4f}) ", style="bold")
    
    # Separator
    tokens_text.append("| ", style="dim")
    
    # Context usage
    context_pct = interaction_input_tokens / get_model_input_tokens(model) * 100
    tokens_text.append("Context: ", style="bold")
    tokens_text.append(f"{context_pct:.1f}% ", style="bold")
    
    # Context indicator
    if context_pct < 50:
        indicator = "🟩"
        color_local = "green"
    elif context_pct < 80:
        indicator = "🟨"
        color_local = "yellow"
    else:
        indicator = "🟥"
        color_local = "red"
    
    tokens_text.append(f"{indicator}", style=color_local)

    return tokens_text

def parse_message_content(message):
    """
    Parse a message object to extract its content.
    Sample of message object: 
    Message(
        content='Hello! How can I assist you today?', 
        role='assistant', 
        tool_calls=None, 
        function_call=None, 
        provider_specific_fields={'refusal': None}, 
        annotations=[]
        ) 
    
    Args:
        message: Can be a string or a Message object with content attribute
        
    Returns:
        str: The extracted content as a string
    """
    # Check if this is a duplicate print from OpenAIChatCompletionsModel
    # If the message has already been displayed, return empty string to avoid duplication
    # This is a hacky approach but should work
    
    # If message is already a string, return it
    if isinstance(message, str):
        return message
        
    # If message is a Message object with content attribute
    if hasattr(message, 'content') and message.content is not None:
        return message.content
        
    # If message is a dict with content key
    if isinstance(message, dict) and 'content' in message:
        return message['content']
        
    # If we can't extract content, convert to string
    return str(message)

def parse_message_tool_call(message, tool_output=None):
    """
    Parse a message object to extract its content and tool calls.
    Displays tool calls in the format: tool_name(command=command, args=args)
    and optionally shows the tool output in a separate panel.
    
    Args:
        message: A Message object or dict with content and tool_calls attributes
        tool_output: String containing the output from the tool execution
        
    Returns:
        tuple: (content, tool_panels) where content is the message text and
               tool_panels is a list of panels representing tool calls and outputs
    """
    content = ""
    tool_panels = []
    
    # Extract the content text first (LLM's inference)
    if isinstance(message, str):
        content = message
    elif hasattr(message, 'content') and message.content is not None:
        content = message.content
    elif isinstance(message, dict) and 'content' in message:
        content = message['content']
    
    # Extract tool calls
    tool_calls = None
    if hasattr(message, 'tool_calls') and message.tool_calls:
        tool_calls = message.tool_calls
    elif isinstance(message, dict) and 'tool_calls' in message and message['tool_calls']:
        tool_calls = message['tool_calls']
    
    # Process tool calls if they exist
    if tool_calls:
        from rich.panel import Panel
        from rich.text import Text
        from rich.box import ROUNDED
        
        for tool_call in tool_calls:
            # Extract tool name and arguments
            tool_name = None
            args_dict = {}
            
            # Handle different formats of tool_call objects
            if hasattr(tool_call, 'function'):
                if hasattr(tool_call.function, 'name'):
                    tool_name = tool_call.function.name
                if hasattr(tool_call.function, 'arguments'):
                    try:
                        import json
                        args_dict = json.loads(tool_call.function.arguments)
                    except:
                        args_dict = {"raw_arguments": tool_call.function.arguments}
            elif isinstance(tool_call, dict):
                if 'function' in tool_call:
                    if 'name' in tool_call['function']:
                        tool_name = tool_call['function']['name']
                    if 'arguments' in tool_call['function']:
                        try:
                            import json
                            args_dict = json.loads(tool_call['function']['arguments'])
                        except:
                            args_dict = {"raw_arguments": tool_call['function']['arguments']}
            
            # Create a panel for this tool call if we have a valid name
            if tool_name:
                # Format in the style shown in screenshot: tool_name(command=command, args=args)
                tool_text = Text()
                
                # Start with the tool name in green
                tool_text.append(f"{tool_name}", style="green")
                
                # Create the arguments list in the format (key=value, key=value)
                args_parts = []
                for key, value in args_dict.items():
                    # Format based on value type
                    if isinstance(value, bool):
                        args_parts.append(f"{key}={value}")
                    elif value == "" or value is None:
                        args_parts.append(f"{key}=")
                    else:
                        # If the value contains spaces or special chars, wrap it in quotes
                        if isinstance(value, str) and (' ' in value or '/' in value):
                            args_parts.append(f'{key}="{value}"')
                        else:
                            args_parts.append(f"{key}={value}")
                
                # Add the arguments in parentheses after the tool name
                if args_parts:
                    tool_text.append("(", style="yellow")
                    tool_text.append(", ".join(args_parts), style="yellow")
                    tool_text.append(")", style="yellow")
                
                # Create the tool call panel (blue border)
                tool_panel = Panel(
                    tool_text,
                    border_style="blue",
                    box=ROUNDED,
                    padding=(1, 2),
                    title="[bold]Tool Execution[/bold]",
                    title_align="left",
                    expand=True
                )
                
                tool_panels.append(tool_panel)
                
                # If there's a tool output, create a separate panel for it
                if tool_output and tool_output.strip():
                    output_panel = Panel(
                        Text(tool_output, style="yellow"),
                        border_style="red",
                        box=ROUNDED,
                        padding=(1, 2),
                        title="[bold]Tool Output[/bold]",
                        title_align="left",
                        expand=True
                    )
                    tool_panels.append(output_panel)
    
    return content, tool_panels

def cli_print_agent_messages(agent_name, message, counter, model, debug,  # pylint: disable=too-many-arguments,too-many-locals,unused-argument # noqa: E501
                             interaction_input_tokens=None,
                             interaction_output_tokens=None,
                             interaction_reasoning_tokens=None,
                             total_input_tokens=None,
                             total_output_tokens=None,
                             total_reasoning_tokens=None,
                             interaction_cost=None,
                             total_cost=None,
                             tool_output=None):  # New parameter for tool output
    """Print agent messages/thoughts with enhanced visual formatting."""
    # Use the model from environment variable if available
    model_override = os.getenv('CAI_MODEL')
    if model_override:
        model = model_override

    timestamp = datetime.now().strftime("%H:%M:%S")

    # Create a more hacker-like header
    text = Text()
    
    # Check if the message has tool calls
    has_tool_calls = False
    if hasattr(message, 'tool_calls') and message.tool_calls:
        has_tool_calls = True
    elif isinstance(message, dict) and 'tool_calls' in message and message['tool_calls']:
        has_tool_calls = True
    
    # Parse the message based on whether it has tool calls
    if has_tool_calls:
        parsed_message, tool_panels = parse_message_tool_call(message, tool_output)
    else:
        parsed_message = parse_message_content(message)
        tool_panels = []

    # Special handling for Reasoner Agent
    if agent_name == "Reasoner Agent":
        text.append(f"[{counter}] ", style="bold red")
        text.append(f"Agent: {agent_name} ", style="bold yellow")
        if parsed_message:
            text.append(f">> {parsed_message} ", style="green")
        text.append(f"[{timestamp}", style="dim")
        if model:
            text.append(f" ({os.getenv('CAI_SUPPORT_MODEL')})",
                        style="bold blue")
        text.append("]", style="dim")
    else:
        text.append(f"[{counter}] ", style="bold cyan")
        text.append(f"Agent: {agent_name} ", style="bold green")
        if parsed_message:
            text.append(f">> {parsed_message} ", style="yellow")
        text.append(f"[{timestamp}", style="dim")
        if model:
            text.append(f" ({model})", style="bold magenta")
        text.append("]", style="dim")

    # Add token information with enhanced formatting
    tokens_text = None
    if (interaction_input_tokens is not None and  # pylint: disable=R0916
            interaction_output_tokens is not None and
            interaction_reasoning_tokens is not None and
            total_input_tokens is not None and
            total_output_tokens is not None and
            total_reasoning_tokens is not None):

        tokens_text = _create_token_display(
            interaction_input_tokens,
            interaction_output_tokens,
            interaction_reasoning_tokens,
            total_input_tokens,
            total_output_tokens,
            total_reasoning_tokens,
            model,
            interaction_cost,
            total_cost
        )
        text.append(tokens_text)

    # Create a panel for better visual separation
    panel = Panel(
        text,
        border_style="red" if agent_name == "Reasoner Agent" else "blue",
        box=ROUNDED,
        padding=(0, 1),
        title=("[bold]Reasoning Analysis[/bold]"
               if agent_name == "Reasoner Agent"
               else "[bold]Agent Interaction[/bold]"),
        title_align="left"
    )
    console.print("\n")
    console.print(panel)
    
    # If there are tool panels, print them after the main message panel
    for tool_panel in tool_panels:
        console.print(tool_panel)

def create_agent_streaming_context(agent_name, counter, model):
    """Create a streaming context object that maintains state for streaming agent output."""
    from rich.live import Live
    import shutil
    
    # Use the model from environment variable if available
    model_override = os.getenv('CAI_MODEL')
    if model_override:
        model = model_override
        
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Determine terminal size for best display
    terminal_width, _ = shutil.get_terminal_size((100, 24))
    panel_width = min(terminal_width - 4, 120)  # Keep some margin
    
    # Create base header for the panel
    header = Text()
    header.append(f"[{counter}] ", style="bold cyan")
    header.append(f"Agent: {agent_name} ", style="bold green")
    header.append(f">> ", style="yellow")
    
    # Create the content area for streaming text
    content = Text("")
    
    # Add timestamp and model info
    footer = Text()
    footer.append(f"\n[{timestamp}", style="dim")
    if model:
        footer.append(f" ({model})", style="bold magenta")
    footer.append("]", style="dim")
    
    # Create the panel (initial state)
    panel = Panel(
        Text.assemble(header, content, footer),
        border_style="blue",
        box=ROUNDED,
        padding=(1, 2),  # Add more padding for better readability
        title="[bold]Agent Streaming Response[/bold]",
        title_align="left",
        width=panel_width,
        expand=True  # Allow panel to expand to terminal width
    )
    
    # Start the live display with a higher refresh rate
    live = Live(panel, refresh_per_second=20, console=console)
    live.start()
    
    # Return context object with all the elements needed for updating
    return {
        "live": live,
        "panel": panel,
        "header": header,
        "content": content,
        "footer": footer,
        "timestamp": timestamp,
        "model": model,
        "agent_name": agent_name,
        "panel_width": panel_width
    }

def update_agent_streaming_content(context, text_delta):
    """Update the streaming content with new text."""
    # Parse the text_delta to get just the content if needed
    parsed_delta = parse_message_content(text_delta)
    
    # Add the parsed text to the content
    context["content"].append(parsed_delta)
    
    # Update the live display with the latest content
    updated_panel = Panel(
        Text.assemble(context["header"], context["content"], context["footer"]),
        border_style="blue",
        box=ROUNDED,
        padding=(1, 2),  # Match padding from creation
        title="[bold]Agent Streaming Response[/bold]",
        title_align="left",
        width=context.get("panel_width", 100),
        expand=True  # Allow panel to expand to terminal width
    )
    
    # Force an update with the new panel
    context["live"].update(updated_panel)
    context["panel"] = updated_panel

def finish_agent_streaming(context, final_stats=None):
    """Finish the streaming session and display final stats if available."""
    # If we have token stats, add them
    tokens_text = None
    if final_stats:
        #print(f"\nDEBUG finish_agent_streaming: Received final_stats: {final_stats}")
        
        interaction_input_tokens = final_stats.get("interaction_input_tokens")
        interaction_output_tokens = final_stats.get("interaction_output_tokens")
        interaction_reasoning_tokens = final_stats.get("interaction_reasoning_tokens")
        total_input_tokens = final_stats.get("total_input_tokens")
        total_output_tokens = final_stats.get("total_output_tokens")
        total_reasoning_tokens = final_stats.get("total_reasoning_tokens")
        
        # Ensure costs are properly extracted and preserved as floats
        interaction_cost = float(final_stats.get("interaction_cost", 0.0))
        total_cost = float(final_stats.get("total_cost", 0.0))
        
        if (interaction_input_tokens is not None and
                interaction_output_tokens is not None and
                interaction_reasoning_tokens is not None and
                total_input_tokens is not None and
                total_output_tokens is not None and
                total_reasoning_tokens is not None):
            
            # Only calculate costs if they weren't provided or are zero
            if interaction_cost is None or interaction_cost == 0.0:
                interaction_cost = calculate_model_cost(context["model"], interaction_input_tokens, interaction_output_tokens)
            if total_cost is None or total_cost == 0.0:
                total_cost = calculate_model_cost(context["model"], total_input_tokens, total_output_tokens)
            
            tokens_text = _create_token_display(
                interaction_input_tokens,
                interaction_output_tokens,
                interaction_reasoning_tokens,
                total_input_tokens,
                total_output_tokens,
                total_reasoning_tokens,
                context["model"],
                interaction_cost,
                total_cost
            )
    
    # Create the final panel with stats
    final_panel = Panel(
        Text.assemble(
            context["header"], 
            context["content"], 
            Text("\n\n"), 
            tokens_text if tokens_text else Text(""),
            context["footer"]
        ),
        border_style="blue",
        box=ROUNDED,
        padding=(1, 2),  # Match padding from creation
        title="[bold]Agent Streaming Response[/bold]",
        title_align="left",
        width=context.get("panel_width", 100),
        expand=True
    )
    
    # Update one last time
    context["live"].update(final_panel)
    
    # Ensure updates are displayed before stopping
    import time
    time.sleep(0.5)
    
    # Stop the live display
    context["live"].stop()

def calculate_model_cost(model_name, input_tokens, output_tokens):
    """
    Calculate the cost for a given model based on token usage.
    
    Args:
        model_name: The name of the model being used
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens used
        
    Returns:
        float: The calculated cost in dollars
    """
    # Fetch model pricing data from LiteLLM GitHub repository
    LITELLM_URL = (
        "https://raw.githubusercontent.com/BerriAI/litellm/main/"
        "model_prices_and_context_window.json"
    )
    
    try:
        import requests
        response = requests.get(LITELLM_URL, timeout=2)
        if response.status_code == 200:
            model_pricing_data = response.json()
            
            # Get pricing info for the model
            pricing_info = model_pricing_data.get(model_name, {})
            input_cost_per_token = pricing_info.get("input_cost_per_token", 0)
            output_cost_per_token = pricing_info.get("output_cost_per_token", 0)
            
            # Calculate costs
            input_cost = input_tokens * input_cost_per_token
            output_cost = output_tokens * output_cost_per_token
            
            return input_cost + output_cost
    except Exception:
        # If we can't fetch pricing data, return 0
        pass
    
    return 0.0