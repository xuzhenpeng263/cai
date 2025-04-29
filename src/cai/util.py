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
import atexit
from dataclasses import dataclass, field
from typing import Dict, Optional
import time


# Shared stats tracking object to maintain consistent costs across calls
@dataclass
class CostTracker:
    # Session-level stats
    session_total_cost: float = 0.0
    
    # Current agent stats
    current_agent_total_cost: float = 0.0
    current_agent_input_tokens: int = 0
    current_agent_output_tokens: int = 0
    current_agent_reasoning_tokens: int = 0
    
    # Current interaction stats
    interaction_input_tokens: int = 0
    interaction_output_tokens: int = 0
    interaction_reasoning_tokens: int = 0
    interaction_cost: float = 0.0
    
    # Calculation cache
    model_pricing_cache: Dict[str, tuple] = field(default_factory=dict)
    calculated_costs_cache: Dict[str, float] = field(default_factory=dict)
    
    # Track the last calculation to debug inconsistencies
    last_interaction_cost: float = 0.0
    last_total_cost: float = 0.0
    
    def reset_interaction_stats(self):
        """Reset stats for a new interaction"""
        self.interaction_input_tokens = 0
        self.interaction_output_tokens = 0
        self.interaction_reasoning_tokens = 0
        self.interaction_cost = 0.0
    
    def update_session_cost(self, new_cost: float) -> None:
        """Add cost to session total and log the update"""
        old_total = self.session_total_cost
        self.session_total_cost += new_cost
        
    def log_final_cost(self) -> None:
        """Display final cost information at exit"""
        print(f"\nTotal CAI Session Cost: ${self.session_total_cost:.6f}")
    
    def get_model_pricing(self, model_name: str) -> tuple:
        """Get and cache pricing information for a model"""
        # Use the centralized function to standardize model names
        model_name = get_model_name(model_name)
        
        # Check cache first
        if model_name in self.model_pricing_cache:
            return self.model_pricing_cache[model_name]
        
        # Fetch from LiteLLM API
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
                
                # Cache the results
                self.model_pricing_cache[model_name] = (input_cost_per_token, output_cost_per_token)
                return input_cost_per_token, output_cost_per_token
        except Exception as e:
            print(f"  WARNING: Error fetching model pricing: {str(e)}")
        
        # Default values if pricing not found
        default_pricing = (0, 0)
        self.model_pricing_cache[model_name] = default_pricing
        return default_pricing
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int, 
                       label: Optional[str] = None, force_calculation: bool = False) -> float:
        """Calculate and cache cost for a given model and token counts"""
        # Standardize model name using the central function
        model_name = get_model_name(model)
        
        # Generate a cache key
        cache_key = f"{model_name}_{input_tokens}_{output_tokens}"
        
        # Return cached result if available (unless force_calculation is True)
        if cache_key in self.calculated_costs_cache and not force_calculation:
            return self.calculated_costs_cache[cache_key]
        
        # Get pricing information
        input_cost_per_token, output_cost_per_token = self.get_model_pricing(model_name)
        
        # Calculate costs - use high precision for calculations
        input_cost = input_tokens * input_cost_per_token
        output_cost = output_tokens * output_cost_per_token
        total_cost = input_cost + output_cost

        # Cache the result with full precision
        self.calculated_costs_cache[cache_key] = total_cost
        
        return total_cost
    
    def process_interaction_cost(self, model: str, 
                                input_tokens: int, 
                                output_tokens: int, 
                                reasoning_tokens: int = 0,
                                provided_cost: Optional[float] = None) -> float:
        """Process and track costs for a new interaction"""
        # Standardize model name
        model_name = get_model_name(model)
        
        # Update token counts
        self.interaction_input_tokens = input_tokens
        self.interaction_output_tokens = output_tokens
        self.interaction_reasoning_tokens = reasoning_tokens
        
        # Use provided cost or calculate
        if provided_cost is not None and provided_cost > 0:
            self.interaction_cost = float(provided_cost)
        else:
            self.interaction_cost = self.calculate_cost(
                model_name, input_tokens, output_tokens, 
                label="OFFICIAL CALCULATION: Interaction")
        
        self.last_interaction_cost = self.interaction_cost
        
        return self.interaction_cost
    
    def process_total_cost(self, model: str, 
                          total_input_tokens: int, 
                          total_output_tokens: int,
                          total_reasoning_tokens: int = 0,
                          provided_cost: Optional[float] = None) -> float:
        """Process and track costs for total (cumulative) usage"""
        # Standardize model name
        model_name = get_model_name(model)
        
        # Update token counts
        self.current_agent_input_tokens = total_input_tokens
        self.current_agent_output_tokens = total_output_tokens
        self.current_agent_reasoning_tokens = total_reasoning_tokens
        
        # Get previous total and add current interaction cost
        previous_total = self.current_agent_total_cost
        
        # Add the new interaction cost
        if provided_cost is not None and provided_cost > 0:
            # If a total cost is explicitly provided, use it
            new_total_cost = float(provided_cost)
            # Calculate how much was added in this interaction
            cost_diff = new_total_cost - previous_total
        else:
            # Simply add the current interaction cost to the previous total
            cost_diff = self.interaction_cost
            new_total_cost = previous_total + cost_diff
        
        # Only add to session total if there's genuinely new cost (and it's positive)
        if cost_diff > 0:
            self.update_session_cost(cost_diff)
        
        # Update the current agent's total cost
        self.current_agent_total_cost = new_total_cost
        
        # Track the last total for debugging
        self.last_total_cost = new_total_cost
        
        return new_total_cost

# Initialize the global cost tracker
COST_TRACKER = CostTracker()

# Register exit handler for final cost display
atexit.register(COST_TRACKER.log_final_cost)
theme = Theme({
    "timestamp": "#00BCD4",
    "agent": "#4CAF50",
    "arrow": "#FFFFFF",
    "content": "#ECEFF1",
    "tool": "#F44336",

    "cost": "#009688",
    "args_str": "#FFC107",

    "border": "#2196F3",
    "border_state": "#FFD700",
    "model": "#673AB7",
    "dim": "#9E9E9E",
    "current_token_count": "#E0E0E0",
    "total_token_count": "#757575",
    "context_tokens": "#0A0A0A",

    "success": "#4CAF50",
    "warning": "#FF9800",
    "error": "#F44336"
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

def get_model_name(model):
    """
    Extract a string model name from various model inputs.
    Centralizes model name standardization to avoid inconsistencies (e.g. avoid passing model object instead of string name).
    Args:
        model: String model name or model object
        
    Returns:
        str: Standardized model name string
    """
    if isinstance(model, str):
        return model
    # If not a string, use environment variable
    return os.environ.get('CAI_MODEL', 'qwen2.5:72b')

def get_model_pricing(model_name):
    """
    Get pricing information for a model, using the CostTracker's implementation.
    This is a global helper that delegates to the CostTracker instance.
    
    Args:
        model_name: String name of the model
        
    Returns:
        tuple: (input_cost_per_token, output_cost_per_token)
    """
    # Standardize model name
    model_name = get_model_name(model_name)
    
    # Use the CostTracker's implementation to maintain consistency and use its cache
    return COST_TRACKER.get_model_pricing(model_name)

def calculate_model_cost(model, input_tokens, output_tokens):
    """
    Calculate the cost for a given model based on token usage.
    
    Args:
        model: The model name or object
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens used
        
    Returns:
        float: The calculated cost in dollars
    """
    # Use the CostTracker to handle duplicates
    return COST_TRACKER.calculate_cost(
        model, 
        input_tokens, 
        output_tokens,
        label="COST CALCULATION",
        force_calculation=False  # Let it use the cache for duplicates
    )

def _create_token_display(
    interaction_input_tokens,
    interaction_output_tokens,
    interaction_reasoning_tokens,
    total_input_tokens,
    total_output_tokens,
    total_reasoning_tokens,
    model,
    interaction_cost=None,
    total_cost=None
) -> Text:
    # Standardize model name
    model_name = get_model_name(model)
    
    # Process interaction cost
    current_cost = COST_TRACKER.process_interaction_cost(
        model_name, 
        interaction_input_tokens, 
        interaction_output_tokens,
        interaction_reasoning_tokens,
        interaction_cost
    )
    
    # Process total cost 
    total_cost_value = COST_TRACKER.process_total_cost(
        model_name,
        total_input_tokens,
        total_output_tokens,
        total_reasoning_tokens,
        total_cost
    )
    
    # Create display text
    tokens_text = Text(justify="left")
    tokens_text.append(" ", style="bold")
    
    # Current interaction tokens
    tokens_text.append("Current: ", style="bold")
    tokens_text.append(f"I:{interaction_input_tokens} ", style="green")
    tokens_text.append(f"O:{interaction_output_tokens} ", style="red")
    tokens_text.append(f"R:{interaction_reasoning_tokens} ", style="yellow")
    tokens_text.append(f"(${current_cost:.4f}) ", style="bold")
    
    # Separator
    tokens_text.append("| ", style="dim")
    
    # Total tokens for this agent run
    tokens_text.append("Total: ", style="bold")
    tokens_text.append(f"I:{total_input_tokens} ", style="green")
    tokens_text.append(f"O:{total_output_tokens} ", style="red")
    tokens_text.append(f"R:{total_reasoning_tokens} ", style="yellow")
    tokens_text.append(f"(${total_cost_value:.4f}) ", style="bold")
    
    # Separator
    tokens_text.append("| ", style="dim")
    
    # Session total across all agents
    tokens_text.append("Session: ", style="bold magenta")
    tokens_text.append(f"${COST_TRACKER.session_total_cost:.4f}", style="bold magenta")
    
    # Context usage
    tokens_text.append(" | ", style="dim")
    context_pct = interaction_input_tokens / get_model_input_tokens(model_name) * 100
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
    Parse a message object to extract its textual content.
    Only processes messages that don't have tool calls.
    
    Args:
        message: Can be a string or a Message object with content attribute
        
    Returns:
        str: The extracted content as a string
    """
    # Check if this is a duplicate print from OpenAIChatCompletionsModel    
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
    Displays tool calls in the format: tool_name({"command":"","args":"","ctf":{},"async_mode":false,"session_id":""}) 
    and shows the tool output in a separated panel.
    
    Args:
        message: A Message object or dict with content and tool_calls attributes
        tool_output: String containing the output from the tool execution
        
    Returns:
        tuple: (content, tool_panels) where content is the message text and
               tool_panels is a list of panels representing tool calls and outputs
    """
    content = ""
    tool_panels = []
    
    # Extract the content text (LLM's inference)
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
        from rich.console import Group
        
        for tool_call in tool_calls:
            # Extract tool name and arguments
            tool_name = None
            args_dict = {}
            call_id = None
            
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
            
            # Create a panel for this tool call if name is not None
            # NOTE: Tool execution panel will be handled in cli_print_tool_output
            # Pass on tool info to generate panels for display in cli_print_agent_messages
            if tool_name and tool_output:
                # Create content for the panel - just showing the output, not the tool call
                panel_content = []
                
                # Add tool output to the panel
                output_text = Text()
                output_text.append("Output:", style="bold #C0C0C0")  # Silver/gray
                output_text.append(f"\n{tool_output}", style="#C0C0C0")  # Silver/gray
                
                panel_content.append(output_text)
                
                # Create a panel with just the output
                tool_panel = Panel(
                    Group(*panel_content),
                    border_style="blue",
                    box=ROUNDED,
                    padding=(1, 2),
                    title="[bold]Tool Output[/bold]",  # Changed title to indicate this is just output
                    title_align="left",
                    expand=True
                )
                
                tool_panels.append(tool_panel)
                
                # Store the call_id with tool name to help cli_print_tool_output avoid duplicates
                if not hasattr(parse_message_tool_call, '_processed_calls'):
                    parse_message_tool_call._processed_calls = set()
                
                call_key = call_id if call_id else f"{tool_name}:{args_dict}"
                parse_message_tool_call._processed_calls.add(call_key)
    
    return content, tool_panels

# Add this function to detect tool output panels
def is_tool_output_message(message):
    """Check if a message appears to be a tool output panel display message."""
    if isinstance(message, str):
        msg_lower = message.lower()
        return ("call id:" in msg_lower and "output:" in msg_lower) or msg_lower.startswith("tool output")
    return False

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
    # Debug prints to trace the function calls
    if debug:
        if isinstance(message, str):
            print(f"DEBUG cli_print_agent_messages: Received string message: {message[:50]}...")
        if tool_output:
            print(f"DEBUG cli_print_agent_messages: Received tool_output: {tool_output[:50]}...")
    
    # Use the model from environment variable if available
    model_override = os.getenv('CAI_MODEL')
    if model_override:
        model = model_override

    timestamp = datetime.now().strftime("%H:%M:%S")

    # Create header
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
    elif not parsed_message: 
        # When parsed_message is empty, only include timestamp and model info
        text.append(f"Agent: {agent_name} ", style="bold green")
        text.append(f"[{timestamp}", style="dim")
        if model:
            text.append(f" ({model})", style="bold magenta")
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
        # Only append token information if there is a parsed message
        if parsed_message:
            text.append(tokens_text)

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
    #console.print("\n")
    console.print(panel)
    
    # If there are tool panels, print them after the main message panel
    # But only in non-streaming mode to avoid duplicates
    is_streaming_enabled = os.getenv('CAI_STREAM', 'false').lower() == 'true'
    if tool_panels and not is_streaming_enabled:
        for tool_panel in tool_panels:
            console.print(tool_panel)

def create_agent_streaming_context(agent_name, counter, model):
    """Create a streaming context object that maintains state for streaming agent output."""
    from rich.live import Live
    import shutil
    
    # Use the model from env if available
    model_override = os.getenv('CAI_MODEL')
    if model_override:
        model = model_override
        
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Terminal size for better display
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
        padding=(1, 2),  
        title="[bold]Agent Streaming Response[/bold]",
        title_align="left",
        width=panel_width,
        expand=True 
    )
    
    # Create Live display object but don't start it until we have content
    live = Live(panel, refresh_per_second=20, console=console, auto_refresh=False)
    
    return {
        "live": live,
        "panel": panel,
        "header": header,
        "content": content,
        "footer": footer,
        "timestamp": timestamp,
        "model": model,
        "agent_name": agent_name,
        "panel_width": panel_width,
        "is_started": False  # Track if we've started the display
    }

def update_agent_streaming_content(context, text_delta):
    """Update the streaming content with new text."""
    # Parse the text_delta to get just the content if needed
    parsed_delta = parse_message_content(text_delta)
    
    # Skip empty updates to avoid showing an empty panel
    if not parsed_delta or parsed_delta.strip() == "":
        return
    
    # Add the parsed text to the content
    context["content"].append(parsed_delta)
    
    # Update the live display with the latest content
    updated_panel = Panel(
        Text.assemble(context["header"], context["content"], context["footer"]),
        border_style="blue",
        box=ROUNDED,
        padding=(1, 2), 
        title="[bold]Agent Streaming Response[/bold]",
        title_align="left",
        width=context.get("panel_width", 100),
        expand=True
    )
    
    # Check if we need to start the display
    if not context.get("is_started", False):
        context["live"].start()
        context["is_started"] = True
    
    # Force an update with the new panel
    context["live"].update(updated_panel)
    context["panel"] = updated_panel

def finish_agent_streaming(context, final_stats=None):
    """Finish the streaming session and display final stats if available."""
    # Check if there's actual content to display - don't show empty panels
    if not context["content"] or context["content"].plain == "":
        # If the display was never started, nothing to do
        if not context.get("is_started", False):
            return
        # Otherwise, stop the display without showing final panel
        context["live"].stop()
        return
    
    # If we have token stats, add them
    tokens_text = None
    if final_stats:
        interaction_input_tokens = final_stats.get("interaction_input_tokens")
        interaction_output_tokens = final_stats.get("interaction_output_tokens")
        interaction_reasoning_tokens = final_stats.get("interaction_reasoning_tokens")
        total_input_tokens = final_stats.get("total_input_tokens")
        total_output_tokens = final_stats.get("total_output_tokens")
        total_reasoning_tokens = final_stats.get("total_reasoning_tokens")
        
        # Ensure costs are properly extracted and preserved as floats
        interaction_cost = float(final_stats.get("interaction_cost", 0.0))
        total_cost = float(final_stats.get("total_cost", 0.0))
        
        model_name = context.get("model", "")
        # If model is not a string, use env
        if not isinstance(model_name, str):
            model_name = os.environ.get('CAI_MODEL', 'gpt-4o-mini')
        
        if (interaction_input_tokens is not None and
                interaction_output_tokens is not None and
                interaction_reasoning_tokens is not None and
                total_input_tokens is not None and
                total_output_tokens is not None and
                total_reasoning_tokens is not None):
            
            # Only calculate costs if they weren't provided or are zero
            if interaction_cost is None or interaction_cost == 0.0:
                interaction_cost = calculate_model_cost(model_name, interaction_input_tokens, interaction_output_tokens)
            if total_cost is None or total_cost == 0.0:
                total_cost = calculate_model_cost(model_name, total_input_tokens, total_output_tokens)
            
            tokens_text = _create_token_display(
                interaction_input_tokens,
                interaction_output_tokens,
                interaction_reasoning_tokens,
                total_input_tokens,
                total_output_tokens,
                total_reasoning_tokens,
                model_name,  # string model name!
                interaction_cost,
                total_cost
            )
    
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
        padding=(1, 2), 
        title="[bold]Agent Streaming Response[/bold]",
        title_align="left",
        width=context.get("panel_width", 100),
        expand=True
    )
    
    # Update one last time
    context["live"].update(final_panel)
    
    # Ensure updates are displayed before stopping
    time.sleep(0.5)
    
    # Stop the live display
    context["live"].stop()

def cli_print_tool_output(tool_name="", args="", output="", call_id=None, execution_info=None, token_info=None):
    """
    Print a tool call output to the command line.
    Similar to cli_print_tool_call but for the output of the tool.
    
    Args:
        tool_name: Name of the tool
        args: Arguments passed to the tool
        output: The output of the tool
        call_id: Optional call ID for streaming updates
        execution_info: Optional execution information
        token_info: Optional token information with keys:
            - interaction_input_tokens, interaction_output_tokens, interaction_reasoning_tokens
            - total_input_tokens, total_output_tokens, total_reasoning_tokens
            - model: model name string
            - interaction_cost, total_cost: optional cost values
    """
    # If it's an empty output, don't print anything
    if not output and not call_id:
        return
    
    
    # CRITICAL CHECK: When in streaming mode (CAI_STREAM=true), ONLY show output panels
    # for streaming updates (those with call_id). This prevents duplicate output panels.
    is_streaming_enabled = os.getenv('CAI_STREAM', 'false').lower() == 'true'
    if is_streaming_enabled and not call_id:
        # Skip all non-streaming tool output in streaming mode
        return
    
    # Track seen call IDs to prevent duplicate panels
    if not hasattr(cli_print_tool_output, '_seen_calls'):
        cli_print_tool_output._seen_calls = {}
    
    # For streaming updates, only show updates for the same call_id
    # but allow the first appearance of each call_id
    if call_id:
        call_key = f"{call_id}:{output[:20]}"  # Use first 20 chars as fingerprint with call_id
        
        # Skip if we've seen this exact output for this call_id before
        if call_key in cli_print_tool_output._seen_calls:
            return
            
        # Mark as seen
        cli_print_tool_output._seen_calls[call_key] = True
        
        # Limit cache size to prevent memory growth
        if len(cli_print_tool_output._seen_calls) > 1000:
            # Keep only the most recent 500 entries
            cli_print_tool_output._seen_calls = {
                k: cli_print_tool_output._seen_calls[k] 
                for k in list(cli_print_tool_output._seen_calls.keys())[-500:]
            }
    
    # Try to use Rich for better formatting if available
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        from rich.box import ROUNDED
        from rich.console import Group
        
        # Create a console for output
        console = Console(theme=theme)
        
        # Format arguments for display
        # Parse JSON string if args is a string
        if isinstance(args, str) and args.strip().startswith('{'):
            try:
                import json
                args = json.loads(args)
            except:
                # Keep as is if not valid JSON
                pass
        
        # Format arguments as a clean string
        if isinstance(args, dict):
            # Only include non-empty values and exclude async_mode=false
            arg_parts = []
            for key, value in args.items():
                # Skip empty values
                if value == "" or value == {} or value is None:
                    continue
                # Skip async_mode=false (default)
                if key == "async_mode" and value is False:
                    continue
                # Format the value
                if isinstance(value, str):
                    arg_parts.append(f"{key}={value}")
                else:
                    arg_parts.append(f"{key}={value}")
            args_str = ", ".join(arg_parts)
        else:
            args_str = str(args)
        
        # Helper function to format time in a human-readable way
        def format_time(seconds):
            if seconds is None:
                return "N/A"
            
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                minutes = int(seconds / 60)
                seconds_remainder = seconds % 60
                return f"{minutes}m {seconds_remainder:.1f}s"
            else:
                hours = int(seconds / 3600)
                minutes = int((seconds % 3600) / 60)
                return f"{hours}h {minutes}m"
        
        # Get session timing information
        try:
            from cai.cli import GLOBAL_START_TIME, START_TIME
            total_time = time.time() - GLOBAL_START_TIME if GLOBAL_START_TIME else None
            session_time = time.time() - START_TIME if START_TIME else None
        except ImportError:
            total_time = None
            session_time = None
        
        # Extract execution timing info
        print(execution_info)
        tool_time = None
        status = None
        if execution_info:
            # Prefer 'tool_time' if present, else fallback to 'time_taken'
            tool_time = execution_info.get('tool_time')
            if tool_time is None:
                tool_time = execution_info.get('time_taken', 0)
            status = execution_info.get('status', 'completed')
        
        # Create header for all panel displays (both streaming and non-streaming)
        header = Text()
        header.append(tool_name, style="#00BCD4")
        header.append("(", style="yellow")
        header.append(args_str, style="yellow")
        header.append(")", style="yellow")
        
        # Add timing information directly in the header
        timing_info = []
        if total_time:
            timing_info.append(f"Total: {format_time(total_time)}")
        if tool_time:
            timing_info.append(f"Tool: {format_time(tool_time)}")
        if timing_info:
            header.append(f" [{' | '.join(timing_info)}]", style="cyan")
           
        # Add completion status if available - REMOVED, just showing timing now
        # if status:
        #     if status == 'completed':
        #         header.append(f" [Completed]", style="green")
        #     elif status == 'running':
        #         header.append(f" [Running]", style="yellow")
        #     elif status == 'error':
        #         header.append(f" [Error]", style="red")
        #     elif status == 'timeout':
        #         header.append(f" [Timeout]", style="red")
        #     else:
        #         header.append(f" [{status.title()}]", style="dim")
        
        # For streaming mode with call_id, use Rich Live display
        if call_id:
            # Create token information if available
            token_content = None
            if token_info:
                model = token_info.get('model', '')
                interaction_input_tokens = token_info.get('interaction_input_tokens', 0)
                interaction_output_tokens = token_info.get('interaction_output_tokens', 0)
                interaction_reasoning_tokens = token_info.get('interaction_reasoning_tokens', 0)
                total_input_tokens = token_info.get('total_input_tokens', 0)
                total_output_tokens = token_info.get('total_output_tokens', 0)
                total_reasoning_tokens = token_info.get('total_reasoning_tokens', 0)
                
                if (interaction_input_tokens > 0 or total_input_tokens > 0):
                    token_text = _create_token_display(
                        interaction_input_tokens,
                        interaction_output_tokens,
                        interaction_reasoning_tokens,
                        total_input_tokens,
                        total_output_tokens,
                        total_reasoning_tokens,
                        model,
                        token_info.get('interaction_cost'),
                        token_info.get('total_cost')
                    )
                    token_content = Text("\n\n")
                    token_content.append(token_text)
            
            # Create content text with the output
            content = Text(output)
            
            # Create the panel for display, including token info if available
            panel_content = [header, Text("\n\n"), content]
            if token_content:
                panel_content.append(token_content)
                
            # Create title - simple title with no timing info
            title = "[bold blue]Tool Output[/bold blue]"
            
            panel = Panel(
                Text.assemble(*panel_content),
                title=title,
                border_style="blue",
                padding=(1, 2),
                box=ROUNDED,
                title_align="left"
            )
            
            # Display using Rich
            console.print(panel)
            return
            
        # For non-streaming output, also use a blue panel with the same format
        # Create token information if available
        token_text = None
        if token_info:
            model = token_info.get('model', '')
            interaction_input_tokens = token_info.get('interaction_input_tokens', 0)
            interaction_output_tokens = token_info.get('interaction_output_tokens', 0)
            interaction_reasoning_tokens = token_info.get('interaction_reasoning_tokens', 0)
            total_input_tokens = token_info.get('total_input_tokens', 0)
            total_output_tokens = token_info.get('total_output_tokens', 0)
            total_reasoning_tokens = token_info.get('total_reasoning_tokens', 0)
            interaction_cost = token_info.get('interaction_cost')
            total_cost = token_info.get('total_cost')
            
            # Generate token display with CostTracker
            if (interaction_input_tokens > 0 or total_input_tokens > 0):
                token_text = _create_token_display(
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
        
        # Now create the panel content, starting with the header
        panel_content = [header, Text("\n")]
        
        # Add token display if available
        if token_text:
            panel_content.append(token_text)
            panel_content.append(Text("\n"))  # Add spacing after token display
            
        # Add the output
        if output:
            output_text = Text(output)
            panel_content.append(output_text)
        
        # If no content was added but we have output, add it directly
        if len(panel_content) == 2 and output:  # Only header and newline
            panel_content.append(Text(output))
        
        # Create title - simple title with no timing info
        title = "[bold blue]Tool Output[/bold blue]"
        
        # Create the final panel - always blue now
        panel = Panel(
            Group(*panel_content),
            title=title,
            border_style="blue",
            padding=(1, 2),
            box=ROUNDED
        )
        
        # Display the panel
        console.print(panel)
        
    except ImportError:
        # Fall back to simple formatting if Rich is not available
        # Format arguments in the cleaner format
        # Parse JSON string if args is a string
        if isinstance(args, str) and args.strip().startswith('{'):
            try:
                import json
                args = json.loads(args)
            except:
                # Keep as is if not valid JSON
                pass
        
        # Format arguments as a clean string
        if isinstance(args, dict):
            # Only include non-empty values and exclude async_mode=false
            arg_parts = []
            for key, value in args.items():
                # Skip empty values
                if value == "" or value == {} or value is None:
                    continue
                # Skip async_mode=false (default)
                if key == "async_mode" and value is False:
                    continue
                # Format the value
                if isinstance(value, str):
                    arg_parts.append(f"{key}={value}")
                else:
                    arg_parts.append(f"{key}={value}")
            args_str = ", ".join(arg_parts)
        else:
            args_str = str(args)
        
        # Helper function to format time in a human-readable way
        def format_time(seconds):
            if seconds is None:
                return "N/A"
            
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                minutes = int(seconds / 60)
                seconds_remainder = seconds % 60
                return f"{minutes}m {seconds_remainder:.1f}s"
            else:
                hours = int(seconds / 3600)
                minutes = int((seconds % 3600) / 60)
                return f"{hours}h {minutes}m"
        
        # Get session timing information
        try:
            from cai.cli import GLOBAL_START_TIME, START_TIME
            total_time = time.time() - GLOBAL_START_TIME if GLOBAL_START_TIME else None
            session_time = time.time() - START_TIME if START_TIME else None
        except ImportError:
            total_time = None
            session_time = None
        
        # For non-streaming output, use the original formatting
        tool_call = f"{tool_name}({args_str})"
        
        # Get tool execution time if available
        tool_time_str = ""
        execution_status = ""
        if execution_info:
            time_taken = execution_info.get('time_taken', 0)
            status = execution_info.get('status', 'completed')
            
            # Add execution info to the tool call display
            if time_taken:
                tool_time_str = f"Tool: {format_time(time_taken)}"
                execution_status = f" [{status} in {time_taken:.2f}s]"
            else:
                execution_status = f" [{status}]"
        
        # Create timing display string
        timing_info = []
        if total_time:
            timing_info.append(f"Total: {format_time(total_time)}")
        if tool_time:
            timing_info.append(f"Tool: {format_time(tool_time)}")
        if timing_info:
            header.append(f" [{' | '.join(timing_info)}]", style="cyan")
            
        timing_display = f" [{' | '.join(timing_info)}]" if timing_info else ""
        
        # Show tool name, args, execution status and timing display
        print(color(f"Tool Output: {tool_call}{timing_display}{execution_status}", fg="blue"))
        
        # If we have token info, display it using the consistent format from _create_token_display
        if token_info:
            model = token_info.get('model', '')
            interaction_input_tokens = token_info.get('interaction_input_tokens', 0)
            interaction_output_tokens = token_info.get('interaction_output_tokens', 0)
            interaction_reasoning_tokens = token_info.get('interaction_reasoning_tokens', 0)
            total_input_tokens = token_info.get('total_input_tokens', 0)
            total_output_tokens = token_info.get('total_output_tokens', 0)
            total_reasoning_tokens = token_info.get('total_reasoning_tokens', 0)
            interaction_cost = token_info.get('interaction_cost')
            total_cost = token_info.get('total_cost')
            
            # If we have complete token information, display it
            if (interaction_input_tokens > 0 or total_input_tokens > 0):
                # Manually create formatted output similar to _create_token_display
                print(color(f"  Current: I:{interaction_input_tokens} O:{interaction_output_tokens} R:{interaction_reasoning_tokens}", fg="cyan"))
                
                # Calculate or use provided costs
                current_cost = COST_TRACKER.process_interaction_cost(
                    model, 
                    interaction_input_tokens, 
                    interaction_output_tokens,
                    interaction_reasoning_tokens,
                    interaction_cost
                )
                total_cost_value = COST_TRACKER.process_total_cost(
                    model,
                    total_input_tokens,
                    total_output_tokens,
                    total_reasoning_tokens,
                    total_cost
                )
                print(color(f"  Cost: Current ${current_cost:.4f} | Total ${total_cost_value:.4f} | Session ${COST_TRACKER.session_total_cost:.4f}", fg="cyan"))
                
                # Show context usage
                context_pct = interaction_input_tokens / get_model_input_tokens(model) * 100
                indicator = "🟩" if context_pct < 50 else "🟨" if context_pct < 80 else "🟥"
                print(color(f"  Context: {context_pct:.1f}% {indicator}", fg="cyan"))
        
        # Print the actual output
        print(output)
        print()