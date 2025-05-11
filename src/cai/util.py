"""
Util model for CAI
"""
import os
import sys
import importlib.resources
import pathlib
import json
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
import threading

# Global timing variables for tracking active and idle time
_active_timer_start = None
_active_time_total = 0.0
_idle_timer_start = None
_idle_time_total = 0.0
_timing_lock = threading.Lock()

# Set up a global tracker for live streaming panels
_LIVE_STREAMING_PANELS = {}

def start_active_timer():
    """
    Start measuring active time (when LLM is processing or tool is executing).
    Pauses the idle timer if it's running.
    """
    global _active_timer_start, _idle_timer_start, _idle_time_total
    
    with _timing_lock:
        # If idle timer is running, pause it and accumulate time
        if _idle_timer_start is not None:
            idle_duration = time.time() - _idle_timer_start
            _idle_time_total += idle_duration
            _idle_timer_start = None
            
        # Start active timer if not already running
        if _active_timer_start is None:
            _active_timer_start = time.time()

def stop_active_timer():
    """
    Stop measuring active time and accumulate the total.
    Restarts the idle timer.
    """
    global _active_timer_start, _active_time_total, _idle_timer_start
    
    with _timing_lock:
        # If active timer is running, pause it and accumulate time
        if _active_timer_start is not None:
            active_duration = time.time() - _active_timer_start
            _active_time_total += active_duration
            _active_timer_start = None
            
        # Start idle timer if not already running
        if _idle_timer_start is None:
            _idle_timer_start = time.time()

def start_idle_timer():
    """
    Start measuring idle time (when waiting for user input).
    Pauses the active timer if it's running.
    """
    global _idle_timer_start, _active_timer_start, _active_time_total
    
    with _timing_lock:
        # If active timer is running, pause it and accumulate time
        if _active_timer_start is not None:
            active_duration = time.time() - _active_timer_start
            _active_time_total += active_duration
            _active_timer_start = None
            
        # Start idle timer if not already running
        if _idle_timer_start is None:
            _idle_timer_start = time.time()

def stop_idle_timer():
    """
    Stop measuring idle time and accumulate the total.
    Restarts the active timer.
    """
    global _idle_timer_start, _idle_time_total, _active_timer_start
    
    with _timing_lock:
        # If idle timer is running, pause it and accumulate time
        if _idle_timer_start is not None:
            idle_duration = time.time() - _idle_timer_start
            _idle_time_total += idle_duration
            _idle_timer_start = None
            
        # Start active timer if not already running
        if _active_timer_start is None:
            _active_timer_start = time.time()

def get_active_time():
    """
    Get the total active time (LLM processing, tool execution).
    Returns a formatted string like "1h 30m 45s" or "45s" or "5m 30s".
    """
    global _active_time_total, _active_timer_start
    
    with _timing_lock:
        # Calculate total active time including current active period if running
        total_active_seconds = _active_time_total
        if _active_timer_start is not None:
            current_active_duration = time.time() - _active_timer_start
            total_active_seconds += current_active_duration
    
    # Format the time string
    hours, remainder = divmod(int(total_active_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def get_idle_time():
    """
    Get the total idle time (waiting for user input).
    Returns a formatted string like "1h 30m 45s" or "45s" or "5m 30s".
    """
    global _idle_time_total, _idle_timer_start
    
    with _timing_lock:
        # Calculate total idle time including current idle period if running
        total_idle_seconds = _idle_time_total
        if _idle_timer_start is not None:
            current_idle_duration = time.time() - _idle_timer_start
            total_idle_seconds += current_idle_duration
    
    # Format the time string
    hours, remainder = divmod(int(total_idle_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def get_active_time_seconds():
    """
    Get the total active time in seconds for precise measurement.
    Returns a float representing the total number of seconds.
    """
    global _active_time_total, _active_timer_start
    
    with _timing_lock:
        # Calculate total active time including current active period if running
        total_active_seconds = _active_time_total
        if _active_timer_start is not None:
            current_active_duration = time.time() - _active_timer_start
            total_active_seconds += current_active_duration
    
    return total_active_seconds

def get_idle_time_seconds():
    """
    Get the total idle time in seconds for precise measurement.
    Returns a float representing the total number of seconds.
    """
    global _idle_time_total, _idle_timer_start
    
    with _timing_lock:
        # Calculate total idle time including current idle period if running
        total_idle_seconds = _idle_time_total
        if _idle_timer_start is not None:
            current_idle_duration = time.time() - _idle_timer_start
            total_idle_seconds += current_idle_duration
    
    return total_idle_seconds

# Initialize idle timer at module load - system starts in idle state
start_idle_timer()

# Instead of direct import
try:
    from cai.cli import START_TIME
except ImportError:
    START_TIME = None

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
        # Skip displaying cost if already shown in the session summary
        if os.environ.get("CAI_COST_DISPLAYED", "").lower() == "true":
            return
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

# Start of Selection
def visualize_agent_graph(start_agent):
    """
    Visualize agent graph showing all bidirectional connections between agents.
    Uses Rich library for pretty printing.
    """
    console = Console()
    if start_agent is None:
        console.print("[red]No agent provided to visualize.[/red]")
        return

    tree = Tree(f"🤖 {start_agent.name} (Current Agent)", guide_style="bold blue")

    visited = set()
    agent_nodes = {}
    agent_positions = {}
    position_counter = 0

    def add_agent_node(agent, parent=None, is_transfer=False):
        """Add an agent node and track for cross-connections."""
        nonlocal position_counter
        if agent is None:
            return None
        aid = id(agent)
        if aid in visited:
            if is_transfer and parent:
                original_pos = agent_positions.get(aid)
                parent.add(f"[cyan]↩ Return to {agent.name} (Agent #{original_pos})[/cyan]")
            return agent_nodes.get(aid)

        visited.add(aid)
        position_counter += 1
        agent_positions[aid] = position_counter

        if is_transfer and parent:
            node = parent
        elif parent:
            node = parent.add(f"[green]{agent.name} (#{position_counter})[/green]")
        else:
            node = tree
        agent_nodes[aid] = node

        # Add tools
        tools_node = node.add("[yellow]Tools[/yellow]")
        for tool in getattr(agent, "tools", []):
            tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", "")
            tools_node.add(f"[blue]{tool_name}[/blue]")

        # Add handoffs
        transfers_node = node.add("[magenta]Handoffs[/magenta]")
        for handoff_fn in getattr(agent, "handoffs", []):
            if callable(handoff_fn):
                try:
                    next_agent = handoff_fn()
                    if next_agent:
                        transfer_node = transfers_node.add(f"🤖 {next_agent.name}")
                        add_agent_node(next_agent, transfer_node, True)
                except Exception:
                    continue

        return node

    # Start traversal from the root agent
    add_agent_node(start_agent)
    console.print(tree)
# End of Selectio

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
    OpenAI API message format, with special attention to tool call sequencing.

    Adjusts the message list to comply with the following rules:
        1. Each assistant message with tool_calls must be immediately
           followed by a tool message for each tool_call, in order.
        2. If a tool message is missing or misplaced, a synthetic one is inserted.
           Actual but misplaced/orphaned tool messages are discarded.
        3. Empty user messages are removed. System messages get "" if content is None.
        4. Content of tool messages is ensured to be a string.
        5. Basic deduplication of consecutive identical messages is performed.

    Args:
        messages (List[dict]): List of message dictionaries.

    Returns:
        List[dict]: Sanitized list of messages.
    """
    if not messages:
        return []

    # Pass 1: Initial filtering and basic content adjustments (same as before)
    temp_messages = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")
        if role == "user" and (content is None or str(content).strip() == ""):
            continue
        if role == "system" and content is None:
            msg["content"] = ""
        if role == "assistant" and tool_calls and content == "":
             msg["content"] = None
        if role == "assistant" and tool_calls:
            for tc_idx, tc in enumerate(tool_calls):
                if not tc.get("id"):
                    import uuid 
                    generated_id = f"generated_id_{uuid.uuid4().hex[:8]}_{tc_idx}"
                    tc["id"] = generated_id
                    tc_name = tc.get("function", {}).get("name", "unknown")
                    # print(color(f"Warning: Generated missing tool_call_id '{generated_id}' for tool '{tc_name}'. Input messages should have tool_call_ids.", "yellow"))
        temp_messages.append(msg)
    
    if not temp_messages:
        return []

    # Pass 2: Enforce strict tool_call -> tool_result sequencing and remove orphans.
    final_messages_pass2 = []
    temp_msg_idx = 0
    # Keep track of tool_call_ids that are expecting a result from the current assistant batch
    expecting_results_for_ids = [] 

    while temp_msg_idx < len(temp_messages):
        current_msg = temp_messages[temp_msg_idx]

        if current_msg.get("role") == "assistant" and current_msg.get("tool_calls"):
            final_messages_pass2.append(current_msg) # Add assistant msg
            temp_msg_idx += 1 # Consume assistant msg from temp_messages
            
            # This assistant message made tool calls, so we now expect their results.
            expecting_results_for_ids = [(tc.get("id"), tc.get("function", {}).get("name", "unknown")) for tc in current_msg.get("tool_calls", [])]

            for tc_id, tool_name in expecting_results_for_ids:
                # Check if the *next* message in temp_messages is the correct tool result
                if (temp_msg_idx < len(temp_messages) and 
                    temp_messages[temp_msg_idx].get("role") == "tool" and 
                    temp_messages[temp_msg_idx].get("tool_call_id") == tc_id):
                    # Correct tool result found and is next, add it.
                    final_messages_pass2.append(temp_messages[temp_msg_idx])
                    temp_msg_idx += 1 # Consume this tool result from temp_messages
                else:
                    # Expected tool result is not immediately next or is missing.
                    # Insert a synthetic one.
                    synthetic_tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": f"Synthetic placeholder: Result for {tool_name} (ID: {tc_id}). Original result missing or misplaced immediately after call."
                    }
                    final_messages_pass2.append(synthetic_tool_msg)
            expecting_results_for_ids = [] # Reset expectations after processing this assistant's calls

        elif current_msg.get("role") == "tool":
            # A tool message is encountered. It should only be here if it was part of an expected sequence handled above.
            # If `expecting_results_for_ids` is empty, it means we are not currently inside an assistant->tool sequence.
            # Any tool message encountered now is either a duplicate of one already processed (if it matched an expectation)
            # or it's orphaned/misplaced. We discard it to prevent sequence errors.
            # The previous block (assistant with tool_calls) would have consumed this tool message if it was correctly placed.
            # So, if we reach this `elif` for a "tool" message, it means it was NOT consumed as an expected result.
            # print(color(f"Warning: Discarding tool message (ID: {current_msg.get('tool_call_id')}) as it was not an expected immediate result.", "yellow"))
            temp_msg_idx += 1 # Consume (discard) this tool message from temp_messages
        
        else: # User, System, or Assistant (text-only, no tool_calls processed in this iteration)
            final_messages_pass2.append(current_msg)
            temp_msg_idx += 1
            expecting_results_for_ids = [] # Text-only assistant resets expectation of tool results.

    # Pass 3: Content normalization (ensure required content fields are present and appropriately typed)
    # ... (This part remains the same as your existing Pass 3)
    for msg in final_messages_pass2:
        role = msg.get("role")
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")

        if role == "assistant":
            if tool_calls and content is None:
                if msg.get("content") == "": msg["content"] = None
            elif not tool_calls and content is None:
                msg["content"] = ""
        elif role == "tool":
            if content is None:
                msg["content"] = f"No output from tool {msg.get('tool_call_id', 'unknown_id')}"
            elif not isinstance(content, str):
                 msg["content"] = str(content) 
        elif role in ["user", "system"]:
            if content is None:
                msg["content"] = ""
            elif isinstance(content, list):
                for part_idx, part in enumerate(content):
                    if isinstance(part, dict) and part.get("type") == "text" and part.get("text") is None:
                        content[part_idx]["text"] = ""

    # Pass 4: Basic deduplication of consecutive identical messages (remains the same)
    if not final_messages_pass2:
        return []
    deduplicated_messages = [final_messages_pass2[0]]
    for i in range(1, len(final_messages_pass2)):
        prev_msg = deduplicated_messages[-1]
        curr_msg = final_messages_pass2[i]
        is_duplicate = False
        if prev_msg.get("role") == curr_msg.get("role"):
            if curr_msg.get("role") == "tool":
                if prev_msg.get("tool_call_id") == curr_msg.get("tool_call_id") and \
                   prev_msg.get("content") == curr_msg.get("content"):
                    is_duplicate = True
            elif curr_msg.get("role") == "assistant" and curr_msg.get("tool_calls"):
                if (prev_msg.get("role") == "assistant" and prev_msg.get("tool_calls") and 
                    prev_msg.get("tool_calls") == curr_msg.get("tool_calls") and
                    prev_msg.get("content") == curr_msg.get("content")):
                    is_duplicate = True
            elif prev_msg.get("content") == curr_msg.get("content") and not curr_msg.get("tool_calls") and not prev_msg.get("tool_calls"):
                is_duplicate = True
        if not is_duplicate:
            deduplicated_messages.append(curr_msg)
            
    return deduplicated_messages

def cli_print_tool_call(tool_name="", args="", output="", prefix="  "):
    """Print a tool call with pretty formatting"""
    if not tool_name:
        return

    print(f"{prefix}{color('Tool Call:', fg='cyan')}")
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
                             tool_output=None,  # New parameter for tool output
                             suppress_empty=False):  # New parameter to suppress empty panels
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
        
    # Skip empty panels - THIS IS THE KEY CHANGE
    # If suppress_empty is True and there's no parsed message and no tool panels, 
    # don't create an empty panel to avoid cluttering during streaming
    if suppress_empty and not parsed_message and not tool_panels:
        return
        
    # Also skip if the only message is "null" or empty
    if parsed_message == "null" or parsed_message == "":
        if suppress_empty and not tool_panels:
            return

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
    if tool_panels:
        for tool_panel in tool_panels:
            console.print(tool_panel)

def create_agent_streaming_context(agent_name, counter, model):
    """
    Create a streaming context object that maintains state for streaming agent output.
    
    Args:
        agent_name: The name of the agent to display
        counter: The interaction counter (turn number)
        model: The model name
        
    Returns:
        A dictionary with the streaming context
    """
    # Add a static variable to track active streaming contexts and prevent duplicates
    if not hasattr(create_agent_streaming_context, "_active_streaming"):
        create_agent_streaming_context._active_streaming = {}
    
    # If there's already an active streaming context with the same counter, return it
    context_key = f"{agent_name}_{counter}"
    if context_key in create_agent_streaming_context._active_streaming:
        return create_agent_streaming_context._active_streaming[context_key]
    
    try:
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
        live = Live(panel, refresh_per_second=10, console=console, auto_refresh=True, vertical_overflow="visible")
        
        context = {
            "live": live,
            "panel": panel,
            "header": header,
            "content": content,
            "footer": footer,
            "timestamp": timestamp,
            "model": model,
            "agent_name": agent_name,
            "panel_width": panel_width,
            "is_started": False,  # Track if we've started the display
            "error": None,  # Track any errors
        }
        
        # Store the context for potential reuse
        create_agent_streaming_context._active_streaming[context_key] = context
        
        return context
    except Exception as e:
        # If rich display fails, return None and log the error
        import sys
        print(f"Error creating streaming context: {e}", file=sys.stderr)
        return None

def update_agent_streaming_content(context, text_delta):
    """
    Update the streaming content with new text.
    
    Args:
        context: The streaming context created by create_agent_streaming_context
        text_delta: The new text to add
    """
    if not context:
        return False
        
    try:
        # Parse the text_delta to get just the content if needed
        parsed_delta = parse_message_content(text_delta)
        
        # Skip empty updates to avoid showing an empty panel
        if not parsed_delta or parsed_delta.strip() == "":
            return True
        
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
            try:
                context["live"].start()
                context["is_started"] = True
            except Exception as e:
                context["error"] = str(e)
                return False
        
        # Force an update with the new panel
        context["live"].update(updated_panel)
        context["panel"] = updated_panel
        context["live"].refresh()
        return True
    except Exception as e:
        # If there's an error, set it in the context
        context["error"] = str(e)
        return False

def finish_agent_streaming(context, final_stats=None):
    """
    Finish the streaming session and display final stats if available.
    
    Args:
        context: The streaming context to finish
        final_stats: Optional dictionary with token statistics and costs
    """
    if not context:
        return False
    
    # Clean up tracking of this context
    if hasattr(create_agent_streaming_context, "_active_streaming"):
        for key, value in list(create_agent_streaming_context._active_streaming.items()):
            if value is context:
                del create_agent_streaming_context._active_streaming[key]
                break
        
    try:
        # Check if there's actual content to display - don't show empty panels
        if not context["content"] or context["content"].plain == "":
            # If the display was never started, nothing to do
            if not context.get("is_started", False):
                return True
            # Otherwise, stop the display without showing final panel
            try:
                context["live"].stop()
            except Exception:
                pass
            return True
        
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
        time.sleep(0.1)
        
        # Stop the live display
        try:
            context["live"].stop()
        except Exception as e:
            context["error"] = str(e)
            
        return True
    except Exception as e:
        # If there's an error, print it if the context hasn't already tracked one
        if not context.get("error"):
            context["error"] = str(e)
            
        # Try to stop the live display even if there was an error
        try:
            if context.get("is_started", False) and context.get("live"):
                context["live"].stop()
        except Exception:
            pass
            
        return False

def cli_print_tool_output(tool_name="", args="", output="", call_id=None, execution_info=None, token_info=None, streaming=False):
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
        streaming: Flag indicating if this is part of a streaming output
    """
    # If it's an empty output, don't print anything except for streaming sessions
    if not output and not call_id and not streaming:
        return
    
    # Set up global tracker for streaming sessions
    if not hasattr(cli_print_tool_output, '_streaming_sessions'):
        cli_print_tool_output._streaming_sessions = {}
    
    # Track seen call IDs to prevent duplicate panels for non-streaming outputs
    if not hasattr(cli_print_tool_output, '_seen_calls'):
        cli_print_tool_output._seen_calls = {}
        
    # Track all displayed commands to prevent duplicates
    if not hasattr(cli_print_tool_output, '_displayed_commands'):
        cli_print_tool_output._displayed_commands = set()
    
    # --- Consistent Command Key Generation ---
    effective_command_args_str = ""
    if isinstance(args, dict):
        # If args is a dictionary, extract the 'args' field.
        effective_command_args_str = args.get("args", "")
    elif isinstance(args, str):
        # If args is a string, it might be a JSON representation or a plain string.
        try:
            parsed_json_args = json.loads(args)
            if isinstance(parsed_json_args, dict):
                # Parsed as JSON dict, get the 'args' field.
                effective_command_args_str = parsed_json_args.get("args", "")
            else:
                # Parsed as JSON, but not a dict (e.g., a JSON string literal).
                effective_command_args_str = parsed_json_args if isinstance(parsed_json_args, str) else args
        except json.JSONDecodeError:
            # Not a JSON string, treat 'args' as a plain string.
            effective_command_args_str = args
    
    command_key = f"{tool_name}:{effective_command_args_str}"
    # --- End of Command Key Generation ---
        
    # Check for duplicate display conditions
    if streaming:
        # For streaming updates, track and update the single streaming session
        if call_id:
            # If this is a new streaming session, record it
            if call_id not in cli_print_tool_output._streaming_sessions:
                cli_print_tool_output._streaming_sessions[call_id] = {
                    'tool_name': tool_name,
                    'args': args, # Store original args for display formatting
                    'buffer': output if output else "",
                    'start_time': time.time(),
                    'last_update': time.time(),
                    'command_key': command_key, # Store the generated key
                    'is_complete': False
                }
                # Add the command key to displayed commands
                if command_key not in cli_print_tool_output._displayed_commands:
                    cli_print_tool_output._displayed_commands.add(command_key)
            else:
                # Update the existing session
                session = cli_print_tool_output._streaming_sessions[call_id]
                # Always replace buffer with latest output for consistency
                session['buffer'] = output
                session['last_update'] = time.time()
                if execution_info and execution_info.get('is_final', False):
                    session['is_complete'] = True
                    
            # For streaming outputs, we'll use Rich Live panel if available
            try:
                from rich.console import Console
                from rich.live import Live
                from rich.panel import Panel
                from rich.text import Text
                from rich.box import ROUNDED
                
                # Access the global live panel dictionary
                global _LIVE_STREAMING_PANELS
                
                # Create the header, content, and panel
                # Pass the original 'args' (dict or string) to _create_tool_panel_content for formatting
                current_args_for_display = cli_print_tool_output._streaming_sessions[call_id]['args']
                header, content = _create_tool_panel_content(
                    tool_name, 
                    current_args_for_display, 
                    cli_print_tool_output._streaming_sessions[call_id]['buffer'],
                    execution_info,
                    token_info
                )
                
                # Determine panel style based on status
                status = "running"
                if execution_info:
                    status = execution_info.get('status', 'running')
                
                border_style = "yellow"  # Default for running
                if status == "completed":
                    border_style = "green"
                elif status in ["error", "timeout"]:
                    border_style = "red"
                    
                # Create panel title based on status
                if status == "running":
                    title = "[bold yellow]Tool Execution [Running][/bold yellow]"
                elif status == "completed":
                    title = "[bold green]Tool Execution [Completed][/bold green]"
                elif status == "error":
                    title = "[bold red]Tool Execution [Error][/bold red]"
                elif status == "timeout":
                    title = "[bold red]Tool Execution [Timeout][/bold red]"
                else:
                    title = "[bold blue]Tool Execution[/bold blue]"
                
                # Create the panel
                panel = Panel(
                    content,
                    title=title,
                    border_style=border_style,
                    padding=(1, 2),
                    box=ROUNDED,
                    title_align="left"
                )
                
                # If we already have a live panel for this call_id, update it
                if call_id in _LIVE_STREAMING_PANELS:
                    live = _LIVE_STREAMING_PANELS[call_id]
                    live.update(panel)
                    
                    # If this is the final update, stop the live panel after a short delay
                    if execution_info and execution_info.get('is_final', False):
                        # Give a moment for the final panel to be seen
                        time.sleep(0.2)
                        live.stop()
                        # Remove from the active panel dictionary
                        del _LIVE_STREAMING_PANELS[call_id]
                else:
                    # Create a new live panel
                    console = Console(theme=theme)
                    
                    # Set refresh rate - lower for generic_linux_command
                    refresh_rate = 4  # Default refresh rate
                    
                    # Override the refresh rate for specific tools
                    if tool_name == "generic_linux_command":
                        refresh_rate = 2  # Lower refresh rate for terminal commands
                    
                    # Custom refresh rate from execution_info takes precedence
                    if execution_info and "refresh_rate" in execution_info:
                        refresh_rate = execution_info.get("refresh_rate")
                    
                    # Create live panel with the appropriate refresh rate
                    live = Live(panel, console=console, refresh_per_second=refresh_rate, auto_refresh=True)
                    
                    # Start and store the live panel
                    live.start()
                    _LIVE_STREAMING_PANELS[call_id] = live
                
                # Return early for streaming updates
                return
                
            except ImportError:
                # Fall back to simple updates without Rich
                pass
    else:
        # For non-streaming outputs, check if we've already seen this command
        if command_key in cli_print_tool_output._displayed_commands:
            # Command has already been displayed (likely through streaming), skip duplicate display
            return
            
        # Add to displayed commands since we're going to show it
        # This handles the case where a command is non-streaming from the start
        cli_print_tool_output._displayed_commands.add(command_key)
            
    # For non-streaming updates with call_id, check if already seen
    # This _seen_calls logic is an additional layer for non-streaming calls that might have call_ids
    # but might be distinct from the primary _displayed_commands check based on command_key.
    if call_id and not streaming:
        # Create a more specific key for _seen_calls if needed, possibly including output fingerprint
        seen_call_key = f"{call_id}:{command_key}:{output[:20]}" 
        
        if seen_call_key in cli_print_tool_output._seen_calls:
            return
            
        cli_print_tool_output._seen_calls[seen_call_key] = True
        
    # Standard tool output display for non-streaming or when rich is not available
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        from rich.box import ROUNDED
        from rich.console import Group
        
        # Create a console for output
        console = Console(theme=theme)
        
        # Get the panel content
        header, content = _create_tool_panel_content(tool_name, args, output, execution_info, token_info)
        
        # Determine border style
        border_style = "blue"  # Default for non-streaming
        
        # Create the panel
        panel = Panel(
            content,
            title="[bold blue]Tool Output[/bold blue]",
            border_style=border_style,
            padding=(1, 2),
            box=ROUNDED,
            title_align="left"
        )
        
        # Display the panel
        console.print(panel)
        
    except ImportError:
        # Fall back to simple formatting if Rich is not available
        _print_simple_tool_output(tool_name, args, output, execution_info, token_info)

# Helper function to create tool panel content
def _create_tool_panel_content(tool_name, args, output, execution_info=None, token_info=None):
    """Create the header and content for a tool output panel."""
    from rich.text import Text
    from rich.panel import Panel
    from rich.box import ROUNDED
    from rich.console import Group
    
    # Format arguments for display
    args_str = _format_tool_args(args)
    
    # Get timing information
    timing_info, tool_time = _get_timing_info(execution_info)
    
    # Create header
    header = Text()
    header.append(tool_name, style="#00BCD4")
    header.append("(", style="yellow")
    header.append(args_str, style="yellow")
    header.append(")", style="yellow")
    
    # Add timing information
    if timing_info:
        header.append(f" [{' | '.join(timing_info)}]", style="cyan")
    
    # Add environment info if available
    if execution_info and execution_info.get('environment'):
        env = execution_info.get('environment')
        host = execution_info.get('host', '')
        if host:
            header.append(f" [{env}:{host}]", style="magenta")
        else:
            header.append(f" [{env}]", style="magenta")
    
    # Add status information if available
    if execution_info:
        status = execution_info.get('status', None)
        if status == "completed":
            header.append(" [Completed]", style="green")
        elif status == "running":
            header.append(" [Running]", style="yellow")
        elif status == "error":
            header.append(" [Error]", style="red")
        elif status == "timeout":
            header.append(" [Timeout]", style="red")
    
    # Create token information if available
    token_content = _create_token_info_display(token_info)
    
    # Prepare the content for the panel
    group_content = [header, Text("\n\n")]
    
    # Calculate title width for simplified display option
    title_width = len(header.plain)
    max_title_width = 100  # Maximum width before simplifying display
    
    # Extract filtered args for special tool displays
    filtered_args = {}
    if isinstance(args, dict):
        filtered_args = args
    else:
        try:
            # Try to parse the args as JSON if it's a string
            import json
            if isinstance(args, str) and args.strip().startswith('{'):
                filtered_args = json.loads(args)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    
    # Special handling for execute_code tool
    if tool_name == "execute_code" and "language" in filtered_args:
        try:
            from rich.syntax import Syntax  # pylint: disable=import-outside-toplevel,import-error # noqa: E402,E501
            
            language = filtered_args.get("language", "python")
            code = filtered_args.get("code", "")
            
            # Create syntax highlighted code panel
            syntax = Syntax(code, language, theme="monokai", line_numbers=True,
                           background_color="#272822", indent_guides=True)
            code_panel = Panel(
                syntax,
                title="Code",
                border_style="arrow", 
                title_align="left",
                box=ROUNDED,
                padding=(1, 2)
            )
            
            # Create output panel
            output_panel = Panel(
                Text(output, style="yellow"),
                title="Output",
                border_style="border",
                title_align="left", 
                box=ROUNDED,
                padding=(1, 2)
            )
            
            # Don't show code in arguments for execute_code tool
            if title_width > max_title_width:
                # Just show the tool name without the code in args
                simplified_text = Text()
                simplified_text.append(f"{tool_name}(", style="bold cyan")
                simplified_text.append("...", style="yellow")
                simplified_text.append(")", style="bold cyan")
                
                # Show timeout and filename in the simplified display
                timeout = filtered_args.get("timeout", 100)
                filename = filtered_args.get("filename", "exploit")
                
                # Format the simplified timing display
                total_elapsed = "N/A"
                tool_elapsed = "N/A"
                if timing_info:
                    for info in timing_info:
                        if info.startswith("Total:"):
                            total_elapsed = info.replace("Total: ", "")
                        elif info.startswith("Tool:"):
                            tool_elapsed = info.replace("Tool: ", "")
                
                simplified_text.append(
                    f" [File: {filename} | Timeout: {timeout}s | "
                    f"Total: {total_elapsed} | Tool: {tool_elapsed}]",
                    style="bold magenta")
                
                group_content[0] = simplified_text
            
            # Add the code and output panels
            group_content.extend([
                code_panel,
                output_panel
            ])
            
            # Add token info if available
            if token_content:
                group_content.append(token_content)
                
            # Create the full content
            content = Group(*group_content)
            return header, content
            
        except Exception as e:  # pylint: disable=broad-exception-caught # noqa: E722,E501
            # Fallback if syntax highlighting fails
            # Just continue with standard output display
            pass
    
    # Special handling for generic_linux_command tool in bash output
    elif tool_name == "generic_linux_command":
        try:
            from rich.syntax import Syntax  # pylint: disable=import-outside-toplevel,import-error # noqa: E402,E501
            
            # Create a syntax highlighted bash output
            bash_syntax = Syntax(output, "bash", theme="monokai", 
                              background_color="#272822", word_wrap=True)
            
            # Add the syntax highlighted output
            group_content.append(bash_syntax)
            
            # Add token info if available
            if token_content:
                group_content.append(Text("\n"))
                group_content.append(token_content)
                
            # Create the full content
            content = Group(*group_content)
            return header, content
            
        except Exception as e:  # pylint: disable=broad-exception-caught # noqa: E722,E501
            # Fallback if syntax highlighting fails
            # Just continue with standard output display
            pass
    
    # Standard output display for other tools
    # Assemble the full content
    content = Text()
    content.append(header)
    content.append("\n\n")
    content.append(output)
    
    # Add token info if available
    if token_content:
        content.append("\n\n")
        content.append(token_content)
    
    return header, content

# Helper function to format tool arguments
def _format_tool_args(args):
    """Format tool arguments as a clean string."""
    # If args is already a string, it might be pre-formatted or a simple arg string
    if isinstance(args, str):
        # If it looks like a JSON dict string, try to parse and format nicely
        if args.strip().startswith('{') and args.strip().endswith('}'):
            try:

                parsed_dict = json.loads(args)
                # Recursively call with the parsed dict for consistent formatting
                return _format_tool_args(parsed_dict) 
            except json.JSONDecodeError:
                # Not valid JSON, or not a dict; return as is
                return args
        else:
            # Simple string arg, return as is
            return args
    
    # Format arguments from a dictionary
    if isinstance(args, dict):
        # Only include non-empty values and exclude special flags
        arg_parts = []
        for key, value in args.items():
            # Skip empty values
            if value == "" or value == {} or value is None:
                continue
            # Skip special flags
            if key in ["async_mode", "streaming"] and not value:
                continue
            # Format the value
            if isinstance(value, str):
                arg_parts.append(f"{key}={value}")
            else:
                arg_parts.append(f"{key}={value}")
        return ", ".join(arg_parts)
    else:
        return str(args)

# Helper function to get timing information
def _get_timing_info(execution_info=None):
    """Get timing information for display."""
    import time
    
    # Get session timing information
    try:
        from cai.cli import START_TIME
        total_time = time.time() - START_TIME if START_TIME else None
    except ImportError:
        total_time = None
    
    # Extract execution timing info
    tool_time = None
    if execution_info:
        tool_time = execution_info.get('tool_time')
    
    # Format timing info for display
    timing_info = []
    if total_time:
        timing_info.append(f"Total: {format_time(total_time)}")
    if tool_time:
        timing_info.append(f"Tool: {format_time(tool_time)}")
    
    return timing_info, tool_time

# Helper function to create token info display
def _create_token_info_display(token_info=None):
    """Create token information display text."""
    if not token_info:
        return None
    
    from rich.text import Text
    
    model = token_info.get('model', '')
    interaction_input_tokens = token_info.get('interaction_input_tokens', 0)
    interaction_output_tokens = token_info.get('interaction_output_tokens', 0)
    interaction_reasoning_tokens = token_info.get('interaction_reasoning_tokens', 0)
    total_input_tokens = token_info.get('total_input_tokens', 0)
    total_output_tokens = token_info.get('total_output_tokens', 0)
    total_reasoning_tokens = token_info.get('total_reasoning_tokens', 0)
    
    # Only continue if we have actual token information
    if not (interaction_input_tokens > 0 or total_input_tokens > 0):
        return None
    
    # Create token display
    return _create_token_display(
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

# Helper function for simple tool output without Rich
def _print_simple_tool_output(tool_name, args, output, execution_info=None, token_info=None):
    """Print tool output without Rich formatting."""
    # Format arguments
    args_str = _format_tool_args(args)
    
    # Get tool execution time if available
    tool_time_str = ""
    execution_status = ""
    if execution_info:
        time_taken = execution_info.get('time_taken', 0) or execution_info.get('tool_time', 0)
        status = execution_info.get('status', 'completed')
        
        # Add execution info to the tool call display
        if time_taken:
            tool_time_str = f"Tool: {format_time(time_taken)}"
            execution_status = f" [{status} in {time_taken:.2f}s]"
        else:
            execution_status = f" [{status}]"
    
    # Create timing display string
    timing_info, _ = _get_timing_info(execution_info)
    timing_display = f" [{' | '.join(timing_info)}]" if timing_info else ""
    
    # Show tool name, args, execution status and timing display
    tool_call = f"{tool_name}({args_str})"
    print(color(f"Tool Output: {tool_call}{timing_display}{execution_status}", fg="blue"))
    
    # If we have token info, display it
    if token_info:
        model = token_info.get('model', '')
        interaction_input_tokens = token_info.get('interaction_input_tokens', 0)
        interaction_output_tokens = token_info.get('interaction_output_tokens', 0)
        interaction_reasoning_tokens = token_info.get('interaction_reasoning_tokens', 0)
        total_input_tokens = token_info.get('total_input_tokens', 0)
        total_output_tokens = token_info.get('total_output_tokens', 0)
        total_reasoning_tokens = token_info.get('total_reasoning_tokens', 0)
        
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
                token_info.get('interaction_cost')
            )
            total_cost_value = COST_TRACKER.process_total_cost(
                model,
                total_input_tokens,
                total_output_tokens,
                total_reasoning_tokens,
                token_info.get('total_cost')
            )
            print(color(f"  Cost: Current ${current_cost:.4f} | Total ${total_cost_value:.4f} | Session ${COST_TRACKER.session_total_cost:.4f}", fg="cyan"))
            
            # Show context usage
            context_pct = interaction_input_tokens / get_model_input_tokens(model) * 100
            indicator = "🟩" if context_pct < 50 else "🟨" if context_pct < 80 else "🟥"
            print(color(f"  Context: {context_pct:.1f}% {indicator}", fg="cyan"))
    
    # Print the actual output
    print(output)
    print()

# Add a new function to start a streaming tool execution
def start_tool_streaming(tool_name, args, call_id=None):
    """
    Start a streaming tool execution session.
    This allows for progressive updates during tool execution.
    
    Args:
        tool_name: Name of the tool being executed
        args: Arguments to the tool (dictionary or string)
        call_id: Optional call ID for this execution. If not provided, one will be generated.
        
    Returns:
        call_id: The call ID for this streaming session (can be used for updates)
    """
    import time
    import uuid
    
    # Generate a command key to check for duplicates - match format used in cli_print_tool_output
    if isinstance(args, dict):
        cmd = args.get("command", "")
        cmd_args = args.get("args", "")
        command_key = f"{tool_name}:{cmd_args}"
    else:
        command_key = f"{tool_name}:{args}"
    
    # Check if we've already seen this exact command recently
    if not hasattr(start_tool_streaming, '_recent_commands'):
        start_tool_streaming._recent_commands = {}
        
    # If we have an existing active streaming session for this command, reuse its call_id
    # This prevents duplicate panels when the same command runs multiple times
    for existing_call_id, info in list(start_tool_streaming._recent_commands.items()):
        # Only consider recent commands (last 10 seconds)
        timestamp = info.get('timestamp', 0)
        if time.time() - timestamp < 10.0:
            existing_command_key = info.get('command_key', '')
            # Get the existing session info if available
            if (hasattr(cli_print_tool_output, '_streaming_sessions') and 
                existing_call_id in cli_print_tool_output._streaming_sessions):
                session = cli_print_tool_output._streaming_sessions[existing_call_id]
                # If this is the same command and not complete, reuse the call_id
                if existing_command_key == command_key and not session.get('is_complete', False):
                    return existing_call_id
    
    # Generate a call_id if not provided
    if not call_id:
        cmd_part = ""
        if isinstance(args, dict) and "command" in args:
            cmd_part = f"{args['command']}_"
        call_id = f"cmd_{cmd_part}{str(uuid.uuid4())[:8]}"
    
    # Track this call_id with command key for better duplicate detection
    start_tool_streaming._recent_commands[call_id] = {
        'timestamp': time.time(),
        'command_key': command_key
    }
    
    # Cleanup old entries to prevent memory growth
    current_time = time.time()
    start_tool_streaming._recent_commands = {
        k: v for k, v in start_tool_streaming._recent_commands.items() 
        if current_time - v.get('timestamp', 0) < 30  # Keep entries from last 30 seconds
    }
    
    # Custom starting message for generic_linux_command
    initial_output = "Starting tool execution..."
    tool_specific_info = {}
    
    # Special handling for generic_linux_command - show command being executed
    if tool_name == "generic_linux_command":
        # Extract the command being executed for display
        if isinstance(args, str):
            cmd_display = args
        else:
            # If it's a dictionary, try to combine command and args
            command_val = args.get("command", "")
            args_val = args.get("args", "")
            cmd_display = f"{command_val} {args_val}".strip()
        
        initial_output = f"Executing: {cmd_display}\n\nWaiting for output..."
        tool_specific_info = {"refresh_rate": 2}  # Lower refresh rate for terminal output
    
    # Show initial message with "Starting..." output
    cli_print_tool_output(
        tool_name=tool_name,
        args=args,
        output=initial_output,
        call_id=call_id,
        execution_info={"status": "running", "start_time": time.time(), **tool_specific_info},
        streaming=True
    )
    
    return call_id

# Add a function to update a streaming tool execution
def update_tool_streaming(tool_name, args, output, call_id):
    """
    Update a streaming tool execution with new output.
    
    Args:
        tool_name: Name of the tool being executed
        args: Arguments to the tool (dictionary or string)
        output: New output to display
        call_id: The call ID for this streaming session
        
    Returns:
        None
    """
    # Update the streaming output
    cli_print_tool_output(
        tool_name=tool_name,
        args=args,
        output=output,
        call_id=call_id,
        execution_info={"status": "running", "replace_buffer": True},
        streaming=True
    )

# Add a function to complete a streaming tool execution
def finish_tool_streaming(tool_name, args, output, call_id, execution_info=None, token_info=None):
    """
    Complete a streaming tool execution.
    
    Args:
        tool_name: Name of the tool being executed
        args: Arguments to the tool (dictionary or string)
        output: Final output to display
        call_id: The call ID for this streaming session
        execution_info: Optional execution information
        token_info: Optional token information
        
    Returns:
        None
    """
    import time
    
    # Prepare execution info with completion status
    if execution_info is None:
        execution_info = {}
    
    # Add completion markers
    execution_info["status"] = execution_info.get("status", "completed")
    execution_info["is_final"] = True
    execution_info["replace_buffer"] = True
    
    # Calculate execution time if start_time is in the streaming session
    if hasattr(cli_print_tool_output, '_streaming_sessions') and call_id in cli_print_tool_output._streaming_sessions:
        session = cli_print_tool_output._streaming_sessions[call_id]
        if 'start_time' in session and 'tool_time' not in execution_info:
            execution_info["tool_time"] = time.time() - session['start_time']
    
    # Show the final output
    cli_print_tool_output(
        tool_name=tool_name,
        args=args,
        output=output,
        call_id=call_id,
        execution_info=execution_info,
        token_info=token_info,
        streaming=True
    )
    
    # Mark the streaming session as complete
    if hasattr(cli_print_tool_output, '_streaming_sessions') and call_id in cli_print_tool_output._streaming_sessions:
        cli_print_tool_output._streaming_sessions[call_id]['is_complete'] = True

# Function to add a message to history if it's not a duplicate
def add_to_message_history(msg):
    """Add a message to history with refined deduplication logic."""
    if not message_history:
        message_history.append(msg)
        return

    is_duplicate = False
    last_msg = message_history[-1]

    # General check for exact same message object or deep equality with the last one
    if msg == last_msg:
        is_duplicate = True
    else:
        current_role = msg.get("role")
        last_role = last_msg.get("role")

        if current_role == last_role:
            if current_role in ["system", "user"]:
                if msg.get("content") == last_msg.get("content"):
                    is_duplicate = True
            elif current_role == "assistant":
                if msg.get("tool_calls") and last_msg.get("tool_calls"):
                    # Compare entire tool_calls structure and content
                    if (msg.get("tool_calls") == last_msg.get("tool_calls") and
                            msg.get("content") == last_msg.get("content")):
                        is_duplicate = True
                elif not msg.get("tool_calls") and not last_msg.get("tool_calls"):
                    # Assistant text messages
                    if msg.get("content") == last_msg.get("content"):
                        is_duplicate = True
                # If one has tool_calls and the other doesn't, they are not duplicates
            elif current_role == "tool":
                if (msg.get("tool_call_id") == last_msg.get("tool_call_id") and
                        msg.get("content") == last_msg.get("content")):
                    is_duplicate = True

    if not is_duplicate:
        message_history.append(msg)