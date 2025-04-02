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
                tc_id, []).append(
                (i, "tool", None))
    # Step 3: Mark invalid or extra occurrences for removal
    removal_messages = set()  # Indices of messages (tool type) to remove
    # Maps message index (assistant) to set of indices (in tool_calls) to
    # remove
    removal_assistant_entries = {}
    for tc_id, occurrences in tool_calls_occurrences.items():
        # Only 2 occurrences allowed. Mark extras for removal.
        valid_occurrences = occurrences[:2]
        extra_occurrences = occurrences[2:]
        for occ in extra_occurrences:
            msg_idx, typ, j = occ
            if typ == "assistant":
                removal_assistant_entries.setdefault(msg_idx, set()).add(j)
            elif typ == "tool":
                removal_messages.add(msg_idx)
        # If valid occurrences aren't exactly 2 (i.e., a lonely tool call),
        # mark for removal
        if len(valid_occurrences) != 2:
            for occ in valid_occurrences:
                msg_idx, typ, j = occ
                if typ == "assistant":
                    removal_assistant_entries.setdefault(
                        msg_idx, set()).add(j)
                elif typ == "tool":
                    removal_messages.add(msg_idx)
        else:
            # If exactly 2 occurrences, ensure both have content
            remove_pair = False
            for occ in valid_occurrences:
                msg_idx, typ, _ = occ
                msg_content = messages[msg_idx].get("content")
                if msg_content is None or not msg_content.strip():
                    remove_pair = True
                    break
            if remove_pair:
                for occ in valid_occurrences:
                    msg_idx, typ, j = occ
                    if typ == "assistant":
                        removal_assistant_entries.setdefault(
                            msg_idx, set()).add(j)
                    elif typ == "tool":
                        removal_messages.add(msg_idx)
    # Step 4: Build new message list applying removals
    new_messages = []
    for i, msg in enumerate(messages):
        # Skip if message (tool type) is marked for removal
        if i in removal_messages:
            continue
        # For assistant messages, remove marked tool_calls
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            new_tool_calls = []
            for j, tc in enumerate(msg["tool_calls"]):
                if j not in removal_assistant_entries.get(i, set()):
                    new_tool_calls.append(tc)
            msg["tool_calls"] = new_tool_calls
        # If after modification message has no content and no tool_calls,
        # discard it
        msg_content = msg.get("content")
        if ((msg_content is None or not msg_content.strip()) and
                not msg.get("tool_calls")):
            continue
        new_messages.append(msg)
    return new_messages
