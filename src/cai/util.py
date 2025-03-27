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

def get_ollama_api_base() -> str:
    """
    Get the Ollama API base URL from the environment variable.
    """
    return os.getenv("OLLAMA_API_BASE", "http://host.docker.internal:8000/v1")

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
