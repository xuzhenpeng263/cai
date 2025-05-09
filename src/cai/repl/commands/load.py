"""
Load command for CAI REPL.

This module provides commands for loading a jsonl into 
the context of the current session.
"""
import os
import signal
from typing import (
    List,
    Optional
)
from rich.console import Console  # pylint: disable=import-error
from cai.repl.commands.base import Command, register_command
from cai.sdk.agents.models.openai_chatcompletions import message_history
from cai.sdk.agents.run_to_jsonl import get_token_stats, load_history_from_jsonl

console = Console()


class LoadCommand(Command):
    """Command for loading a jsonl into the context of the current session."""

    def __init__(self):
        """Initialize the load command."""
        super().__init__(
            name="/load",
            description="Load a jsonl into the context of the current session",
            aliases=["/l"]
        )

    def handle(self, args: Optional[List[str]] = None) -> bool:
        """Handle the load command.

        Args:
            args: Optional list of command arguments

        Returns:
            True if the command was handled successfully, False otherwise
        """
        return self.handle_load_command(args)

    def handle_load_command(self, args: List[str]) -> bool:
        """Load a jsonl into the context of the current session.

        Args:
            args: List containing the PID to kill

        Returns:
            bool: True if the jsonl was loaded successfully
        """
        if not args:
            console.print("[red]Error: No jsonl file specified[/red]")
            return False

        try:
            jsonl_file = args[0]
            # Try to load the jsonl file
            try:
                messages = load_history_from_jsonl(jsonl_file)
                console.print(f"[green]Jsonl file {jsonl_file} loaded[/green]")
            except BaseException:  # pylint: disable=broad-exception-caught
                # If killing the process group fails, try killing just the
                # process
                console.print(f"[red]Error: Failed to load jsonl file {jsonl_file}[/red]")

            # fetch messages from JSONL file and add them to message_history
            import pprint
            pprint.pprint(messages)

            # for message in messages:
            #     message_history.append(message)


        except Exception as e:  # pylint: disable=broad-exception-caught
            console.print(f"[red]Error loading jsonl file: {str(e)}[/red]")
            return False


# Register the command
register_command(LoadCommand())
