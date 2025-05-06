"""
Basic utilities for executing tools
inside or outside of virtual containers.
"""

import subprocess  # nosec B404
import threading
import os
import pty
import signal
import time
import uuid
import sys
from wasabi import color  # pylint: disable=import-error
from cai.util  import format_time


# Instead of direct import
try:
    from cai.cli import START_TIME
except ImportError:
    START_TIME = None

# Global dictionary to store active sessions
ACTIVE_SESSIONS = {}


class ShellSession:  # pylint: disable=too-many-instance-attributes
    """Class to manage interactive shell sessions"""

    def __init__(self, command, session_id=None, ctf=None):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.command = command
        self.ctf = ctf
        self.process = None
        self.master = None
        self.slave = None
        self.output_buffer = []
        self.is_running = False
        self.last_activity = time.time()

    def start(self):
        """Start the shell session"""
        if self.ctf:
            # For CTF environments
            self.is_running = True
            self.output_buffer.append(
                f"[Session {
                    self.session_id}] Started CTF command: {
                    self.command}")
            try:
                output = self.ctf.get_shell(self.command)
                self.output_buffer.append(output)
            except Exception as e:  # pylint: disable=broad-except
                self.output_buffer.append(f"Error: {str(e)}")
            self.is_running = False
            return

        # For local environment
        try:
            # Create a pseudo-terminal
            self.master, self.slave = pty.openpty()

            # Start the process
            self.process = subprocess.Popen(  # pylint: disable=subprocess-popen-preexec-fn, consider-using-with # noqa: E501
                self.command,
                shell=True,  # nosec B602
                stdin=self.slave,
                stdout=self.slave,
                stderr=self.slave,
                preexec_fn=os.setsid,  # Create a new process group
                universal_newlines=True
            )

            self.is_running = True
            self.output_buffer.append(
                f"[Session {
                    self.session_id}] Started: {
                    self.command}")

            # Start a thread to read output
            threading.Thread(target=self._read_output, daemon=True).start()
        except Exception as e:  # pylint: disable=broad-except
            self.output_buffer.append(f"Error starting session: {str(e)}")
            self.is_running = False

    def _read_output(self):
        """Read output from the process"""
        try:
            while self.is_running:
                try:
                    output = os.read(self.master, 1024).decode()
                    if output:
                        self.output_buffer.append(output)
                        self.last_activity = time.time()
                except OSError:
                    # No data available or terminal closed
                    time.sleep(0.1)
                    if not self.is_process_running():
                        self.is_running = False
                        break
        except Exception as e:  # pylint: disable=broad-except
            self.output_buffer.append(f"Error reading output: {str(e)}")
            self.is_running = False

    def is_process_running(self):
        """Check if the process is still running"""
        if not self.process:
            return False
        return self.process.poll() is None

    def send_input(self, input_data):
        """Send input to the process"""
        if not self.is_running:
            return "Session is not running"

        try:
            if self.ctf:
                # For CTF environments
                output = self.ctf.get_shell(input_data)
                self.output_buffer.append(output)
                return "Input sent to CTF session"

            # For local environment
            input_data = input_data.rstrip() + "\n"
            os.write(self.master, input_data.encode())
            self.last_activity = time.time()
            return "Input sent to session"
        except Exception as e:  # pylint: disable=broad-except
            return f"Error sending input: {str(e)}"

    def get_output(self, clear=True):
        """Get and optionally clear the output buffer"""
        output = "\n".join(self.output_buffer)
        if clear:
            self.output_buffer = []
        return output

    def terminate(self):
        """Terminate the session"""
        if not self.is_running:
            return "Session already terminated"

        try:
            self.is_running = False

            if self.process:
                # Try to terminate the process group
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                except BaseException:  # pylint: disable=bare-except,broad-except # noqa: E501
                    # If that fails, try to terminate just the process
                    self.process.terminate()

                # Clean up resources
                if self.master:
                    os.close(self.master)
                if self.slave:
                    os.close(self.slave)

            return f"Session {self.session_id} terminated"
        except Exception as e:  # pylint: disable=broad-except
            return f"Error terminating session: {str(e)}"


def create_shell_session(command, ctf=None):
    """Create a new shell session"""
    session = ShellSession(command, ctf=ctf)
    session.start()
    ACTIVE_SESSIONS[session.session_id] = session
    return session.session_id


def list_shell_sessions():
    """List all active shell sessions"""
    result = []
    for session_id, session in list(ACTIVE_SESSIONS.items()):
        # Clean up terminated sessions
        if not session.is_running:
            del ACTIVE_SESSIONS[session_id]
            continue

        result.append({
            "session_id": session_id,
            "command": session.command,
            "running": session.is_running,
            "last_activity": time.strftime(
                "%H:%M:%S",
                time.localtime(session.last_activity))
        })
    return result


def send_to_session(session_id, input_data):
    """Send input to a specific session"""
    if session_id not in ACTIVE_SESSIONS:
        return f"Session {session_id} not found"

    session = ACTIVE_SESSIONS[session_id]
    return session.send_input(input_data)


def get_session_output(session_id, clear=True):
    """Get output from a specific session"""
    if session_id not in ACTIVE_SESSIONS:
        return f"Session {session_id} not found"

    session = ACTIVE_SESSIONS[session_id]
    return session.get_output(clear)


def terminate_session(session_id):
    """Terminate a specific session"""
    if session_id not in ACTIVE_SESSIONS:
        return f"Session {session_id} not found"

    session = ACTIVE_SESSIONS[session_id]
    result = session.terminate()
    del ACTIVE_SESSIONS[session_id]
    return result


def _run_ctf(ctf, command, stdout=False, timeout=100, stream=False, call_id=None):
    try:
        # Check the type of ctf object to handle various formats properly
        if ctf is None:
            return f"Error: CTF environment is None"
        
        # Handle string CTF values (like "CTF_ENV")
        if isinstance(ctf, str):
            # If in local mode, fallback to running local command
            return _run_local(command, stdout, timeout, stream, call_id)
        
        # Handle dict format that might come from Ollama/Qwen models
        if isinstance(ctf, dict):
            # Check if this dict has a get_shell key or function
            if callable(getattr(ctf, 'get_shell', None)):
                # This is a proper CTF object
                output = ctf.get_shell(command, timeout=timeout)
                if stdout:
                    print("\033[32m" + output + "\033[0m")
                return output
            else:
                # This is a dict but not a proper CTF object, fallback to local
                return _run_local(command, stdout, timeout, stream, call_id)
        
        # Original code path for proper CTF objects
        if hasattr(ctf, 'get_shell') and callable(ctf.get_shell):
            output = ctf.get_shell(command, timeout=timeout)
            if stdout:
                print("\033[32m" + output + "\033[0m")
            return output
            
        # Fallback for any other case
        return _run_local(command, stdout, timeout, stream, call_id)
    except Exception as e:  # pylint: disable=broad-except
        print(color(f"Error executing CTF command: {e}", fg="red"))
        # Fallback to local execution
        return _run_local(command, stdout, timeout, stream, call_id)


def _run_local(command, stdout=False, timeout=100, stream=False, call_id=None, tool_name=None):
    # If streaming is enabled and we have a call_id
    if stream and call_id:
        return _run_local_streamed(command, call_id, timeout, tool_name)
    
    try:
        # nosec B602 - shell=True is required for command chaining
        result = subprocess.run(
            command,
            shell=True,  # nosec B602
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout)
        output = result.stdout if result.stdout else result.stderr
        if stdout:
            print("\033[32m" + output + "\033[0m")
            
        # Skip passing output to cli_print_tool_output when CAI_STREAM=true
        # This prevents duplicate output in streaming mode
        is_streaming_enabled = os.getenv('CAI_STREAM', 'false').lower() == 'true'
        if not is_streaming_enabled:
            # Optional: Add cli_print_tool_output call here if needed for non-streaming
            pass
            
        return output
    except subprocess.TimeoutExpired as e:
        error_output = e.stdout.decode() if e.stdout else str(e)
        if stdout:
            print("\033[32m" + error_output + "\033[0m")
        return error_output
    except Exception as e:  # pylint: disable=broad-except
        error_msg = f"Error executing local command: {e}"
        print(color(error_msg, fg="red"))
        return error_msg


def _run_local_streamed(command, call_id, timeout=100, tool_name=None):
    """Run a local command with streaming output to the Tool output panel"""
    try:
        # Try to import Rich for nice display
        try:
            from rich.console import Console
            from rich.live import Live
            from rich.panel import Panel
            from rich.text import Text
            from rich.box import ROUNDED
            console = Console()
            rich_available = True
        except ImportError:
            rich_available = False
            from cai.util import cli_print_tool_output
            
        output_buffer = []
        
        # Start the process
        process = subprocess.Popen(
            command,
            shell=True,  # nosec B602
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # If tool_name is not provided, derive it from the command
        if tool_name is None:
            # Just use the first command as the tool name
            tool_name = command.strip().split()[0] + "_command"
        
        # Create panel content for Rich display
        if rich_available:
            # Parse command into command and args
            parts = command.strip().split(' ', 1)
            cmd = parts[0] if parts else ""
            args = parts[1] if len(parts) > 1 else ""
            
            # Format clean arguments, following the same rules as cli_print_tool_output
            arg_parts = []
            if cmd:
                arg_parts.append(f"command={cmd}")
            if args and args.strip():  # Only add args if non-empty
                arg_parts.append(f"args={args}")
            args_str = ", ".join(arg_parts)
            
            header = Text()
            header.append(tool_name, style="#00BCD4")
            header.append("(", style="yellow")
            header.append(args_str, style="yellow")
            header.append(")", style="yellow")
            tool_time = 0 
            start_time = time.time()
            total_time = time.time() - START_TIME if START_TIME else 0
            timing_info = []
            if total_time:
                timing_info.append(f"Total: {format_time(total_time)}")
            if tool_time:
                timing_info.append(f"Tool: {format_time(tool_time)}")
            if timing_info:
                header.append(f" [{' | '.join(timing_info)}]", style="cyan")

            content = Text()
            
            # Get console width and calculate appropriate panel width
            console_width = console.width if hasattr(console, 'width') else 100
            panel_width = min(int(console_width * 0.95), console_width - 4)  # Use 95% of width or leave 4 chars margin
            
            # Create the panel with a specific width to avoid overflow
            panel = Panel(
                Text.assemble(header, "\n\n", content),
                title="[bold green]Tool Execution[/bold green]",
                subtitle="[bold green]Live Output[/bold green]",
                border_style="green",
                padding=(1, 2),
                box=ROUNDED,
                width=panel_width
            )
            
            # Print clear separator before panel to avoid overlap with previous content
            console.print("\n\n")
            
            # Start Live display
            with Live(panel, console=console, refresh_per_second=4) as live:
                # Stream stdout in real-time
                start_time = time.time()
                for line in iter(process.stdout.readline, ''):
                    if not line:
                        break

                    # Add to output collection
                    output_buffer.append(line)

                    # Update content with new line
                    content.append(line, style="bright_white")

                    # Update tool_time and header with new timing info
                    tool_time = time.time() - start_time
                    total_time = time.time() - START_TIME if START_TIME else 0
                    # Remove any previous timing info from header (rebuild header)
                    timing_info = []
                    if total_time:
                        timing_info.append(f"Total: {format_time(total_time)}")
                    if tool_time:
                        timing_info.append(f"Tool: {format_time(tool_time)}")
                    # Rebuild header to update timing
                    header = Text()
                    header.append(tool_name, style="#00BCD4")
                    header.append("(", style="yellow")
                    header.append(args_str, style="yellow")
                    header.append(")", style="yellow")
                    if timing_info:
                        header.append(f" [{' | '.join(timing_info)}]", style="cyan")

                    panel = Panel(
                        Text.assemble(header, "\n\n", content),
                        title="[bold green]Tool Execution[/bold green]",
                        subtitle="[bold green]Live Output[/bold green]",
                        border_style="green",
                        padding=(1, 2),
                        box=ROUNDED,
                        width=panel_width
                    )
                    live.update(panel)
                # Check if process is done
                process.stdout.close()
                return_code = process.wait(timeout=timeout)
                
                # Get any stderr output
                stderr_data = process.stderr.read()
                if stderr_data:
                    content.append("\nERROR OUTPUT:\n", style="red")
                    content.append(stderr_data, style="red")
                    output_buffer.append("\nERROR OUTPUT:\n" + stderr_data)
                    panel = Panel(
                        Text.assemble(header, "\n\n", content),
                        title="[bold green]Tool Execution[/bold green]",
                        subtitle="[bold green]Live Output[/bold green]",
                        border_style="green",
                        padding=(1, 2),
                        box=ROUNDED,
                        width=panel_width
                    )
                    live.update(panel)
                
                # Add completion message
                panel = Panel(
                    Text.assemble(header, "\n\n", content),
                    title="[bold green]Tool Execution[/bold green]",
                    border_style="green",
                    padding=(1, 2),
                    box=ROUNDED,
                    width=panel_width
                )
                live.update(panel)
                
                # Wait a moment for the panel to be displayed properly
                time.sleep(0.5)
                
                # Print clear separator after panel to avoid overlap with next content
                console.print("\n\n")
        else:
            # Fallback to simpler streaming with cli_print_tool_output
            # Parse command into command and args (same as rich mode)
            parts = command.strip().split(' ', 1)
            cmd = parts[0] if parts else ""
            args = parts[1] if len(parts) > 1 else ""
            
            # Create a dictionary with only non-empty values (following the same rules)
            tool_args = {}
            if cmd:
                tool_args["command"] = cmd
            if args and args.strip():
                tool_args["args"] = args
            # Note: Omitted empty values and async_mode=False as it's default
            
            # Print separator to avoid overlap with previous content
            print("\n")
            
            # Initial notification - just once
            cli_print_tool_output(tool_name, tool_args, "Command started...", call_id=call_id)
            
            # Buffer for collecting output 
            buffer_size = 0
            update_interval = 10  # lines
            
            # Stream stdout in real-time
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break
                
                # Add to output collection
                output_buffer.append(line)
                buffer_size += 1
                
                # Only update the output periodically to reduce panel refresh rate
                if buffer_size >= update_interval:
                    current_output = ''.join(output_buffer)
                    cli_print_tool_output(tool_name, tool_args, current_output, call_id=call_id)
                    buffer_size = 0
            
            # Check if process is done
            process.stdout.close()
            return_code = process.wait(timeout=timeout)
            
            # Get any stderr output
            stderr_data = process.stderr.read()
            if stderr_data:
                output_buffer.append("\nERROR OUTPUT:\n" + stderr_data)
            
            # Final output update - always show the final result
            final_output = ''.join(output_buffer)
            if return_code != 0:
                final_output += f"\nCommand exited with code {return_code}"
                
            cli_print_tool_output(tool_name, tool_args, final_output, call_id=call_id)
            
            # Print separator to avoid overlap with next content
            print("\n")
        
        # Return the full output
        return ''.join(output_buffer)
        
    except subprocess.TimeoutExpired:
        error_msg = f"Command timed out after {timeout} seconds"
        output_buffer.append("\n" + error_msg)
        final_output = ''.join(output_buffer)
        
        # Update tool output panel with timeout message
        if not rich_available:
            tool_args = {"command": command}
            cli_print_tool_output(tool_name, tool_args, final_output, call_id=call_id)
        
        return final_output
        
    except Exception as e:  # pylint: disable=broad-except
        error_msg = f"Error executing command: {str(e)}"
        print(color(error_msg, fg="red"))
        
        # Update tool output panel with error message if simple streaming
        if not rich_available:
            tool_args = {"command": command}
            cli_print_tool_output(tool_name, tool_args, error_msg, call_id=call_id)
        
        return error_msg


def run_command(command, ctf=None, stdout=False,  # pylint: disable=too-many-arguments # noqa: E501
                async_mode=False, session_id=None,
                timeout=100, stream=False, call_id=None, tool_name=None):
    """
    Run command either in CTF container or on the local attacker machine

    Args:
        command: The command to execute
        ctf: CTF environment object (if running in CTF)
        stdout: Whether to print output to stdout
        async_mode: Whether to run the command asynchronously
        session_id: ID of an existing session to send the command to
        timeout: Command timeout in seconds
        stream: Whether to stream output in real-time
        call_id: Unique ID for the command execution (for streaming)
        tool_name: Name of the tool being executed (for display in streaming output).
                  If None, the tool name will be derived from the command.

    Returns:
        str: Command output, status message, or session ID
    """
    # If session_id is provided, send command to that session
    if session_id:
        if session_id not in ACTIVE_SESSIONS:
            return f"Session {session_id} not found"

        result = send_to_session(session_id, command)
        if stdout:
            output = get_session_output(session_id, clear=False)
            print("\033[32m" + output + "\033[0m")
        return result

    # If async_mode, create a new session
    if async_mode:
        session_id = create_shell_session(command, ctf)
        if stdout:
            # Wait a moment for initial output
            time.sleep(0.5)
            output = get_session_output(session_id, clear=False)
            print("\033[32m" + output + "\033[0m")
        return f"Created session {session_id}. Use this ID to interact with the session."

    # Generate a call_id if we're streaming and one wasn't provided
    if stream and not call_id:
        call_id = str(uuid.uuid4())[:8]
        
    # Determine whether to use CTF or local execution
    use_ctf = False
    if ctf is not None:
        # Check if ctf is a proper object that can handle shell commands
        if (hasattr(ctf, 'get_shell') and callable(ctf.get_shell)) or isinstance(ctf, dict) or isinstance(ctf, str):
            use_ctf = True
    
    # Run the command using the appropriate method
    if use_ctf:
        try:
            # Try with CTF first
            return _run_ctf(ctf, command, stdout, timeout, stream, call_id)
        except Exception as e:  # pylint: disable=broad-except
            # Fallback to local if CTF fails
            print(color(f"CTF execution failed, falling back to local: {str(e)}", fg="yellow"))
            return _run_local(command, stdout, timeout, stream, call_id, tool_name)
    else:
        # Use local execution
        return _run_local(command, stdout, timeout, stream, call_id, tool_name)
