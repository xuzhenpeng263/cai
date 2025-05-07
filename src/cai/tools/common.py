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

def _get_workspace_dir() -> str:
    """Determines the target workspace directory based on env vars for host."""
    base_dir_env = os.getenv("CAI_WORKSPACE_DIR")
    workspace_name = os.getenv("CAI_WORKSPACE")

    # Determine the base directory
    if base_dir_env:
        base_dir = os.path.abspath(base_dir_env)
    else: # Default base directory is 'workspaces' 
        if workspace_name:
            base_dir = os.path.join(os.getcwd(), "workspaces")
        else: # If no workspace name is set, the workspace IS the CWD.
             return os.getcwd()

    # If a workspace name is provided, append it to the base directory
    if workspace_name:
        if not all(c.isalnum() or c in ['_', '-'] for c in workspace_name):
            print(color(f"Invalid CAI_WORKSPACE name '{workspace_name}'. "
                        f"Using directory '{base_dir}' instead.", fg="yellow"))
            target_dir = base_dir
        else:
             target_dir = os.path.join(base_dir, workspace_name)
    else:
         target_dir = base_dir

    # Ensure the final target directory exists on the host
    try:
        abs_target_dir = os.path.abspath(target_dir)
        os.makedirs(abs_target_dir, exist_ok=True)
        return abs_target_dir
    except OSError as e:
        print(color(f"Error creating/accessing host workspace directory '{abs_target_dir}': {e}",
                    fg="red"))
        print(color(f"Falling back to current directory: {os.getcwd()}", fg="yellow"))
        return os.getcwd()

def _get_container_workspace_path() -> str:
    """Determines the target workspace path inside the container."""
    workspace_name = os.getenv("CAI_WORKSPACE") 
    if workspace_name:
        if not all(c.isalnum() or c in ['_', '-'] for c in workspace_name):
            print(color(f"Invalid CAI_WORKSPACE name '{workspace_name}' for container. "
                        f"Using '/workspace'.", fg="yellow"))
            return "/"
        # Standard path inside CAI containers
        return f"/workspace/workspaces/{workspace_name}"
    else:
        return "/"

class ShellSession:  # pylint: disable=too-many-instance-attributes
    """Class to manage interactive shell sessions"""

    def __init__(self, command, session_id=None, ctf=None, workspace_dir=None, container_id=None): # noqa E501
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.command = command  
        self.ctf = ctf
        self.container_id = container_id 
        # Determine workspace based on context (container, ctf or local host)
        if self.container_id:
            self.workspace_dir = _get_container_workspace_path()
        elif self.ctf:
            self.workspace_dir = workspace_dir or _get_workspace_dir()
        else:
            self.workspace_dir = _get_workspace_dir()
        self.process = None
        self.master = None
        self.slave = None
        self.output_buffer = []
        self.is_running = False
        self.last_activity = time.time()

    def start(self):
        """Start the shell session in the appropriate environment."""
        start_message_cmd = self.command 

        # --- Start in Container ---
        if self.container_id:
            try:
                self.master, self.slave = pty.openpty()
                docker_cmd_list = [
                    "docker", "exec", "-i",
                    "-w", self.workspace_dir, 
                    self.container_id,
                    "sh", "-c", # Use shell to handle complex commands if needed
                    self.command # The actual command to run
                ]
                self.process = subprocess.Popen(
                    docker_cmd_list,
                    stdin=self.slave,
                    stdout=self.slave,
                    stderr=self.slave,
                    preexec_fn=os.setsid,
                    universal_newlines=True
                )
                self.is_running = True
                self.output_buffer.append(
                    f"[Session {self.session_id}] Started in container {self.container_id[:12]}: "
                    f"{start_message_cmd} in {self.workspace_dir}")
                threading.Thread(target=self._read_output, daemon=True).start()
            except Exception as e:
                self.output_buffer.append(f"Error starting container session: {str(e)}")
                self.is_running = False
            return

        # --- Start in CTF ---
        if self.ctf:
            self.is_running = True
            self.output_buffer.append(
                f"[Session {
                    self.session_id}] Started CTF command: {
                    self.command}")
            try:
                output = self.ctf.get_shell(self.command)
                self.output_buffer.append(output)
            except Exception as e:  # pylint: disable=broad-except
                self.output_buffer.append(f"Error executing CTF command: {str(e)}")
            self.is_running = False 
            return

        # --- Start Locally (Host) ---
        try:
            self.master, self.slave = pty.openpty()
            self.process = subprocess.Popen(  # pylint: disable=subprocess-popen-preexec-fn, consider-using-with # noqa: E501
                self.command, 
                shell=True,  # nosec B602 
                stdin=self.slave,
                stdout=self.slave,
                stderr=self.slave,
                cwd=self.workspace_dir, 
                preexec_fn=os.setsid,
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
            self.output_buffer.append(f"Error starting local session: {str(e)}")
            self.is_running = False

    def _read_output(self):
        """Read output from the process"""
        try:
            while self.is_running and self.master is not None:
                try:
                    # Check if process has exited before reading
                    if self.process and self.process.poll() is not None:
                        self.is_running = False
                        break
                    # Read the output
                    output = os.read(self.master, 1024).decode()
                    if output:
                        self.output_buffer.append(output)
                        self.last_activity = time.time()
                    else:
                        self.is_running = False
                        break
                except Exception as read_err: 
                     self.output_buffer.append(f"Error reading output buffer: {str(read_err)}")
                     self.is_running = False
                     break
                # Add a small sleep to prevent busy-waiting if no output
                if self.is_process_running():
                     time.sleep(0.05)
        except Exception as e:
            self.output_buffer.append(f"Error in read_output loop: {str(e)}")
            self.is_running = False
    

    def is_process_running(self):
        """Check if the process is still running"""
        # For CTF or container
        if self.container_id or self.ctf: 
            return self.is_running
        # For local host
        if not self.process:
            return False
        return self.process.poll() is None

    def send_input(self, input_data):
        """Send input to the process (local or container)"""
        if not self.is_running: # For CTF or container
            if self.process and self.process.poll() is None:
                self.is_running = True 
            else:  # For local host
                 return "Session is not running"

        try:
            # --- Send to CTF ---
            if self.ctf:
                output = self.ctf.get_shell(input_data)
                self.output_buffer.append(output)
                return "Input sent to CTF session"

            # --- Send to Local or Container PTY ---
            if self.master is not None:
                input_data_bytes = (input_data.rstrip() + "\n").encode()
                bytes_written = os.write(self.master, input_data_bytes)
                if bytes_written != len(input_data_bytes):
                     self.output_buffer.append(f"[Session {self.session_id}] Warning: Partial input write.")
                self.last_activity = time.time()
                return "Input sent to session"
            else:
                return "Session PTY not available for input"
        except Exception as e:  # pylint: disable=broad-except
            self.output_buffer.append(f"Error sending input: {str(e)}")
            return f"Error sending input: {str(e)}"

    def get_output(self, clear=True):
        """Get and optionally clear the output buffer"""
        output = "\n".join(self.output_buffer)
        if clear:
            self.output_buffer = []
        return output

    def terminate(self):
        """Terminate the session"""
        session_id_short = self.session_id[:8]
        termination_message = f"Session {session_id_short} terminated"
        
        if not self.is_running:
             if self.process and self.process.poll() is None:
                 pass # Process is running, proceed with termination
             else:
                 return f"Session {session_id_short} already terminated or finished."

        try:
            self.is_running = False

            if self.process:
                # Try to terminate the process group
                try:
                    pgid = os.getpgid(self.process.pid)
                    os.killpg(pgid, signal.SIGTERM) 
                except ProcessLookupError:
                     pass # Process already gone
                except subprocess.TimeoutExpired:
                     print(color(f"Session {session_id_short} did not terminate gracefully, sending SIGKILL...", fg="yellow")) # noqa E501
                     try:
                          if pgid:
                              os.killpg(pgid, signal.SIGKILL) # Force kill
                          else:
                              self.process.kill()
                     except ProcessLookupError:
                          pass # Already gone
                     except Exception as kill_err:
                          termination_message = f" (Error during SIGKILL: {kill_err})"
                except Exception as term_err: # Catch other errors during SIGTERM
                     termination_message = f" (Error during SIGTERM: {term_err})"
                     try:
                         self.process.kill()
                     except Exception: pass # Ignore nested errors


                # Final check
                if self.process.poll() is None:
                     print(color(f"Session {session_id_short} process {self.process.pid} may still be running after termination attempts.", fg="red")) # noqa E501
                     termination_message += " (Warning: Process may still be running)"


            # Clean up PTY resources if they exist
            if self.master:
                try: os.close(self.master)
                except OSError: pass
                self.master = None
            if self.slave:
                try: os.close(self.slave)
                except OSError: pass
                self.slave = None
                
            return termination_message
        except Exception as e:  # pylint: disable=broad-except
            return f"Error terminating session {session_id_short}: {str(e)}"


def create_shell_session(command, ctf=None, container_id=None, **kwargs):
    """Create a new shell session in the correct workspace/environment."""
    if container_id:
        session = ShellSession(command, ctf=ctf, container_id=container_id)
    else:
        workspace_dir = _get_workspace_dir()
        session = ShellSession(command, ctf=ctf, workspace_dir=workspace_dir)

    session.start()
    if session.is_running or (ctf and not session.is_running): 
         ACTIVE_SESSIONS[session.session_id] = session
         return session.session_id
    else:
         error_msg = session.get_output(clear=True)
         print(color(f"Failed to start session: {error_msg}", fg="red"))
         return f"Failed to start session: {error_msg}"


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
        return f"Session {session_id} not found or already terminated."

    session = ACTIVE_SESSIONS[session_id]
    result = session.terminate()
    if session_id in ACTIVE_SESSIONS:
        del ACTIVE_SESSIONS[session_id]
    return result


def _run_ctf(ctf, command, stdout=False, timeout=100, workspace_dir=None):
    """Runs command in CTF env, changing to workspace_dir first."""
    target_dir = workspace_dir or _get_workspace_dir()
    full_command = f"cd '{target_dir}' && {command}"
    original_cmd_for_msg = command # For logging
    context_msg = f"(ctf:{target_dir})"
    try:
        output = ctf.get_shell(full_command, timeout=timeout)
        if stdout:
            print(f"\033[32m{context_msg} $ {original_cmd_for_msg}\n{output}\033[0m") # noqa E501
        return output
    except Exception as e:  # pylint: disable=broad-except
        error_msg = f"Error executing CTF command '{original_cmd_for_msg}' in '{target_dir}': {e}" # noqa E501
        print(color(error_msg, fg="red"))
        return error_msg

def _run_ssh(command, stdout=False, timeout=100, workspace_dir=None):
    """Runs command via SSH. Assumes SSH agent or passwordless setup unless sshpass is used externally.""" # noqa E501
    ssh_user = os.environ.get('SSH_USER')
    ssh_host = os.environ.get('SSH_HOST')
    ssh_pass = os.environ.get('SSH_PASS') 
    remote_command = command
    original_cmd_for_msg = command
    context_msg = f"({ssh_user}@{ssh_host})"

    # Construct base SSH command list
    if ssh_pass:
        ssh_cmd_list = ["sshpass", "-p", ssh_pass, "ssh", f"{ssh_user}@{ssh_host}"] # noqa E501
    else:
        ssh_cmd_list = ["ssh", f"{ssh_user}@{ssh_host}"]
    ssh_cmd_list.append(remote_command)

    try:
        # Use subprocess.run with list of args for better security than shell=True
        result = subprocess.run(
            ssh_cmd_list,
            capture_output=True,
            text=True,
            check=False, # Don't raise exception on non-zero exit code
            timeout=timeout
        )
        output = result.stdout if result.stdout else result.stderr
        if stdout:
            print(f"\033[32m{context_msg} $ {original_cmd_for_msg}\n{output}\033[0m") # noqa E501
        # Return combined output, potentially including errors
        return output.strip()
    except subprocess.TimeoutExpired as e:
        error_output = e.stdout if e.stdout else str(e)
        timeout_msg = f"Timeout executing SSH command: {error_output}"
        if stdout:
            print(f"\033[33m{context_msg} $ {original_cmd_for_msg}\nTIMEOUT\n{error_output}\033[0m") # noqa E501
        return timeout_msg
    except FileNotFoundError:
         # Handle case where ssh or sshpass isn't installed
         error_msg = f"'sshpass' or 'ssh' command not found. Ensure they are installed and in PATH." # noqa E501
         print(color(error_msg, fg="red"))
         return error_msg
    except Exception as e:  # pylint: disable=broad-except
        error_msg = f"Error executing SSH command '{original_cmd_for_msg}' on {ssh_host}: {e}" # noqa E501
        print(color(error_msg, fg="red"))
        return error_msg


def _run_local(command, stdout=False, timeout=100, stream=False, call_id=None, tool_name=None, workspace_dir=None):
    """Runs command locally in the specified workspace_dir."""
    # If streaming is enabled and we have a call_id
    if stream and call_id:
        return _run_local_streamed(command, call_id, timeout, tool_name, workspace_dir)
    
    target_dir = workspace_dir or _get_workspace_dir()
    original_cmd_for_msg = command # For logging
    context_msg = f"(local:{target_dir})"
    try:
        result = subprocess.run(
            command,
            shell=True,  # nosec B602
            capture_output=True,
            text=True,
            check=False, 
            timeout=timeout,
            cwd=target_dir 
        )
        output = result.stdout if result.stdout else result.stderr
        if stdout:
            print(f"\033[32m{context_msg} $ {original_cmd_for_msg}\n{output}\033[0m") # noqa E501
            
        # Skip passing output to cli_print_tool_output when CAI_STREAM=true
        # This prevents duplicate output in streaming mode
        is_streaming_enabled = os.getenv('CAI_STREAM', 'false').lower() == 'true'
        if not is_streaming_enabled:
            # Optional: Add cli_print_tool_output call here if needed for non-streaming
            pass
            
        return output.strip()
    except subprocess.TimeoutExpired as e:
        error_output = e.stdout if e.stdout else str(e)
        if stdout:
            print("\033[32m" + error_output + "\033[0m")
            return error_output
    except Exception as e:  # pylint: disable=broad-except
            error_msg = f"Error executing local command: {e}"
            print(color(error_msg, fg="red"))
            return error_msg


def _run_local_streamed(command, call_id, timeout=100, tool_name=None, workspace_dir=None):
    """Run a local command with streaming output to the Tool output panel."""
    target_dir = workspace_dir or _get_workspace_dir()
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
            bufsize=1,
            cwd=target_dir  # Set CWD for local process
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
            total_time = 0
            if START_TIME is not None:
                total_time = time.time() - START_TIME 
            timing_info = []
            if total_time:
                timing_info.append(f"Total: {format_time(total_time)}")
            if tool_time:
                timing_info.append(f"Tool: {format_time(tool_time)}")
            if timing_info:
                header.append(f" [{' | '.join(timing_info)}]", style="cyan")

            content = Text()
            
            panel = Panel(
                Text.assemble(header, "\n\n", content),
                title="[bold green]Tool Execution[/bold green]",
                subtitle="[bold green]Live Output[/bold green]",
                border_style="green",
                padding=(1, 2),
                box=ROUNDED
            )
            
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
                    total_time = 0
                    if START_TIME is not None:
                        total_time = time.time() - START_TIME 
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
                        box=ROUNDED
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
                        box=ROUNDED
                    )
                    live.update(panel)
                
                # Add completion message
                panel = Panel(
                    Text.assemble(header, "\n\n", content),
                    title="[bold green]Tool Execution[/bold green]",
                    border_style="green",
                    padding=(1, 2),
                    box=ROUNDED
                )
                live.update(panel)
                
                # Wait a moment for the panel to be displayed properly
                time.sleep(0.5)
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
    Run command in the appropriate environment (Docker, CTF, SSH, Local)
    and workspace.

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
        str: Command output, status message, or session ID.
    """
    # If session_id is provided, send command to that session
    if session_id:
        if session_id not in ACTIVE_SESSIONS:
            return f"Session {session_id} not found"
        session = ACTIVE_SESSIONS[session_id]
        result = session.send_input(command) # Send the raw command string
        if stdout:
            output = get_session_output(session_id, clear=False)
            env_type = "Local"
            if session.container_id:
                 env_type = f"Container({session.container_id[:12]})"
            elif session.ctf:
                 env_type = "CTF"
            print(f"\033[32m(Session {session_id} in {env_type}:{session.workspace_dir}) >> {command}\n{output}\033[0m") # noqa E501
        return result # Return the result of sending input ("Input sent..." or error)

    # Generate a call_id if we're streaming and one wasn't provided
    if stream and not call_id:
        call_id = str(uuid.uuid4())[:8]

    # 2. Determine Execution Environment (Container > CTF > SSH > Local)
    active_container = os.getenv("CAI_ACTIVE_CONTAINER", "")
    is_ssh_env = all(os.getenv(var) for var in ['SSH_USER', 'SSH_HOST'])

    # --- Docker Container Execution ---
    if active_container and not ctf and not is_ssh_env:
        container_id = active_container
        container_workspace = _get_container_workspace_path()
        context_msg = f"(docker:{container_id[:12]}:{container_workspace})"

        # Handle Async Session Creation in Container
        if async_mode:
            # Create a session specifically for the container environment
            new_session_id = create_shell_session(command, container_id=container_id) # noqa E501
            if "Failed" in new_session_id: # Check if session creation failed
                 return new_session_id
            if stdout:
                # Wait a moment for initial output
                time.sleep(0.2)
                output = get_session_output(new_session_id, clear=False)
                print(f"\033[32m(Started Session {new_session_id} in {context_msg})\n{output}\033[0m") # noqa E501
            return f"Started async session {new_session_id} in container {container_id[:12]}. Use this ID to interact." # noqa E501

        # Handle Streaming Container Execution - not yet implemented for containers
        if stream:
            # For now, display that streaming isn't supported for containers
            from cai.util import cli_print_tool_output
            if call_id and tool_name:
                tool_args = {"command": command, "container": container_id[:12]}
                cli_print_tool_output(
                    tool_name, 
                    tool_args, 
                    "Streaming not yet supported for container execution. Running normally...",
                    call_id=call_id
                )

        # Handle Synchronous Execution in Container
        try:
            # Ensure container workspace exists (best effort)
            # Consider moving this to workspace set/container activation
            mkdir_cmd = ["docker", "exec", container_id, "mkdir", "-p", container_workspace] # noqa E501
            subprocess.run(mkdir_cmd, capture_output=True, text=True, check=False, timeout=10) # noqa E501

            # Construct the docker exec command with workspace context
            cmd_list = [
                "docker", "exec",
                "-w", container_workspace, # Set working directory
                container_id,
                "sh", "-c", command # Execute command via shell
            ]
            result = subprocess.run(
                cmd_list,
                capture_output=True,
                text=True,
                check=False, # Don't raise exception on non-zero exit
                timeout=timeout
            )

            output = result.stdout if result.stdout else result.stderr
            output = output.strip() # Clean trailing newline

            if stdout:
                print(f"\033[32m{context_msg} $ {command}\n{output}\033[0m") # noqa E501

            # Check if command failed specifically because container isn't running
            if result.returncode != 0 and "is not running" in result.stderr:
                print(color(f"{context_msg} Container is not running. Attempting execution on host instead.", fg="yellow")) # noqa E501
                 # Fallback to local execution, preserving workspace context
                return _run_local(command, stdout, timeout, stream, call_id, tool_name, _get_workspace_dir()) # noqa E501

            return output # Return combined stdout/stderr

        except subprocess.TimeoutExpired:
            timeout_msg = "Timeout executing command in container."
            if stdout:
                print(f"\033[33m{context_msg} $ {command}\nTIMEOUT\033[0m") # noqa E501
                print(color("Attempting execution on host instead.", fg="yellow"))
             # Fallback to local execution on timeout
            return _run_local(command, stdout, timeout, stream, call_id, tool_name, _get_workspace_dir()) # noqa E501
        except Exception as e:  # pylint: disable=broad-except
            error_msg = f"Error executing command in container: {str(e)}"
            print(color(f"{context_msg} {error_msg}", fg="red"))
            print(color("Attempting execution on host instead.", fg="yellow"))
             # Fallback to local execution on other errors
            return _run_local(command, stdout, timeout, stream, call_id, tool_name, _get_workspace_dir()) # noqa E501

    # --- CTF Execution ---
    if ctf:
        # Handling streaming for CTF - not fully implemented yet
        if stream:
            from cai.util import cli_print_tool_output
            if call_id and tool_name:
                tool_args = {"command": command, "ctf": True}
                cli_print_tool_output(
                    tool_name, 
                    tool_args, 
                    "Streaming not yet supported for CTF execution. Running normally...",
                    call_id=call_id
                )
        
        # _run_ctf handles workspace internally using _get_workspace_dir() default
        return _run_ctf(ctf, command, stdout, timeout)  # Pass None for workspace_dir

    # --- SSH Execution ---
    if is_ssh_env:
        # Async for SSH would require session management via SSH client features
        if async_mode:
            return "Async mode not fully supported for SSH environment via this function yet."
        
        # Handling streaming for SSH - not fully implemented yet
        if stream:
            from cai.util import cli_print_tool_output
            if call_id and tool_name:
                tool_args = {"command": command, "ssh": True}
                cli_print_tool_output(
                    tool_name, 
                    tool_args, 
                    "Streaming not yet supported for SSH execution. Running normally...",
                    call_id=call_id
                )
        
        # _run_ssh handles command execution, workspace is relative to remote home
        return _run_ssh(command, stdout, timeout)  # Workspace dir less relevant here

    # --- Local Execution (Default Fallback) ---
    # Let _run_local handle determining the host workspace
    # Handle Async Session Creation Locally
    if async_mode:
        # create_shell_session uses _get_workspace_dir() when container_id is None
        new_session_id = create_shell_session(command)
        if isinstance(new_session_id, str) and "Failed" in new_session_id:  # Check failure
            return new_session_id
        # Retrieve the actual workspace dir the session is using
        session = ACTIVE_SESSIONS.get(new_session_id)
        actual_workspace = session.workspace_dir if session else "unknown"
        if stdout:
            time.sleep(0.2)  # Allow session buffer to populate
            output = get_session_output(new_session_id, clear=False)
            print(f"\033[32m(Started Session {new_session_id} in local:{actual_workspace})\n{output}\033[0m")
        return f"Started async session {new_session_id} locally. Use this ID to interact."

    # Handle Synchronous Execution Locally using _run_local default with streaming support
    return _run_local(command, stdout, timeout, stream, call_id, tool_name, None)
