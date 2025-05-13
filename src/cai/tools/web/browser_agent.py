import os
import asyncio
import threading
from typing import Optional, List, Dict, Any, Union
from concurrent.futures import TimeoutError

from dotenv import load_dotenv
from cai.sdk.agents import function_tool


async def _run_browser_agent(
    task: str,
    model: str = None,
    headless: bool = False,
    debug: bool = False,
    memory: bool = False,
    input_data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Run a browser agent task asynchronously.

    Args:
        task (str): The task description for the browser agent.
        model (str): The LLM model to use. If None, uses CAI_MODEL from env.
        headless (bool): Whether to run the browser in headless mode.
        debug (bool): Whether to enable debug mode.
        memory (bool): Whether to enable memory features.
        input_data (Optional[Dict[str, Any]]): Additional input data for the task.

    Returns:
        str: The result of the browser agent task.
    """
    try:
        # Dynamically import browser_use to avoid import errors if not installed
        from browser_use import Agent
        from langchain_openai import ChatOpenAI
        
        # Use CAI_MODEL environment variable if model not specified
        if model is None:
            model = os.getenv("CAI_MODEL", "gpt-4o")
        
        # Initialize the agent
        llm = ChatOpenAI(model=model)
        
        agent_kwargs = {
            "headless": headless,
            "debug": debug,
        }
        
        if memory:
            agent_kwargs["memory"] = True
            
        if input_data:
            agent_kwargs["input_data"] = input_data
            
        agent = Agent(
            task=task,
            llm=llm,
            **agent_kwargs
        )
        
        # Run the agent and capture the result
        result = await agent.run()
        return result
        
    except ImportError as e:
        return (
            f"Error: Required packages not installed. Please install with:\n"
            f"pip install browser-use\n"
            f"For memory functionality: pip install \"browser-use[memory]\"\n"
            f"Then install the browser: playwright install chromium --with-deps\n"
            f"Original error: {str(e)}"
        )
    except Exception as e:
        return f"Error running browser agent: {str(e)}"


def _run_in_new_loop(coro):
    """Run a coroutine in a new event loop in a separate thread."""
    result = None
    exception = None
    
    def run_in_thread():
        nonlocal result, exception
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(coro)
            loop.close()
        except Exception as e:
            exception = e
    
    thread = threading.Thread(target=run_in_thread)
    thread.start()
    thread.join(timeout=300)  # 5 minute timeout
    
    if thread.is_alive():
        return "Error: Browser automation timed out after 5 minutes"
    if exception:
        return f"Error running browser agent: {str(exception)}"
    return result


def _run_browser_agent_sync(
    task: str,
    model: str = None,
    headless: bool = False,
    debug: bool = False,
    memory: bool = False,
    input_data: Optional[Dict[str, Any]] = None
) -> str:
    """Synchronous wrapper for _run_browser_agent"""
    coro = _run_browser_agent(
        task=task,
        model=model,
        headless=headless,
        debug=debug,
        memory=memory,
        input_data=input_data
    )
    
    # Check if we're already in an event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an event loop, use a new thread with a new event loop
            return _run_in_new_loop(coro)
        else:
            # We have a loop but it's not running
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists in this thread
        return asyncio.run(coro)


@function_tool
def automate_browser(
    task: str,
    model: str = None,
    headless: bool = False,
    debug: bool = False,
    memory: bool = False,
    input_data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Automate browser tasks using AI. This tool allows an AI agent to control
    a web browser to perform various tasks like web scraping, form filling,
    data extraction, shopping, and more.
    
    Args:
        task (str): A detailed description of what the browser should do.
                    Be specific about the websites to visit and actions to take.
        model (str): The LLM model to use. If None, uses CAI_MODEL from environment.
                     Examples include 'gpt-4o', 'claude-3-opus', etc.
        headless (bool): Whether to run the browser in headless mode (invisible).
                         Default is False (visible browser).
        debug (bool): Whether to enable debug mode with detailed logs.
                     Default is False.
        memory (bool): Whether to enable memory features for longer tasks.
                      Default is False.
        input_data (Optional[Dict[str, Any]]): Additional data to provide to the task.
                                              Useful for passing structured information.
    
    Returns:
        str: The result of the browser automation task, including any data collected
             or confirmation of actions taken.
    
    Example:
        automate_browser(
            task="Go to example.com, find the contact form, and submit a message with name 'John' and email 'john@example.com'",
            headless=False,
            debug=True
        )
    """
    # Load environment variables to ensure API keys are available
    load_dotenv()
    
    # Run the browser agent synchronously
    return _run_browser_agent_sync(
        task=task,
        model=model,
        headless=headless,
        debug=debug,
        memory=memory,
        input_data=input_data
    ) 