"""
A simple example to test the one_tool agent.

This script demonstrates how to initialize and run the one_tool agent
using Runner.run() with a simple hello message to verify everything
is working correctly.
"""

import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from cai.sdk.agents import Runner, set_default_openai_client
from cai.agents import get_agent_by_name
from cai.util import fix_litellm_transcription_annotations, color


# Load environment variables
load_dotenv()

# Initialize OpenAI client
external_client = AsyncOpenAI(
    base_url=os.getenv('LITELLM_BASE_URL', 'http://localhost:4000'),
    api_key=os.getenv('LITELLM_API_KEY', 'key')
)
set_default_openai_client(external_client)

async def main():
    # Apply litellm patch to fix the __annotations__ error
    patch_applied = fix_litellm_transcription_annotations()
    if not patch_applied:
        print(color("Something went wrong patching LiteLLM fix_litellm_transcription_annotations", color="red"))
        
    # Get the one_tool agent
    agent = get_agent_by_name("one_tool_agent")
    
    print("Testing one_tool agent with a simple hello message...")
    print(f"Using model: {os.getenv('CAI_MODEL', 'default')}")
    
    # Run the agent with a simple test message
    result = await Runner.run(agent, "Hello! Can you introduce yourself and explain what you can do?")
    
    # Print the result
    print("\nAgent response:")
    print("-" * 40)
    print(result.final_output)
    print("-" * 40)
    print("\nTest completed successfully!")

if __name__ == "__main__":
    asyncio.run(main()) 