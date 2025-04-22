"""
CTF Flag Discriminator Agent with test
"""
import os
from cai.sdk.agents import Agent
from openai import AsyncOpenAI

model = os.getenv('CAI_MODEL', "qwen2.5:14b")
flag_discriminator = Agent(
    model="gpt-4o" if os.getenv('CAI_MODEL') == "o3-mini" else model,
    name="Flag discriminator",
    description="Agent focused on extracting the flag from the output",
    instructions="""INSTRUCTIONS:
                1. You are an agent tailored to extract the flag from a given output.
                2. ONLY return the flag, nothing else.
                3. Flags can be in any format (not always CTF{...}).
                4. If you do not find a flag, call `ctf_agent` to continue investigating.
                """,
)

# Transfer Function
def transfer_to_flag_discriminator(**kwargs):  # pylint: disable=W0613
    """Transfer flag discriminator.
    Accepts any keyword arguments but ignores them."""
    return flag_discriminator
