from __future__ import annotations

import os
from typing import Any

from cai.sdk.agents import Agent
from cai.sdk.agents.models import ModelProvider, OpenAIProvider, ZhipuAIProvider, DeepSeekProvider, GeminiProvider


def get_model_provider_for_model(model_name: str) -> ModelProvider:
    """Get the appropriate model provider for a given model name.
    
    Args:
        model_name: The name of the model to use
        
    Returns:
        The appropriate ModelProvider instance
    """
    # Check if this is a DeepSeek model
    if model_name.startswith("deepseek-"):
        # For DeepSeek models, create a DeepSeekProvider
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY environment variable is required for DeepSeek models"
            )
        return DeepSeekProvider(api_key=deepseek_api_key)
    
    # Check if this is a ZhipuAI model
    if model_name.startswith("glm-"):
        # For ZhipuAI models, create a ZhipuAIProvider
        zhipuai_api_key = os.getenv("ZHIPUAI_API_KEY")
        if not zhipuai_api_key:
            raise ValueError(
                "ZHIPUAI_API_KEY environment variable is required for ZhipuAI models"
            )
        return ZhipuAIProvider(api_key=zhipuai_api_key)
    
    # Check if this is a Gemini model
    if model_name.startswith("gemini-"):
        # For Gemini models, create a GeminiProvider
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required for Gemini models"
            )
        return GeminiProvider(api_key=google_api_key)
    
    # For all other models, use OpenAIProvider as default
    # This includes alias models, OpenAI models, Anthropic models, OpenRouter models, etc.
    
    # Check for OpenRouter API configuration
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url = os.getenv("OPENROUTER_API_BASE")
    
    if openrouter_api_key and openrouter_base_url:
        # Use OpenRouter configuration with ChatCompletions API (not Responses API)
        return OpenAIProvider(api_key=openrouter_api_key, base_url=openrouter_base_url, use_responses=False)
    
    # Default OpenAI configuration
    return OpenAIProvider()


def update_agent_models_recursively(agent: Agent[Any], new_model: str, visited=None) -> None:
    """Recursively update the model for an agent and all agents in its handoffs.

    Args:
        agent: The agent to update
        new_model: The new model string to set
        visited: Set of agent names already visited to prevent infinite loops
    """
    if visited is None:
        visited = set()

    # Avoid infinite loops by tracking visited agents
    if agent.name in visited:
        return
    visited.add(agent.name)

    # Update the main agent's model
    if hasattr(agent, "model") and hasattr(agent.model, "model"):
        agent.model.model = new_model
        # Also ensure the agent name is set correctly in the model
        if hasattr(agent.model, "agent_name"):
            agent.model.agent_name = agent.name
        
        # IMPORTANT: Clear any cached state in the model that might be model-specific
        # This ensures the model doesn't have stale state from the previous model
        if hasattr(agent.model, "_client"):
            # Force recreation of the client on next use
            agent.model._client = None
        if hasattr(agent.model, "_converter"):
            # Reset the converter's state
            if hasattr(agent.model._converter, "recent_tool_calls"):
                agent.model._converter.recent_tool_calls.clear()
            if hasattr(agent.model._converter, "tool_outputs"):
                agent.model._converter.tool_outputs.clear()

    # Update models for all handoff agents
    if hasattr(agent, "handoffs"):
        for handoff_item in agent.handoffs:
            # Handle both direct Agent references and Handoff objects
            if hasattr(handoff_item, "on_invoke_handoff"):
                # This is a Handoff object
                # For handoffs created with the handoff() function, the agent is stored
                # in the closure of the on_invoke_handoff function
                # We can try to extract it from the function's closure
                try:
                    # Get the closure variables of the handoff function
                    if (
                        hasattr(handoff_item.on_invoke_handoff, "__closure__")
                        and handoff_item.on_invoke_handoff.__closure__
                    ):
                        for cell in handoff_item.on_invoke_handoff.__closure__:
                            if hasattr(cell.cell_contents, "model") and hasattr(
                                cell.cell_contents, "name"
                            ):
                                # This looks like an agent
                                handoff_agent = cell.cell_contents
                                update_agent_models_recursively(handoff_agent, new_model, visited)
                                break
                except Exception:
                    # If we can't extract the agent from closure, skip it
                    pass
            elif hasattr(handoff_item, "model"):
                # This is a direct Agent reference
                update_agent_models_recursively(handoff_item, new_model, visited)