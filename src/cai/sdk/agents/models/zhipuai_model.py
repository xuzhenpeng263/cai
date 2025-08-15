from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

from zai import ZhipuAiClient

from ..agent_output import AgentOutputSchema
from ..handoffs import Handoff
from ..items import ModelResponse, TResponseInputItem, TResponseStreamEvent
from ..tool import Tool
from .interface import Model, ModelTracing

if TYPE_CHECKING:
    from ..model_settings import ModelSettings


class ZhipuAIModel(Model):
    """ZhipuAI Model Implementation"""

    def __init__(
        self,
        model: str,
        api_key: str,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._client = ZhipuAiClient(api_key=api_key)

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
    ) -> ModelResponse:
        """Get a response from the ZhipuAI model."""
        # Prepare messages
        messages = self._prepare_messages(system_instructions, input)
        
        # Prepare tools
        tools_for_api = self._prepare_tools(tools, handoffs)
        
        # Prepare request parameters
        request_params = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        
        # Add tools if available
        if tools_for_api:
            request_params["tools"] = tools_for_api
            request_params["tool_choice"] = "auto"
            
        # Add model settings
        if model_settings.temperature is not None:
            request_params["temperature"] = model_settings.temperature
        if model_settings.max_tokens is not None:
            request_params["max_tokens"] = model_settings.max_tokens
            
        # Add thinking parameter for GLM-4.5
        if "glm-4.5" in self.model:
            request_params["thinking"] = {
                "type": "enabled",
            }
        
        # Make the API call
        response = self._client.chat.completions.create(**request_params)
        
        # Process the response
        return self._process_response(response)

    def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
    ) -> AsyncIterator[TResponseStreamEvent]:
        """Stream a response from the ZhipuAI model."""
        # Prepare messages
        messages = self._prepare_messages(system_instructions, input)
        
        # Prepare tools
        tools_for_api = self._prepare_tools(tools, handoffs)
        
        # Prepare request parameters
        request_params = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        
        # Add tools if available
        if tools_for_api:
            request_params["tools"] = tools_for_api
            request_params["tool_choice"] = "auto"
            
        # Add model settings
        if model_settings.temperature is not None:
            request_params["temperature"] = model_settings.temperature
        if model_settings.max_tokens is not None:
            request_params["max_tokens"] = model_settings.max_tokens
            
        # Add thinking parameter for GLM-4.5
        if "glm-4.5" in self.model:
            request_params["thinking"] = {
                "type": "enabled",
            }
        
        # Make the API call
        response = self._client.chat.completions.create(**request_params)
        
        # Process the streaming response
        return self._process_streaming_response(response)

    def _prepare_messages(self, system_instructions: str | None, input: str | list[TResponseInputItem]) -> list[dict]:
        """Prepare messages for the API call using the proven OpenAI converter."""
        # Import the converter from OpenAI model
        from .openai_chatcompletions import _Converter
        
        # Create a converter instance
        converter = _Converter()
        
        # Use the proven items_to_messages method
        messages = converter.items_to_messages(input)
        
        # Add system message if provided
        if system_instructions:
            # Check if we already have a system message
            has_system = any(msg.get("role") == "system" for msg in messages)
            if not has_system:
                messages.insert(0, {
                    "role": "system",
                    "content": system_instructions
                })
        
        # Convert from OpenAI format to simple dict format for ZhipuAI
        simple_messages = []
        for msg in messages:
            simple_msg = {
                "role": msg["role"],
                "content": msg.get("content")
            }
            
            # Handle tool calls
            if "tool_calls" in msg and msg["tool_calls"]:
                simple_msg["tool_calls"] = []
                for tool_call in msg["tool_calls"]:
                    simple_msg["tool_calls"].append({
                        "id": tool_call["id"],
                        "type": tool_call["type"],
                        "function": {
                            "name": tool_call["function"]["name"],
                            "arguments": tool_call["function"]["arguments"]
                        }
                    })
            
            # Handle tool call ID for tool messages
            if "tool_call_id" in msg:
                simple_msg["tool_call_id"] = msg["tool_call_id"]
            
            simple_messages.append(simple_msg)
        
        return simple_messages

    def _prepare_tools(self, tools: list[Tool], handoffs: list[Handoff]) -> list[dict]:
        """Prepare tools for the API call."""
        tools_for_api = []
        
        # Add regular tools
        for tool in tools:
            if hasattr(tool, 'name') and hasattr(tool, 'description') and hasattr(tool, 'params_json_schema'):
                tools_for_api.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.params_json_schema,
                    }
                })
        
        # Add handoff tools
        for handoff in handoffs:
            tools_for_api.append({
                "type": "function",
                "function": {
                    "name": handoff.tool_name,
                    "description": handoff.tool_description,
                    "parameters": handoff.input_json_schema,
                }
            })
        
        return tools_for_api

    def _process_response(self, response) -> ModelResponse:
        """Process the API response."""
        # Extract content and tool calls
        choice = response.choices[0]
        message = choice.message
        
        # Prepare output items
        output_items = []
        
        # Add text content
        if message.content:
            # For GLM-4.5 with thinking enabled, the content might be a list with different types
            if isinstance(message.content, list):
                # Handle the case where content is a list of dicts (like with thinking enabled)
                for item in message.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        output_items.append({
                            "type": "output_text",
                            "text": item.get("text", "")
                        })
                    elif isinstance(item, str):
                        # Handle the case where some items in the list are plain strings
                        output_items.append({
                            "type": "output_text",
                            "text": item
                        })
            else:
                # Standard case - content is a string
                # For GLM-4.5, we need to create the proper structure
                if "glm-4.5" in self.model:
                    output_items.append({
                        "type": "output_text",
                        "text": message.content
                    })
                else:
                    # For other models, create a proper content structure
                    output_items.append({
                        "type": "message",
                        "content": [
                            {
                                "type": "text",
                                "text": message.content
                            }
                        ]
                    })
        
        # Add tool calls if present
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                output_items.append({
                    "type": "function_call",
                    "call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                })
        
        # Extract usage information
        usage = response.usage if hasattr(response, 'usage') else None
        input_tokens = usage.prompt_tokens if usage and hasattr(usage, 'prompt_tokens') else 0
        output_tokens = usage.completion_tokens if usage and hasattr(usage, 'completion_tokens') else 0
        
        # Create proper Usage object instead of dict
        from ..usage import Usage
        usage_obj = Usage(
            requests=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ) if usage else None
        
        return ModelResponse(
            output=output_items,
            usage=usage_obj,
            referenceable_id=None,
        )

    async def _process_streaming_response(self, response) -> AsyncIterator[TResponseStreamEvent]:
        """Process the streaming API response."""
        # Import required response event types
        from openai.types.responses import (
            ResponseTextDeltaEvent,
            ResponseFunctionCallArgumentsDeltaEvent,
            ResponseCompletedEvent,
            ResponseCreatedEvent,
        )
        from openai.types import Response
        import time
        
        # Yield response created event first
        dummy_response = Response(
            id="zhipuai_response",
            created_at=int(time.time()),
            model=self.model,
            object="response",
            output=[],
        )
        yield ResponseCreatedEvent(
            type="response.created",
            response=dummy_response,
        )
        
        async for chunk in response:
            # Process each chunk and yield appropriate events
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                
                # Yield text delta events
                if delta.content:
                    yield ResponseTextDeltaEvent(
                        type="response.output_text.delta",
                        delta=delta.content,
                        item_id="zhipuai_response",
                        output_index=0,
                        content_index=0,
                    )
                
                # Yield tool call events if present
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    for i, tool_call in enumerate(delta.tool_calls):
                        yield ResponseFunctionCallArgumentsDeltaEvent(
                            type="response.function_call_arguments.delta",
                            delta=tool_call.function.arguments if hasattr(tool_call.function, 'arguments') else "",
                            item_id="zhipuai_response",
                            output_index=0,
                        )
        
        # Yield completion event
        yield ResponseCompletedEvent(
            type="response.completed",
            response=dummy_response,
        )