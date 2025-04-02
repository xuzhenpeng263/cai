from __future__ import annotations

import dataclasses
import json
import time
import os
import litellm

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast, overload
from cai.util import get_ollama_api_base, fix_message_list
from wasabi import color

from openai import NOT_GIVEN, AsyncOpenAI, AsyncStream, NotGiven
from openai.types import ChatModel
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.completion_usage import CompletionUsage
from openai.types.responses import (
    EasyInputMessageParam,
    Response,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFileSearchToolCallParam,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseInputContentParam,
    ResponseInputImageParam,
    ResponseInputTextParam,
    ResponseOutputItem,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputMessageParam,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseRefusalDeltaEvent,
    ResponseTextDeltaEvent,
    ResponseUsage,
)
from openai.types.responses.response_input_param import FunctionCallOutput, ItemReference, Message
from openai.types.responses.response_usage import OutputTokensDetails

# Create custom InputTokensDetails class since it's not available in current OpenAI version
from openai._models import BaseModel
class InputTokensDetails(BaseModel):
    prompt_tokens: int
    """The number of prompt tokens."""
    cached_tokens: int = 0
    """The number of cached tokens."""

from .. import _debug
from ..agent_output import AgentOutputSchema
from ..exceptions import AgentsException, UserError
from ..handoffs import Handoff
from ..items import ModelResponse, TResponseInputItem, TResponseOutputItem, TResponseStreamEvent
from ..logger import logger
from ..tool import FunctionTool, Tool
from ..tracing import generation_span
from ..tracing.span_data import GenerationSpanData
from ..tracing.spans import Span
from ..usage import Usage
from ..version import __version__
from .fake_id import FAKE_RESPONSES_ID
from .interface import Model, ModelTracing

if TYPE_CHECKING:
    from ..model_settings import ModelSettings


# Suppress debug info from litellm
litellm.suppress_debug_info = True

_USER_AGENT = f"Agents/Python {__version__}"
_HEADERS = {"User-Agent": _USER_AGENT}


@dataclass
class _StreamingState:
    started: bool = False
    text_content_index_and_output: tuple[int, ResponseOutputText] | None = None
    refusal_content_index_and_output: tuple[int, ResponseOutputRefusal] | None = None
    function_calls: dict[int, ResponseFunctionToolCall] = field(default_factory=dict)


class OpenAIChatCompletionsModel(Model):
    def __init__(
        self,
        model: str | ChatModel,
        openai_client: AsyncOpenAI,
    ) -> None:
        self.model = model
        self._client = openai_client
        # Check if we're using OLLAMA models
        self.is_ollama = os.getenv('OLLAMA') is not None and os.getenv('OLLAMA').lower() != 'false'

    def _non_null_or_not_given(self, value: Any) -> Any:
        return value if value is not None else NOT_GIVEN

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
        with generation_span(
            model=str(self.model),
            model_config=dataclasses.asdict(model_settings)
            | {"base_url": str(self._client.base_url)},
            disabled=tracing.is_disabled(),
        ) as span_generation:
            response = await self._fetch_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                span_generation,
                tracing,
                stream=False,
            )

            if _debug.DONT_LOG_MODEL_DATA:
                logger.debug("Received model response")
            else:
                logger.debug(
                    f"LLM resp:\n{json.dumps(response.choices[0].message.model_dump(), indent=2)}\n"
                )

            usage = (
                Usage(
                    requests=1,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
                if response.usage
                else Usage()
            )
            if tracing.include_data():
                span_generation.span_data.output = [response.choices[0].message.model_dump()]
            span_generation.span_data.usage = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            }

            items = _Converter.message_to_output_items(response.choices[0].message)

            return ModelResponse(
                output=items,
                usage=usage,
                referenceable_id=None,
            )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
    ) -> AsyncIterator[TResponseStreamEvent]:
        """
        Yields a partial message as it is generated, as well as the usage information.
        """
        with generation_span(
            model=str(self.model),
            model_config=dataclasses.asdict(model_settings)
            | {"base_url": str(self._client.base_url)},
            disabled=tracing.is_disabled(),
        ) as span_generation:
            response, stream = await self._fetch_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                span_generation,
                tracing,
                stream=True,
            )

            usage: CompletionUsage | None = None
            state = _StreamingState()

            async for chunk in stream:
                if not state.started:
                    state.started = True
                    yield ResponseCreatedEvent(
                        response=response,
                        type="response.created",
                    )

                # The usage is only available in the last chunk
                if hasattr(chunk, 'usage'):
                    usage = chunk.usage
                # For Ollama/LiteLLM streams that don't have usage attribute
                else:
                    usage = None

                # Handle different stream chunk formats
                if hasattr(chunk, 'choices') and chunk.choices:
                    choices = chunk.choices
                elif hasattr(chunk, 'delta') and chunk.delta:
                    # Some providers might return delta directly
                    choices = [{"delta": chunk.delta}]
                elif isinstance(chunk, dict) and 'choices' in chunk:
                    choices = chunk['choices']
                else:
                    # Skip chunks that don't contain choice data
                    continue
                
                if not choices or len(choices) == 0:
                    continue
                
                # Get the delta content
                delta = None
                if hasattr(choices[0], 'delta'):
                    delta = choices[0].delta
                elif isinstance(choices[0], dict) and 'delta' in choices[0]:
                    delta = choices[0]['delta']
                
                if not delta:
                    continue

                # Handle text
                content = None
                if hasattr(delta, 'content') and delta.content is not None:
                    content = delta.content
                elif isinstance(delta, dict) and 'content' in delta and delta['content'] is not None:
                    content = delta['content']
                
                if content:
                    if not state.text_content_index_and_output:
                        # Initialize a content tracker for streaming text
                        state.text_content_index_and_output = (
                            0 if not state.refusal_content_index_and_output else 1,
                            ResponseOutputText(
                                text="",
                                type="output_text",
                                annotations=[],
                            ),
                        )
                        # Start a new assistant message stream
                        assistant_item = ResponseOutputMessage(
                            id=FAKE_RESPONSES_ID,
                            content=[],
                            role="assistant",
                            type="message",
                            status="in_progress",
                        )
                        # Notify consumers of the start of a new output message + first content part
                        yield ResponseOutputItemAddedEvent(
                            item=assistant_item,
                            output_index=0,
                            type="response.output_item.added",
                        )
                        yield ResponseContentPartAddedEvent(
                            content_index=state.text_content_index_and_output[0],
                            item_id=FAKE_RESPONSES_ID,
                            output_index=0,
                            part=ResponseOutputText(
                                text="",
                                type="output_text",
                                annotations=[],
                            ),
                            type="response.content_part.added",
                        )
                    # Emit the delta for this segment of content
                    yield ResponseTextDeltaEvent(
                        content_index=state.text_content_index_and_output[0],
                        delta=content,
                        item_id=FAKE_RESPONSES_ID,
                        output_index=0,
                        type="response.output_text.delta",
                    )
                    # Accumulate the text into the response part
                    state.text_content_index_and_output[1].text += content

                # Handle refusals (model declines to answer)
                refusal_content = None
                if hasattr(delta, 'refusal') and delta.refusal:
                    refusal_content = delta.refusal
                elif isinstance(delta, dict) and 'refusal' in delta and delta['refusal']:
                    refusal_content = delta['refusal']
                
                if refusal_content:
                    if not state.refusal_content_index_and_output:
                        # Initialize a content tracker for streaming refusal text
                        state.refusal_content_index_and_output = (
                            0 if not state.text_content_index_and_output else 1,
                            ResponseOutputRefusal(refusal="", type="refusal"),
                        )
                        # Start a new assistant message if one doesn't exist yet (in-progress)
                        assistant_item = ResponseOutputMessage(
                            id=FAKE_RESPONSES_ID,
                            content=[],
                            role="assistant",
                            type="message",
                            status="in_progress",
                        )
                        # Notify downstream that assistant message + first content part are starting
                        yield ResponseOutputItemAddedEvent(
                            item=assistant_item,
                            output_index=0,
                            type="response.output_item.added",
                        )
                        yield ResponseContentPartAddedEvent(
                            content_index=state.refusal_content_index_and_output[0],
                            item_id=FAKE_RESPONSES_ID,
                            output_index=0,
                            part=ResponseOutputText(
                                text="",
                                type="output_text",
                                annotations=[],
                            ),
                            type="response.content_part.added",
                        )
                    # Emit the delta for this segment of refusal
                    yield ResponseRefusalDeltaEvent(
                        content_index=state.refusal_content_index_and_output[0],
                        delta=refusal_content,
                        item_id=FAKE_RESPONSES_ID,
                        output_index=0,
                        type="response.refusal.delta",
                    )
                    # Accumulate the refusal string in the output part
                    state.refusal_content_index_and_output[1].refusal += refusal_content

                # Handle tool calls
                # Because we don't know the name of the function until the end of the stream, we'll
                # save everything and yield events at the end
                tool_calls = None
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    tool_calls = delta.tool_calls
                elif isinstance(delta, dict) and 'tool_calls' in delta and delta['tool_calls']:
                    tool_calls = delta['tool_calls']
                
                if tool_calls:
                    for tc_delta in tool_calls:
                        tc_index = tc_delta.index if hasattr(tc_delta, 'index') else tc_delta.get('index', 0)
                        if tc_index not in state.function_calls:
                            state.function_calls[tc_index] = ResponseFunctionToolCall(
                                id=FAKE_RESPONSES_ID,
                                arguments="",
                                name="",
                                type="function_call",
                                call_id="",
                            )
                        
                        tc_function = None
                        if hasattr(tc_delta, 'function'):
                            tc_function = tc_delta.function
                        elif isinstance(tc_delta, dict) and 'function' in tc_delta:
                            tc_function = tc_delta['function']
                            
                        if tc_function:
                            # Handle both object and dict formats
                            args = ""
                            if hasattr(tc_function, 'arguments'):
                                args = tc_function.arguments or ""
                            elif isinstance(tc_function, dict) and 'arguments' in tc_function:
                                args = tc_function.get('arguments', "") or ""
                                
                            name = ""
                            if hasattr(tc_function, 'name'):
                                name = tc_function.name or ""
                            elif isinstance(tc_function, dict) and 'name' in tc_function:
                                name = tc_function.get('name', "") or ""
                                
                            state.function_calls[tc_index].arguments += args
                            state.function_calls[tc_index].name += name
                        
                        # Handle call_id in both formats
                        call_id = ""
                        if hasattr(tc_delta, 'id'):
                            call_id = tc_delta.id or ""
                        elif isinstance(tc_delta, dict) and 'id' in tc_delta:
                            call_id = tc_delta.get('id', "") or ""
                            
                        state.function_calls[tc_index].call_id += call_id

            function_call_starting_index = 0
            if state.text_content_index_and_output:
                function_call_starting_index += 1
                # Send end event for this content part
                yield ResponseContentPartDoneEvent(
                    content_index=state.text_content_index_and_output[0],
                    item_id=FAKE_RESPONSES_ID,
                    output_index=0,
                    part=state.text_content_index_and_output[1],
                    type="response.content_part.done",
                )

            if state.refusal_content_index_and_output:
                function_call_starting_index += 1
                # Send end event for this content part
                yield ResponseContentPartDoneEvent(
                    content_index=state.refusal_content_index_and_output[0],
                    item_id=FAKE_RESPONSES_ID,
                    output_index=0,
                    part=state.refusal_content_index_and_output[1],
                    type="response.content_part.done",
                )

            # Actually send events for the function calls
            for function_call in state.function_calls.values():
                # First, a ResponseOutputItemAdded for the function call
                yield ResponseOutputItemAddedEvent(
                    item=ResponseFunctionToolCall(
                        id=FAKE_RESPONSES_ID,
                        call_id=function_call.call_id,
                        arguments=function_call.arguments,
                        name=function_call.name,
                        type="function_call",
                    ),
                    output_index=function_call_starting_index,
                    type="response.output_item.added",
                )
                # Then, yield the args
                yield ResponseFunctionCallArgumentsDeltaEvent(
                    delta=function_call.arguments,
                    item_id=FAKE_RESPONSES_ID,
                    output_index=function_call_starting_index,
                    type="response.function_call_arguments.delta",
                )
                # Finally, the ResponseOutputItemDone
                yield ResponseOutputItemDoneEvent(
                    item=ResponseFunctionToolCall(
                        id=FAKE_RESPONSES_ID,
                        call_id=function_call.call_id,
                        arguments=function_call.arguments,
                        name=function_call.name,
                        type="function_call",
                    ),
                    output_index=function_call_starting_index,
                    type="response.output_item.done",
                )

            # Finally, send the Response completed event
            outputs: list[ResponseOutputItem] = []
            if state.text_content_index_and_output or state.refusal_content_index_and_output:
                assistant_msg = ResponseOutputMessage(
                    id=FAKE_RESPONSES_ID,
                    content=[],
                    role="assistant",
                    type="message",
                    status="completed",
                )
                if state.text_content_index_and_output:
                    assistant_msg.content.append(state.text_content_index_and_output[1])
                if state.refusal_content_index_and_output:
                    assistant_msg.content.append(state.refusal_content_index_and_output[1])
                outputs.append(assistant_msg)

                # send a ResponseOutputItemDone for the assistant message
                yield ResponseOutputItemDoneEvent(
                    item=assistant_msg,
                    output_index=0,
                    type="response.output_item.done",
                )

            for function_call in state.function_calls.values():
                outputs.append(function_call)

            final_response = response.model_copy()
            final_response.output = outputs

            final_response.usage = (
                ResponseUsage(
                    input_tokens=usage.prompt_tokens if usage and hasattr(usage, 'prompt_tokens') else 0,
                    output_tokens=usage.completion_tokens if usage and hasattr(usage, 'completion_tokens') else 0,
                    total_tokens=usage.total_tokens if usage and hasattr(usage, 'total_tokens') else 0,
                    output_tokens_details=OutputTokensDetails(
                        reasoning_tokens=usage.completion_tokens_details.reasoning_tokens
                        if usage and hasattr(usage, 'completion_tokens_details') 
                        and usage.completion_tokens_details
                        and hasattr(usage.completion_tokens_details, 'reasoning_tokens')
                        and usage.completion_tokens_details.reasoning_tokens
                        else 0
                    ),
                    input_tokens_details={
                        "prompt_tokens": usage.prompt_tokens if usage and hasattr(usage, 'prompt_tokens') else 0,
                        "cached_tokens": usage.prompt_tokens_details.cached_tokens
                        if usage and hasattr(usage, 'prompt_tokens_details')
                        and usage.prompt_tokens_details
                        and hasattr(usage.prompt_tokens_details, 'cached_tokens')
                        and usage.prompt_tokens_details.cached_tokens
                        else 0
                    },
                )
                if usage is not None
                else ResponseUsage(
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                    input_tokens_details={"prompt_tokens": 0, "cached_tokens": 0}
                )
            )

            yield ResponseCompletedEvent(
                response=final_response,
                type="response.completed",
            )
            if tracing.include_data():
                span_generation.span_data.output = [final_response.model_dump()]

            if usage:
                span_generation.span_data.usage = {
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                }
            else:
                span_generation.span_data.usage = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                }

    @overload
    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        span: Span[GenerationSpanData],
        tracing: ModelTracing,
        stream: Literal[True],
    ) -> tuple[Response, AsyncStream[ChatCompletionChunk]]: ...

    @overload
    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        span: Span[GenerationSpanData],
        tracing: ModelTracing,
        stream: Literal[False],
    ) -> ChatCompletion: ...

    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        span: Span[GenerationSpanData],
        tracing: ModelTracing,
        stream: bool = False,
    ) -> ChatCompletion | tuple[Response, AsyncStream[ChatCompletionChunk]]:

        # start by re-fetching self.is_ollama
        self.is_ollama = os.getenv('OLLAMA') is not None and os.getenv('OLLAMA').lower() == 'true'

        converted_messages = _Converter.items_to_messages(input)

        if system_instructions:
            converted_messages.insert(
                0,
                {
                    "content": system_instructions,
                    "role": "system",
                },
            )
        if tracing.include_data():
            span.span_data.input = converted_messages

        parallel_tool_calls = (
            True if model_settings.parallel_tool_calls and tools and len(tools) > 0 else NOT_GIVEN
        )
        tool_choice = _Converter.convert_tool_choice(model_settings.tool_choice)
        response_format = _Converter.convert_response_format(output_schema)
        converted_tools = [ToolConverter.to_openai(tool) for tool in tools] if tools else []

        for handoff in handoffs:
            converted_tools.append(ToolConverter.convert_handoff_tool(handoff))

        if _debug.DONT_LOG_MODEL_DATA:
            logger.debug("Calling LLM")
        else:
            logger.debug(
                f"{json.dumps(converted_messages, indent=2)}\n"
                f"Tools:\n{json.dumps(converted_tools, indent=2)}\n"
                f"Stream: {stream}\n"
                f"Tool choice: {tool_choice}\n"
                f"Response format: {response_format}\n"
                f"Using OLLAMA: {self.is_ollama}\n"
            )

        # Match the behavior of Responses where store is True when not given
        store = model_settings.store if model_settings.store is not None else True

        # Prepare kwargs for the API call
        kwargs = {
            "model": self.model,
            "messages": converted_messages,
            "tools": converted_tools or NOT_GIVEN,
            "temperature": self._non_null_or_not_given(model_settings.temperature),
            "top_p": self._non_null_or_not_given(model_settings.top_p),
            "frequency_penalty": self._non_null_or_not_given(model_settings.frequency_penalty),
            "presence_penalty": self._non_null_or_not_given(model_settings.presence_penalty),
            "max_tokens": self._non_null_or_not_given(model_settings.max_tokens),
            "tool_choice": tool_choice,
            "response_format": response_format,
            "parallel_tool_calls": parallel_tool_calls,
            "stream": stream,
            "stream_options": {"include_usage": True} if stream else NOT_GIVEN,
            "store": store,
            "extra_headers": _HEADERS,
        }

        # Model adjustments
        if any(x in self.model for x in ["claude"]):
            litellm.drop_params = True

            # Error encountered: Error code: 400 - {'error': {'code': 'invalid_request_error', 
            #     'message': "'tool_choice' is only allowed when 'tools' are specified", 
            #     'type': 'invalid_request_error', 'param': None}}
            #
            # if has no tools, remove tool_choice
            if not converted_tools:
                kwargs.pop("tool_choice", None)
                        
            # BadRequestError encountered: litellm.BadRequestError: AnthropicException - 
            # b'{"type":"error","error":
            #     {"type":"invalid_request_error","message":"store: Extra inputs are not permitted"}}'
            #
            kwargs.pop("store", None)
            
            # Filter out NotGiven values to avoid JSON serialization issues
            filtered_kwargs = {}
            for key, value in kwargs.items():
                if value is not NOT_GIVEN:
                    filtered_kwargs[key] = value
            kwargs = filtered_kwargs

        try:
            if self.is_ollama:
                return await self._fetch_response_litellm_ollama(kwargs, model_settings, tool_choice, stream, parallel_tool_calls)
            else:
                return await self._fetch_response_litellm_openai(kwargs, model_settings, tool_choice, stream, parallel_tool_calls)

                # ret = await self._get_client().chat.completions.create(**kwargs)

                # if isinstance(ret, ChatCompletion):
                #     return ret

                # response = Response(
                #     id=FAKE_RESPONSES_ID,
                #     created_at=time.time(),
                #     model=self.model,
                #     object="response",
                #     output=[],
                #     tool_choice=cast(Literal["auto", "required", "none"], tool_choice)
                #     if tool_choice != NOT_GIVEN
                #     else "auto",
                #     top_p=model_settings.top_p,
                #     temperature=model_settings.temperature,
                #     tools=[],
                #     parallel_tool_calls=parallel_tool_calls or False,
                # )
                # return response, ret
                
        except litellm.exceptions.BadRequestError as e:
            print(color("BadRequestError encountered: " + str(e), fg="yellow"))
            if "LLM Provider NOT provided" in str(e):
                # Create a copy of params to avoid overwriting the original
                # ones
                try:
                    return await self._fetch_response_litellm_ollama(kwargs, model_settings, tool_choice, stream, parallel_tool_calls)
                except litellm.exceptions.BadRequestError as e:  # pylint: disable=W0621,C0301 # noqa: E501
                    #
                    # CTRL-C handler for ollama models
                    #
                    if "invalid message content type" in str(e):
                        kwargs["messages"] = fix_message_list(
                            kwargs["messages"])
                        return await self._fetch_response_litellm_ollama(kwargs, model_settings, tool_choice, stream, parallel_tool_calls)
                    else:
                        raise e

            elif ("An assistant message with 'tool_calls'" in str(e) or
                "`tool_use` blocks must be followed by a user message with `tool_result`" in str(e)):  # noqa: E501 # pylint: disable=C0301
                print(f"Error: {str(e)}")
                # NOTE: EDGE CASE: Report Agent CTRL C error
                #
                # This fix CTRL-C error when message list is incomplete
                # When a tool is not finished but the LLM generates a tool call
                kwargs["messages"] = fix_message_list(kwargs["messages"])
                return await self._fetch_response_litellm_openai(kwargs, model_settings, tool_choice, stream, parallel_tool_calls)

            # this captures an error related to the fact
            # that the messages list contains an empty
            # content position
            elif "expected a string, got null" in str(e):
                print(f"Error: {str(e)}")
                # Fix for null content in messages
                kwargs["messages"] = [
                    msg if msg.get("content") is not None else
                    {**msg, "content": ""} for msg in kwargs["messages"]
                ]
                return await self._fetch_response_litellm_openai(kwargs, model_settings, tool_choice, stream, parallel_tool_calls)

            # Handle Anthropic error for empty text content blocks
            elif ("text content blocks must be non-empty" in str(e) or
                "cache_control cannot be set for empty text blocks" in str(e)):  # noqa
                print(f"Error: {str(e)}")
                # Fix for empty content in messages for Anthropic models
                kwargs["messages"] = [
                    msg if msg.get("content") not in [None, ""] else
                    {
                        **msg,
                        "content": "Empty content block"
                    } for msg in kwargs["messages"]
                ]
                return await self._fetch_response_litellm_openai(kwargs, model_settings, tool_choice, stream, parallel_tool_calls)
            else:
                raise e
        except litellm.exceptions.RateLimitError as e:
            print("Rate Limit Error:" + str(e))
            # Try to extract retry delay from error response or use default
            retry_delay = 60  # Default delay in seconds
            try:
                # Extract the JSON part from the error message
                json_str = str(e.message).split('VertexAIException - ')[-1]
                error_details = json.loads(json_str)

                retry_info = next(
                    (detail for detail in error_details.get('error', {}).get('details', [])
                        if detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo'),
                    None
                )
                if retry_info and 'retryDelay' in retry_info:
                    retry_delay = int(retry_info['retryDelay'].rstrip('s'))
            except Exception as parse_error:
                print(f"Could not parse retry delay, using default: {parse_error}")

            print(f"Waiting {retry_delay} seconds before retrying...")
            time.sleep(retry_delay)

        # fall back to ollama if openai API fails
        except Exception as e:  # pylint: disable=W0718
            print(color("Error encountered: " + str(e), fg="yellow"))
            try:
                return await self._fetch_response_litellm_ollama(kwargs, model_settings, tool_choice, stream, parallel_tool_calls)
            except Exception as execp:  # pylint: disable=W0718
                print("Error: " + str(execp))
                return None

    async def _fetch_response_litellm_openai(
        self,
        kwargs: dict,
        model_settings: ModelSettings,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven,
        stream: bool,
        parallel_tool_calls: bool
    ) -> ChatCompletion | tuple[Response, AsyncStream[ChatCompletionChunk]]:
        """Handle standard LiteLLM API calls for OpenAI and compatible models."""
        if stream:
            # Standard LiteLLM handling for streaming
            ret = litellm.completion(**kwargs)
            stream_obj = await litellm.acompletion(**kwargs)

            response = Response(
                id=FAKE_RESPONSES_ID,
                created_at=time.time(),
                model=self.model,
                object="response",
                output=[],
                tool_choice="auto" if tool_choice is None or tool_choice == NOT_GIVEN else cast(Literal["auto", "required", "none"], tool_choice),
                top_p=model_settings.top_p,
                temperature=model_settings.temperature,
                tools=[],
                parallel_tool_calls=parallel_tool_calls or False,
            )
            return response, stream_obj
        else:
            # Standard OpenAI handling for non-streaming
            ret = litellm.completion(**kwargs)
            return ret

    async def _fetch_response_litellm_ollama(
        self,
        kwargs: dict,
        model_settings: ModelSettings,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven,
        stream: bool,
        parallel_tool_calls: bool
    ) -> ChatCompletion | tuple[Response, AsyncStream[ChatCompletionChunk]]:
        # Filter out parameters not supported by Ollama
        ollama_supported_params = {
            "model": kwargs["model"],
            "messages": kwargs["messages"],
            "temperature": kwargs["temperature"] if kwargs["temperature"] is not NOT_GIVEN else None,
            "top_p": kwargs["top_p"] if kwargs["top_p"] is not NOT_GIVEN else None,
            "max_tokens": kwargs["max_tokens"] if kwargs["max_tokens"] is not NOT_GIVEN else None,
            "stream": kwargs["stream"],
            "extra_headers": kwargs["extra_headers"]
        }

        # Modify the messages to remove system message for Ollama
        if ollama_supported_params["messages"] and ollama_supported_params["messages"][0].get("role") == "system":
            # Extract the system message
            system_content = ollama_supported_params["messages"][0].get("content", "")
            # Remove it from the messages
            ollama_supported_params["messages"] = ollama_supported_params["messages"][1:]
            # If there are user messages, prepend system to first user
            if ollama_supported_params["messages"] and ollama_supported_params["messages"][0].get("role") == "user":
                # Prepend the system instruction to the first user message, with a separator
                user_content = ollama_supported_params["messages"][0].get("content", "")
                if isinstance(user_content, str):
                    ollama_supported_params["messages"][0]["content"] = f"System: {system_content}\n\nUser: {user_content}"

        # Remove None values
        ollama_kwargs = {k: v for k, v in ollama_supported_params.items() if v is not None}

        if stream:
            # For streaming with Ollama, we need to create a Response object first
            response = Response(
                id=FAKE_RESPONSES_ID,
                created_at=time.time(),
                model=self.model,
                object="response",
                output=[],
                tool_choice="auto" if tool_choice is None or tool_choice == NOT_GIVEN else cast(Literal["auto", "required", "none"], tool_choice),
                top_p=model_settings.top_p,
                temperature=model_settings.temperature,
                tools=[],
                parallel_tool_calls=parallel_tool_calls or False,
            )
            # Get the streaming object
            stream_obj = await litellm.acompletion(
                **ollama_kwargs,
                api_base=get_ollama_api_base().rstrip('/v1'),
                custom_llm_provider="ollama"
            )
            return response, stream_obj
        else:
            # Non-streaming mode
            ret = litellm.completion(
                **ollama_kwargs,
                api_base=get_ollama_api_base().rstrip('/v1'),
                custom_llm_provider="ollama"
            )
            return ret

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI()
        return self._client


class _Converter:
    @classmethod
    def convert_tool_choice(
        cls, tool_choice: Literal["auto", "required", "none"] | str | None
    ) -> ChatCompletionToolChoiceOptionParam | NotGiven:
        if tool_choice is None:
            return "auto"
        elif tool_choice == "auto":
            return "auto"
        elif tool_choice == "required":
            return "required"
        elif tool_choice == "none":
            return "none"
        else:
            return {
                "type": "function",
                "function": {
                    "name": tool_choice,
                },
            }

    @classmethod
    def convert_response_format(
        cls, final_output_schema: AgentOutputSchema | None
    ) -> ResponseFormat | NotGiven:
        if not final_output_schema or final_output_schema.is_plain_text():
            return None

        return {
            "type": "json_schema",
            "json_schema": {
                "name": "final_output",
                "strict": final_output_schema.strict_json_schema,
                "schema": final_output_schema.json_schema(),
            },
        }

    @classmethod
    def message_to_output_items(cls, message: ChatCompletionMessage) -> list[TResponseOutputItem]:
        items: list[TResponseOutputItem] = []

        message_item = ResponseOutputMessage(
            id=FAKE_RESPONSES_ID,
            content=[],
            role="assistant",
            type="message",
            status="completed",
        )
        if message.content:
            message_item.content.append(
                ResponseOutputText(text=message.content, type="output_text", annotations=[])
            )
        if hasattr(message, 'refusal') and message.refusal:
            message_item.content.append(
                ResponseOutputRefusal(refusal=message.refusal, type="refusal")
            )
        if hasattr(message, 'audio') and message.audio:
            raise AgentsException("Audio is not currently supported")

        if message_item.content:
            items.append(message_item)

        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                items.append(
                    ResponseFunctionToolCall(
                        id=FAKE_RESPONSES_ID,
                        call_id=tool_call.id,
                        arguments=tool_call.function.arguments,
                        name=tool_call.function.name,
                        type="function_call",
                    )
                )

        return items

    @classmethod
    def maybe_easy_input_message(cls, item: Any) -> EasyInputMessageParam | None:
        if not isinstance(item, dict):
            return None

        keys = item.keys()
        # EasyInputMessageParam only has these two keys
        if keys != {"content", "role"}:
            return None

        role = item.get("role", None)
        if role not in ("user", "assistant", "system", "developer"):
            return None

        if "content" not in item:
            return None

        return cast(EasyInputMessageParam, item)

    @classmethod
    def maybe_input_message(cls, item: Any) -> Message | None:
        if (
            isinstance(item, dict)
            and item.get("type") == "message"
            and item.get("role")
            in (
                "user",
                "system",
                "developer",
            )
        ):
            return cast(Message, item)

        return None

    @classmethod
    def maybe_file_search_call(cls, item: Any) -> ResponseFileSearchToolCallParam | None:
        if isinstance(item, dict) and item.get("type") == "file_search_call":
            return cast(ResponseFileSearchToolCallParam, item)
        return None

    @classmethod
    def maybe_function_tool_call(cls, item: Any) -> ResponseFunctionToolCallParam | None:
        if isinstance(item, dict) and item.get("type") == "function_call":
            return cast(ResponseFunctionToolCallParam, item)
        return None

    @classmethod
    def maybe_function_tool_call_output(
        cls,
        item: Any,
    ) -> FunctionCallOutput | None:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            return cast(FunctionCallOutput, item)
        return None

    @classmethod
    def maybe_item_reference(cls, item: Any) -> ItemReference | None:
        if isinstance(item, dict) and item.get("type") == "item_reference":
            return cast(ItemReference, item)
        return None

    @classmethod
    def maybe_response_output_message(cls, item: Any) -> ResponseOutputMessageParam | None:
        # ResponseOutputMessage is only used for messages with role assistant
        if (
            isinstance(item, dict)
            and item.get("type") == "message"
            and item.get("role") == "assistant"
        ):
            return cast(ResponseOutputMessageParam, item)
        return None

    @classmethod
    def extract_text_content(
        cls, content: str | Iterable[ResponseInputContentParam]
    ) -> str | list[ChatCompletionContentPartTextParam]:
        all_content = cls.extract_all_content(content)
        if isinstance(all_content, str):
            return all_content
        out: list[ChatCompletionContentPartTextParam] = []
        for c in all_content:
            if c.get("type") == "text":
                out.append(cast(ChatCompletionContentPartTextParam, c))
        return out

    @classmethod
    def extract_all_content(
        cls, content: str | Iterable[ResponseInputContentParam]
    ) -> str | list[ChatCompletionContentPartParam]:
        if isinstance(content, str):
            return content
        out: list[ChatCompletionContentPartParam] = []

        for c in content:
            if isinstance(c, dict) and c.get("type") == "input_text":
                casted_text_param = cast(ResponseInputTextParam, c)
                out.append(
                    ChatCompletionContentPartTextParam(
                        type="text",
                        text=casted_text_param["text"],
                    )
                )
            elif isinstance(c, dict) and c.get("type") == "input_image":
                casted_image_param = cast(ResponseInputImageParam, c)
                if "image_url" not in casted_image_param or not casted_image_param["image_url"]:
                    raise UserError(
                        f"Only image URLs are supported for input_image {casted_image_param}"
                    )
                out.append(
                    ChatCompletionContentPartImageParam(
                        type="image_url",
                        image_url={
                            "url": casted_image_param["image_url"],
                            "detail": casted_image_param["detail"],
                        },
                    )
                )
            elif isinstance(c, dict) and c.get("type") == "input_file":
                raise UserError(f"File uploads are not supported for chat completions {c}")
            else:
                raise UserError(f"Unknown content: {c}")
        return out

    @classmethod
    def items_to_messages(
        cls,
        items: str | Iterable[TResponseInputItem],
    ) -> list[ChatCompletionMessageParam]:
        """
        Convert a sequence of 'Item' objects into a list of ChatCompletionMessageParam.

        Rules:
        - EasyInputMessage or InputMessage (role=user) => ChatCompletionUserMessageParam
        - EasyInputMessage or InputMessage (role=system) => ChatCompletionSystemMessageParam
        - EasyInputMessage or InputMessage (role=developer) => ChatCompletionDeveloperMessageParam
        - InputMessage (role=assistant) => Start or flush a ChatCompletionAssistantMessageParam
        - response_output_message => Also produces/flushes a ChatCompletionAssistantMessageParam
        - tool calls get attached to the *current* assistant message, or create one if none.
        - tool outputs => ChatCompletionToolMessageParam
        """

        if isinstance(items, str):
            return [
                ChatCompletionUserMessageParam(
                    role="user",
                    content=items,
                )
            ]

        result: list[ChatCompletionMessageParam] = []
        current_assistant_msg: ChatCompletionAssistantMessageParam | None = None

        def flush_assistant_message() -> None:
            nonlocal current_assistant_msg
            if current_assistant_msg is not None:
                # The API doesn't support empty arrays for tool_calls
                if not current_assistant_msg.get("tool_calls"):
                    del current_assistant_msg["tool_calls"]
                result.append(current_assistant_msg)
                current_assistant_msg = None

        def ensure_assistant_message() -> ChatCompletionAssistantMessageParam:
            nonlocal current_assistant_msg
            if current_assistant_msg is None:
                current_assistant_msg = ChatCompletionAssistantMessageParam(role="assistant")
                current_assistant_msg["tool_calls"] = []
            return current_assistant_msg

        for item in items:
            # 1) Check easy input message
            if easy_msg := cls.maybe_easy_input_message(item):
                role = easy_msg["role"]
                content = easy_msg["content"]

                if role == "user":
                    flush_assistant_message()
                    msg_user: ChatCompletionUserMessageParam = {
                        "role": "user",
                        "content": cls.extract_all_content(content),
                    }
                    result.append(msg_user)
                elif role == "system":
                    flush_assistant_message()
                    msg_system: ChatCompletionSystemMessageParam = {
                        "role": "system",
                        "content": cls.extract_text_content(content),
                    }
                    result.append(msg_system)
                elif role == "developer":
                    flush_assistant_message()
                    msg_developer: ChatCompletionDeveloperMessageParam = {
                        "role": "developer",
                        "content": cls.extract_text_content(content),
                    }
                    result.append(msg_developer)
                elif role == "assistant":
                    flush_assistant_message()
                    msg_assistant: ChatCompletionAssistantMessageParam = {
                        "role": "assistant",
                        "content": cls.extract_text_content(content),
                    }
                    result.append(msg_assistant)
                else:
                    raise UserError(f"Unexpected role in easy_input_message: {role}")

            # 2) Check input message
            elif in_msg := cls.maybe_input_message(item):
                role = in_msg["role"]
                content = in_msg["content"]
                flush_assistant_message()

                if role == "user":
                    msg_user = {
                        "role": "user",
                        "content": cls.extract_all_content(content),
                    }
                    result.append(msg_user)
                elif role == "system":
                    msg_system = {
                        "role": "system",
                        "content": cls.extract_text_content(content),
                    }
                    result.append(msg_system)
                elif role == "developer":
                    msg_developer = {
                        "role": "developer",
                        "content": cls.extract_text_content(content),
                    }
                    result.append(msg_developer)
                else:
                    raise UserError(f"Unexpected role in input_message: {role}")

            # 3) response output message => assistant
            elif resp_msg := cls.maybe_response_output_message(item):
                flush_assistant_message()
                new_asst = ChatCompletionAssistantMessageParam(role="assistant")
                contents = resp_msg["content"]

                text_segments = []
                for c in contents:
                    if c["type"] == "output_text":
                        text_segments.append(c["text"])
                    elif c["type"] == "refusal":
                        new_asst["refusal"] = c["refusal"]
                    elif c["type"] == "output_audio":
                        # Can't handle this, b/c chat completions expects an ID which we dont have
                        raise UserError(
                            f"Only audio IDs are supported for chat completions, but got: {c}"
                        )
                    else:
                        raise UserError(f"Unknown content type in ResponseOutputMessage: {c}")

                if text_segments:
                    combined = "\n".join(text_segments)
                    new_asst["content"] = combined

                new_asst["tool_calls"] = []
                current_assistant_msg = new_asst

            # 4) function/file-search calls => attach to assistant
            elif file_search := cls.maybe_file_search_call(item):
                asst = ensure_assistant_message()
                tool_calls = list(asst.get("tool_calls", []))
                new_tool_call = ChatCompletionMessageToolCallParam(
                    id=file_search["id"],
                    type="function",
                    function={
                        "name": "file_search_call",
                        "arguments": json.dumps(
                            {
                                "queries": file_search.get("queries", []),
                                "status": file_search.get("status"),
                            }
                        ),
                    },
                )
                tool_calls.append(new_tool_call)
                asst["tool_calls"] = tool_calls

            elif func_call := cls.maybe_function_tool_call(item):
                asst = ensure_assistant_message()
                tool_calls = list(asst.get("tool_calls", []))
                new_tool_call = ChatCompletionMessageToolCallParam(
                    id=func_call["call_id"],
                    type="function",
                    function={
                        "name": func_call["name"],
                        "arguments": func_call["arguments"],
                    },
                )
                tool_calls.append(new_tool_call)
                asst["tool_calls"] = tool_calls
            # 5) function call output => tool message
            elif func_output := cls.maybe_function_tool_call_output(item):
                flush_assistant_message()
                msg: ChatCompletionToolMessageParam = {
                    "role": "tool",
                    "tool_call_id": func_output["call_id"],
                    "content": func_output["output"],
                }
                result.append(msg)

            # 6) item reference => handle or raise
            elif item_ref := cls.maybe_item_reference(item):
                raise UserError(
                    f"Encountered an item_reference, which is not supported: {item_ref}"
                )

            # 7) If we haven't recognized it => fail or ignore
            else:
                raise UserError(f"Unhandled item type or structure: {item}")

        flush_assistant_message()
        return result


class ToolConverter:
    @classmethod
    def to_openai(cls, tool: Tool) -> ChatCompletionToolParam:
        if isinstance(tool, FunctionTool):
            return {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.params_json_schema,
                },
            }

        raise UserError(
            f"Hosted tools are not supported with the ChatCompletions API. FGot tool type: "
            f"{type(tool)}, tool: {tool}"
        )

    @classmethod
    def convert_handoff_tool(cls, handoff: Handoff[Any]) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": handoff.tool_name,
                "description": handoff.tool_description,
                "parameters": handoff.input_json_schema,
            },
        }
