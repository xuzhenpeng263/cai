from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Dict, List, Union

from google import genai
from google.genai import types

from ..agent_output import AgentOutputSchema
from ..handoffs import Handoff
from ..items import ModelResponse, TResponseInputItem, TResponseStreamEvent
from ..usage import Usage
from openai.types.responses import ResponseOutputMessage, ResponseOutputText
from ..tool import Tool
from .interface import Model, ModelTracing

if TYPE_CHECKING:
    from ..model_settings import ModelSettings


class GeminiModel(Model):
    """Gemini模型实现."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        agent_name: str = "AI Assistant",
    ) -> None:
        """初始化Gemini模型.

        Args:
            model: 模型名称，如 "gemini-2.5-flash"
            api_key: Google AI API密钥
            agent_name: 代理名称
        """
        self._model = model
        self._api_key = api_key
        self._agent_name = agent_name
        self._client = genai.Client(api_key=api_key)

    def _convert_messages_to_gemini_format(
        self, input_items: str | list[TResponseInputItem]
    ) -> str:
        """将输入转换为Gemini可接受的格式."""
        if isinstance(input_items, str):
            return input_items
        
        # 提取文本内容
        content_parts = []
        for item in input_items:
            if isinstance(item, dict):
                if item.get("role") == "user":
                    if isinstance(item.get("content"), str):
                        content_parts.append(item["content"])
                    elif isinstance(item.get("content"), list):
                        for content_item in item["content"]:
                            if isinstance(content_item, dict) and content_item.get("type") == "text":
                                content_parts.append(content_item.get("text", ""))
        
        return "\n".join(content_parts) if content_parts else ""

    def _convert_gemini_response_to_model_response(
        self, response: Any
    ) -> ModelResponse:
        """将Gemini响应转换为ModelResponse格式."""
        content = response.text if hasattr(response, 'text') else str(response)
        
        # 创建ResponseOutputText对象
        text_output = ResponseOutputText(
            type="output_text",
            text=content,
            annotations=[]
        )
        
        # 创建ResponseOutputMessage对象
        message_output = ResponseOutputMessage(
            id="gemini_response",
            role="assistant",
            content=[text_output],
            status="completed",
            type="message"
        )
        
        # 创建Usage对象
        usage_info = Usage(
            requests=1,
            input_tokens=getattr(response, 'usage', {}).get('prompt_tokens', 0),
            output_tokens=getattr(response, 'usage', {}).get('completion_tokens', 0),
            total_tokens=getattr(response, 'usage', {}).get('prompt_tokens', 0) + getattr(response, 'usage', {}).get('completion_tokens', 0)
        )
        
        return ModelResponse(
            output=[message_output],
            usage=usage_info,
            referenceable_id=None
        )

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
        """获取Gemini模型响应."""
        # 构建提示
        prompt_parts = []
        if system_instructions:
            prompt_parts.append(f"系统指令: {system_instructions}")
        
        user_content = self._convert_messages_to_gemini_format(input)
        if user_content:
            prompt_parts.append(f"用户输入: {user_content}")
        
        prompt = "\n\n".join(prompt_parts)
        
        # 配置生成参数
        config = types.GenerateContentConfig(
            temperature=getattr(model_settings, 'temperature', 0.7),
            max_output_tokens=getattr(model_settings, 'max_tokens', 4096),
        )
        
        # 生成响应
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config
        )
        
        return self._convert_gemini_response_to_model_response(response)

    def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
    ) -> Any:
        """Gemini流式响应暂不支持，返回普通响应."""
        # Gemini API的流式支持较为复杂，这里先实现基础版本
        async def _stream_wrapper():
            response = await self.get_response(
                system_instructions, input, model_settings, tools, output_schema, handoffs, tracing
            )
            yield {
                "type": "text",
                "text": response.content[0]["text"] if response.content else "",
                "model": self._model,
            }
        
        return _stream_wrapper()