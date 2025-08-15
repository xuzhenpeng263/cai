from __future__ import annotations

import os
from typing import TYPE_CHECKING

import httpx
from openai import AsyncOpenAI

from .interface import Model, ModelProvider
from .openai_chatcompletions import OpenAIChatCompletionsModel

if TYPE_CHECKING:
    pass

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekProvider(ModelProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Create a new DeepSeek provider.

        Args:
            api_key: The API key to use for the DeepSeek client. If not provided, we will use the
                DEEPSEEK_API_KEY environment variable.
            base_url: The base URL to use for the DeepSeek client. If not provided, we will use the
                default DeepSeek base URL (https://api.deepseek.com).
        """
        self._api_key = api_key or DEEPSEEK_API_KEY
        self._base_url = base_url or DEEPSEEK_BASE_URL
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        """Get or create the DeepSeek client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                http_client=httpx.AsyncClient(),
            )
        return self._client

    def get_model(self, model_name: str | None) -> Model:
        """Get a DeepSeek model by name.
        
        Args:
            model_name: The name of the model to get. Defaults to "deepseek-chat" if not provided.
                        Supported models: "deepseek-chat", "deepseek-reasoner"
        
        Returns:
            The DeepSeek model instance.
        """
        if model_name is None:
            model_name = "deepseek-chat"

        # Create OpenAI client configured for DeepSeek API
        client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            http_client=httpx.AsyncClient(),
        )
        
        return OpenAIChatCompletionsModel(model=model_name, openai_client=client)