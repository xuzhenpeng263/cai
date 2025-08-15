from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .gemini_model import GeminiModel
from .interface import ModelProvider

if TYPE_CHECKING:
    from .interface import Model

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


class GeminiProvider(ModelProvider):
    """Google Gemini模型提供者."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
    ) -> None:
        """创建新的Gemini提供者.

        Args:
            api_key: Google AI API密钥。如果未提供，将使用GOOGLE_API_KEY环境变量。
        """
        self._api_key = api_key or GOOGLE_API_KEY

    def get_model(self, model_name: str | None) -> Model:
        """获取Gemini模型实例.
        
        Args:
            model_name: 模型名称。默认为"gemini-2.5-flash"。
                        支持的模型: "gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"
        
        Returns:
            Gemini模型实例。
        """
        if model_name is None:
            model_name = "gemini-2.5-flash"

        return GeminiModel(
            model=model_name,
            api_key=self._api_key,
            agent_name="AI Assistant"
        )