from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .interface import ModelProvider
from .zhipuai_model import ZhipuAIModel

if TYPE_CHECKING:
    from .interface import Model

ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY", "")


class ZhipuAIProvider(ModelProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
    ) -> None:
        """Create a new ZhipuAI provider.

        Args:
            api_key: The API key to use for the ZhipuAI client. If not provided, we will use the
                ZHIPUAI_API_KEY environment variable.
        """
        self._api_key = api_key or ZHIPUAI_API_KEY

    def get_model(self, model_name: str | None) -> Model:
        if model_name is None:
            model_name = "glm-4.5"

        return ZhipuAIModel(model=model_name, api_key=self._api_key)