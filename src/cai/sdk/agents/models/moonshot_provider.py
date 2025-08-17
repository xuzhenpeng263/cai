from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .interface import ModelProvider
from .moonshot_model import MoonshotModel

if TYPE_CHECKING:
    from .interface import Model

MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY", "")


class MoonshotProvider(ModelProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
    ) -> None:
        """Create a new Moonshot provider.

        Args:
            api_key: The API key to use for the Moonshot client. If not provided, we will use the
                MOONSHOT_API_KEY environment variable.
        """
        self._api_key = api_key or MOONSHOT_API_KEY

    def get_model(self, model_name: str | None) -> Model:
        if model_name is None:
            model_name = "kimi-k2-0711-preview"

        return MoonshotModel(model=model_name, api_key=self._api_key)