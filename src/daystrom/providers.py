import logging
import os
from dataclasses import dataclass
from enum import Enum
from functools import cache

import httpx

log = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    name: str
    display_name: str
    input_cost: float | None
    """cost per million tokens"""
    output_cost: float | None
    """cost per million tokens"""
    context_limit: int | None
    output_limit: int | None


class ProviderMetadata:
    name: str
    display_name: str
    base_url: str | None
    env_names: list[str]

    def __init__(
        self,
        name: str,
        display_name: str,
        base_url: str | None = None,
    ) -> None:
        self.name = name
        self.display_name = display_name
        self.base_url = base_url
        self.models: dict[str, ModelMetadata] = {}
        self.env_names = []

        base_data = self.get_data()
        if base_data:
            provider_data = base_data.get(self.name)
            if provider_data:
                self.env_names = provider_data.get("env", [])
                for model, metadata in provider_data["models"].items():
                    cost = metadata.get("cost")
                    limit = metadata.get("limit")
                    model_metadata = ModelMetadata(
                        name=metadata.get("id"),
                        display_name=metadata.get("name"),
                        input_cost=cost.get("input") if cost else None,
                        output_cost=cost.get("output") if cost else None,
                        context_limit=limit.get("context") if limit else None,
                        output_limit=limit.get("output") if limit else None,
                    )
                    self.models[model] = model_metadata

    @classmethod
    @cache
    def get_data(cls) -> dict | None:
        metadata_api_url = "https://models.dev/api.json"
        data = None
        try:
            response = httpx.get(metadata_api_url, timeout=3.0)
            response.raise_for_status()
            data = response.json()
        except Exception:
            err = "Failed to pull model metadata from models.dev. Proceeding without..."
            log.error(err)
        return data

    def get_api_key(self):
        api_key = None
        for env_name in self.env_names:
            api_key = os.getenv(env_name)
            if api_key:
                break
        return api_key


class Provider(Enum):
    OPENAI = ProviderMetadata(
        name="openai",
        display_name="Open AI",
        base_url="https://api.openai.com/v1",
    )
    OPENROUTER = ProviderMetadata(
        name="openrouter",
        display_name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
    )
