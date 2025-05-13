# src/llm_portal/adapters/embedding_providers/base.py
from abc import ABC, abstractmethod
from typing import List


class LLMProvider(ABC):

    def __init__(self, provider_name:str):
        self._provider_name = provider_name
        self._embedding_models = {}

    @abstractmethod
    def generate_embeddings(self, text: str, model: str = None) -> List[float]:
        """Generate embeddings for the given text using the specified model"""
        pass

    def _validate_embedding_model(self, model: str) -> None:
        """Validate the model name."""
        if model not in self._embedding_models:
            raise ValueError(f"Model {model} is not supported. Supported models are: {list(self._embedding_models.keys())}")


    @property
    def provider_name(self) -> str:
        """Return the name of this embedding provider"""
        return self._provider_name

    @property
    @abstractmethod
    def available_models(self) -> List[str]:
        """Return a list of available embedding models for this provider"""
        pass

    @provider_name.setter
    def provider_name(self, value):
        self._provider_name = value

    def model_dimensions(self, model_name) -> int:
        return self._embedding_models[model_name]["dimensions"]
