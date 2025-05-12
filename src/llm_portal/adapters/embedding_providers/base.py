# src/llm_portal/adapters/embedding_providers/base.py
from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    @abstractmethod
    def generate_embeddings(self, text: str, model: str = None) -> List[float]:
        """Generate embeddings for the given text using the specified model"""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this embedding provider"""
        pass

    @property
    @abstractmethod
    def available_models(self) -> List[str]:
        """Return a list of available embedding models for this provider"""
        pass
