import logging
from typing import Any, Dict

# Import providers
from src.services.providers.openai_provider import OpenAIProvider
from src.services.providers.vertexai_provider import VertexAIProvider


class EmbeddingService:
    def __init__(self):
        self.providers = {
            "openai": OpenAIProvider(),
            "vertexai": VertexAIProvider()
        }

        # Mapping of model names to providers
        self.model_provider_map = {
            # OpenAI models
            **OpenAIProvider().model_configs,
            # Google/VertexAI models
            **VertexAIProvider().model_configs,
        }

    async def generate_embedding(self, text: str, model_name: str) -> Dict[str, Any]:
        """
        Generate text embedding using the specified model
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        if model_name not in self.model_provider_map:
            supported_models = ", ".join(self.model_provider_map.keys())
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {supported_models}")

        provider_name = self.model_provider_map[model_name]
        provider = self.providers[provider_name]

        try:
            embedding_vector = await provider.get_embedding(text, model_name)

            return {
                "embedding": embedding_vector,
                "dimensions": len(embedding_vector),
                "model_used": model_name,
                "provider": provider_name
            }
        except Exception as e:
            logging.error(f"Error in {provider_name} embedding generation: {str(e)}")
            raise Exception(f"Failed to generate embedding with {provider_name}: {str(e)}")


# Dependency for FastAPI
def get_embedding_service():
    return EmbeddingService()
