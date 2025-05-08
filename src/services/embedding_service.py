import logging

from src.models.embedding_schemas import EmbeddingError, EmbeddingRequest, EmbeddingResponse

# Import providers
# from src.services.providers.openai_provider import OpenAIProvider
from src.services.providers.vertexai_provider import VertexAIProvider


class EmbeddingService:

    # Constants for error messages
    ERROR_UNSUPPORTED_MODEL = "Unsupported model: {model}. Supported models: {supported}"
    ERROR_EMBEDDING_GENERATION = "Failed to generate embedding with {provider}: {error}"

    def __init__(self):
        self.providers = {
            # "openai": OpenAIProvider(),
            "vertexai": VertexAIProvider()
        }

        # Mapping of model names to providers
        self.model_provider_map = {
            # OpenAI models
            # Google/VertexAI models
            **{key: "vertexai" for key in VertexAIProvider().model_configs.keys()},
            # **{key: "openai" for key in OpenAIProvider().model_configs.keys()},
        }

    def _validate_model(self, model_name: str):
        """
        Validate the model name and return the provider name.

        Args:
            model_name: The model to use for embedding

        Returns:
            The provider name for the requested model

        Raises:
            ValueError: If model is not supported
        """
        if model_name not in self.model_provider_map:
            supported_models = ", ".join(self.model_provider_map.keys())
            raise ValueError(
                self.ERROR_UNSUPPORTED_MODEL.format(model=model_name, supported=supported_models)
            )

        return self.model_provider_map[model_name]

    async def generate_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate text embedding using the specified model.

        Args:
            request (EmbeddingRequest): Request object containing text and model name


        Returns:
            Dictionary containing embedding results

        Raises:
            ValueError: If inputs are invalid
            EmbeddingError: If embedding generation fails
        """


        provider_name = self._validate_model(request.model_name)
        provider = self.providers[provider_name]

        try:
            embedding_vector = await provider.get_embedding(request.user_text, request.model_name)

            return EmbeddingResponse(
                embedding=embedding_vector,
                dimensions=len(embedding_vector),
                model_used=request.model_name,
                provider=provider_name,
            )

        except Exception as e:
            error_msg = f"Error in {provider_name} embedding generation: {str(e)}"
            logging.error(error_msg)
            raise EmbeddingError(
                self.ERROR_EMBEDDING_GENERATION.format(provider=provider_name, error=str(e))
            ) from e


# Dependency for FastAPI
def get_embedding_service():
    return EmbeddingService()
