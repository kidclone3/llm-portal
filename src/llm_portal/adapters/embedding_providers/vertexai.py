from typing import List

from .base import EmbeddingProvider
from google.oauth2 import service_account
import vertexai
from vertexai.language_models import TextEmbeddingModel

class VertexAIEmbeddingProvider(EmbeddingProvider):
    """Vertex AI embedding provider."""

    def __init__(self, project_id: str, credentials_path: str, location: str = "us-central1"):
        super().__init__()
        self.project_id = project_id
        self.location = location
        self.credentials_path = credentials_path
        self._models = {}

    def generate_embeddings(self, text: str, model: str = None) -> List[float]:
        """Generate embeddings for the given text using Vertex AI."""
        from google.cloud import aiplatform

        # Initialize the Vertex AI client
        aiplatform.init(project=self.project_id, location=self.location)

        # Create an embedding model
        if model is None:
            model = "textembedding-gecko"

        embedding_model = aiplatform.Model(model_name=model)

        # Generate embeddings
        embeddings = embedding_model.predict([text])

        return embeddings[0]

    @property
    def provider_name(self) -> str:
        return "vertexai"

    @property
    def available_models(self) -> List[str]:
        return list(self._models.keys())
