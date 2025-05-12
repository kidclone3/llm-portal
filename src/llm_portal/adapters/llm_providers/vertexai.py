from typing import List

import utils
from vertexai.language_models import TextEmbeddingModel

from .base import LLMProvider

config = utils.get_config()

class VertexAIProvider(LLMProvider):
    """Vertex AI embedding provider."""

    def __init__(self):
        super().__init__("vertexai")
        self.project_id = config["google"]["vertexai"]["project_id"]
        self.location = config["google"]["vertexai"]["project_location"]
        self.credentials_path = config["google"]["vertexai"]["credentials_path"]
        self._embedding_models = {
            "text-embedding-005": {
                "dimensions": 768
            },
            "text-multilingual-embedding-002": {
                "dimensions": 768
            },
            "text-embedding-large-exp-03-07": {
                "dimensions": 768  # Default dimensions
            }}

    def generate_embeddings(self, text: str, model: str = None) -> List[float]:
        """
        Generates text embeddings using a specified pre-trained text embedding model.

        Args:
            text (str): The input text for which embeddings are to be generated.
            model (str, optional): The name or identifier of the pre-trained model to use.
                If not provided, a default embedding model will be used.

        Returns:
            List[float]: A list of floats representing the embedding vector for the given
            input text.

        Raises:
            Exception: If the embedding process fails or no embeddings are returned by the
            model, an exception is raised with details about the failure.
        """

        self._validate_embedding_model(model)

        try:
            # Initialize the embedding model
            model = TextEmbeddingModel.from_pretrained(model)

            # Generate embeddings asynchronously
            embeddings = model.get_embeddings([text])

            # Return the embedding vector
            if embeddings and len(embeddings) > 0:
                return embeddings[0].values
            else:
                raise Exception("No embedding returned from Vertex AI")
        except Exception as e:
            # logging.error(f"Vertex AI embedding error: {str(e)}")
            raise Exception(f"Vertex AI embedding failed: {str(e)}")

    @property
    def available_models(self) -> List[str]:
        return list(self._embedding_models.keys())
