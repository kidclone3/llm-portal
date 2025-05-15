from typing import List

import utils
from vertexai.language_models import TextEmbeddingModel

from .base import LLMProvider

config = utils.get_config()

class VertexAIProvider(LLMProvider):
    """Vertex AI embedding provider."""

    def __init__(self):
        super().__init__("vertexai")
        assert config.get("vertexai") is not None, "Vertex AI config is missing"
        assert config["vertexai"].get("project_id") is not None, "Project ID is missing"
        assert config["vertexai"].get("project_location") is not None, "Project location is missing"
        assert config["vertexai"].get("credentials_path") is not None, "Credentials path is missing"


        self.project_id = config["vertexai"]["project_id"]
        self.location = config["vertexai"]["project_location"]
        self.credentials_path = config["vertexai"]["credentials_path"]
        # I think this model is configurable
        self._embedding_models = {
            "text-embedding-005": {
                "dimensions": 768
            },
            "text-multilingual-embedding-002": {
                "dimensions": 768
            },
            "text-embedding-large-exp-03-07": {
                "dimensions": 768
            }}

    def generate_embeddings(self, list_texts: list[str], model: str = None) -> List[float]:
        """
        Generates text embeddings using a specified pre-trained text embedding model.

        Args:
            list_texts (list[str]): The list of input texts for which embeddings are to be generated.
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
            embedding_model = TextEmbeddingModel.from_pretrained(model)

            embeddings = embedding_model.get_embeddings(list_texts)

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
