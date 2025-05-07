import os
import logging
from typing import List, Dict, Any
import asyncio
from google.oauth2 import service_account
import vertexai
from vertexai.language_models import TextEmbeddingModel

class VertexAIProvider:
    def __init__(self):
        # Get Google credentials from environment variables
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if not self.project_id:
            logging.warning("GOOGLE_CLOUD_PROJECT environment variable not set")

        # Initialize Vertex AI
        try:
            if credentials_path and os.path.exists(credentials_path):
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path
                )
                vertexai.init(
                    project=self.project_id,
                    location=self.location,
                    credentials=credentials
                )
            else:
                # Use default credentials
                vertexai.init(
                    project=self.project_id,
                    location=self.location
                )
        except Exception as e:
            logging.error(f"Error initializing Vertex AI: {str(e)}")

        # Model configurations
        self.model_configs = {
            "text-embedding-005": {
                "dimensions": 768
            },
            "text-multilingual-embedding-002": {
                "dimensions": 768
            },
            "text-embedding-large-exp-03-07": {
                "dimensions": 768  # Default dimensions
            }
        }

    async def get_embedding(self, text: str, model_name: str) -> List[float]:
        """
        Generate embedding using Vertex AI
        """
        if not self.project_id:
            raise ValueError("Google Cloud project not configured")

        if model_name not in self.model_configs:
            supported_models = ", ".join(self.model_configs.keys())
            raise ValueError(f"Unsupported Vertex AI model: {model_name}. Supported models: {supported_models}")

        try:
            # Initialize the embedding model
            model = TextEmbeddingModel.from_pretrained(model_name)

            # Generate embeddings asynchronously
            embeddings = await model.get_embeddings_async([text])

            # Return the embedding vector
            if embeddings and len(embeddings) > 0:
                return embeddings[0].values
            else:
                raise Exception("No embedding returned from Vertex AI")
        except Exception as e:
            logging.error(f"Vertex AI embedding error: {str(e)}")
            raise Exception(f"Vertex AI embedding failed: {str(e)}")