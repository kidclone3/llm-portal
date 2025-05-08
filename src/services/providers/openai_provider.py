import logging
import os
from typing import List

from openai import AsyncOpenAI


class OpenAIProvider:
    def __init__(self):
        # Get API key from environment variable
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            logging.warning("OPENAI_API_KEY environment variable not set")

        self.client = AsyncOpenAI(api_key=self.api_key)

        # Model configurations
        self.model_configs = {
            "text-embedding-ada-002": {
                "dimensions": 1536
            },
            "text-embedding-3-small": {
                "dimensions": 1536
            },
            "text-embedding-3-large": {
                "dimensions": 3072
            }
        }

    async def get_embedding(self, text: str, model_name: str) -> List[float]:
        """
        Generate embedding using OpenAI API
        """
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")

        if model_name not in self.model_configs:
            supported_models = ", ".join(self.model_configs.keys())
            raise ValueError(f"Unsupported OpenAI model: {model_name}. Supported models: {supported_models}")

        try:
            response = await self.client.embeddings.create(
                model=model_name,
                input=text
            )

            # Extract the embedding vector from the response
            embedding_vector = response.data[0].embedding

            return embedding_vector
        except Exception as e:
            logging.error(f"OpenAI embedding error: {str(e)}")
            raise Exception(f"OpenAI embedding failed: {str(e)}")
