from typing import List

import core
import pydantic

__all__ = [
    "InputTextCommand",
    "EmbeddingResult",
]

class InputTextCommand(core.Command):
    """
    Input command

    Args:
        text (str): The text to be processed
        provider_name (str): The name of the provider used for embedding
        embedding_model (str): The model used for embedding the text
    """
    text: str
    provider_name: str
    embedding_model: str


class EmbeddingResult(pydantic.BaseModel):
    """
    Embedding result

    Args:
        embedding (List[float]): The embedding vector
        dimensions (int): The number of dimensions in the embedding
        embedding_model (str): The model used for generating the embedding
        provider (str): The provider of the embedding model
    """
    embedding: List[float]
    dimensions: int
    embedding_model: str
    provider: str
