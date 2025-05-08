from typing import List

from pydantic import BaseModel, Field, field_validator


class EmbeddingError(Exception):
    """Custom exception for embedding generation errors."""
    pass

class EmbeddingRequest(BaseModel):
    user_text: str = Field(..., description="Text to generate embeddings for")
    model_name: str = Field(..., description="Model to use for embedding generation")

    @field_validator("user_text")
    def validate_text_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Text cannot be empty")
        return value.strip()

    @field_validator("model_name")
    def validate_model_exists(cls, value: str, info) -> str:
        # This is a placeholder for runtime validation
        # Actual validation happens in the service
        return value


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimensions: int
    model_used: str
    provider: str
