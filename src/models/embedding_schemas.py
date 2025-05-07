from typing import List

from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    user_text: str
    model_name: str

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimensions: int
    model_used: str
    provider: str