import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from src.models.embedding_schemas import EmbeddingRequest, EmbeddingResponse
from src.services.embedding_service import EmbeddingService, get_embedding_service


router = APIRouter()


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embedding(
    request: EmbeddingRequest,
    embedding_service: Annotated[EmbeddingService , Depends(get_embedding_service)]
):
    try:
        result = await embedding_service.generate_embedding(
            request.user_text,
            request.model_name
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding")
