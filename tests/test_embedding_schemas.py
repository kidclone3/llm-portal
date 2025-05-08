from pydantic import ValidationError
import pytest

from src.models.embedding_schemas import EmbeddingRequest, EmbeddingResponse


class TestEmbeddingSchemas:
    def test_valid_embedding_request(self):
        """Test that a valid EmbeddingRequest can be created"""
        request_data = {
            "user_text": "This is a test text",
            "model_name": "text-embedding-ada-002"
        }

        # This should not raise any exception
        request = EmbeddingRequest(**request_data)

        assert request.user_text == "This is a test text"
        assert request.model_name == "text-embedding-ada-002"

    def test_invalid_embedding_request_missing_text(self):
        """Test that EmbeddingRequest raises error when missing user_text"""
        request_data = {
            "model_name": "text-embedding-ada-002"
        }

        with pytest.raises(ValidationError):
            EmbeddingRequest(**request_data)

    def test_invalid_embedding_request_missing_model(self):
        """Test that EmbeddingRequest raises error when missing model_name"""
        request_data = {
            "user_text": "This is a test text"
        }

        with pytest.raises(ValidationError):
            EmbeddingRequest(**request_data)

    def test_valid_embedding_response(self):
        """Test that a valid EmbeddingResponse can be created"""
        response_data = {
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
            "dimensions": 5,
            "model_used": "text-embedding-ada-002",
            "provider": "openai"
        }

        # This should not raise any exception
        response = EmbeddingResponse(**response_data)

        assert response.embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert response.dimensions == 5
        assert response.model_used == "text-embedding-ada-002"
        assert response.provider == "openai"

    def test_invalid_embedding_response_wrong_type(self):
        """Test that EmbeddingResponse raises error with wrong data types"""
        response_data = {
            "embedding": "not a list",  # Should be a list
            "dimensions": 5,
            "model_used": "text-embedding-ada-002",
            "provider": "openai"
        }

        with pytest.raises(ValidationError):
            EmbeddingResponse(**response_data)
