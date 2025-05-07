import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

# Import the FastAPI app and necessary modules
from src.main import app
from src.controllers.embedding_controller import create_embedding
from src.services.embedding_service import EmbeddingService
from src.models.embedding_schemas import EmbeddingRequest

# Setup test client
client = TestClient(app)

class TestEmbeddingController:

    @pytest.mark.asyncio
    @patch('services.embedding_service.get_embedding_service')
    async def test_create_embedding_success(self, mock_get_embedding_service, monkeypatch):
        """Test successful embedding creation"""
        # Mock the embedding service
        mock_service = AsyncMock(spec=EmbeddingService)
        mock_get_embedding_service.return_value = mock_service

        # Setup the mock response
        mock_result = {
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
            "dimensions": 5,
            "model_used": "text-embedding-ada-002",
            "provider": "openai"
        }
        mock_service.generate_embedding.return_value = mock_result

        # Test request data
        request = EmbeddingRequest(
            user_text="This is a test",
            model_name="text-embedding-ada-002"
        )

        # Call the endpoint directly with dependency override
        response = await create_embedding(request, mock_service)

        # Assert service was called with correct params
        mock_service.generate_embedding.assert_called_once_with(
            "This is a test", "text-embedding-ada-002"
        )

        # Assert response is correct
        assert response == mock_result

    @pytest.mark.asyncio
    @patch('services.embedding_service.get_embedding_service')
    async def test_create_embedding_value_error(self, mock_get_embedding_service):
        """Test handling of ValueError from service"""
        # Mock the embedding service
        mock_service = AsyncMock(spec=EmbeddingService)
        mock_get_embedding_service.return_value = mock_service

        # Setup the mock to raise ValueError
        mock_service.generate_embedding.side_effect = ValueError("Invalid model name")

        # Test request data
        request = EmbeddingRequest(
            user_text="This is a test",
            model_name="invalid-model"
        )

        # Call the endpoint and expect HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await create_embedding(request, mock_service)

        # Assert the correct status code and detail
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "Invalid model name"

    @pytest.mark.asyncio
    @patch('services.embedding_service.get_embedding_service')
    async def test_create_embedding_generic_error(self, mock_get_embedding_service):
        """Test handling of generic Exception from service"""
        # Mock the embedding service
        mock_service = AsyncMock(spec=EmbeddingService)
        mock_get_embedding_service.return_value = mock_service

        # Setup the mock to raise a generic Exception
        mock_service.generate_embedding.side_effect = Exception("API connection failed")

        # Test request data
        request = EmbeddingRequest(
            user_text="This is a test",
            model_name="text-embedding-ada-002"
        )

        # Call the endpoint and expect HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await create_embedding(request, mock_service)

        # Assert the correct status code and detail
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to generate embedding"

    def test_create_embedding_integration(self):
        """Test the API endpoint through the test client"""

        # Use the test client to make a request
        with patch('services.embedding_service.EmbeddingService.generate_embedding') as mock_generate:
            # Mock the service method to avoid actual API calls
            mock_generate.return_value = {
                "embedding": [0.1, 0.2, 0.3],
                "dimensions": 3,
                "model_used": "text-embedding-ada-002",
                "provider": "openai"
            }

            # Make request to the API
            response = client.post(
                "/api/v1/embeddings",
                json={
                    "user_text": "Test text",
                    "model_name": "text-embedding-ada-002"
                }
            )

            # Assert response
            assert response.status_code == 200
            data = response.json()
            assert data["embedding"] == [0.1, 0.2, 0.3]
            assert data["dimensions"] == 3
            assert data["model_used"] == "text-embedding-ada-002"
            assert data["provider"] == "openai"