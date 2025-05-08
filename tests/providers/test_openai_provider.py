import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.providers.openai_provider import OpenAIProvider


class TestOpenAIProvider:

    def setup_method(self):
        """Set up test fixtures before each test method"""
        # Save original environment
        self.original_env = os.environ.copy()
        # Set API key for testing
        os.environ["OPENAI_API_KEY"] = "test-api-key"

        # Create provider instance
        self.provider = OpenAIProvider()

        # Mock the AsyncOpenAI client
        self.client_mock = MagicMock()
        self.provider.client = self.client_mock

    def teardown_method(self):
        """Tear down test fixtures after each test method"""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

    @pytest.mark.asyncio
    async def test_get_embedding_ada_002(self):
        """Test get_embedding with text-embedding-ada-002 model"""
        # Setup mock response
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Create mock for the embeddings.create method
        embeddings_mock = AsyncMock()
        self.client_mock.embeddings = embeddings_mock

        # Setup the response structure that matches OpenAI's format
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = mock_embedding

        embeddings_mock.create.return_value = mock_response

        # Call the provider method
        result = await self.provider.get_embedding("Test text", "text-embedding-ada-002")

        # Assert API was called correctly
        embeddings_mock.create.assert_called_once_with(
            model="text-embedding-ada-002",
            input="Test text"
        )

        # Assert result matches mock embedding
        assert result == mock_embedding

    @pytest.mark.asyncio
    async def test_get_embedding_no_api_key(self):
        """Test get_embedding with no API key configured"""
        # Remove API key
        os.environ.pop("OPENAI_API_KEY", None)

        # Create a new provider with no API key
        provider = OpenAIProvider()

        # Call and expect exception
        with pytest.raises(ValueError) as exc_info:
            await provider.get_embedding("Test text", "text-embedding-ada-002")

        assert "OpenAI API key not configured" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_embedding_unsupported_model(self):
        """Test get_embedding with unsupported model"""
        with pytest.raises(ValueError) as exc_info:
            await self.provider.get_embedding("Test text", "unsupported-model")

        assert "Unsupported OpenAI model" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_embedding_api_error(self):
        """Test handling of API errors"""
        # Setup mock to raise exception
        embeddings_mock = AsyncMock()
        self.client_mock.embeddings = embeddings_mock
        embeddings_mock.create.side_effect = Exception("API Error")

        # Call and expect exception
        with pytest.raises(Exception) as exc_info:
            await self.provider.get_embedding("Test text", "text-embedding-ada-002")

        assert "OpenAI embedding failed" in str(exc_info.value)
