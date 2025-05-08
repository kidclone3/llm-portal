import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from vertexai.language_models import TextEmbeddingModel

from src.services.providers.vertexai_provider import VertexAIProvider


class TestVertexAIProvider:

    MODEL_NAMES = [
        "text-embedding-005",
        "text-multilingual-embedding-002",
        "text-embedding-large-exp-03-07"
    ]
    def setup_method(self):
        """Set up test fixtures before each test method"""
        from dotenv import load_dotenv
        load_dotenv()

        # Save original environment
        self.original_env = os.environ.copy()

        # Patch vertexai.init to prevent actual initialization
        self.init_patch = patch("vertexai.init")
        self.mock_init = self.init_patch.start()

        # Create provider instance
        self.provider = VertexAIProvider()

        # Mock model configurations
        self.model_configs = {
            "text-embedding-005": {
                "dimensions": 768
            },
            "text-multilingual-embedding-002": {
                "dimensions": 768
            },
            "text-embedding-large-exp-03-07": {
                "dimensions": 768
            }
        }


    def teardown_method(self):
        """Tear down test fixtures after each test method"""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

        # Stop patches
        self.init_patch.stop()


    @pytest.fixture
    def provider(self):
        """Create a VertexAIProvider instance for testing."""
        # Set required environment variables for testing
        os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
        os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

        with patch("vertexai.init"):
            provider = VertexAIProvider()
            # Verify model configs are set
            assert "text-embedding-005" in provider.model_configs
            return provider


    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_name", MODEL_NAMES)
    async def test_get_embedding_successful(self, model_name, provider):
        """Test get_embedding with successful API response."""
        # Test data
        test_text = "This is a test text"
        test_model = model_name
        expected_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Create mock response
        mock_embedding = MagicMock()
        mock_embedding.values = expected_embedding
        mock_embeddings = [mock_embedding]

        # Setup mocks
        mock_model = AsyncMock()
        mock_model.get_embeddings_async.return_value = mock_embeddings

        with patch.object(TextEmbeddingModel, "from_pretrained", return_value=mock_model):
            # Call method
            result = await provider.get_embedding(test_text, test_model)

            # Verify results
            assert result == expected_embedding
            TextEmbeddingModel.from_pretrained.assert_called_once_with(test_model)
            mock_model.get_embeddings_async.assert_called_once_with([test_text])


    @pytest.mark.asyncio
    async def test_get_embedding_no_project_id(self):
        """Test get_embedding with no project ID configured"""
        # Remove project ID
        os.environ.pop("GOOGLE_CLOUD_PROJECT", None)

        # Create a new provider with no project ID
        with patch("vertexai.init"):
            provider = VertexAIProvider()

        # Call and expect exception
        with pytest.raises(ValueError) as exc_info:
            await provider.get_embedding("Test text", "text-embedding-005")

        assert "Google Cloud project not configured" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_embedding_unsupported_model(self):
        """Test get_embedding with unsupported model"""
        with pytest.raises(ValueError) as exc_info:
            await self.provider.get_embedding("Test text", "unsupported-model")

        assert "Unsupported Vertex AI model" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
    @patch("asyncio.get_event_loop")
    async def test_get_embedding_api_error(self, mock_get_loop, mock_from_pretrained):
        """Test handling of API errors"""
        # Setup mock to raise exception
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop
        mock_loop.run_in_executor.return_value = asyncio.Future()
        mock_loop.run_in_executor.return_value.set_exception(Exception("API Error"))

        # Call and expect exception
        with pytest.raises(Exception) as exc_info:
            await self.provider.get_embedding("Test text", "text-embedding-005")

        assert "Vertex AI embedding failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_embedding_async_without_mock(self):
        """Test get_embedding without mocking"""
        # Create a new provider instance
        provider = VertexAIProvider()

        # Call the method and expect it to raise an exception
        result = await provider.get_embedding("Test text", "text-embedding-005")
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 768
