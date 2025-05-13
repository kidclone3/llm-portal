
import pytest

from llm_portal.domains import commands
from llm_portal.service.handlers.command import generate_text_embeddings


class TestCommandHandlers:
    @pytest.fixture(autouse=True)
    def setup_method(self, in_memory_uow):
        self.uow = in_memory_uow

    def test_generate_embedding_returns_embedding_result(self, mock_llm_provider_factory):
        """Test that generate_embedding returns an EmbeddingResult with a proper structure"""
        # Arrange
        test_text = "This is a test text"
        test_model = "fake-model-1"
        command = commands.InputTextCommand(text=test_text, embedding_model=test_model, provider_name="fake-provider")

        # Act
        result = generate_text_embeddings(command, self.uow)

        # Verify the mock was called with the right parameters
        mock_llm_provider_factory.assert_called_once_with("fake-provider")

        # Assert
        assert isinstance(result, commands.EmbeddingResult)

        assert result.provider == "fake-provider"
        assert result.embedding_model == test_model
        assert result.dimensions == 10
        assert result.embedding == [0.1 * len(test_text) for _ in range(10)]



    def test_generate_embedding_with_empty_text(self, mock_llm_provider_factory):
        """Test that generate_embedding handles empty text input properly"""
        # Arrange
        test_text = ""
        test_model = "fake-model-1"
        command = commands.InputTextCommand(text=test_text, embedding_model=test_model, provider_name="fake-provider")

        # Act
        result = generate_text_embeddings(command, self.uow)

        mock_llm_provider_factory.assert_called_once_with("fake-provider")

        # Assert
        assert isinstance(result, commands.EmbeddingResult)
        assert result.embedding == []  # Empty embedding for empty text
        assert result.dimensions == 0

