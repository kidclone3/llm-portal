import pytest
from unittest.mock import Mock, patch
import core

from llm_portal.domains import commands
from llm_portal.service.handlers.command import text_embedding


@pytest.fixture
def uow():
    """Create a mock unit of work for testing"""
    mock_uow = Mock(spec=core.UnitOfWork)
    mock_uow.__enter__ = Mock(return_value=mock_uow)
    mock_uow.__exit__ = Mock(return_value=None)
    mock_uow.embeddings_repository = Mock()
    return mock_uow


@patch('core.llm.generate_embeddings')
def test_generate_embedding_returns_embedding_result(mock_generate_embeddings, uow):
    """Test that generate_embedding returns an EmbeddingResult with proper structure"""
    # Arrange
    test_text = "This is a test text"
    test_model = "test-embedding-model"
    mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_generate_embeddings.return_value = mock_embedding
    command = commands.InputTextCommand(text=test_text, embedding_model=test_model)

    # Act
    result = text_embedding.generate_text_embeddings(command, uow)

    # Assert
    assert isinstance(result, commands.EmbeddingResult)
    assert result.embedding == mock_embedding
    assert result.dimensions == len(mock_embedding)
    assert result.embedding_model == test_model
    assert result.provider is not None
    mock_generate_embeddings.assert_called_once_with(test_text, model=test_model)
    uow.embeddings_repository.save_embeddings.assert_called_once()


@patch('core.llm.generate_embeddings')
def test_generate_embedding_with_empty_text(mock_generate_embeddings, uow):
    """Test that generate_embedding handles empty text input properly"""
    # Arrange
    test_text = ""
    test_model = "test-embedding-model"
    mock_embedding = []  # Empty embedding for empty text
    mock_generate_embeddings.return_value = mock_embedding
    command = commands.InputTextCommand(text=test_text, embedding_model=test_model)

    # Act
    result = text_embedding.generate_text_embeddings(command, uow)

    # Assert
    assert isinstance(result, commands.EmbeddingResult)
    assert result.embedding == mock_embedding
    assert result.dimensions == 0
    mock_generate_embeddings.assert_called_once_with(test_text, model=test_model)
