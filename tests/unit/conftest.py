from typing import List, Optional
from unittest.mock import patch

import core
import pytest
import utils

from llm_portal.adapters.llm_providers import LLMProvider
from llm_portal import bootstrap


class InMemoryEmbeddingsRepository:
    """
    In-memory implementation of an embeddings repository for testing purposes.
    Stores embeddings in a list rather than a database.
    """
    def __init__(self):
        self.embeddings = []
        self.id_counter = 1

    def save_embeddings(self, embedding):
        """Save embeddings to the in-memory store"""
        # If the embedding has no ID, assign one (mimicking database behavior)
        if hasattr(embedding, "id") and embedding.id is None:
            embedding.id = self.id_counter
            self.id_counter += 1

        self.embeddings.append(embedding)
        return embedding

    def get_by_id(self, embedding_id):
        """Retrieve embeddings by ID"""
        for embedding in self.embeddings:
            if hasattr(embedding, "id") and embedding.id == embedding_id:
                return embedding
        return None

    def get_all(self):
        """Get all stored embeddings"""
        return self.embeddings


class InMemoryUnitOfWork(core.UnitOfWork):
    """
    In-memory implementation of UnitOfWork for testing.
    Simulates database operations without requiring a real database.
    """
    def __init__(self):
        super().__init__()
        self.config = utils.get_config()
        self.factory = None
        self.embeddings_repository = InMemoryEmbeddingsRepository()
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.commit()
        else:
            self.rollback()

    def commit(self):
        """Simulate committing changes to a database"""
        self.committed = True

    def rollback(self):
        """Simulate rolling back changes"""
        self.committed = False

    def setup_database(self):
        """No-op for in-memory implementation"""
        pass


class FakeLLMProvider(LLMProvider):
    def __init__(self, provider_name: str):
        super().__init__("fake-provider")

    @property
    def available_models(self) -> List[str]:
        return ["fake-model-1", "fake-model-2"]
    def generate_embeddings(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Generate embeddings using a fake model.
        """
        if not text:
            return []

        # Generate a fake embedding based on the text length
        return [0.1 * len(text) for _ in range(10)]

@pytest.fixture
def mock_llm_provider_factory():
    """Replace the real LLM provider factory with a fake implementation."""
    patcher = patch("llm_portal.service.handlers.command.llm_provider_factory")
    mock_factory = patcher.start()
    mock_factory.side_effect = lambda provider_name, **kwargs: FakeLLMProvider(provider_name)

    # This ensures the patch is stopped even if the test fails
    yield mock_factory

    patcher.stop()


@pytest.fixture
def in_memory_uow():
    """
    Create an in-memory unit of work for testing.
    """
    return InMemoryUnitOfWork()

@pytest.fixture
def fake_message_bus():
    test_dependencies = {
        "uow": InMemoryUnitOfWork(),
        "publisher": None,
    }

    # Patch the dependencies.DEPENDENCIES that's imported in bootstrap.py
    with patch("llm_portal.dependencies.DEPENDENCIES", test_dependencies):
        # Now bootstrap will use our test_dependencies
        return bootstrap

