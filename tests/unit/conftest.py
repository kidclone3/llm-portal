import pytest
import core
from typing import List, Optional, Any

from llm_portal.adapters.llm_providers import LLMProvider


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
        if hasattr(embedding, 'id') and embedding.id is None:
            embedding.id = self.id_counter
            self.id_counter += 1
            
        self.embeddings.append(embedding)
        return embedding
    
    def get_by_id(self, embedding_id):
        """Retrieve embeddings by ID"""
        for embedding in self.embeddings:
            if hasattr(embedding, 'id') and embedding.id == embedding_id:
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
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @property
    def provider_name(self) -> str:
        return self.name

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

@pytest.fixture(autouse=True)
def mock_llm_provider_factory(monkeypatch):
    def fake_factory(provider_name: str, **kwargs):
        return FakeLLMProvider(provider_name)
    monkeypatch.setattr(
        "llm_portal.adapters.provider_factory.llm_provider_factory",
        fake_factory
    )

@pytest.fixture
def setup_test_environment():
    """
    Configure the test environment by replacing core services with test doubles.
    Restores original implementations after the test.
    """

    # override factory


@pytest.fixture
def in_memory_uow():
    """
    Create an in-memory unit of work for testing.
    """
    return InMemoryUnitOfWork()


@pytest.fixture
def integration_uow():
    """
    Create a real UoW with a test database connection.
    """
    try:
        # Try to create a UoW with an in-memory SQLite database
        from core import create_uow
        uow = create_uow(db_url="sqlite:///:memory:")
        
        # Set up any necessary database schema
        with uow:
            if hasattr(uow, 'setup_database'):
                uow.setup_database()
        
        return uow
    except (ImportError, AttributeError):
        # If core doesn't provide a create_uow function, fall back to in-memory
        pytest.skip("Integration UoW not available - using in-memory UoW instead")
        return InMemoryUnitOfWork()


@pytest.fixture
def stub_embedding_provider():
    """
    Create a stub embedding provider with default dimensions.
    """
    return StubEmbeddingProvider()
