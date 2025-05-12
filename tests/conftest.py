import pytest
import core
from typing import List, Optional, Any


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


class StubEmbeddingProvider:
    """
    A deterministic embedding provider for testing.
    Generates predictable embeddings based on input text.
    """
    def __init__(self, dimensions: int = 10):
        self.dimensions = dimensions
    
    def generate_embeddings(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Generate deterministic embeddings based on input text.
        Empty text returns an empty list.
        Non-empty text returns a list of dimension values following a pattern.
        """
        if not text:
            return []
        
        # Create a deterministic but unique pattern based on text length and content
        seed = sum(ord(c) for c in text[:5]) if text else 0
        
        # Generate embeddings of fixed dimension
        return [
            0.1 * ((i + 1) * (seed % 10 + 1) / 10) 
            for i in range(self.dimensions)
        ]


@pytest.fixture
def setup_test_environment():
    """
    Configure the test environment by replacing core services with test doubles.
    Restores original implementations after the test.
    """
    # Store original implementations
    original_generate_embeddings = None
    if hasattr(core.llm, 'generate_embeddings'):
        original_generate_embeddings = core.llm.generate_embeddings
    
    # Create and set up test implementations
    embedding_provider = StubEmbeddingProvider()
    
    # Replace real implementations with test doubles
    if hasattr(core, 'llm'):
        core.llm.generate_embeddings = embedding_provider.generate_embeddings
    else:
        # If core.llm doesn't exist, create it as a module-like object
        class LLMModule:
            pass
        
        llm_module = LLMModule()
        llm_module.generate_embeddings = embedding_provider.generate_embeddings
        core.llm = llm_module
    
    yield embedding_provider
    
    # Restore original implementations
    if original_generate_embeddings is not None:
        core.llm.generate_embeddings = original_generate_embeddings


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
        
        # Set up any needed database schema
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
