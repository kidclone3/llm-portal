
import pytest
from unittest.mock import patch

# This file contains pytest fixtures that can be reused across test files

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Set up environment variables for all tests"""
    with patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-api-key',
        'GOOGLE_CLOUD_LOCATION': 'us-central1'
    }):
        yield