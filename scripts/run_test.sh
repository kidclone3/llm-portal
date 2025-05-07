#!/usr/bin/zsh
# Basic Test Command

pytest

# Run tests with detailed output
pytest -v

# Run tests and show coverage report
pytest --cov=. --cov-report=term

# Run tests and generate HTML coverage report
pytest --cov=. --cov-report=html

# Run tests with JUnit XML report for CI systems
pytest --junitxml=test-results.xml

# Generate coverage report with XML format (useful for CI/CD tools)
pytest --cov=. --cov-report=xml:coverage.xml

# Specific commands to test different components
pytest tests/test_embedding_schemas.py
pytest tests/test_embedding_controller.py
pytest tests/test_embedding_service.py
pytest tests/test_openai_provider.py
pytest tests/test_vertexai_provider.py

# Run all tests and generate a detailed coverage report in HTML format
pytest --cov=. --cov-report=html --cov-report=