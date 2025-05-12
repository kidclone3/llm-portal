import core
from typing import List

from llm_portal.domains import commands


def generate_text_embeddings(command: commands.InputTextCommand, uow: core.UnitOfWork) -> commands.EmbeddingResult:
    """
    Generate text embeddings for the given command.

    Args:
        command (commands.InputTextCommand): The command containing text to embed
        uow (core.UnitOfWork): Unit of work for database operations

    Returns:
        commands.EmbeddingResult: The embedding result containing the vector and metadata
    """
    with uow:
        # Get the text and model from the command
        text = command.text
        embedding_model = command.embedding_model

        # Generate embeddings using the LLM
        embedding_vector = core.llm.generate_embeddings(text, model=embedding_model)
        provider_name = "default_provider"  # This could be obtained from config or model info

        # Create embedding result
        result = commands.EmbeddingResult(
            embedding=embedding_vector,
            dimensions=len(embedding_vector),
            embedding_model=embedding_model,
            provider=provider_name,
        )

        # Store the embeddings in the database
        uow.embeddings_repository.save_embeddings(result)
        
        return result