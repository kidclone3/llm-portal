import core
from typing import List, Callable, TypeVar

from llm_portal.domains import commands, models
from llm_portal.adapters.provider_factory import llm_provider_factory

TCommand = TypeVar("TCommand", bound=core.Command)
TResult = TypeVar("TResult")

CommandHandler = Callable[[TCommand, core.UnitOfWork], TResult]

def generate_text_embeddings(command: commands.InputTextCommand, uow: core.UnitOfWork):
    """
    Generate text embeddings for the given command.

    Args:
        command (commands.InputTextCommand): The command containing text to embed
        uow (core.UnitOfWork): Unit of work for database operations

    Returns:
        commands.EmbeddingResult: The embedding result containing the vector and metadata
    """
    with uow:
        llm_provider = llm_provider_factory(command.provider_name)

        # Generate embeddings using the LLM
        embedding_vector = llm_provider.generate_embeddings([command.text], command.embedding_model)

        # Create embedding result
        result = models.EmbeddedResult(
            id=command._id,
            text=command.text,
            vector=embedding_vector,
            model=command.embedding_model,
            provider=command.provider_name,
            dimensions=llm_provider.model_dimensions(command.embedding_model),
        )

        uow.repo.add(result)
        uow.commit()

COMMAND_HANDLERS: dict[type[core.Command], CommandHandler] = {
    commands.InputTextCommand: generate_text_embeddings,
}