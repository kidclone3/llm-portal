from llm_portal.adapters.llm_providers import VertexAIProvider, LLMProvider


def llm_provider_factory(
    provider_name: str,
    **kwargs
) -> "LLMProvider":
    """
    Factory function to create an instance of a specific LLM provider.

    Args:
        provider_name (str): The name of the LLM provider.
        **kwargs: Additional keyword arguments for provider initialization.

    Returns:
        LLMProvider: An instance of the specified LLM provider.
    """


    providers = {
        "vertexai": VertexAIProvider,
    }

    if provider_name not in providers:
        raise ValueError(f"Unsupported provider: {provider_name}")

    return providers[provider_name](**kwargs)