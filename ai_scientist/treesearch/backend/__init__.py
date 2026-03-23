from . import backend_openai
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md
from ai_scientist.llm.client import max_output_token_limit

def get_ai_client(model: str, **model_kwargs):
    """
    Get the appropriate AI client based on the model string.

    Args:
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        **model_kwargs: Additional keyword arguments for model configuration.
    Returns:
        An instance of the appropriate AI client.
    """
    return backend_openai.get_ai_client(model=model, **model_kwargs)

def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    fallback_model: str | None = None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI format)
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        fallback_model (str | None, optional): Optional explicit backup model on the same backend route.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
        "fallback_model": fallback_model,
        "max_tokens": max_tokens or max_output_token_limit(),
    }

    output, req_time, in_tok_count, out_tok_count, info = backend_openai.query(
        system_message=compile_prompt_to_md(system_message) if system_message else None,
        user_message=compile_prompt_to_md(user_message) if user_message else None,
        func_spec=func_spec,
        **model_kwargs,
    )

    return output
