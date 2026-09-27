from .claude import ClaudeLLM, split_sentences_stream  # noqa: F401


def create_llm(cfg, system_prompt: str):
    return ClaudeLLM(cfg, system_prompt)
