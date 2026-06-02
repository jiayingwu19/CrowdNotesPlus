"""High-level orchestration helpers for common CrowdNotesPlus workflows."""

from __future__ import annotations

import os

from .config import NotesConfig
from .generation import step1_check_rag_relevance, step2_run_notes_generate, step2_run_notes_generate_batch, step3_check_expression_correctness
from .rag import run_notes_rag_from_unified, run_notes_rag_from_unified_web
from .raw_collection import clean_raw_text, collect_raw_unified_v3, recover_failed_with_jina_using_fetch_html, retry_invalid_in_unified
from .utility_selection import run_directly_selection, run_utility_selection_noreplacement


def config_from_env(**overrides) -> NotesConfig:
    """Create a configuration object using provider credentials from environment variables."""
    provider = overrides.pop("provider", os.getenv("CROWDNOTES_PROVIDER", "openai"))
    api_key = overrides.pop("api_key", None)
    if api_key is None:
        api_key = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "gemini": os.getenv("GOOGLE_API_KEY"),
            "openrouter": os.getenv("OPENROUTER_API_KEY"),
        }.get(provider.lower())
    model = overrides.pop("model", os.getenv("CROWDNOTES_MODEL", "gpt-4.1-mini"))
    return NotesConfig(provider=provider, model=model, api_key=api_key, **overrides)


async def run_pipeline(cfg: NotesConfig) -> None:
    """Dispatch a configured workflow by `cfg.mode`."""
    if cfg.mode == "web search":
        await run_directly_selection(cfg)
    elif cfg.mode == "utility":
        await run_utility_selection_noreplacement(cfg)
    elif cfg.mode == "collect":
        await collect_raw_unified_v3(cfg)
    elif cfg.mode == "rag":
        await run_notes_rag_from_unified_web(cfg)
    elif cfg.mode == "generate":
        await step3_check_expression_correctness(cfg)
    elif cfg.mode == "full":
        await collect_raw_unified_v3(cfg)
        await recover_failed_with_jina_using_fetch_html(cfg)
        await clean_raw_text(cfg)
        await run_notes_rag_from_unified(cfg)
        await step1_check_rag_relevance(cfg)
        await step2_run_notes_generate(cfg)
        await step3_check_expression_correctness(cfg)
    else:
        raise ValueError("Unsupported mode. Use one of: web search, utility, collect, rag, generate, full.")
