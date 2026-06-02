"""Central configuration models for CrowdNotesPlus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class NotesConfig:
    """Runtime settings shared by collection, retrieval, generation, and selection steps."""

    provider: str
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    port: Optional[int] = None

    input_path: str = ""
    step1_output_path: str = ""
    step2_output_path: str = ""
    step3_output_path: str = ""
    rag_output_path: str = "rag.jsonl"
    raw_unified_path: Optional[str] = None
    web_step1_output_path: str = ""
    web_step2_output_path: str = ""
    web_step3_output_path: str = ""
    web_raw_output_path: str = ""
    failed_unified_path: Optional[str] = None
    utility_output_path: Optional[str] = None

    mode: str = "full"

    tweet_key: str = "tweetText"
    note_key: str = "summary"
    id_key: str = "noteId"

    semaphore_size: int = 50
    max_conn: int = 50
    http_concurrency: int = 8
    http_timeout: int = 25
    samples_concurrency: int = 32

    max_tokens: int = 4096
    temperature: float = 0.2
    top_p: float = 1.0

    url_cost_mode: str = "one_char_per_url"
    url_cost_value: int = 23
    min_budget: int = 60

    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    per_url_top_k: int = 2
    max_snippets: Optional[int] = 12

    use_token_chunking: bool = True
    tokenizer_backend: str = "tiktoken:cl100k_base"
    token_chunk_size: int = 512
    token_chunk_overlap: int = 64
    chunk_size: int = 900
    chunk_overlap: int = 150

    batch_size: Optional[int] = 1000
    batch_pause_s: float = 5.0
    rpm_limit: int = 140

    min_text_chars: int = 20
    max_retries: int = 3
    retry_backoff_s: float = 3.0
    default_query: str = "Please provide accurate background information and conclusions based on the source."
