# CrowdNotesPlus

## Project layout

```text
CrowdNotesPlus/
├── README.md
└── crowdnotesplus/
    ├── __init__.py
    ├── __main__.py
    ├── cleaning.py
    ├── config.py
    ├── fetching.py
    ├── generation.py
    ├── llm.py
    ├── pipeline.py
    ├── prompts.py
    ├── rag.py
    ├── raw_collection.py
    ├── retrieval.py
    ├── text_utils.py
    └── utility_selection.py
```

## File responsibilities

### `crowdnotesplus/config.py`

Defines `NotesConfig`, the central configuration dataclass used by every workflow step. It contains provider settings, input/output paths, concurrency limits, generation parameters, RAG parameters, retry settings, and batching settings.

### `crowdnotesplus/llm.py`

Contains all asynchronous LLM client adapters:

- `OpenAIClient`
- `AnthropicClient`
- `GeminiClient`
- `VLLMClient`
- `OpenRouterClient`

It also provides `create_llm_client(cfg)`, which builds the correct client from `NotesConfig`.

### `crowdnotesplus/text_utils.py`

Contains reusable low-level helpers:

- URL extraction and URL stripping
- JSONL reading and rewriting
- UTC timestamp creation
- tokenizer loading
- character-based and token-based chunking
- text extraction fallback helpers
- note budget calculation

### `crowdnotesplus/cleaning.py`

Cleans Markdown text returned by Jina Reader. It removes navigation blocks, image-only lines, raw links, Markdown tables, list-like boilerplate, and trailing reference sections.

### `crowdnotesplus/fetching.py`

Handles network fetching and text extraction from URLs. It includes:

- browser-like request headers
- retryable HTTP GET
- binary-content sniffing
- normal HTML extraction
- Jina Reader fallback extraction
- optional Jina rate limiting
- URL-to-text fetch helpers

### `crowdnotesplus/retrieval.py`

Contains embedding-based retrieval logic. `Retriever` embeds chunks with `sentence-transformers` and selects the most relevant snippets per URL using cosine similarity.

### `crowdnotesplus/prompts.py`

Stores prompt builders and shared system prompts for:

- Community Note generation
- RAG relevance checks
- expression correctness checks

### `crowdnotesplus/raw_collection.py`

Collects, retries, and cleans raw URL text. This module owns the workflow steps that produce or update `raw_unified.jsonl`-style files.

Important functions:

- `collect_raw_unified`
- `collect_raw_unified_v2`
- `collect_raw_unified_v3`
- `recover_failed_with_jina_using_fetch_html`
- `retry_invalid_in_unified`
- `clean_raw_text`

### `crowdnotesplus/rag.py`

Builds RAG-ready JSONL records from collected raw text. It maps raw text by item ID, chunks source text, retrieves the best snippets, calculates note budgets, and writes RAG records.

Important functions:

- `run_notes_rag_from_unified`
- `run_notes_rag_from_unified_web`

### `crowdnotesplus/generation.py`

Runs LLM-based generation and validation steps.

Important functions:

- `step1_check_rag_relevance`
- `step2_run_notes_generate`
- `step2_run_notes_generate_batch`
- `step3_check_expression_correctness`

### `crowdnotesplus/utility_selection.py`

Selects useful source URLs from web-search candidates.

Important functions:

- `run_utility_selection_noreplacement`
- `run_utility_selection_noreplacement_seq`
- `run_directly_selection`

### `crowdnotesplus/pipeline.py`

Provides high-level workflow orchestration. Use `config_from_env()` to create a configuration from environment variables, then pass it to `run_pipeline(cfg)`.

### `crowdnotesplus/__main__.py`

Provides a small command-line entry point so the package can be run with:

```bash
python -m crowdnotesplus --mode rag --input-path input.jsonl --raw-unified-path raw_unified.jsonl --rag-output-path rag.jsonl
```

## Environment variables

Set the API key for the provider you want to use:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
export OPENROUTER_API_KEY="your-key"
```

You only need the variable for the provider you choose.

## Basic usage

```python
import asyncio
from crowdnotesplus import NotesConfig
from crowdnotesplus.rag import run_notes_rag_from_unified_web

cfg = NotesConfig(
    provider="openai",
    model="gpt-4.1-mini",
    api_key="your-key",
    input_path="input.jsonl",
    raw_unified_path="raw_unified.jsonl",
    rag_output_path="rag.jsonl",
    tweet_key="tweet_hint",
    note_key="summary",
    id_key="id",
)

asyncio.run(run_notes_rag_from_unified_web(cfg))
```

## Typical workflow

1. Collect raw source text from URLs.
2. Clean raw source text.
3. Build RAG records with retrieved snippets.
4. Check whether snippets are useful.
5. Generate the final note text.
6. Check whether the generated note distorts the sources.

Example:

```python
import asyncio
from crowdnotesplus.pipeline import config_from_env, run_pipeline

cfg = config_from_env(
    mode="full",
    input_path="input.jsonl",
    raw_unified_path="raw_unified.jsonl",
    rag_output_path="rag.jsonl",
    step1_output_path="step1.jsonl",
    step2_output_path="step2.jsonl",
    step3_output_path="step3.jsonl",
    tweet_key="tweet_hint",
    note_key="summary",
    id_key="id",
)

asyncio.run(run_pipeline(cfg))
```

## Input and output expectations

Most workflow steps read and write JSONL files.

A typical input row should contain:

```json
{"id": "example-id", "tweet_hint": "tweet text", "summary": "note text with source URLs"}
```

The exact field names are controlled by `NotesConfig.tweet_key`, `NotesConfig.note_key`, and `NotesConfig.id_key`.

