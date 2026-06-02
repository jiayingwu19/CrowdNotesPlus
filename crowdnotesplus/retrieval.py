"""Embedding-based snippet retrieval for RAG."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import aiohttp
import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .fetching import fetch_html
from .text_utils import chunk_text, chunk_text_by_tokens, extract_main_text, get_tokenizer


class Retriever:

    def __init__(self, model_name: str):
        self.embedder = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def top_k_per_url(self, chunks: List[Dict[str, Any]], query: str, per_url_k: int, max_snippets: int) -> List[Dict[str, Any]]:
        if not chunks:
            return []
        texts = [c['text'] for c in chunks]
        urls = [c['url'] for c in chunks]
        c_vec = self.embed(texts)
        q_vec = self.embed([query])
        sims = cosine_similarity(c_vec, q_vec).reshape(-1)
        by_url: Dict[str, List[int]] = {}
        for i, u in enumerate(urls):
            by_url.setdefault(u, []).append(i)
        selected_idx: List[int] = []
        for u, idxs in by_url.items():
            local_scores = sims[idxs]
            order = np.argsort(-local_scores)[:per_url_k]
            local_sel = [idxs[j] for j in order]
            selected_idx.extend(local_sel)
        selected_idx = sorted(selected_idx, key=lambda i: -sims[i])
        if max_snippets and max_snippets > 0:
            selected_idx = selected_idx[:max_snippets]
        return [chunks[i] for i in selected_idx]

async def gather_chunks_and_filter_urls(urls: List[str], *, http_concurrency: int, http_timeout: int, chunk_size: int, chunk_overlap: int, use_token_chunking: bool=False, tokenizer_backend: str='tiktoken:cl100k_base', token_chunk_size: int=512, token_chunk_overlap: int=64) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    if use_token_chunking:
        encode, decode, tk_name = get_tokenizer(tokenizer_backend)
    else:
        encode = decode = tk_name = None
    connector = aiohttp.TCPConnector(limit=http_concurrency, ssl=False)
    chunks: List[Dict[str, Any]] = []
    valid_urls: List[str] = []
    invalid_urls: List[str] = []
    async with aiohttp.ClientSession(connector=connector) as session:
        htmls = await asyncio.gather(*[fetch_html(session, u, timeout=http_timeout) for u in urls], return_exceptions=True)
    for u, html in zip(urls, htmls):
        if isinstance(html, Exception) or not html:
            invalid_urls.append(u)
            continue
        try:
            text = extract_main_text(html)
            if not text or len(text.strip()) < 200:
                invalid_urls.append(u)
                continue
            if use_token_chunking:
                pieces = chunk_text_by_tokens(text, encode, decode, size=token_chunk_size, overlap=token_chunk_overlap)
            else:
                pieces = chunk_text(text, size=chunk_size, overlap=chunk_overlap)
            if not pieces:
                invalid_urls.append(u)
                continue
            for i, ch in enumerate(pieces):
                chunks.append({'url': u, 'chunk_id': i, 'text': ch})
            valid_urls.append(u)
        except Exception:
            invalid_urls.append(u)
    return (chunks, valid_urls, invalid_urls)
