"""RAG record construction from collected raw source text."""

from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

from tqdm import tqdm

from .config import NotesConfig
from .retrieval import Retriever
from .text_utils import (
    _iter_jsonl,
    calc_budget,
    chunk_text,
    chunk_text_by_tokens,
    extract_urls,
    get_tokenizer,
    read_items_jsonl,
    strip_urls,
)


async def run_notes_rag_from_unified(cfg) -> None:
    assert getattr(cfg, 'raw_unified_path', None), 'cfg.raw_unified_path is not set.'
    os.makedirs(os.path.dirname(cfg.rag_output_path) or '.', exist_ok=True)
    raw_by_id: Dict[Any, Dict[str, str]] = defaultdict(dict)
    for rec in _iter_jsonl(cfg.raw_unified_path):
        if rec.get('status') == 'ok':
            _id = rec.get('id')
            u = rec.get('url')
            rt = (rec.get('raw_text') or '').strip()
            if _id is not None and u and rt:
                raw_by_id[_id][u] = rt
    items = read_items_jsonl(cfg.input_path)
    retriever = Retriever(cfg.embed_model)
    out = open(cfg.rag_output_path, 'w', encoding='utf-8')
    if getattr(cfg, 'use_token_chunking', False):
        encode, decode, _ = get_tokenizer(cfg.tokenizer_backend)
    else:
        encode = decode = None
    total = len(items)
    ok_cnt = 0
    invalid_cnt = 0
    for idx, it in enumerate(tqdm(items), 1):
        note = str(it.get(cfg.note_key, '') or '')
        tweet_hint = str(it.get(cfg.tweet_key, '') or '')
        _id = it.get(cfg.id_key)
        raw_urls = extract_urls(note)
        if not raw_urls:
            record = {'id': _id, 'tweet_hint': tweet_hint, 'original_note': note, 'valid_urls': [], 'budget': cfg.min_budget, 'snippets': [], 'status': 'cannot_extract_valid_urls'}
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            invalid_cnt += 1
            continue
        raw_map_for_id: Dict[str, str] = raw_by_id.get(_id, {})
        valid_urls = [u for u in raw_urls if u in raw_map_for_id]
        invalid_urls = [u for u in raw_urls if u not in raw_map_for_id]
        if not valid_urls:
            record = {'id': _id, 'tweet_hint': tweet_hint, 'original_note': note, 'valid_urls': [], 'invalid_urls': raw_urls, 'budget': cfg.min_budget, 'snippets': [], 'status': 'invalid_no_valid_urls'}
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            invalid_cnt += 1
            continue
        chunks: List[Dict[str, Any]] = []
        for u in valid_urls:
            raw_text = raw_map_for_id[u]
            if getattr(cfg, 'use_token_chunking', False):
                pieces = chunk_text_by_tokens(raw_text, encode, decode, size=cfg.token_chunk_size, overlap=cfg.token_chunk_overlap)
            else:
                pieces = chunk_text(raw_text, size=cfg.chunk_size, overlap=cfg.chunk_overlap)
            for i, ch in enumerate(pieces):
                chunks.append({'url': u, 'chunk_id': i, 'text': ch})
        if not chunks:
            record = {'id': _id, 'tweet_hint': tweet_hint, 'original_note': note, 'valid_urls': [], 'invalid_urls': raw_urls, 'budget': cfg.min_budget, 'snippets': [], 'status': 'invalid_no_valid_urls'}
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            invalid_cnt += 1
            continue
        budget = calc_budget(valid_urls, cfg.url_cost_mode, cfg.url_cost_value, cfg.min_budget)
        query = tweet_hint.strip() or strip_urls(note)
        selected = retriever.top_k_per_url(chunks, query, per_url_k=cfg.per_url_top_k, max_snippets=cfg.max_snippets)
        record = {'id': _id, 'tweet_hint': tweet_hint, 'original_note': note, 'valid_urls': valid_urls, 'invalid_urls': invalid_urls, 'budget': budget, 'snippets': selected, 'status': 'ok'}
        out.write(json.dumps(record, ensure_ascii=False) + '\n')
        ok_cnt += 1
        if getattr(cfg, 'batch_size', None) and getattr(cfg, 'batch_pause_s', 0) > 0:
            if idx % cfg.batch_size == 0:
                await asyncio.sleep(cfg.batch_pause_s)
    out.close()
    print(f'✅ Step2 done. RAG -> {cfg.rag_output_path} (ok={ok_cnt}, invalid={invalid_cnt}, total_items={total})')

async def run_notes_rag_from_unified_web(cfg) -> None:
    assert getattr(cfg, 'raw_unified_path', None), 'cfg.raw_unified_path is not set.'
    os.makedirs(os.path.dirname(cfg.rag_output_path) or '.', exist_ok=True)
    raw_by_id: Dict[Any, Dict[str, str]] = defaultdict(dict)
    ok_records_read = 0
    for rec in _iter_jsonl(cfg.raw_unified_path):
        if rec.get('status') == 'ok':
            _id = rec.get('id')
            u = rec.get('url')
            rt = (rec.get('raw_text') or '').strip()
            if _id is not None and u and rt:
                raw_by_id[_id][u] = rt
                ok_records_read += 1
    items = read_items_jsonl(cfg.input_path)
    total_items = len(items)
    retriever = Retriever(cfg.embed_model)
    use_token_chunking = bool(getattr(cfg, 'use_token_chunking', False))
    if use_token_chunking:
        encode, decode, _ = get_tokenizer(cfg.tokenizer_backend)
    else:
        encode = decode = None
    default_query = getattr(cfg, 'default_query', 'Please provide accurate background information and conclusions based on the source.')
    out = open(cfg.rag_output_path, 'w', encoding='utf-8')
    ok_cnt = 0
    no_raw_cnt = 0
    no_chunks_cnt = 0
    for idx, it in enumerate(tqdm(items), 1):
        _id = it.get(getattr(cfg, 'id_key', 'id'))
        tweet_hint = str(it.get(getattr(cfg, 'tweet_key', 'tweet_hint'), '') or '').strip()
        url_map_for_id: Dict[str, str] = raw_by_id.get(_id, {})
        valid_urls: List[str] = list(url_map_for_id.keys())
        if not valid_urls:
            record = {'id': _id, 'tweet_hint': tweet_hint, 'valid_urls': [], 'budget': getattr(cfg, 'min_budget', 0), 'snippets': [], 'status': 'invalid_no_raw_text'}
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            no_raw_cnt += 1
            continue
        chunks: List[Dict[str, Any]] = []
        for u in valid_urls:
            raw_text = url_map_for_id[u]
            if use_token_chunking:
                pieces = chunk_text_by_tokens(raw_text, encode, decode, size=cfg.token_chunk_size, overlap=cfg.token_chunk_overlap)
            else:
                pieces = chunk_text(raw_text, size=cfg.chunk_size, overlap=cfg.chunk_overlap)
            for i, ch in enumerate(pieces):
                chunks.append({'url': u, 'chunk_id': i, 'text': ch})
        if not chunks:
            record = {'id': _id, 'tweet_hint': tweet_hint, 'valid_urls': valid_urls, 'budget': getattr(cfg, 'min_budget', 0), 'snippets': [], 'status': 'no_chunks'}
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            no_chunks_cnt += 1
            continue
        budget = calc_budget(valid_urls, cfg.url_cost_mode, cfg.url_cost_value, cfg.min_budget)
        query = tweet_hint.strip() if tweet_hint else default_query
        selected = retriever.top_k_per_url(chunks, query, per_url_k=cfg.per_url_top_k, max_snippets=cfg.max_snippets)
        record = {'id': _id, 'tweet_hint': tweet_hint, 'valid_urls': valid_urls, 'budget': budget, 'snippets': selected, 'status': 'ok'}
        out.write(json.dumps(record, ensure_ascii=False) + '\n')
        ok_cnt += 1
    out.close()
    print(f'✅ Step2 done. RAG -> {cfg.rag_output_path} (ok={ok_cnt}, invalid_no_raw_text={no_raw_cnt}, no_chunks={no_chunks_cnt}, total_items={total_items}, ok_records_read_from_unified={ok_records_read})')
