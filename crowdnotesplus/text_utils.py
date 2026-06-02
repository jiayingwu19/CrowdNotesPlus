"""Reusable text, URL, JSONL, and chunking helpers."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

from bs4 import BeautifulSoup
import trafilatura


URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)


def get_tokenizer(tokenizer_backend: str):
    backend = (tokenizer_backend or '').strip().lower()
    if backend.startswith('tiktoken'):
        try:
            import tiktoken
            name = backend.split(':', 1)[1] if ':' in backend else 'cl100k_base'
            enc = tiktoken.get_encoding(name)

            def encode_fn(s: str) -> List[int]:
                return enc.encode(s or '', disallowed_special=())

            def decode_fn(toks: List[int]) -> str:
                return enc.decode(toks or [])
            return (encode_fn, decode_fn, f'tiktoken:{name}')
        except Exception:
            pass
    from transformers import AutoTokenizer
    name = backend.split(':', 1)[1] if ':' in backend else 'gpt2'
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)

    def encode_fn(s: str) -> List[int]:
        return tok.encode(s or '', add_special_tokens=False)

    def decode_fn(toks: List[int]) -> str:
        return tok.decode(toks or [], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return (encode_fn, decode_fn, f'hf:{name}')

def extract_urls(text: str) -> List[str]:
    return URL_RE.findall(text or '')

def strip_urls(text: str) -> str:
    return URL_RE.sub('', text or '').strip()

def fallback_text(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    txt = soup.get_text('\n')
    return re.sub('\\n{2,}', '\n', txt).strip()

def extract_main_text(html: str) -> str:
    txt = trafilatura.extract(html, include_links=False, include_tables=False)
    if txt and len(txt.strip()) > 200:
        return txt.strip()
    return fallback_text(html)

def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    out, i, n = ([], 0, len(text))
    while i < n:
        j = min(n, i + size)
        out.append(text[i:j])
        if j == n:
            break
        i = max(j - overlap, i + 1)
    return out

def chunk_text_by_tokens(text: str, encode, decode, size: int, overlap: int) -> List[str]:
    toks = encode(text or '')
    n = len(toks)
    if n == 0:
        return []
    chunks = []
    start = 0
    while start < n:
        end = min(n, start + size)
        piece = toks[start:end]
        chunks.append(decode(piece))
        if end == n:
            break
        start = max(end - overlap, start + 1)
    return chunks

def read_items_jsonl(path: str) -> List[Dict]:
    items: List[Dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def calc_budget(urls: List[str], mode: str, cost_value: int, min_budget: int) -> int:
    if mode == 'one_char_per_url':
        budget = 280 - len(urls)
    else:
        budget = 280 - cost_value * len(urls)
    return max(min_budget, budget)

def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _atomic_rewrite_jsonl(src_path: str, updater):
    dirname = os.path.dirname(src_path) or '.'
    os.makedirs(dirname, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix='.rewrite_', dir=dirname, text=True)
    os.close(fd)
    try:
        with open(tmp_path, 'w', encoding='utf-8') as out:
            for rec in _iter_jsonl(src_path):
                new_rec = updater(rec)
                if new_rec is None:
                    continue
                out.write(json.dumps(new_rec, ensure_ascii=False) + '\n')
        os.replace(tmp_path, src_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise
