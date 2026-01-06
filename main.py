import os
import re
import json
import asyncio
import textwrap
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Iterable
from datetime import datetime, timezone
from collections import defaultdict
import time
import tempfile

import aiohttp
import aiofiles
from yarl import URL
from bs4 import BeautifulSoup
import trafilatura
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from httpx import AsyncClient, Limits
from tenacity import retry, stop_after_attempt, wait_exponential
from dataclasses import dataclass
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_async
from url_extract_v3 import fetch_html
from clean_text import clean_jina_markdown
from utility_judgement import run_utility_selection_noreplacement, run_directly_selection

import debugpy
def init_debug():
    debugpy.listen(5680)
    print("🧠 Waiting for debugger to attach on port 5679...")
    debugpy.wait_for_client()
    print("✅ Debugger attached!")

class BaseLLM:
    """Unified asynchronous chat interface."""
    async def aclose(self):
        pass
    async def chat(self, system: str, user: str, max_tokens: int = 8, temperature: float = 0.0, top_p: float = 1.0) -> str:
        raise NotImplementedError

# --- OpenAI (GPT) ---
class OpenAIClient(BaseLLM):
    def __init__(self, api_key: str, model: str, base_url: Optional[str], max_conn: int):
        from openai import AsyncOpenAI
        self.model = model
        self.http = AsyncClient(timeout=30.0, limits=Limits(max_connections=max_conn, max_keepalive_connections=20))
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, http_client=self.http)

    async def aclose(self):
        await self.http.aclose()

    @retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=20))
    async def chat(self, system: str, user: str, max_tokens: int = 8, temperature: float = 0.0, top_p: float = 1.0) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            # max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        # print(resp)
        return (resp.choices[0].message.content or "").strip()

# --- Anthropic (Claude) ---
class AnthropicClient(BaseLLM):
    def __init__(self, api_key: str, model: str, base_url: Optional[str], max_conn: int):
        from anthropic import AsyncAnthropic
        self.model = model
        self.client = AsyncAnthropic(api_key=api_key, base_url=base_url)
        self.http = AsyncClient(timeout=30.0, limits=Limits(max_connections=max_conn, max_keepalive_connections=20))

    async def aclose(self):
        await self.http.aclose()

    @retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=20))
    async def chat(self, system: str, user: str, max_tokens: int = 8, temperature: float = 0.0, top_p: float = 1.0) -> str:
        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            system=system if system else None,
            messages=[{"role": "user", "content": user}],
            thinking={"type": "disabled",} # close thinking
        )
        parts = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "\n".join(parts).strip()

# --- Google (Gemini) ---
class GeminiClient(BaseLLM):
    def __init__(self, api_key: str, model: str, base_url: Optional[str], max_conn: int):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model_name = model
        self.genai = genai
        self.model = genai.GenerativeModel(model)

    async def aclose(self):
        pass

    @retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=20))
    async def chat(self, system: str, user: str, max_tokens: int = 8, temperature: float = 0.0, top_p: float = 1.0) -> str:
        prompt = (system.strip() + "\n\n" + user.strip()).strip()

        def _call_sync():
            resp = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_output_tokens": max_tokens,
                },
            )
            try:
                return (resp.text or "").strip()
            except Exception:
                if hasattr(resp, "candidates") and resp.candidates:
                    parts = []
                    for cand in resp.candidates:
                        try:
                            parts.append(cand.content.parts[0].text)
                        except Exception:
                            pass
                    return "\n".join([p for p in parts if p]).strip()
                return ""
        return await asyncio.to_thread(_call_sync)
    
# --- VLLM ---
class VLLMClient(BaseLLM):
    def __init__(self, port: int):
        from openai import AsyncOpenAI
        openai_api_key = "EMPTY"
        openai_api_base = f"http://localhost:{port}/v1"

        self.client = AsyncOpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=36000.0,
        )

    async def chat(self, system: str, user: str, max_tokens: int = 8, temperature: float = 0.0, top_p: float = 1.0) -> str:
        models = await self.client.models.list()
        model = models.data[0].id
        prompt = (system.strip() + "\n\n" + user.strip()).strip()
        completion = await self.client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt,
            }],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        content = completion.choices[0].message.content
        result = (content or "")
        return result  
    
# --- OpenRouter (xAI Grok) ---
class OpenRouterClient(BaseLLM):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = "https://openrouter.ai/api/v1",
        max_conn: int = 100,
        http_referer: Optional[str] = None,
        x_title: Optional[str] = None,
    ):
        from openai import AsyncOpenAI
        self.model = f"x-ai/{model}"
        self.http = AsyncClient(timeout=30.0, limits=Limits(max_connections=max_conn, max_keepalive_connections=20))
        base_url = "https://openrouter.ai/api/v1"

        default_headers = {}
        if http_referer:
            default_headers["HTTP-Referer"] = http_referer
        if x_title:
            default_headers["X-Title"] = x_title

        self.client = AsyncOpenAI(
            api_key=api_key, 
            base_url=base_url,      
            http_client=self.http,
            default_headers=default_headers or None,
        )

    async def aclose(self):
        await self.http.aclose()

    @retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=20))
    async def chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 8,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return (resp.choices[0].message.content or "").strip()

# ====== Config ======
@dataclass
class NotesConfig:
    # Provider & model
    provider: str
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    port: Optional[int] = None

    # I/O
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

    # mode
    mode: str = "full" 

    # key name
    tweet_key: str = "tweetText"
    note_key: str  = "summary"
    id_key: str    = "noteId"

    # Concurrency & HTTP
    semaphore_size: int = 50
    max_conn: int = 50
    http_concurrency: int = 8
    http_timeout: int = 25

    # Generation Parameters
    max_tokens: int = 4096
    temperature: float = 0.2
    top_p: float = 1.0

    # Budget Rules
    url_cost_mode: str = "one_char_per_url"  # "one_char_per_url" | "x_cost_per_url"
    url_cost_value: int = 23
    min_budget: int = 60

    # RAG
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    per_url_top_k: int = 2
    max_snippets: int = 12

    use_token_chunking: bool = True
    tokenizer_backend: str = "tiktoken:cl100k_base"
    token_chunk_size: int = 512
    token_chunk_overlap: int = 64
    chunk_size: int = 900
    chunk_overlap: int = 150

    batch_size: Optional[int] = 1000
    batch_pause_s: float = 5.0


def get_tokenizer(tokenizer_backend: str):
    backend = (tokenizer_backend or "").strip().lower()

    if backend.startswith("tiktoken"):
        try:
            import tiktoken
            name = backend.split(":", 1)[1] if ":" in backend else "cl100k_base"
            enc = tiktoken.get_encoding(name)

            def encode_fn(s: str) -> List[int]:
                return enc.encode(s or "", disallowed_special=())
            def decode_fn(toks: List[int]) -> str:
                return enc.decode(toks or [])
            return encode_fn, decode_fn, f"tiktoken:{name}"
        except Exception:
            pass 

    from transformers import AutoTokenizer
    name = backend.split(":", 1)[1] if ":" in backend else "gpt2"
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)

    def encode_fn(s: str) -> List[int]:
        return tok.encode(s or "", add_special_tokens=False)
    def decode_fn(toks: List[int]) -> str:
        return tok.decode(toks or [], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return encode_fn, decode_fn, f"hf:{name}"

URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)

def extract_urls(text: str) -> List[str]:
    return URL_RE.findall(text or "")

def strip_urls(text: str) -> str:
    return URL_RE.sub("", text or "").strip()

def fallback_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]): tag.decompose()
    txt = soup.get_text("\n")
    return re.sub(r"\n{2,}", "\n", txt).strip()

def extract_main_text(html: str) -> str:
    txt = trafilatura.extract(html, include_links=False, include_tables=False)
    if txt and len(txt.strip()) > 200:
        return txt.strip()
    return fallback_text(html)

def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    out, i, n = [], 0, len(text)
    while i < n:
        j = min(n, i + size)
        out.append(text[i:j])
        if j == n: break
        i = max(j - overlap, i + 1)
    return out

def chunk_text_by_tokens(
    text: str,
    encode,
    decode,
    size: int,
    overlap: int
) -> List[str]:
    toks = encode(text or "")
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

class Retriever:
    def __init__(self, model_name: str):
        self.embedder = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def top_k_per_url(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        per_url_k: int,
        max_snippets: int
    ) -> List[Dict[str, Any]]:
        if not chunks: return []

        texts = [c["text"] for c in chunks]
        urls  = [c["url"]  for c in chunks]
        c_vec = self.embed(texts)
        q_vec = self.embed([query])
        sims  = cosine_similarity(c_vec, q_vec).reshape(-1)

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

SYSTEM_PROMPT = "Community notes is a collaborative way to add helpful context to posts and keep people better informed. Now you are a highly experienced community note writer."

def build_user_prompt(query: str, snippets: List[Dict[str, Any]], budget_chars: int) -> str:
    joined = "\n\n".join(
        f"[S{i+1}] {s['url']} (chunk {s['chunk_id']})\n{s['text']}"
        for i, s in enumerate(snippets)
    )

    prompt = f"""Task: Write a community note based ONLY on the source snippets below.
Hard constraints:
- The note MUST be in English.
- DO NOT include any URLs in the note.
- The note MUST be a single line (no line breaks, no bullets).
- Note length MUST be ≤ {budget_chars} characters. Do not exceed this budget.
- Be specific, objective, and verifiable.

Tweet:
{query}

Source snippets:
{joined}

Output only the note content. Remember: length ≤ {budget_chars}, no URLs.
"""

    return prompt


def read_items_jsonl(path: str) -> List[Dict]:
    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            items.append(json.loads(line))
    return items

async def gather_chunks_and_filter_urls(
    urls: List[str],
    *,
    http_concurrency: int,
    http_timeout: int,
    chunk_size: int,
    chunk_overlap: int,
    use_token_chunking: bool = False,
    tokenizer_backend: str = "tiktoken:cl100k_base",
    token_chunk_size: int = 512,
    token_chunk_overlap: int = 64,
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    if use_token_chunking:
        encode, decode, tk_name = get_tokenizer(tokenizer_backend)
    else:
        encode = decode = tk_name = None  # type: ignore

    connector = aiohttp.TCPConnector(limit=http_concurrency, ssl=False)
    chunks: List[Dict[str, Any]] = []
    valid_urls: List[str] = []
    invalid_urls: List[str] = []

    async with aiohttp.ClientSession(connector=connector) as session:
        htmls = await asyncio.gather(
            *[fetch_html(session, u, timeout=http_timeout) for u in urls],
            return_exceptions=True
        )

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
                pieces = chunk_text_by_tokens(
                    text, encode, decode,
                    size=token_chunk_size,
                    overlap=token_chunk_overlap
                )
            else:
                pieces = chunk_text(text, size=chunk_size, overlap=chunk_overlap)

            if not pieces:
                invalid_urls.append(u)
                continue

            for i, ch in enumerate(pieces):
                chunks.append({"url": u, "chunk_id": i, "text": ch})
            valid_urls.append(u)

        except Exception:
            invalid_urls.append(u)

    return chunks, valid_urls, invalid_urls

def calc_budget(urls: List[str], mode: str, cost_value: int, min_budget: int) -> int:
    if mode == "one_char_per_url":
        budget = 280 - len(urls)
    else:
        budget = 280 - cost_value * len(urls)
    return max(min_budget, budget)

def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _atomic_rewrite_jsonl(src_path: str, updater):
    dirname = os.path.dirname(src_path) or "."
    os.makedirs(dirname, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".rewrite_", dir=dirname, text=True)
    os.close(fd)

    try:
        with open(tmp_path, "w", encoding="utf-8") as out:
            for rec in _iter_jsonl(src_path):
                new_rec = updater(rec)
                if new_rec is None:
                    continue
                out.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
        os.replace(tmp_path, src_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise

async def collect_raw_unified(cfg) -> None:
    assert getattr(cfg, "raw_unified_path", None), "cfg.raw_unified_path is not set"
    min_text_chars = int(getattr(cfg, "min_text_chars", 20))

    items = read_items_jsonl(cfg.input_path)
    os.makedirs(os.path.dirname(cfg.raw_unified_path) or ".", exist_ok=True)

    with open(cfg.raw_unified_path, "w", encoding="utf-8") as _:
        pass

    out = open(cfg.raw_unified_path, "a", encoding="utf-8")

    connector = aiohttp.TCPConnector(limit=cfg.http_concurrency, ssl=False)
    total = len(items)
    ok_lines = 0
    invalid_lines = 0
    no_url_items = 0

    async with aiohttp.ClientSession(connector=connector) as session:
        for idx, it in enumerate(tqdm(items), 1):
            note = str(it.get(cfg.note_key, "") or "")
            _id  = it.get(cfg.id_key)
            raw_urls: List[str] = extract_urls(note)

            if not raw_urls:
                no_url_items += 1
                continue

            results = await asyncio.gather(
                *[fetch_html(session, u, timeout=cfg.http_timeout) for u in raw_urls],
                return_exceptions=True
            )

            now = _iso_now()
            for u, r in zip(raw_urls, results):
                if isinstance(r, Exception):
                    out.write(json.dumps({
                        "id": _id, "url": u, "status": "invalid",
                        "error": repr(r)[:500],
                        "meta": {"source": "step1", "timestamp": now}
                    }, ensure_ascii=False) + "\n")
                    invalid_lines += 1
                    continue

                txt = (r or "").strip()
                if not txt:
                    out.write(json.dumps({
                        "id": _id, "url": u, "status": "invalid",
                        "error": "empty",
                        "meta": {"source": "step1", "timestamp": now}
                    }, ensure_ascii=False) + "\n")
                    invalid_lines += 1
                    continue

                if len(txt) < min_text_chars:
                    out.write(json.dumps({
                        "id": _id, "url": u, "status": "invalid",
                        "error": f"too_short(<{min_text_chars})",
                        "meta": {"source": "step1", "timestamp": now}
                    }, ensure_ascii=False) + "\n")
                    invalid_lines += 1
                    continue

                out.write(json.dumps({
                    "id": _id, "url": u, "status": "ok",
                    "raw_text": txt,
                    "meta": {"source": "step1", "timestamp": now}
                }, ensure_ascii=False) + "\n")
                ok_lines += 1

            if getattr(cfg, "batch_size", None) and getattr(cfg, "batch_pause_s", 0) > 0:
                if (idx % cfg.batch_size) == 0:
                    await asyncio.sleep(cfg.batch_pause_s)

    out.close()
    print(f"📝 Step1 done. unified -> {cfg.raw_unified_path} (ok={ok_lines}, invalid={invalid_lines}, no_url_items={no_url_items}, total_items={total})")

async def collect_raw_unified_v2(cfg: NotesConfig) -> None:
    assert getattr(cfg, "raw_unified_path", None), "cfg.raw_unified_path is not set"
    min_text_chars = int(getattr(cfg, "min_text_chars", 20))
    max_retries = int(getattr(cfg, "max_retries", 3))
    retry_backoff_s = float(getattr(cfg, "retry_backoff_s", 3.0))
    http_timeout = getattr(cfg, "http_timeout", 20)
    http_concurrency = getattr(cfg, "http_concurrency", 16)

    items = read_items_jsonl(cfg.web_step2_output_path)
    os.makedirs(os.path.dirname(cfg.raw_unified_path) or ".", exist_ok=True)

    with open(cfg.raw_unified_path, "w", encoding="utf-8"):
        pass
    out_ok = open(cfg.raw_unified_path, "a", encoding="utf-8")

    failed_path = getattr(cfg, "failed_unified_path", None)
    if not failed_path:
        base = cfg.raw_unified_path
        if base.endswith(".jsonl"):
            failed_path = base[:-6] + ".failed.jsonl"
        else:
            failed_path = base + ".failed.jsonl"
    os.makedirs(os.path.dirname(failed_path) or ".", exist_ok=True)
    with open(failed_path, "w", encoding="utf-8"):
        pass
    out_failed = open(failed_path, "a", encoding="utf-8")

    total_items = len(items)
    ok_lines = 0             
    failed_samples = 0           
    no_url_items = 0             
    processed_items = 0

    connector = aiohttp.TCPConnector(limit=http_concurrency, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        for idx, it in enumerate(tqdm(items), 1):
            processed_items += 1
            note: str = str(it.get(getattr(cfg, "note_key", "note"), "") or "")
            _id = it.get(getattr(cfg, "id_key", "id"))
            valid_counts = int(it.get("valid_counts", 1))
            if valid_counts <= 0:
                valid_counts = 1

            raw_urls: List[str] = extract_urls(note)

            if not raw_urls:
                no_url_items += 1
                now = _iso_now()
                out_failed.write(json.dumps({
                    "id": _id,
                    "failed_urls": [],
                    "missing": valid_counts,
                    "meta": {"source": "step1", "timestamp": now}
                }, ensure_ascii=False) + "\n")
                failed_samples += 1
                if getattr(cfg, "batch_size", None) and getattr(cfg, "batch_pause_s", 0) > 0:
                    if (idx % cfg.batch_size) == 0:
                        await asyncio.sleep(cfg.batch_pause_s)
                continue

            successes = 0
            failed_urls_this_item: List[str] = []
            now = _iso_now()

            for u in raw_urls:
                if successes >= valid_counts:
                    break

                txt: Optional[str] = None
                last_exception: Optional[Exception] = None
                for attempt in range(1, max_retries + 1):
                    try:
                        r = await fetch_html(session, u, timeout=http_timeout, use_jina=True)
                        txt = (r or "").strip()
                        if not txt:
                            raise ValueError("empty")
                        if len(txt) < min_text_chars:
                            raise ValueError(f"too_short(<{min_text_chars})")
                        filter_list = ["403: Forbidden", "404: Not Found", "Page Not Found", "CAPTCHA", "429: Too Many Requests"]
                        if any(kw in txt for kw in filter_list):
                            raise ValueError("The text contains filtered keywords; retrieval failed.")
                        break
                    except Exception as e:
                        last_exception = e
                        if attempt < max_retries and retry_backoff_s > 0:
                            await asyncio.sleep(retry_backoff_s)

                if txt is not None and len(txt) >= min_text_chars:
                    out_ok.write(json.dumps({
                        "id": _id,
                        "url": u,
                        "status": "ok",
                        "raw_text": txt,
                        "meta": {"source": "step1", "timestamp": now}
                    }, ensure_ascii=False) + "\n")
                    ok_lines += 1
                    successes += 1
                else:
                    failed_urls_this_item.append(u)

            if successes < valid_counts:
                missing = valid_counts - successes
                out_failed.write(json.dumps({
                    "id": _id,
                    "failed_urls": failed_urls_this_item,
                    "missing": missing,
                    "meta": {"source": "step1", "timestamp": now}
                }, ensure_ascii=False) + "\n")
                failed_samples += 1

            if getattr(cfg, "batch_size", None) and getattr(cfg, "batch_pause_s", 0) > 0:
                if (idx % cfg.batch_size) == 0:
                    await asyncio.sleep(cfg.batch_pause_s)

    out_ok.close()
    out_failed.close()

    print(
        "📝 Step1 done. unified -> "
        f"{cfg.raw_unified_path} (ok_urls={ok_lines}, "
        f"failed_samples={failed_samples}, no_url_items={no_url_items}, total_items={total_items})\n"
        f"❗ Failed details -> {failed_path}"
    )

async def collect_raw_unified_v3(cfg: NotesConfig) -> None:
    assert getattr(cfg, "raw_unified_path", None), "cfg.raw_unified_path is not set"

    min_text_chars = int(getattr(cfg, "min_text_chars", 20))
    max_retries = int(getattr(cfg, "max_retries", 3))
    retry_backoff_s = float(getattr(cfg, "retry_backoff_s", 3.0))
    http_timeout = int(getattr(cfg, "http_timeout", 20))
    http_concurrency = int(getattr(cfg, "http_concurrency", 16))
    samples_concurrency = int(getattr(cfg, "samples_concurrency", 32))
    api_key = None

    items = read_items_jsonl(cfg.web_step3_output_path)

    os.makedirs(os.path.dirname(cfg.raw_unified_path) or ".", exist_ok=True)
    failed_path = getattr(cfg, "failed_unified_path", None)
    if not failed_path:
        base = cfg.raw_unified_path
        failed_path = (base[:-6] + ".failed.jsonl") if base.endswith(".jsonl") else (base + ".failed.jsonl")
    os.makedirs(os.path.dirname(failed_path) or ".", exist_ok=True)

    async with aiofiles.open(cfg.raw_unified_path, "w", encoding="utf-8") as f:
        await f.flush()
    async with aiofiles.open(failed_path, "w", encoding="utf-8") as f:
        await f.flush()

    file_lock = asyncio.Lock()

    async def write_ok_line(obj: dict):
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        async with file_lock:
            async with aiofiles.open(cfg.raw_unified_path, "a", encoding="utf-8") as f:
                await f.write(line)

    async def write_fail_line(obj: dict):
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        async with file_lock:
            async with aiofiles.open(failed_path, "a", encoding="utf-8") as f:
                await f.write(line)

    total_items = len(items)
    ok_lines = 0
    failed_samples = 0
    no_url_items = 0
    stats_lock = asyncio.Lock()

    connector = aiohttp.TCPConnector(limit=http_concurrency, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(samples_concurrency)

        async def process_one_sample(it) -> Tuple[int, int, int]:
            note: str = str(it.get(getattr(cfg, "note_key", "note"), "") or "")
            _id = it.get(getattr(cfg, "id_key", "id"))
            valid_counts = int(it.get("valid_counts", 1))
            if valid_counts <= 0:
                valid_counts = 1

            raw_urls: List[str] = extract_urls(note)
            now = _iso_now()

            if not raw_urls:
                await write_fail_line({
                    "id": _id,
                    "failed_urls": [],
                    "missing": valid_counts,
                    "meta": {"source": "step1", "timestamp": now}
                })
                return (0, 1, 1)

            successes = 0
            failed_urls_this_item: List[str] = []

            for u in raw_urls:
                if successes >= valid_counts:
                    break

                txt: Optional[str] = None
                last_exception: Optional[Exception] = None
                filter_list = ["403: Forbidden", "404: Not Found", "Page Not Found", "CAPTCHA", "429: Too Many Requests"]

                for attempt in range(1, max_retries + 1):
                    try:
                        r = await fetch_html(session, u, timeout=http_timeout, api_key=api_key, use_jina=True)
                        txt = (r or "").strip()

                        if not txt:
                            raise ValueError("empty")
                        if len(txt) < min_text_chars:
                            raise ValueError(f"too_short(<{min_text_chars})")

                        if any(kw in txt for kw in filter_list):
                            raise ValueError("The text contains filtered keywords; retrieval failed.")

                        break
                    except Exception as e:
                        last_exception = e
                        if attempt < max_retries and retry_backoff_s > 0:
                            await asyncio.sleep(retry_backoff_s)

                if txt is not None and len(txt) >= min_text_chars and (not any(kw in txt for kw in filter_list)):
                    await write_ok_line({
                        "id": _id,
                        "url": u,
                        "status": "ok",
                        "raw_text": txt,
                        "meta": {"source": "step1", "timestamp": now}
                    })
                    successes += 1
                else:
                    failed_urls_this_item.append(u)

            if successes < valid_counts:
                missing = valid_counts - successes
                await write_fail_line({
                    "id": _id,
                    "failed_urls": failed_urls_this_item,
                    "missing": missing,
                    "meta": {"source": "step1", "timestamp": now}
                })
                return (successes, 1, 0)

            return (successes, 0, 0)

        async def worker(it):
            nonlocal ok_lines, failed_samples, no_url_items
            async with sem:
                add_ok, add_fail, add_no_url = await process_one_sample(it)
                async with stats_lock:
                    ok_lines += add_ok
                    failed_samples += add_fail
                    no_url_items += add_no_url

        tasks = [asyncio.create_task(worker(it)) for it in items]

        with tqdm(total=total_items, desc="Step 1 (Concurrent Crawling)") as pbar:
            for coro in asyncio.as_completed(tasks):
                await coro
                pbar.update(1)

    print(
        "📝 Step1 done. unified -> "
        f"{cfg.raw_unified_path} (ok_urls={ok_lines}, "
        f"failed_samples={failed_samples}, no_url_items={no_url_items}, total_items={total_items})\n"
        f"❗ Failed details -> {failed_path}"
    )

async def recover_failed_with_jina_using_fetch_html(cfg) -> None:
    raw_path = getattr(cfg, "raw_unified_path", None)
    assert raw_path, "cfg.raw_unified_path is not set."

    failed_path = getattr(cfg, "failed_unified_path", None)
    if not failed_path:
        base = raw_path
        failed_path = (base[:-6] + ".failed.jsonl") if base.endswith(".jsonl") else (base + ".failed.jsonl")

    if not os.path.exists(failed_path):
        print(f"No failed files found: {failed_path}")
        return

    min_text_chars = int(getattr(cfg, "min_text_chars", 20))
    max_retries = int(getattr(cfg, "max_retries", 3))
    retry_backoff_s = float(getattr(cfg, "retry_backoff_s", 3.0))
    http_timeout = int(getattr(cfg, "http_timeout", 20))
    http_concurrency = int(getattr(cfg, "http_concurrency", 16))
    samples_concurrency = int(getattr(cfg, "samples_concurrency", 32))
    api_key = ""

    async def _read_failed(path: str) -> List[Dict]:
        items = []
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            async for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    obj.setdefault("failed_urls", [])
                    obj.setdefault("missing", max(1, len(obj["failed_urls"])))
                    items.append(obj)
                except Exception:
                    continue
        return items

    failed_items = await _read_failed(failed_path)
    if not failed_items:
        print(f"{failed_path} is empty")
        return

    file_lock = asyncio.Lock()
    async def write_ok(obj: dict):
        async with file_lock:
            async with aiofiles.open(raw_path, "a", encoding="utf-8") as f:
                await f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    connector = aiohttp.TCPConnector(limit=http_concurrency, ssl=False)
    kept_failed: List[Dict] = []

    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(samples_concurrency)

        async def process_one(it: Dict) -> Optional[Dict]:
            _id = it.get("id")
            failed_urls = list(it.get("failed_urls", []))
            missing = int(it.get("missing", 1))
            if not failed_urls:
                return it

            successes = 0
            remaining = []

            for u in failed_urls:
                if successes >= missing:
                    break
                txt = None
                filter_list = ["403: Forbidden","404: Not Found","Page Not Found","CAPTCHA","429: Too Many Requests"]
                # filter_list = ["403: Forbidden","404: Not Found","Page Not Found","429: Too Many Requests"]
                for attempt in range(1, max_retries + 1):
                    try:
                        r = await fetch_html(session, u, timeout=http_timeout, api_key=api_key, use_jina=True)
                        txt = (r or "").strip()
                        if not txt or len(txt) < min_text_chars:
                            raise ValueError("too_short")
                        if any(kw in txt for kw in filter_list):
                            raise ValueError("blocked")
                        break
                    except Exception:
                        if attempt < max_retries and retry_backoff_s > 0:
                            await asyncio.sleep(retry_backoff_s)

                if txt and len(txt) >= min_text_chars and not any(kw in txt for kw in filter_list):
                    await write_ok({
                        "id": _id,
                        "url": u,
                        "status": "ok",
                        "raw_text": txt,
                        "meta": {"source": "recover", "timestamp": _iso_now()}
                    })
                    successes += 1
                else:
                    remaining.append(u)

            if successes >= missing:
                return None
            else:
                return {**it, "failed_urls": remaining, "missing": missing - successes}

        tasks = [asyncio.create_task(process_one(it)) for it in failed_items]

        with tqdm(total=len(tasks), desc="Recover (Jina)") as pbar:
            for coro in asyncio.as_completed(tasks):
                res = await coro
                if res is not None:
                    kept_failed.append(res)
                pbar.update(1)

    async with aiofiles.open(failed_path, "w", encoding="utf-8") as f:
        for obj in kept_failed:
            await f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"🔁 Recover done. Successfully rewritten -> {raw_path}")
    print(f"❗ Still failed {len(kept_failed)} entries -> {failed_path}")


async def retry_invalid_in_unified(cfg) -> None:
    assert getattr(cfg, "raw_unified_path", None), "cfg.raw_unified_path is not set."
    min_text_chars = int(getattr(cfg, "min_text_chars", 20))

    to_retry: List[Tuple[Any, str]] = []
    seen = set()
    for rec in _iter_jsonl(cfg.raw_unified_path):
        if rec.get("status") == "invalid":
            key = (rec.get("id"), rec.get("url"))
            if key not in seen and key[0] is not None and key[1]:
                seen.add(key)
                to_retry.append(key)

    if not to_retry:
        print("🔁 Step1.5: no invalid entries to retry.")
        return


    api_key = ""

    connector = aiohttp.TCPConnector(limit=cfg.http_concurrency, ssl=False)
    updates: Dict[Tuple[Any, str], Dict[str, Any]] = {}
    ok_updates = 0
    async with aiohttp.ClientSession(connector=connector) as session:
        batch = 1
        for i in tqdm(range(0, len(to_retry), batch)):
            batch_keys = to_retry[i:i+batch]
            urls = [u for _, u in batch_keys]
            results = await asyncio.gather(
                *[fetch_html(session, u, timeout=cfg.http_timeout, api_key=api_key, use_jina=True) for u in urls],
                return_exceptions=True
            )
            now = _iso_now()
            for (id_, u), r in zip(batch_keys, results):
                if isinstance(r, Exception):
                    continue  
                txt = (r or "").strip()
                if not txt or len(txt) < min_text_chars:
                    continue  

                filter_list = ["403: Forbidden", "404: Not Found", "Page Not Found", "CAPTCHA", "429: Too Many Requests"]
                if any(kw in txt for kw in filter_list):
                    tqdm.write("The text contains filtered keywords; retrieval failed.")
                    continue

                updates[(id_, u)] = {
                    "id": id_, "url": u, "status": "ok",
                    "raw_text": txt,
                    "meta": {"source": "retry", "timestamp": now}
                }
                ok_updates += 1

    if not updates:
        print("🔁 Step1.5: retries completed, but no successful updates.")
        return

    def _updater(rec: Dict[str, Any]) -> Dict[str, Any]:
        key = (rec.get("id"), rec.get("url"))
        if key in updates:
            return updates[key]
        return rec

    _atomic_rewrite_jsonl(cfg.raw_unified_path, _updater)
    print(f"🔁 Step1.5 done. updated={ok_updates} entries in {cfg.raw_unified_path}")

async def clean_raw_text(cfg) -> None:
    data = []
    with open(cfg.raw_unified_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    for item in tqdm(data):
        if item["status"] == 'ok':
            item["raw_text"] = clean_jina_markdown(item["raw_text"])

    with open(cfg.raw_unified_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"🔁 Step1.75 done. Clean raw text.")

async def run_notes_rag_from_unified(cfg) -> None:

    assert getattr(cfg, "raw_unified_path", None), "cfg.raw_unified_path is not set."
    os.makedirs(os.path.dirname(cfg.rag_output_path) or ".", exist_ok=True)

    raw_by_id: Dict[Any, Dict[str, str]] = defaultdict(dict)
    for rec in _iter_jsonl(cfg.raw_unified_path):
        if rec.get("status") == "ok":
            _id = rec.get("id")
            u   = rec.get("url")
            rt  = (rec.get("raw_text") or "").strip()
            if _id is not None and u and rt:
                raw_by_id[_id][u] = rt

    items = read_items_jsonl(cfg.input_path)
    retriever = Retriever(cfg.embed_model)
    out = open(cfg.rag_output_path, "w", encoding="utf-8")

    # tokenizer
    if getattr(cfg, "use_token_chunking", False):
        encode, decode, _ = get_tokenizer(cfg.tokenizer_backend)
    else:
        encode = decode = None  # type: ignore

    total = len(items)
    ok_cnt = 0
    invalid_cnt = 0

    for idx, it in enumerate(tqdm(items), 1):
        note        = str(it.get(cfg.note_key, "") or "")
        tweet_hint  = str(it.get(cfg.tweet_key, "") or "")
        _id         = it.get(cfg.id_key)

        raw_urls = extract_urls(note)
        if not raw_urls:
            record = {
                "id": _id,
                "tweet_hint": tweet_hint,
                "original_note": note,
                "valid_urls": [],
                "budget": cfg.min_budget,
                "snippets": [],
                "status": "cannot_extract_valid_urls"
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            invalid_cnt += 1
            continue

        raw_map_for_id: Dict[str, str] = raw_by_id.get(_id, {})
        valid_urls   = [u for u in raw_urls if u in raw_map_for_id]
        invalid_urls = [u for u in raw_urls if u not in raw_map_for_id]

        if not valid_urls:
            record = {
                "id": _id,
                "tweet_hint": tweet_hint,
                "original_note": note,
                "valid_urls": [],
                "invalid_urls": raw_urls,
                "budget": cfg.min_budget,
                "snippets": [],
                "status": "invalid_no_valid_urls"
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            invalid_cnt += 1
            continue

        chunks: List[Dict[str, Any]] = []
        for u in valid_urls:
            raw_text = raw_map_for_id[u]
            if getattr(cfg, "use_token_chunking", False):
                pieces = chunk_text_by_tokens(
                    raw_text, encode, decode,
                    size=cfg.token_chunk_size, overlap=cfg.token_chunk_overlap  # type: ignore
                )
            else:
                pieces = chunk_text(raw_text, size=cfg.chunk_size, overlap=cfg.chunk_overlap)

            for i, ch in enumerate(pieces):
                chunks.append({"url": u, "chunk_id": i, "text": ch})

        if not chunks:
            record = {
                "id": _id,
                "tweet_hint": tweet_hint,
                "original_note": note,
                "valid_urls": [],
                "invalid_urls": raw_urls,
                "budget": cfg.min_budget,
                "snippets": [],
                "status": "invalid_no_valid_urls"
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            invalid_cnt += 1
            continue

        budget = calc_budget(valid_urls, cfg.url_cost_mode, cfg.url_cost_value, cfg.min_budget)
        query = (tweet_hint.strip() or strip_urls(note))

        selected = retriever.top_k_per_url(
            chunks, query,
            per_url_k=cfg.per_url_top_k,
            max_snippets=cfg.max_snippets
        )

        record = {
            "id": _id,
            "tweet_hint": tweet_hint,
            "original_note": note,
            "valid_urls": valid_urls,
            "invalid_urls": invalid_urls,
            "budget": budget,
            "snippets": selected,
            "status": "ok"
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        ok_cnt += 1

        if getattr(cfg, "batch_size", None) and getattr(cfg, "batch_pause_s", 0) > 0:
            if (idx % cfg.batch_size) == 0:
                await asyncio.sleep(cfg.batch_pause_s)

    out.close()
    print(f"✅ Step2 done. RAG -> {cfg.rag_output_path} (ok={ok_cnt}, invalid={invalid_cnt}, total_items={total})")


async def run_notes_rag_from_unified_web(cfg) -> None:

    assert getattr(cfg, "raw_unified_path", None), "cfg.raw_unified_path is not set."
    os.makedirs(os.path.dirname(cfg.rag_output_path) or ".", exist_ok=True)

    raw_by_id: Dict[Any, Dict[str, str]] = defaultdict(dict)
    ok_records_read = 0
    for rec in _iter_jsonl(cfg.raw_unified_path):
        if rec.get("status") == "ok":
            _id = rec.get("id")
            u   = rec.get("url")
            rt  = (rec.get("raw_text") or "").strip()
            if _id is not None and u and rt:
                raw_by_id[_id][u] = rt
                ok_records_read += 1

    items = read_items_jsonl(cfg.input_path)
    total_items = len(items)

    retriever = Retriever(cfg.embed_model)
    use_token_chunking = bool(getattr(cfg, "use_token_chunking", False))
    if use_token_chunking:
        encode, decode, _ = get_tokenizer(cfg.tokenizer_backend)
    else:
        encode = decode = None  # type: ignore

    default_query = getattr(cfg, "default_query", "Please provide accurate background information and conclusions based on the source.")

    out = open(cfg.rag_output_path, "w", encoding="utf-8")

    ok_cnt = 0
    no_raw_cnt = 0      
    no_chunks_cnt = 0    

    for idx, it in enumerate(tqdm(items), 1):
        _id = it.get(getattr(cfg, "id_key", "id"))
        tweet_hint = str(it.get(getattr(cfg, "tweet_key", "tweet_hint"), "") or "").strip()

        url_map_for_id: Dict[str, str] = raw_by_id.get(_id, {})
        valid_urls: List[str] = list(url_map_for_id.keys())

        if not valid_urls:
            record = {
                "id": _id,
                "tweet_hint": tweet_hint,
                "valid_urls": [],
                "budget": getattr(cfg, "min_budget", 0),
                "snippets": [],
                "status": "invalid_no_raw_text"
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            no_raw_cnt += 1
            continue

        chunks: List[Dict[str, Any]] = []
        for u in valid_urls:
            raw_text = url_map_for_id[u]
            if use_token_chunking:
                pieces = chunk_text_by_tokens(
                    raw_text, encode, decode,
                    size=cfg.token_chunk_size, overlap=cfg.token_chunk_overlap  # type: ignore
                )
            else:
                pieces = chunk_text(raw_text, size=cfg.chunk_size, overlap=cfg.chunk_overlap)

            for i, ch in enumerate(pieces):
                chunks.append({"url": u, "chunk_id": i, "text": ch})

        if not chunks:
            record = {
                "id": _id,
                "tweet_hint": tweet_hint,
                "valid_urls": valid_urls,
                "budget": getattr(cfg, "min_budget", 0),
                "snippets": [],
                "status": "no_chunks"
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            no_chunks_cnt += 1
            continue

        budget = calc_budget(
            valid_urls,
            cfg.url_cost_mode, cfg.url_cost_value,
            cfg.min_budget
        )
        query = tweet_hint.strip() if tweet_hint else default_query

        selected = retriever.top_k_per_url(
            chunks, query,
            per_url_k=cfg.per_url_top_k,
            max_snippets=cfg.max_snippets
        )

        record = {
            "id": _id,
            "tweet_hint": tweet_hint,
            "valid_urls": valid_urls,
            "budget": budget,
            "snippets": selected,
            "status": "ok"
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        ok_cnt += 1

    out.close()

    print(
        f"✅ Step2 done. RAG -> {cfg.rag_output_path} "
        f"(ok={ok_cnt}, invalid_no_raw_text={no_raw_cnt}, no_chunks={no_chunks_cnt}, "
        f"total_items={total_items}, ok_records_read_from_unified={ok_records_read})"
    )

SYSTEM_PROMPT_GEN = SYSTEM_PROMPT

_DECISION_RE = re.compile(r'\bfinal decision\s*:\s*(yes|no)\b[.! ]*', re.IGNORECASE)

def normalize_yes_no_global(raw: str) -> tuple[str, bool]:

    if not raw:
        return "no", False
    matches = list(_DECISION_RE.finditer(raw))
    if not matches:
        return "no", False
    decision = matches[-1].group(1).lower()
    return decision, True

def build_step1_prompt(query: str, snippets: List[Dict[str, Any]]) -> str:
    joined = "\n\n".join(
        f"[S{i+1}] {s['url']} (chunk {s['chunk_id']})\n{s['text']}"
        for i, s in enumerate(snippets)
    )

    prompt = f"""You are given a Tweet and one or more Source snippets:
Tweet:
{query}

Source snippets:
{joined}

Task: Determine whether any of the Source snippets adds meaningful factual background, clarification, or supporting information that helps better understand or evaluate the claim made in the Tweet.
1. Check each snippet independently.
2. If at least one snippet meets the requirements, output "Final decision: yes"; otherwise output "Final decision: no".
"""

    return prompt

def build_user_prompt_from_snippets(query: str, snippets: List[Dict[str, Any]], budget: int) -> str:
    return build_user_prompt(query, snippets, budget)

def build_step3_prompt(note: str, snippets: List[Dict[str, Any]]) -> str:
    joined = "\n\n".join(
        f"[S{i+1}] {s['url']} (chunk {s['chunk_id']})\n{s['text']}"
        for i, s in enumerate(snippets)
    )

    prompt = f"""You are given a Community note and one or more Source snippets:
Community note:
{note}

Source snippets:
{joined}

Task: Decide whether the Community note distorts the information in any of the provided Source snippets.
1. Check each snippet independently.
2. If at least one distortion is found, output "Final decision: yes"; otherwise output "Final decision: no".
"""
    
    return prompt

async def step1_check_rag_relevance(cfg: NotesConfig):
    api_key = "Your Openai API Key"
    model = "gpt-4.1"
    judge = OpenAIClient(api_key, model, cfg.base_url, cfg.max_conn)

    print("Step1 begins...")

    async def _gen_one(rec: Dict) -> Dict:
        user_prompt = build_step1_prompt(rec["tweet_hint"], rec["snippets"])
        raw = await judge.chat(
            system="You are a very meticulous inspector",
            user=user_prompt,
            max_tokens=8192,
            temperature=0,
            top_p=1.0
        )
        label, ok = normalize_yes_no_global(raw)

        final_label = (str(label).strip().lower() if isinstance(label, str) else "").strip()
        if final_label not in ("yes", "no"):
            final_label = "no"
            ok = False

        return {
            **rec,
            "label": final_label,
            "ok": bool(ok),
            "raw_output": raw
        }

    sem = asyncio.Semaphore(cfg.semaphore_size)
    async def _wrap(rec):
        async with sem:
            return await _gen_one(rec)

    with open(cfg.rag_output_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    total = len(lines)

    tasks = [_wrap(json.loads(ln)) for ln in lines]
    results = await asyncio.gather(*tasks)

    ok_count = sum(1 for r in results if r.get("ok"))
    yes_count = sum(1 for r in results if r.get("label") == "yes")
    no_count = sum(1 for r in results if r.get("label") == "no")

    print(f"[step1] total={total}, ok={ok_count}, yes={yes_count}, no={no_count}")

    # yes_records = [r for r in results if r.get("label") == "yes"]
    with open(cfg.step1_output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[step1]✅ Generated to: {cfg.step1_output_path}  (valid={ok_count}/{total})")

    await judge.aclose()  # type: ignore

async def step2_run_notes_generate(cfg: NotesConfig, llm: Optional[BaseLLM] = None):

    if llm is None:
        p = cfg.provider.lower()
        if p == "openai":
            if not cfg.api_key: raise EnvironmentError("Missing OPENAI_API_KEY")
            llm = OpenAIClient(cfg.api_key, cfg.model, cfg.base_url, cfg.max_conn)  # type: ignore
        elif p == "anthropic":
            if not cfg.api_key: raise EnvironmentError("Missing ANTHROPIC_API_KEY")
            llm = AnthropicClient(cfg.api_key, cfg.model, cfg.base_url, cfg.max_conn)  # type: ignore
        elif p == "gemini":
            if not cfg.api_key: raise EnvironmentError("Missing GOOGLE_API_KEY")
            llm = GeminiClient(cfg.api_key, cfg.model, cfg.base_url, cfg.max_conn)  # type: ignore
        elif p == "vllm":
            if not cfg.port: raise EnvironmentError("Missing VLLM PORT")
            llm = VLLMClient(cfg.port)  # type: ignore
        elif p == "openrouter":
            if not cfg.api_key: raise EnvironmentError("Missing OPENROUTER_API_KEY")
            llm = OpenRouterClient(cfg.api_key, cfg.model, cfg.base_url, cfg.max_conn)
        else:
            raise ValueError("provider must be one of: openai | anthropic | gemini | vllm | openrouter")

    print("Step2 begins...")
    details: List[Dict] = []
    valid_cnt = 0
    total = 0

    async def _gen_one(rec: Dict) -> Dict:
        _id = rec.get("id")
        status = rec.get("status")
        valid_urls: List[str] = rec.get("valid_urls") or []
        budget: int = int(rec.get("budget") or cfg.min_budget)
        tweet_hint: str = rec.get("tweet_hint") or ""
        original_note: str = rec.get("original_note") or ""
        snippets: List[Dict] = rec.get("snippets") or []
        invalid_urls: List[str] = rec.get("invalid_urls") or []

        if status != "ok" or not valid_urls:
            return {
                cfg.id_key: _id,
                "ok": False,
                "reason": "invalid_no_valid_urls",
                "tweet": tweet_hint,
                "original_note": original_note,
                "final_text": "",
                "valid_urls": [],
                "invalid_urls": invalid_urls, 
                "snippets": snippets,
            }

        # TODO: Force the budget to be set to 270.
        budget = 270
        query = (tweet_hint.strip() or strip_urls(original_note))
        user_prompt = build_user_prompt_from_snippets(query, snippets, budget)

        raw = await llm.chat(
            system=SYSTEM_PROMPT_GEN,
            user=user_prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p
        )

        prose = (raw or "").strip()
        prose = URL_RE.sub("", prose).strip()
        if len(prose) > budget:
            prose = prose[:budget].rstrip("，。；,.;:：!！?？ ")
        prose = " ".join(prose.split())

        final_text = (prose + " " + " ".join(valid_urls)).strip()

        return {
            cfg.id_key: _id,
            "ok": True,
            "valid_urls": valid_urls,
            "invalid_urls": invalid_urls,    
            "tweet": tweet_hint,
            "original_note": original_note,
            "final_text": final_text,
            "model_output_raw": raw,
            "snippets": snippets,
        }
    
    sem = asyncio.Semaphore(cfg.semaphore_size)
    async def _wrap(rec):
        async with sem:
            return await _gen_one(rec)

    with open(cfg.step1_output_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    # total = len(lines)
    lines = [json.loads(ln) for ln in lines]
    lines = [ln for ln in lines if ln["label"] == "yes"]
    total = len(lines)
    print(f"Generating {len(lines)} samples ...")

    tasks = [_wrap(ln) for ln in lines]
    results = await asyncio.gather(*tasks)

    for r in results:
        if r.get("ok"): valid_cnt += 1
        details.append(r)

    with open(cfg.step2_output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[step2]✅ Generated to: {cfg.step2_output_path}  (valid={valid_cnt}/{total})")

    await llm.aclose()  # type: ignore

async def step2_run_notes_generate_batch(cfg: NotesConfig, llm: Optional[BaseLLM] = None):

    from aiolimiter import AsyncLimiter
    RATE_LIMIT_PER_MIN = getattr(cfg, "rpm_limit", 140)
    rate_limiter = AsyncLimiter(RATE_LIMIT_PER_MIN, 60)

    if llm is None:
        p = cfg.provider.lower()
        if p == "openai":
            if not cfg.api_key: raise EnvironmentError("Missing OPENAI_API_KEY")
            llm = OpenAIClient(cfg.api_key, cfg.model, cfg.base_url, cfg.max_conn)  # type: ignore
        elif p == "anthropic":
            if not cfg.api_key: raise EnvironmentError("Missing ANTHROPIC_API_KEY")
            llm = AnthropicClient(cfg.api_key, cfg.model, cfg.base_url, cfg.max_conn)  # type: ignore
        elif p == "gemini":
            if not cfg.api_key: raise EnvironmentError("Missing GOOGLE_API_KEY")
            llm = GeminiClient(cfg.api_key, cfg.model, cfg.base_url, cfg.max_conn)  # type: ignore
        elif p == "vllm":
            if not cfg.port: raise EnvironmentError("Missing VLLM PORT")
            llm = VLLMClient(cfg.port)  # type: ignore
        elif p == "openrouter":
            if not cfg.api_key: raise EnvironmentError("Missing OPENROUTER_API_KEY")
            llm = OpenRouterClient(cfg.api_key, cfg.model, cfg.base_url, cfg.max_conn)
        else:
            raise ValueError("provider must be one of: openai | anthropic | gemini | vllm | openrouter")

    print("Step2 begins...")
    details: List[Dict] = []
    valid_cnt = 0
    total = 0

    async def _gen_one(rec: Dict) -> Dict:
        _id = rec.get("id")
        status = rec.get("status")
        valid_urls: List[str] = rec.get("valid_urls") or []
        budget: int = int(rec.get("budget") or cfg.min_budget)
        tweet_hint: str = rec.get("tweet_hint") or ""
        original_note: str = rec.get("original_note") or ""
        snippets: List[Dict] = rec.get("snippets") or []
        invalid_urls: List[str] = rec.get("invalid_urls") or []

        if status != "ok" or not valid_urls:
            return {
                cfg.id_key: _id,
                "ok": False,
                "reason": "invalid_no_valid_urls",
                "tweet": tweet_hint,
                "original_note": original_note,
                "final_text": "",
                "valid_urls": [],
                "invalid_urls": invalid_urls, 
                "snippets": snippets,
            }

        budget = 270
        query = (tweet_hint.strip() or strip_urls(original_note))
        user_prompt = build_user_prompt_from_snippets(query, snippets, budget)

        raw = await llm.chat(
            system=SYSTEM_PROMPT_GEN,
            user=user_prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p
        )

        prose = (raw or "").strip()
        prose = URL_RE.sub("", prose).strip()
        if len(prose) > budget:
            prose = prose[:budget].rstrip("，。；,.;:：!！?？ ")
        prose = " ".join(prose.split())

        final_text = (prose + " " + " ".join(valid_urls)).strip()

        return {
            cfg.id_key: _id,
            "ok": True,
            "valid_urls": valid_urls,
            "invalid_urls": invalid_urls, 
            "tweet": tweet_hint,
            "original_note": original_note,
            "final_text": final_text,
            "model_output_raw": raw,
            "snippets": snippets,
        }

    with open(cfg.step1_output_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    lines = [json.loads(ln) for ln in lines]
    lines = [ln for ln in lines if ln.get("label") == "yes"]
    total = len(lines)
    print(f"Generating {total} samples ...")

    batch_size = cfg.batch_size
    pause_s = cfg.batch_pause_s
    print(f"batch_size: {batch_size}")
    print(f"pause_s: {pause_s}")
    if batch_size <= 0:
        batch_size = total if total > 0 else 1 

    # sem = asyncio.Semaphore(cfg.semaphore_size)
    sem = asyncio.Semaphore(5)

    async def _wrap(rec):
        async with sem:
            async with rate_limiter:
                return await _gen_one(rec)

    open(cfg.step2_output_path, "w", encoding="utf-8").close()

    processed = 0
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = lines[start:end]
        print(f"[step2] Processing batch {start//batch_size + 1} "
              f"({start+1}-{end}/{total}), size={len(batch)}")

        tasks = [_wrap(ln) for ln in batch]
        batch_results = await asyncio.gather(*tasks)

        for r in batch_results:
            if r.get("ok"):
                valid_cnt += 1
            details.append(r)

        with open(cfg.step2_output_path, "a", encoding="utf-8") as f:
            for r in batch_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        processed += len(batch)
        print(f"[step2]✅ Batch done: {processed}/{total} processed, "
              f"valid={valid_cnt}")

        if processed < total and pause_s > 0:
            print(f"[step2]⏸  Pausing {pause_s} seconds before next batch ...")
            await asyncio.sleep(pause_s)

    print(f"[step2]✅ Generated to: {cfg.step2_output_path}  (valid={valid_cnt}/{total})")

    await llm.aclose()  # type: ignore

async def step3_check_expression_correctness(cfg: NotesConfig):
    api_key = "Your Openai API Key"
    model = "gpt-4.1"
    judge = OpenAIClient(api_key, model, cfg.base_url, cfg.max_conn)

    print("Step3 begins...")

    async def _gen_one(rec: Dict) -> Dict:
        user_prompt = build_step3_prompt(rec["final_text"], rec["snippets"])
        raw = await judge.chat(
            system="You are a very meticulous inspector",
            user=user_prompt,
            max_tokens=8192,
            temperature=0,
            top_p=1.0
        )
        label, ok = normalize_yes_no_global(raw)

        final_label = (str(label).strip().lower() if isinstance(label, str) else "").strip()
        if final_label not in ("yes", "no"):
            final_label = "no"
            ok = False

        return {
            **rec,
            "label": final_label,
            "ok": bool(ok),
            "raw_output": raw
        }

    sem = asyncio.Semaphore(cfg.semaphore_size)
    async def _wrap(rec):
        async with sem:
            return await _gen_one(rec)

    with open(cfg.step2_output_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    total = len(lines)

    tasks = [_wrap(json.loads(ln)) for ln in lines]
    results = await asyncio.gather(*tasks)

    ok_count = sum(1 for r in results if r.get("ok"))
    yes_count = sum(1 for r in results if r.get("label") == "yes")
    no_count = sum(1 for r in results if r.get("label") == "no")

    print(f"[step3] total={total}, ok={ok_count}, yes={yes_count}, no={no_count}")

    # yes_records = [r for r in results if r.get("label") == "yes"]
    with open(cfg.step3_output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[step3]✅ Generated to: {cfg.step3_output_path}  (valid={ok_count}/{total})")

    await judge.aclose()  # type: ignore


# ====== Integrated Entry Point（full / rag / generate） ======
async def main_async():
    os.environ["OPENAI_API_KEY"] = "Your OPENAI API KEY"
    os.environ["ANTHROPIC_API_KEY"] = "Your ANTHROPIC API KEY"
    os.environ["GOOGLE_API_KEY"] = "Your GOOGLE API KEY"
    os.environ["OPENROUTER_API_KEY"] = "Your OPENROUTER API KEY" # grok

    # ==== Select Provider & Model ====
    provider = "openai"      # "anthropic" | "gemini" | "openrouter"
    # Examples：
    # - openai: "gpt-4o-mini" or "gpt-4.1-mini"
    # - anthropic: "claude-3-5-sonnet-20240620" / "claude-opus-4-0"
    # - gemini: "gemini-2.5-flash" / "gemini-2.5-pro"
    # - vllm: "Lingshu-7B" / "Lingshu-32B"
    # - openrouter: "grok-4"
    model = "o3"
    model_str = model

    # ==== API Keys ====
    api_key = (
        os.getenv("OPENAI_API_KEY") if provider == "openai" else
        os.getenv("ANTHROPIC_API_KEY") if provider == "anthropic" else
        os.getenv("GOOGLE_API_KEY") if provider == "gemini" else
        os.getenv("OPENROUTER_API_KEY")
    )
    base_url = None

    # ==== I/O Path ====
    class_type = "helpful" # "helpful" | "not_helpful"
    input_path = f""
    raw_unified_path = f""
    rag_path    = f"" 
    step1_output_path = f""
    step2_output_path = f""
    step3_output_path = f""
    web_step1_output_path = f""
    web_step2_output_path = f""
    web_raw_output_path = f""
    web_step3_output_path = f""

    cfg = NotesConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        port=8000,                      # vLLM port
        input_path=input_path,
        step1_output_path=step1_output_path,
        step2_output_path=step2_output_path,
        step3_output_path=step3_output_path,
        rag_output_path=rag_path,       
        raw_unified_path=raw_unified_path,
        web_step1_output_path=web_step1_output_path,
        web_step2_output_path=web_step2_output_path,
        web_raw_output_path=web_raw_output_path,
        web_step3_output_path=web_step3_output_path,

        mode="generate",                    # NEW: "web search" | "rag" | "generate" | "full"

        tweet_key="tweet_hint",
        note_key="summary",
        id_key="id",

        semaphore_size=50,
        max_conn=50,
        http_concurrency=8,
        http_timeout=60,

        max_tokens=4096,
        temperature=0,
        top_p=1.0,

        url_cost_mode="one_char_per_url",
        url_cost_value=23,
        min_budget=60,
        embed_model="sentence-transformers/all-mpnet-base-v2",
        per_url_top_k=1,
        max_snippets=None,
        chunk_size=512,
        chunk_overlap=128,

        use_token_chunking = True,
        tokenizer_backend = "tiktoken:cl100k_base" ,  
        token_chunk_size = 512,
        token_chunk_overlap = 128,

        batch_size=100,
        batch_pause_s=65.0,
    )

    if cfg.mode == "web search":
        # await step1_generate_queries(cfg)
        # await step2_start_web_search(cfg)
        # await integrated_steps_web_search(cfg)
        # await integrated_steps_web_search_new(cfg)
        # await run_utility_selection_noreplacement(cfg)
        await run_directly_selection(cfg)
    if cfg.mode == "rag":
        # await collect_raw_unified_v3(cfg)
        # await recover_failed_with_jina_using_fetch_html(cfg)
        # await clean_raw_text(cfg)
        await run_notes_rag_from_unified_web(cfg)

        # await retry_invalid_in_unified(cfg)
        # await clean_raw_text(cfg)
        # await run_notes_rag_from_unified(cfg)
    elif cfg.mode == "generate":
        # await step1_check_rag_relevance(cfg)
        # await step2_run_notes_generate_batch(cfg) # for gemini pro
        # await step2_run_notes_generate(cfg)
        await step3_check_expression_correctness(cfg)
    else:  # full
        # await collect_raw_unified(cfg)
        # await retry_invalid_in_unified(cfg)
        # await run_notes_rag_from_unified(cfg)
        # await run_notes_generate(cfg)
        pass

if __name__ == "__main__":
    if os.getenv("DEBUGPY") == "1":
        init_debug()
    asyncio.run(main_async())
