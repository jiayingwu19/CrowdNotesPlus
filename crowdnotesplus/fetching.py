"""HTTP fetching and web-page text extraction utilities."""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse

import aiohttp
import trafilatura
from tqdm import tqdm

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    from readability import Document as ReadabilityDocument
except Exception:
    ReadabilityDocument = None

try:
    import justext
except Exception:
    justext = None

from .text_utils import chunk_text, chunk_text_by_tokens, get_tokenizer


DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)
DEFAULT_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
DEFAULT_ACCEPT_LANG = "en-US,en;q=0.9,ja;q=0.7,zh-CN;q=0.6"
RETRY_STATUS = {429, 500, 502, 503, 504, 520, 522, 524}
TEXTUAL_CT_KEYWORDS = ("text/html", "application/xhtml+xml", "text/xml", "application/xml", "text/plain")
class RateLimiter:

    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            self.calls = [t for t in self.calls if now - t < self.period]
            if len(self.calls) >= self.max_calls:
                sleep_for = self.period - (now - self.calls[0])
                await asyncio.sleep(sleep_for)
                now = time.monotonic()
                self.calls = [t for t in self.calls if now - t < self.period]
            self.calls.append(now)

JINA_RATE_LIMITER = RateLimiter(max_calls=400, period=60)


def build_render_url(u: str) -> str:
    p = urlparse(strip_fragment(u))
    base = urlunparse((p.scheme, p.netloc, p.path, '', p.query, ''))
    return f'https://r.jina.ai/{base}'

def strip_fragment(u: str) -> str:
    u, _ = urldefrag(u)
    return u

def make_headers(base_url: Optional[str]=None) -> Dict[str, str]:
    headers = {'User-Agent': DEFAULT_UA, 'Accept': DEFAULT_ACCEPT, 'Accept-Language': DEFAULT_ACCEPT_LANG, 'Cache-Control': 'no-cache', 'Pragma': 'no-cache', 'DNT': '1'}
    if base_url:
        parsed = urlparse(base_url)
        headers['Origin'] = f'{parsed.scheme}://{parsed.netloc}'
        headers['Referer'] = f'{parsed.scheme}://{parsed.netloc}/'
    return headers

def guess_kind_from_url(u: str) -> str:
    u = u.lower()
    if 'youtube.com/watch' in u or 'youtu.be/' in u:
        return 'video'
    if 'twitter.com/' in u or 'x.com/' in u:
        return 'twitter'
    return 'html'

def sniff_is_binary(data: bytes) -> bool:
    if not data:
        return True
    head = data[:8]
    if head.startswith(b'%PDF'):
        return True
    if head.startswith(b'PK\x03\x04'):
        return True
    nul_ratio = data.count(b'\x00') / max(1, len(data))
    if nul_ratio > 0.002:
        return True
    sample = data[:4096]
    text_like = sum((32 <= c <= 126 or c in (9, 10, 13) for c in sample))
    if text_like / max(1, len(sample)) < 0.85:
        return True
    return False

def is_probably_js_heavy(domain: str) -> bool:
    return any((d in domain for d in ['washingtonpost.com', 'nytimes.com', 'reuters.com', 'sciencedirect.com', 'archive.is', 'mountsinai.org', 'foxnews.com', 'newsnetwork.mayoclinic.org']))

def extract_with_readability(html: str) -> Optional[str]:
    if not ReadabilityDocument:
        return None
    try:
        doc = ReadabilityDocument(html)
        return doc.summary(html_partial=True)
    except Exception:
        return None

def extract_with_justext(html: str) -> Optional[str]:
    if not justext:
        return None
    try:
        paragraphs = justext.justext(html.encode('utf-8', 'ignore'), justext.get_stoplist('English'))
        return '\n'.join((p.text for p in paragraphs if not p.is_boilerplate))
    except Exception:
        return None

def html_to_text(html: str) -> str:
    txt = trafilatura.extract(html, include_links=False, include_tables=False)
    if txt and txt.strip():
        return txt.strip()
    r = extract_with_readability(html)
    if r:
        t = trafilatura.extract(r, include_links=False, include_tables=False) or ''
        if t.strip():
            return t.strip()
    j = extract_with_justext(html)
    if j and j.strip():
        return j.strip()
    try:
        from bs4 import BeautifulSoup as BS
        soup = BS(html, 'lxml')
        for tag in soup(['script', 'style', 'noscript']):
            tag.extract()
        body = soup.get_text('\n', strip=True)
        if body and body.strip():
            return body.strip()
    except Exception:
        pass
    return ''

async def http_get(session: aiohttp.ClientSession, url: str, *, timeout: int) -> Tuple[Optional[bytes], Optional[str], Optional[str], int]:
    try:
        async with session.get(url, headers=make_headers(url), timeout=aiohttp.ClientTimeout(total=timeout), allow_redirects=True) as resp:
            ct = resp.headers.get('Content-Type')
            data = await resp.read()
            return (data, ct, str(resp.url), resp.status)
    except Exception:
        return (None, None, None, 0)

async def fetch_with_retries(session: aiohttp.ClientSession, url: str, *, timeout: int, max_retries: int=2):
    url = strip_fragment(url)
    last = (None, None, None, 0)
    for attempt in range(max_retries + 1):
        content, ct, final_u, status = await http_get(session, url, timeout=timeout)
        last = (content, ct, final_u, status)
        if status and status not in RETRY_STATUS and (content is not None):
            return last
        if attempt < max_retries:
            await asyncio.sleep(0.6 * 2 ** attempt)
    return last

async def discover_amp_link(html: str, base_url: str) -> Optional[str]:
    if not BeautifulSoup:
        return None
    try:
        soup = BeautifulSoup(html, 'lxml')
        link = soup.find('link', rel=lambda v: v and 'amphtml' in v.lower())
        if link and link.get('href'):
            return urljoin(base_url, link['href'])
    except Exception:
        pass
    return None

def x_twitter_to_readable(url: str) -> str:
    return f'https://r.jina.ai/http://{strip_fragment(url)}'

async def smart_fetch_text(session: aiohttp.ClientSession, url: str, *, timeout: int, api_key=None, use_jina=False) -> Optional[str]:
    R_JINA_AUTH = api_key
    RENDER_MIN_LEN = 40
    if use_jina:
        kind = guess_kind_from_url(url)
        if kind in {'binary', 'video'}:
            tqdm.write(f'Non-text URL type:{url}')
            return None
        final_u = None
        render_url = build_render_url(final_u or url)
        tqdm.write(f'Using Jina directly - {render_url}')
        try:
            headers = {'Authorization': R_JINA_AUTH} if R_JINA_AUTH else {}
            async with session.get(render_url, headers={**make_headers(url), **headers}, timeout=aiohttp.ClientTimeout(total=timeout), allow_redirects=True) as resp:
                if 200 <= resp.status < 300:
                    data = await resp.read()
                    if data and (not sniff_is_binary(data)):
                        txt = data.decode('utf-8', errors='ignore').strip()
                        return txt if len(txt) >= RENDER_MIN_LEN else None
        except Exception as e:
            tqdm.write(f'Fetch failed with error: {e}, url={render_url}')
            return None
        tqdm.write(f'Fetch did not succeed, url={render_url}')
        return None
    kind = guess_kind_from_url(url)
    if kind in {'binary', 'video'}:
        return None
    content, ct, final_u, status = await fetch_with_retries(session, url, timeout=timeout)
    if not content or not 200 <= (status or 0) < 300:
        render_url = build_render_url(final_u or url)
        tqdm.write(render_url)
        try:
            headers = {'Authorization': R_JINA_AUTH} if R_JINA_AUTH else {}
            async with session.get(render_url, headers={**make_headers(url), **headers}, timeout=aiohttp.ClientTimeout(total=timeout), allow_redirects=True) as resp:
                if 200 <= resp.status < 300:
                    data = await resp.read()
                    if data and (not sniff_is_binary(data)):
                        txt = data.decode('utf-8', errors='ignore').strip()
                        return txt if len(txt) >= RENDER_MIN_LEN else None
        except Exception:
            pass
        return None
    if sniff_is_binary(content):
        return None
    if ct:
        ctl = ct.lower()
        if not any((k in ctl for k in TEXTUAL_CT_KEYWORDS)):
            render_url = build_render_url(final_u or url)
            tqdm.write(render_url)
            try:
                headers = {'Authorization': R_JINA_AUTH} if R_JINA_AUTH else {}
                async with session.get(render_url, headers={**make_headers(url), **headers}, timeout=aiohttp.ClientTimeout(total=timeout), allow_redirects=True) as resp:
                    if 200 <= resp.status < 300:
                        data = await resp.read()
                        if data and (not sniff_is_binary(data)):
                            txt = data.decode('utf-8', errors='ignore').strip()
                            return txt if len(txt) >= RENDER_MIN_LEN else None
            except Exception:
                pass
            return None
    html = content.decode('utf-8', errors='ignore')
    text = html_to_text(html)
    if text and len(text.strip()) >= RENDER_MIN_LEN:
        return text.strip()
    render_url = build_render_url(final_u or url)
    tqdm.write(render_url)
    try:
        headers = {'Authorization': R_JINA_AUTH} if R_JINA_AUTH else {}
        async with session.get(render_url, headers={**make_headers(url), **headers}, timeout=aiohttp.ClientTimeout(total=timeout), allow_redirects=True) as resp:
            if 200 <= resp.status < 300:
                data = await resp.read()
                if data and (not sniff_is_binary(data)):
                    txt = data.decode('utf-8', errors='ignore').strip()
                    return txt if len(txt) >= RENDER_MIN_LEN else None
    except Exception:
        pass
    return None

async def smart_fetch_text_v2(session: aiohttp.ClientSession, url: str, *, timeout: int, api_key=None, use_jina: bool=False) -> Optional[str]:
    R_JINA_AUTH = api_key
    RENDER_MIN_LEN = 40

    async def fetch_via_jina(render_url: str, orig_url_for_headers: str) -> Optional[str]:
        try:
            await JINA_RATE_LIMITER.acquire()
            headers = {'Authorization': R_JINA_AUTH} if R_JINA_AUTH else {}
            async with session.get(render_url, headers={**make_headers(orig_url_for_headers), **headers}, timeout=aiohttp.ClientTimeout(total=timeout), allow_redirects=True) as resp:
                if 200 <= resp.status < 300:
                    data = await resp.read()
                    if data and (not sniff_is_binary(data)):
                        txt = data.decode('utf-8', errors='ignore').strip()
                        return txt if len(txt) >= RENDER_MIN_LEN else None
        except Exception as e:
            tqdm.write(f'Jina rendering failed: {e}, url={render_url}')
        return None
    if use_jina:
        kind = guess_kind_from_url(url)
        if kind in {'binary', 'video'}:
            tqdm.write(f'Non-text URL type: {url}')
            return None
        render_url = build_render_url(url)
        tqdm.write(f'Using Jina directly - {render_url}')
        txt = await fetch_via_jina(render_url, url)
        if txt:
            return txt
        tqdm.write(f'Fetch did not succeed, url= {render_url}')
        return None
    kind = guess_kind_from_url(url)
    if kind in {'binary', 'video'}:
        return None
    content, ct, final_u, status = await fetch_with_retries(session, url, timeout=timeout)
    if not content or not 200 <= (status or 0) < 300:
        render_url = build_render_url(final_u or url)
        tqdm.write(render_url)
        txt = await fetch_via_jina(render_url, url)
        return txt
    if sniff_is_binary(content):
        return None
    if ct:
        ctl = ct.lower()
        if not any((k in ctl for k in TEXTUAL_CT_KEYWORDS)):
            render_url = build_render_url(final_u or url)
            tqdm.write(render_url)
            txt = await fetch_via_jina(render_url, url)
            return txt
    html = content.decode('utf-8', errors='ignore')
    text = html_to_text(html)
    if text and len(text.strip()) >= RENDER_MIN_LEN:
        return text.strip()
    render_url = build_render_url(final_u or url)
    tqdm.write(render_url)
    txt = await fetch_via_jina(render_url, url)
    return txt

async def fetch_html(session: aiohttp.ClientSession, url: str, timeout: int, api_key=None, use_jina=False) -> Optional[str]:
    return await smart_fetch_text_v2(session, url, timeout=timeout, api_key=api_key, use_jina=use_jina)

async def gather_chunks_and_filter_urls_v2(urls: List[str], *, http_concurrency: int, http_timeout: int, chunk_size: int, chunk_overlap: int, use_token_chunking: bool=False, tokenizer_backend: str='tiktoken:cl100k_base', token_chunk_size: int=512, token_chunk_overlap: int=64) -> Tuple[List[Dict[str, Any]], List[str], List[str], Dict[str, str]]:
    if use_token_chunking:
        encode, decode, _ = get_tokenizer(tokenizer_backend)
    else:
        encode = decode = None
    connector = aiohttp.TCPConnector(limit=http_concurrency, ssl=False)
    chunks: List[Dict[str, Any]] = []
    valid_urls: List[str] = []
    invalid_urls: List[str] = []
    raw_text_map: Dict[str, str] = {}
    async with aiohttp.ClientSession(connector=connector) as session:
        texts = await asyncio.gather(*[fetch_html(session, u, timeout=http_timeout) for u in urls], return_exceptions=True)
    for u, text in zip(urls, texts):
        if isinstance(text, Exception) or not text:
            invalid_urls.append(u)
            continue
        try:
            text = text.strip()
            if len(text) < 20:
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
            raw_text_map[u] = text
        except Exception:
            invalid_urls.append(u)
    return (chunks, valid_urls, invalid_urls, raw_text_map)
