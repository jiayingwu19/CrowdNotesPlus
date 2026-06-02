"""Steps for fetching, retrying, and cleaning raw source text."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import aiofiles
import aiohttp
from tqdm import tqdm

from .cleaning import clean_jina_markdown
from .config import NotesConfig
from .fetching import fetch_html
from .text_utils import _atomic_rewrite_jsonl, _iso_now, _iter_jsonl, extract_urls, read_items_jsonl


async def collect_raw_unified(cfg) -> None:
    assert getattr(cfg, 'raw_unified_path', None), 'cfg.raw_unified_path is not set'
    min_text_chars = int(getattr(cfg, 'min_text_chars', 20))
    items = read_items_jsonl(cfg.input_path)
    os.makedirs(os.path.dirname(cfg.raw_unified_path) or '.', exist_ok=True)
    with open(cfg.raw_unified_path, 'w', encoding='utf-8') as _:
        pass
    out = open(cfg.raw_unified_path, 'a', encoding='utf-8')
    connector = aiohttp.TCPConnector(limit=cfg.http_concurrency, ssl=False)
    total = len(items)
    ok_lines = 0
    invalid_lines = 0
    no_url_items = 0
    async with aiohttp.ClientSession(connector=connector) as session:
        for idx, it in enumerate(tqdm(items), 1):
            note = str(it.get(cfg.note_key, '') or '')
            _id = it.get(cfg.id_key)
            raw_urls: List[str] = extract_urls(note)
            if not raw_urls:
                no_url_items += 1
                continue
            results = await asyncio.gather(*[fetch_html(session, u, timeout=cfg.http_timeout) for u in raw_urls], return_exceptions=True)
            now = _iso_now()
            for u, r in zip(raw_urls, results):
                if isinstance(r, Exception):
                    out.write(json.dumps({'id': _id, 'url': u, 'status': 'invalid', 'error': repr(r)[:500], 'meta': {'source': 'step1', 'timestamp': now}}, ensure_ascii=False) + '\n')
                    invalid_lines += 1
                    continue
                txt = (r or '').strip()
                if not txt:
                    out.write(json.dumps({'id': _id, 'url': u, 'status': 'invalid', 'error': 'empty', 'meta': {'source': 'step1', 'timestamp': now}}, ensure_ascii=False) + '\n')
                    invalid_lines += 1
                    continue
                if len(txt) < min_text_chars:
                    out.write(json.dumps({'id': _id, 'url': u, 'status': 'invalid', 'error': f'too_short(<{min_text_chars})', 'meta': {'source': 'step1', 'timestamp': now}}, ensure_ascii=False) + '\n')
                    invalid_lines += 1
                    continue
                out.write(json.dumps({'id': _id, 'url': u, 'status': 'ok', 'raw_text': txt, 'meta': {'source': 'step1', 'timestamp': now}}, ensure_ascii=False) + '\n')
                ok_lines += 1
            if getattr(cfg, 'batch_size', None) and getattr(cfg, 'batch_pause_s', 0) > 0:
                if idx % cfg.batch_size == 0:
                    await asyncio.sleep(cfg.batch_pause_s)
    out.close()
    print(f'📝 Step1 done. unified -> {cfg.raw_unified_path} (ok={ok_lines}, invalid={invalid_lines}, no_url_items={no_url_items}, total_items={total})')

async def collect_raw_unified_v2(cfg: NotesConfig) -> None:
    assert getattr(cfg, 'raw_unified_path', None), 'cfg.raw_unified_path is not set'
    min_text_chars = int(getattr(cfg, 'min_text_chars', 20))
    max_retries = int(getattr(cfg, 'max_retries', 3))
    retry_backoff_s = float(getattr(cfg, 'retry_backoff_s', 3.0))
    http_timeout = getattr(cfg, 'http_timeout', 20)
    http_concurrency = getattr(cfg, 'http_concurrency', 16)
    items = read_items_jsonl(cfg.web_step2_output_path)
    os.makedirs(os.path.dirname(cfg.raw_unified_path) or '.', exist_ok=True)
    with open(cfg.raw_unified_path, 'w', encoding='utf-8'):
        pass
    out_ok = open(cfg.raw_unified_path, 'a', encoding='utf-8')
    failed_path = getattr(cfg, 'failed_unified_path', None)
    if not failed_path:
        base = cfg.raw_unified_path
        if base.endswith('.jsonl'):
            failed_path = base[:-6] + '.failed.jsonl'
        else:
            failed_path = base + '.failed.jsonl'
    os.makedirs(os.path.dirname(failed_path) or '.', exist_ok=True)
    with open(failed_path, 'w', encoding='utf-8'):
        pass
    out_failed = open(failed_path, 'a', encoding='utf-8')
    total_items = len(items)
    ok_lines = 0
    failed_samples = 0
    no_url_items = 0
    processed_items = 0
    connector = aiohttp.TCPConnector(limit=http_concurrency, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        for idx, it in enumerate(tqdm(items), 1):
            processed_items += 1
            note: str = str(it.get(getattr(cfg, 'note_key', 'note'), '') or '')
            _id = it.get(getattr(cfg, 'id_key', 'id'))
            valid_counts = int(it.get('valid_counts', 1))
            if valid_counts <= 0:
                valid_counts = 1
            raw_urls: List[str] = extract_urls(note)
            if not raw_urls:
                no_url_items += 1
                now = _iso_now()
                out_failed.write(json.dumps({'id': _id, 'failed_urls': [], 'missing': valid_counts, 'meta': {'source': 'step1', 'timestamp': now}}, ensure_ascii=False) + '\n')
                failed_samples += 1
                if getattr(cfg, 'batch_size', None) and getattr(cfg, 'batch_pause_s', 0) > 0:
                    if idx % cfg.batch_size == 0:
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
                        txt = (r or '').strip()
                        if not txt:
                            raise ValueError('empty')
                        if len(txt) < min_text_chars:
                            raise ValueError(f'too_short(<{min_text_chars})')
                        filter_list = ['403: Forbidden', '404: Not Found', 'Page Not Found', 'CAPTCHA', '429: Too Many Requests']
                        if any((kw in txt for kw in filter_list)):
                            raise ValueError('The text contains filtered keywords; retrieval failed.')
                        break
                    except Exception as e:
                        last_exception = e
                        if attempt < max_retries and retry_backoff_s > 0:
                            await asyncio.sleep(retry_backoff_s)
                if txt is not None and len(txt) >= min_text_chars:
                    out_ok.write(json.dumps({'id': _id, 'url': u, 'status': 'ok', 'raw_text': txt, 'meta': {'source': 'step1', 'timestamp': now}}, ensure_ascii=False) + '\n')
                    ok_lines += 1
                    successes += 1
                else:
                    failed_urls_this_item.append(u)
            if successes < valid_counts:
                missing = valid_counts - successes
                out_failed.write(json.dumps({'id': _id, 'failed_urls': failed_urls_this_item, 'missing': missing, 'meta': {'source': 'step1', 'timestamp': now}}, ensure_ascii=False) + '\n')
                failed_samples += 1
            if getattr(cfg, 'batch_size', None) and getattr(cfg, 'batch_pause_s', 0) > 0:
                if idx % cfg.batch_size == 0:
                    await asyncio.sleep(cfg.batch_pause_s)
    out_ok.close()
    out_failed.close()
    print(f'📝 Step1 done. unified -> {cfg.raw_unified_path} (ok_urls={ok_lines}, failed_samples={failed_samples}, no_url_items={no_url_items}, total_items={total_items})\n❗ Failed details -> {failed_path}')

async def collect_raw_unified_v3(cfg: NotesConfig) -> None:
    assert getattr(cfg, 'raw_unified_path', None), 'cfg.raw_unified_path is not set'
    min_text_chars = int(getattr(cfg, 'min_text_chars', 20))
    max_retries = int(getattr(cfg, 'max_retries', 3))
    retry_backoff_s = float(getattr(cfg, 'retry_backoff_s', 3.0))
    http_timeout = int(getattr(cfg, 'http_timeout', 20))
    http_concurrency = int(getattr(cfg, 'http_concurrency', 16))
    samples_concurrency = int(getattr(cfg, 'samples_concurrency', 32))
    api_key = None
    items = read_items_jsonl(cfg.web_step3_output_path)
    os.makedirs(os.path.dirname(cfg.raw_unified_path) or '.', exist_ok=True)
    failed_path = getattr(cfg, 'failed_unified_path', None)
    if not failed_path:
        base = cfg.raw_unified_path
        failed_path = base[:-6] + '.failed.jsonl' if base.endswith('.jsonl') else base + '.failed.jsonl'
    os.makedirs(os.path.dirname(failed_path) or '.', exist_ok=True)
    async with aiofiles.open(cfg.raw_unified_path, 'w', encoding='utf-8') as f:
        await f.flush()
    async with aiofiles.open(failed_path, 'w', encoding='utf-8') as f:
        await f.flush()
    file_lock = asyncio.Lock()

    async def write_ok_line(obj: dict):
        line = json.dumps(obj, ensure_ascii=False) + '\n'
        async with file_lock:
            async with aiofiles.open(cfg.raw_unified_path, 'a', encoding='utf-8') as f:
                await f.write(line)

    async def write_fail_line(obj: dict):
        line = json.dumps(obj, ensure_ascii=False) + '\n'
        async with file_lock:
            async with aiofiles.open(failed_path, 'a', encoding='utf-8') as f:
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
            note: str = str(it.get(getattr(cfg, 'note_key', 'note'), '') or '')
            _id = it.get(getattr(cfg, 'id_key', 'id'))
            valid_counts = int(it.get('valid_counts', 1))
            if valid_counts <= 0:
                valid_counts = 1
            raw_urls: List[str] = extract_urls(note)
            now = _iso_now()
            if not raw_urls:
                await write_fail_line({'id': _id, 'failed_urls': [], 'missing': valid_counts, 'meta': {'source': 'step1', 'timestamp': now}})
                return (0, 1, 1)
            successes = 0
            failed_urls_this_item: List[str] = []
            for u in raw_urls:
                if successes >= valid_counts:
                    break
                txt: Optional[str] = None
                last_exception: Optional[Exception] = None
                filter_list = ['403: Forbidden', '404: Not Found', 'Page Not Found', 'CAPTCHA', '429: Too Many Requests']
                for attempt in range(1, max_retries + 1):
                    try:
                        r = await fetch_html(session, u, timeout=http_timeout, api_key=api_key, use_jina=True)
                        txt = (r or '').strip()
                        if not txt:
                            raise ValueError('empty')
                        if len(txt) < min_text_chars:
                            raise ValueError(f'too_short(<{min_text_chars})')
                        if any((kw in txt for kw in filter_list)):
                            raise ValueError('The text contains filtered keywords; retrieval failed.')
                        break
                    except Exception as e:
                        last_exception = e
                        if attempt < max_retries and retry_backoff_s > 0:
                            await asyncio.sleep(retry_backoff_s)
                if txt is not None and len(txt) >= min_text_chars and (not any((kw in txt for kw in filter_list))):
                    await write_ok_line({'id': _id, 'url': u, 'status': 'ok', 'raw_text': txt, 'meta': {'source': 'step1', 'timestamp': now}})
                    successes += 1
                else:
                    failed_urls_this_item.append(u)
            if successes < valid_counts:
                missing = valid_counts - successes
                await write_fail_line({'id': _id, 'failed_urls': failed_urls_this_item, 'missing': missing, 'meta': {'source': 'step1', 'timestamp': now}})
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
        with tqdm(total=total_items, desc='Step 1 (Concurrent Crawling)') as pbar:
            for coro in asyncio.as_completed(tasks):
                await coro
                pbar.update(1)
    print(f'📝 Step1 done. unified -> {cfg.raw_unified_path} (ok_urls={ok_lines}, failed_samples={failed_samples}, no_url_items={no_url_items}, total_items={total_items})\n❗ Failed details -> {failed_path}')

async def recover_failed_with_jina_using_fetch_html(cfg) -> None:
    raw_path = getattr(cfg, 'raw_unified_path', None)
    assert raw_path, 'cfg.raw_unified_path is not set.'
    failed_path = getattr(cfg, 'failed_unified_path', None)
    if not failed_path:
        base = raw_path
        failed_path = base[:-6] + '.failed.jsonl' if base.endswith('.jsonl') else base + '.failed.jsonl'
    if not os.path.exists(failed_path):
        print(f'No failed files found: {failed_path}')
        return
    min_text_chars = int(getattr(cfg, 'min_text_chars', 20))
    max_retries = int(getattr(cfg, 'max_retries', 3))
    retry_backoff_s = float(getattr(cfg, 'retry_backoff_s', 3.0))
    http_timeout = int(getattr(cfg, 'http_timeout', 20))
    http_concurrency = int(getattr(cfg, 'http_concurrency', 16))
    samples_concurrency = int(getattr(cfg, 'samples_concurrency', 32))
    api_key = ''

    async def _read_failed(path: str) -> List[Dict]:
        items = []
        async with aiofiles.open(path, 'r', encoding='utf-8') as f:
            async for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    obj.setdefault('failed_urls', [])
                    obj.setdefault('missing', max(1, len(obj['failed_urls'])))
                    items.append(obj)
                except Exception:
                    continue
        return items
    failed_items = await _read_failed(failed_path)
    if not failed_items:
        print(f'{failed_path} is empty')
        return
    file_lock = asyncio.Lock()

    async def write_ok(obj: dict):
        async with file_lock:
            async with aiofiles.open(raw_path, 'a', encoding='utf-8') as f:
                await f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    connector = aiohttp.TCPConnector(limit=http_concurrency, ssl=False)
    kept_failed: List[Dict] = []
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(samples_concurrency)

        async def process_one(it: Dict) -> Optional[Dict]:
            _id = it.get('id')
            failed_urls = list(it.get('failed_urls', []))
            missing = int(it.get('missing', 1))
            if not failed_urls:
                return it
            successes = 0
            remaining = []
            for u in failed_urls:
                if successes >= missing:
                    break
                txt = None
                filter_list = ['403: Forbidden', '404: Not Found', 'Page Not Found', 'CAPTCHA', '429: Too Many Requests']
                for attempt in range(1, max_retries + 1):
                    try:
                        r = await fetch_html(session, u, timeout=http_timeout, api_key=api_key, use_jina=True)
                        txt = (r or '').strip()
                        if not txt or len(txt) < min_text_chars:
                            raise ValueError('too_short')
                        if any((kw in txt for kw in filter_list)):
                            raise ValueError('blocked')
                        break
                    except Exception:
                        if attempt < max_retries and retry_backoff_s > 0:
                            await asyncio.sleep(retry_backoff_s)
                if txt and len(txt) >= min_text_chars and (not any((kw in txt for kw in filter_list))):
                    await write_ok({'id': _id, 'url': u, 'status': 'ok', 'raw_text': txt, 'meta': {'source': 'recover', 'timestamp': _iso_now()}})
                    successes += 1
                else:
                    remaining.append(u)
            if successes >= missing:
                return None
            else:
                return {**it, 'failed_urls': remaining, 'missing': missing - successes}
        tasks = [asyncio.create_task(process_one(it)) for it in failed_items]
        with tqdm(total=len(tasks), desc='Recover (Jina)') as pbar:
            for coro in asyncio.as_completed(tasks):
                res = await coro
                if res is not None:
                    kept_failed.append(res)
                pbar.update(1)
    async with aiofiles.open(failed_path, 'w', encoding='utf-8') as f:
        for obj in kept_failed:
            await f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    print(f'🔁 Recover done. Successfully rewritten -> {raw_path}')
    print(f'❗ Still failed {len(kept_failed)} entries -> {failed_path}')

async def retry_invalid_in_unified(cfg) -> None:
    assert getattr(cfg, 'raw_unified_path', None), 'cfg.raw_unified_path is not set.'
    min_text_chars = int(getattr(cfg, 'min_text_chars', 20))
    to_retry: List[Tuple[Any, str]] = []
    seen = set()
    for rec in _iter_jsonl(cfg.raw_unified_path):
        if rec.get('status') == 'invalid':
            key = (rec.get('id'), rec.get('url'))
            if key not in seen and key[0] is not None and key[1]:
                seen.add(key)
                to_retry.append(key)
    if not to_retry:
        print('🔁 Step1.5: no invalid entries to retry.')
        return
    api_key = ''
    connector = aiohttp.TCPConnector(limit=cfg.http_concurrency, ssl=False)
    updates: Dict[Tuple[Any, str], Dict[str, Any]] = {}
    ok_updates = 0
    async with aiohttp.ClientSession(connector=connector) as session:
        batch = 1
        for i in tqdm(range(0, len(to_retry), batch)):
            batch_keys = to_retry[i:i + batch]
            urls = [u for _, u in batch_keys]
            results = await asyncio.gather(*[fetch_html(session, u, timeout=cfg.http_timeout, api_key=api_key, use_jina=True) for u in urls], return_exceptions=True)
            now = _iso_now()
            for (id_, u), r in zip(batch_keys, results):
                if isinstance(r, Exception):
                    continue
                txt = (r or '').strip()
                if not txt or len(txt) < min_text_chars:
                    continue
                filter_list = ['403: Forbidden', '404: Not Found', 'Page Not Found', 'CAPTCHA', '429: Too Many Requests']
                if any((kw in txt for kw in filter_list)):
                    tqdm.write('The text contains filtered keywords; retrieval failed.')
                    continue
                updates[id_, u] = {'id': id_, 'url': u, 'status': 'ok', 'raw_text': txt, 'meta': {'source': 'retry', 'timestamp': now}}
                ok_updates += 1
    if not updates:
        print('🔁 Step1.5: retries completed, but no successful updates.')
        return

    def _updater(rec: Dict[str, Any]) -> Dict[str, Any]:
        key = (rec.get('id'), rec.get('url'))
        if key in updates:
            return updates[key]
        return rec
    _atomic_rewrite_jsonl(cfg.raw_unified_path, _updater)
    print(f'🔁 Step1.5 done. updated={ok_updates} entries in {cfg.raw_unified_path}')

async def clean_raw_text(cfg) -> None:
    data = []
    with open(cfg.raw_unified_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    for item in tqdm(data):
        if item['status'] == 'ok':
            item['raw_text'] = clean_jina_markdown(item['raw_text'])
    with open(cfg.raw_unified_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'🔁 Step1.75 done. Clean raw text.')
