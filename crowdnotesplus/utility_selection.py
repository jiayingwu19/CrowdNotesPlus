"""Utility-based source selection for web-search candidates."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Dict, List, Optional

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from .config import NotesConfig
from .llm import BaseLLM, create_llm_client

_INT_RE = re.compile(r'\b\d+\b')


def build_utility_prompt_pick_one(tweet: str, items_remaining: list[dict], round_no: int) -> str:
    """Build a prompt that asks the model to select one useful search result."""
    lines = []
    for idx, item in enumerate(items_remaining, start=1):
        url = (item.get("link") or "").strip()
        title = (item.get("title") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        lines.append(
            f"[{idx}] Title: {title}\n"
            f"     Snippet: {snippet}\n"
            f"     URL: {url}\n"
        )

    return f"""You are selecting one source for Community Note utility.
This is selection round #{round_no}. Choose exactly ONE result that has the highest utility.

Utility should reflect whether the search result is:
- Relevant to the tweet's topic.
- Likely to add meaningful background or clarification.
- Reliable enough to be worth retrieving.

Output format:
- Output exactly one integer, the index of your chosen item (1..{len(items_remaining)}).
- Do not include any extra words, punctuation, or explanation.

Tweet:
{tweet}

Search Results:
{chr(10).join(lines)}
"""


def parse_single_index(raw_text: str, max_n: int) -> Optional[int]:
    """Parse a one-based index from model output and validate it against the candidate count."""
    if not raw_text:
        return None
    match = _INT_RE.search(raw_text.strip())
    if not match:
        return None
    value = int(match.group(0))
    return value if 1 <= value <= max_n else None


def _load_candidate_rows(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            if all(key in record for key in ("tweet_hint", "valid_counts", "last_merged_unique_items")):
                rows.append(record)
    return rows


async def run_utility_selection_noreplacement(cfg: NotesConfig, llm: Optional[BaseLLM] = None) -> None:
    """Select sources with LLM ranking, removing each selected result from later rounds."""
    own_client = llm is None
    llm = llm or create_llm_client(cfg)
    rows = _load_candidate_rows(cfg.web_step2_output_path)
    output_path = cfg.utility_output_path or cfg.web_step3_output_path or "utility_selected.jsonl"
    semaphore = asyncio.Semaphore(cfg.semaphore_size)
    per_round_retries = 2

    async def _process_one(record: Dict) -> Dict:
        async with semaphore:
            tweet = record["tweet_hint"]
            k = int(record["valid_counts"])
            items: List[Dict] = record["last_merged_unique_items"]
            if not isinstance(items, list) or len(items) <= k + 2 or k <= 0:
                raise ValueError("Candidate list must contain more than k + 2 items and k must be positive.")

            remaining = items[:]
            chosen_urls: List[str] = []
            for round_no in range(1, k + 3):
                prompt = build_utility_prompt_pick_one(tweet, remaining, round_no)
                selected_idx = None
                attempt = 0
                while attempt <= per_round_retries and selected_idx is None:
                    raw = await llm.chat(
                        system="You are a careful selector. Output exactly ONE integer as instructed.",
                        user=prompt,
                        max_tokens=getattr(cfg, "max_tokens", 32),
                        temperature=getattr(cfg, "temperature", 0.0),
                        top_p=getattr(cfg, "top_p", 1.0),
                    )
                    selected_idx = parse_single_index(raw, len(remaining))
                    attempt += 1
                if selected_idx is None:
                    raise RuntimeError(f"No valid index returned in round {round_no} after {per_round_retries + 1} attempts.")
                pick = remaining[selected_idx - 1]
                url = (pick.get("link") or "").strip()
                if not url:
                    raise RuntimeError(f"Chosen item has an empty URL in round {round_no}.")
                chosen_urls.append(url)
                del remaining[selected_idx - 1]

            return {
                "id": record.get("id"),
                "tweet_hint": record.get("tweet_hint"),
                "createdAtMillis": record.get("createdAtMillis"),
                "valid_counts": k,
                "summary": " ".join(chosen_urls),
            }

    results = await tqdm_asyncio.gather(*[_process_one(row) for row in rows], total=len(rows), desc="Utility Selection")
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if own_client:
        await llm.aclose()
    print(f"[utility-noreplacement] Wrote {len(results)} rows to {output_path}")


async def run_utility_selection_noreplacement_seq(cfg: NotesConfig, llm: Optional[BaseLLM] = None) -> None:
    """Run no-replacement utility selection one record at a time."""
    own_client = llm is None
    llm = llm or create_llm_client(cfg)
    rows = _load_candidate_rows(cfg.web_step2_output_path)
    output_path = cfg.utility_output_path or cfg.web_step3_output_path or "utility_selected.jsonl"
    results: List[Dict] = []

    for record in tqdm(rows, total=len(rows), desc="Utility Selection (serial)"):
        temp_cfg = cfg
        single_path = output_path + ".tmp.single"
        tweet = record["tweet_hint"]
        k = int(record["valid_counts"])
        items: List[Dict] = record["last_merged_unique_items"]
        if not isinstance(items, list) or len(items) <= k + 2 or k <= 0:
            raise ValueError("Candidate list must contain more than k + 2 items and k must be positive.")
        remaining = items[:]
        chosen_urls: List[str] = []
        for round_no in range(1, k + 3):
            prompt = build_utility_prompt_pick_one(tweet, remaining, round_no)
            selected_idx = None
            for _ in range(3):
                raw = await llm.chat(
                    system="You are a careful selector. Output exactly ONE integer as instructed.",
                    user=prompt,
                    max_tokens=getattr(cfg, "max_tokens", 32),
                    temperature=getattr(cfg, "temperature", 0.0),
                    top_p=getattr(cfg, "top_p", 1.0),
                )
                selected_idx = parse_single_index(raw, len(remaining))
                if selected_idx is not None:
                    break
            if selected_idx is None:
                raise RuntimeError(f"No valid index returned in round {round_no}.")
            pick = remaining[selected_idx - 1]
            url = (pick.get("link") or "").strip()
            if not url:
                raise RuntimeError(f"Chosen item has an empty URL in round {round_no}.")
            chosen_urls.append(url)
            del remaining[selected_idx - 1]
        results.append({
            "id": record.get("id"),
            "tweet_hint": record.get("tweet_hint"),
            "createdAtMillis": record.get("createdAtMillis"),
            "valid_counts": k,
            "summary": " ".join(chosen_urls),
        })

    with open(output_path, "w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if own_client:
        await llm.aclose()
    print(f"[utility-noreplacement-serial] Wrote {len(results)} rows to {output_path}")


async def run_directly_selection(cfg: NotesConfig) -> None:
    """Select the first k + 2 non-empty links from each candidate list."""
    rows = _load_candidate_rows(cfg.web_step2_output_path)
    output_path = cfg.utility_output_path or cfg.web_step3_output_path or "utility_selected.jsonl"

    async def _process_one(record: Dict) -> Dict:
        tweet = record["tweet_hint"]
        k = int(record["valid_counts"])
        items: List[Dict] = record["last_merged_unique_items"]
        if not isinstance(items, list) or k <= 0:
            raise ValueError("Items must be a list and k must be positive.")
        urls_in_order = []
        for item in items:
            url = (item.get("link") or "").strip()
            if url:
                urls_in_order.append(url)
            if len(urls_in_order) >= k + 2:
                break
        if len(urls_in_order) < k + 2:
            raise RuntimeError(f"Not enough valid links: required {k + 2}, found {len(urls_in_order)}.")
        return {
            "id": record.get("id"),
            "tweet_hint": tweet,
            "createdAtMillis": record.get("createdAtMillis"),
            "valid_counts": k,
            "summary": " ".join(urls_in_order[:k + 2]),
        }

    results = await tqdm_asyncio.gather(*[_process_one(row) for row in rows], total=len(rows), desc="Direct Utility Selection")
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[utility-direct] Wrote {len(results)} rows to {output_path}")
