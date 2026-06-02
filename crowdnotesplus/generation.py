"""LLM generation and validation steps for Community Notes."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Dict, List, Optional, Tuple

from .config import NotesConfig
from .llm import BaseLLM, OpenAIClient, create_llm_client
from .prompts import SYSTEM_PROMPT_GEN, build_step1_prompt, build_step3_prompt, build_user_prompt_from_snippets
from .text_utils import URL_RE, strip_urls

_DECISION_RE = re.compile(r'\bfinal decision\s*:\s*(yes|no)\b[.! ]*', re.IGNORECASE)
FULLWIDTH_TRAILING_PUNCTUATION = "\uff0c\u3002\uff1b,.;:\uff1a!\uff01?\uff1f "


def normalize_yes_no_global(raw: str) -> Tuple[str, bool]:
    """Extract the last `Final decision: yes/no` label from a judge response."""
    if not raw:
        return "no", False
    matches = list(_DECISION_RE.finditer(raw))
    if not matches:
        return "no", False
    return matches[-1].group(1).lower(), True


def _create_judge(cfg: NotesConfig) -> BaseLLM:
    """Create the judge client. It reuses the configured API key and connection settings."""
    if (cfg.provider or "").lower() == "openai":
        if not cfg.api_key:
            raise EnvironmentError("Missing OPENAI_API_KEY")
        return OpenAIClient(cfg.api_key, cfg.model or "gpt-4.1", cfg.base_url, cfg.max_conn)
    return create_llm_client(cfg)


async def step1_check_rag_relevance(cfg: NotesConfig, judge: Optional[BaseLLM] = None) -> None:
    """Judge whether retrieved snippets are useful for each tweet."""
    own_client = judge is None
    judge = judge or _create_judge(cfg)
    print("Step1 begins...")

    async def _gen_one(rec: Dict) -> Dict:
        user_prompt = build_step1_prompt(rec["tweet_hint"], rec["snippets"])
        raw = await judge.chat(
            system="You are a very meticulous inspector",
            user=user_prompt,
            max_tokens=8192,
            temperature=0,
            top_p=1.0,
        )
        label, ok = normalize_yes_no_global(raw)
        final_label = (str(label).strip().lower() if isinstance(label, str) else "").strip()
        if final_label not in ("yes", "no"):
            final_label = "no"
            ok = False
        return {**rec, "label": final_label, "ok": bool(ok), "raw_output": raw}

    semaphore = asyncio.Semaphore(cfg.semaphore_size)

    async def _wrap(rec: Dict) -> Dict:
        async with semaphore:
            return await _gen_one(rec)

    with open(cfg.rag_output_path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    total = len(lines)
    results = await asyncio.gather(*[_wrap(json.loads(line)) for line in lines])

    ok_count = sum(1 for row in results if row.get("ok"))
    yes_count = sum(1 for row in results if row.get("label") == "yes")
    no_count = sum(1 for row in results if row.get("label") == "no")
    print(f"[step1] total={total}, ok={ok_count}, yes={yes_count}, no={no_count}")

    with open(cfg.step1_output_path, "w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[step1] Generated to: {cfg.step1_output_path} (valid={ok_count}/{total})")

    if own_client:
        await judge.aclose()


async def step2_run_notes_generate(cfg: NotesConfig, llm: Optional[BaseLLM] = None) -> None:
    """Generate final note text for records that passed the relevance check."""
    own_client = llm is None
    llm = llm or create_llm_client(cfg)
    print("Step2 begins...")

    async def _gen_one(rec: Dict) -> Dict:
        record_id = rec.get("id")
        status = rec.get("status")
        valid_urls: List[str] = rec.get("valid_urls") or []
        budget = int(rec.get("budget") or cfg.min_budget)
        tweet_hint = rec.get("tweet_hint") or ""
        original_note = rec.get("original_note") or ""
        snippets: List[Dict] = rec.get("snippets") or []
        invalid_urls: List[str] = rec.get("invalid_urls") or []

        if status != "ok" or not valid_urls:
            return {
                cfg.id_key: record_id,
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
        query = tweet_hint.strip() or strip_urls(original_note)
        user_prompt = build_user_prompt_from_snippets(query, snippets, budget)
        raw = await llm.chat(
            system=SYSTEM_PROMPT_GEN,
            user=user_prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )

        prose = URL_RE.sub("", (raw or "").strip()).strip()
        if len(prose) > budget:
            prose = prose[:budget].rstrip(FULLWIDTH_TRAILING_PUNCTUATION)
        prose = " ".join(prose.split())
        final_text = (prose + " " + " ".join(valid_urls)).strip()

        return {
            cfg.id_key: record_id,
            "ok": True,
            "valid_urls": valid_urls,
            "invalid_urls": invalid_urls,
            "tweet": tweet_hint,
            "original_note": original_note,
            "final_text": final_text,
            "model_output_raw": raw,
            "snippets": snippets,
        }

    with open(cfg.step1_output_path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line.strip()) for line in handle if line.strip()]
    rows = [row for row in rows if row.get("label") == "yes"]
    total = len(rows)
    print(f"Generating {total} samples ...")

    semaphore = asyncio.Semaphore(cfg.semaphore_size)

    async def _wrap(rec: Dict) -> Dict:
        async with semaphore:
            return await _gen_one(rec)

    results = await asyncio.gather(*[_wrap(row) for row in rows])
    valid_count = sum(1 for row in results if row.get("ok"))

    with open(cfg.step2_output_path, "w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[step2] Generated to: {cfg.step2_output_path} (valid={valid_count}/{total})")

    if own_client:
        await llm.aclose()


async def step2_run_notes_generate_batch(cfg: NotesConfig, llm: Optional[BaseLLM] = None) -> None:
    """Generate notes in batches with an optional requests-per-minute limit."""
    from aiolimiter import AsyncLimiter

    own_client = llm is None
    llm = llm or create_llm_client(cfg)
    rate_limiter = AsyncLimiter(getattr(cfg, "rpm_limit", 140), 60)

    with open(cfg.step1_output_path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line.strip()) for line in handle if line.strip()]
    rows = [row for row in rows if row.get("label") == "yes"]
    total = len(rows)

    batch_size = cfg.batch_size or total or 1
    pause_s = cfg.batch_pause_s
    if batch_size <= 0:
        batch_size = total or 1

    open(cfg.step2_output_path, "w", encoding="utf-8").close()
    semaphore = asyncio.Semaphore(min(cfg.semaphore_size, 5))
    valid_count = 0
    processed = 0

    async def _limited_generate(rec: Dict) -> Dict:
        async with semaphore:
            async with rate_limiter:
                temp_path = cfg.step1_output_path
                return await _generate_one_from_rag_record(cfg, llm, rec)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = rows[start:end]
        print(f"[step2] Processing batch {start // batch_size + 1} ({start + 1}-{end}/{total}), size={len(batch)}")
        batch_results = await asyncio.gather(*[_limited_generate(row) for row in batch])
        valid_count += sum(1 for row in batch_results if row.get("ok"))
        with open(cfg.step2_output_path, "a", encoding="utf-8") as handle:
            for row in batch_results:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        processed += len(batch)
        print(f"[step2] Batch done: {processed}/{total} processed, valid={valid_count}")
        if processed < total and pause_s > 0:
            print(f"[step2] Pausing {pause_s} seconds before next batch ...")
            await asyncio.sleep(pause_s)

    print(f"[step2] Generated to: {cfg.step2_output_path} (valid={valid_count}/{total})")
    if own_client:
        await llm.aclose()


async def _generate_one_from_rag_record(cfg: NotesConfig, llm: BaseLLM, rec: Dict) -> Dict:
    """Generate one note from a RAG record. Shared by normal and batched generation."""
    record_id = rec.get("id")
    status = rec.get("status")
    valid_urls: List[str] = rec.get("valid_urls") or []
    tweet_hint = rec.get("tweet_hint") or ""
    original_note = rec.get("original_note") or ""
    snippets: List[Dict] = rec.get("snippets") or []
    invalid_urls: List[str] = rec.get("invalid_urls") or []

    if status != "ok" or not valid_urls:
        return {
            cfg.id_key: record_id,
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
    query = tweet_hint.strip() or strip_urls(original_note)
    user_prompt = build_user_prompt_from_snippets(query, snippets, budget)
    raw = await llm.chat(
        system=SYSTEM_PROMPT_GEN,
        user=user_prompt,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
    )
    prose = URL_RE.sub("", (raw or "").strip()).strip()
    if len(prose) > budget:
        prose = prose[:budget].rstrip(FULLWIDTH_TRAILING_PUNCTUATION)
    prose = " ".join(prose.split())
    final_text = (prose + " " + " ".join(valid_urls)).strip()
    return {
        cfg.id_key: record_id,
        "ok": True,
        "valid_urls": valid_urls,
        "invalid_urls": invalid_urls,
        "tweet": tweet_hint,
        "original_note": original_note,
        "final_text": final_text,
        "model_output_raw": raw,
        "snippets": snippets,
    }


async def step3_check_expression_correctness(cfg: NotesConfig, judge: Optional[BaseLLM] = None) -> None:
    """Judge whether generated notes distort their supporting snippets."""
    own_client = judge is None
    judge = judge or _create_judge(cfg)
    print("Step3 begins...")

    async def _gen_one(rec: Dict) -> Dict:
        user_prompt = build_step3_prompt(rec["final_text"], rec["snippets"])
        raw = await judge.chat(
            system="You are a very meticulous inspector",
            user=user_prompt,
            max_tokens=8192,
            temperature=0,
            top_p=1.0,
        )
        label, ok = normalize_yes_no_global(raw)
        final_label = (str(label).strip().lower() if isinstance(label, str) else "").strip()
        if final_label not in ("yes", "no"):
            final_label = "no"
            ok = False
        return {**rec, "label": final_label, "ok": bool(ok), "raw_output": raw}

    semaphore = asyncio.Semaphore(cfg.semaphore_size)

    async def _wrap(rec: Dict) -> Dict:
        async with semaphore:
            return await _gen_one(rec)

    with open(cfg.step2_output_path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    results = await asyncio.gather(*[_wrap(json.loads(line)) for line in lines])

    ok_count = sum(1 for row in results if row.get("ok"))
    yes_count = sum(1 for row in results if row.get("label") == "yes")
    no_count = sum(1 for row in results if row.get("label") == "no")
    print(f"[step3] total={len(lines)}, ok={ok_count}, yes={yes_count}, no={no_count}")

    with open(cfg.step3_output_path, "w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[step3] Generated to: {cfg.step3_output_path} (valid={ok_count}/{len(lines)})")

    if own_client:
        await judge.aclose()
