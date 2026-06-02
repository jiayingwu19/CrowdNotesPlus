"""Prompt builders used by RAG generation and validation steps."""

from __future__ import annotations

from typing import Any, Dict, List


SYSTEM_PROMPT = (
    "Community notes is a collaborative way to add helpful context to posts and keep people better informed. "
    "Now you are a highly experienced community note writer."
)
SYSTEM_PROMPT_GEN = SYSTEM_PROMPT


def build_user_prompt(query: str, snippets: List[Dict[str, Any]], budget_chars: int) -> str:
    joined = '\n\n'.join((f"[S{i + 1}] {s['url']} (chunk {s['chunk_id']})\n{s['text']}" for i, s in enumerate(snippets)))
    prompt = f'Task: Write a community note based ONLY on the source snippets below.\nHard constraints:\n- The note MUST be in English.\n- DO NOT include any URLs in the note.\n- The note MUST be a single line (no line breaks, no bullets).\n- Note length MUST be ≤ {budget_chars} characters. Do not exceed this budget.\n- Be specific, objective, and verifiable.\n\nTweet:\n{query}\n\nSource snippets:\n{joined}\n\nOutput only the note content. Remember: length ≤ {budget_chars}, no URLs.\n'
    return prompt

def build_step1_prompt(query: str, snippets: List[Dict[str, Any]]) -> str:
    joined = '\n\n'.join((f"[S{i + 1}] {s['url']} (chunk {s['chunk_id']})\n{s['text']}" for i, s in enumerate(snippets)))
    prompt = f'You are given a Tweet and one or more Source snippets:\nTweet:\n{query}\n\nSource snippets:\n{joined}\n\nTask: Determine whether any of the Source snippets adds meaningful factual background, clarification, or supporting information that helps better understand or evaluate the claim made in the Tweet.\n1. Check each snippet independently.\n2. If at least one snippet meets the requirements, output "Final decision: yes"; otherwise output "Final decision: no".\n'
    return prompt

def build_user_prompt_from_snippets(query: str, snippets: List[Dict[str, Any]], budget: int) -> str:
    return build_user_prompt(query, snippets, budget)

def build_step3_prompt(note: str, snippets: List[Dict[str, Any]]) -> str:
    joined = '\n\n'.join((f"[S{i + 1}] {s['url']} (chunk {s['chunk_id']})\n{s['text']}" for i, s in enumerate(snippets)))
    prompt = f'You are given a Community note and one or more Source snippets:\nCommunity note:\n{note}\n\nSource snippets:\n{joined}\n\nTask: Decide whether the Community note distorts the information in any of the provided Source snippets.\n1. Check each snippet independently.\n2. If at least one distortion is found, output "Final decision: yes"; otherwise output "Final decision: no".\n'
    return prompt
