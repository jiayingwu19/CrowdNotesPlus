"""Command-line entry point for running a configured CrowdNotesPlus workflow."""

from __future__ import annotations

import argparse
import asyncio

from .pipeline import config_from_env, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CrowdNotesPlus workflow.")
    parser.add_argument("--mode", default="rag", help="Workflow mode: collect, rag, generate, utility, web search, or full.")
    parser.add_argument("--input-path", default="", help="Input JSONL path.")
    parser.add_argument("--raw-unified-path", default="raw_unified.jsonl", help="Raw source text JSONL path.")
    parser.add_argument("--rag-output-path", default="rag.jsonl", help="RAG output JSONL path.")
    parser.add_argument("--step1-output-path", default="step1.jsonl", help="Step 1 output JSONL path.")
    parser.add_argument("--step2-output-path", default="step2.jsonl", help="Step 2 output JSONL path.")
    parser.add_argument("--step3-output-path", default="step3.jsonl", help="Step 3 output JSONL path.")
    parser.add_argument("--provider", default="openai", help="LLM provider.")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config_from_env(
        provider=args.provider,
        model=args.model,
        mode=args.mode,
        input_path=args.input_path,
        raw_unified_path=args.raw_unified_path,
        rag_output_path=args.rag_output_path,
        step1_output_path=args.step1_output_path,
        step2_output_path=args.step2_output_path,
        step3_output_path=args.step3_output_path,
    )
    asyncio.run(run_pipeline(cfg))


if __name__ == "__main__":
    main()
