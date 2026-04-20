from __future__ import annotations

import argparse
import json
import sys

from semantic_query_compressor.pipeline import compress_and_evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compress a long query and compare Groq LLM responses.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--query", help="Long query text. Must contain at least 100 words.")
    input_group.add_argument("--file", help="Path to a UTF-8 text file containing the long query.")
    parser.add_argument("--target-ratio", type=float, default=0.35, help="Approximate sentence keep ratio.")
    parser.add_argument(
        "--compression-method",
        choices=("amr", "llm"),
        default="amr",
        help="Use the local AMR-inspired compressor or LLM-based compression.",
    )
    parser.add_argument("--pretty", action="store_true", help="Print formatted JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    query = args.query
    if args.file:
        with open(args.file, "r", encoding="utf-8") as handle:
            query = handle.read()

    try:
        result = compress_and_evaluate(
            query,
            target_ratio=args.target_ratio,
            compression_method=args.compression_method,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    indent = 2 if args.pretty else None
    print(json.dumps(result.as_dict(), indent=indent, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
