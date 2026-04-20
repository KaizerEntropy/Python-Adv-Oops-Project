from __future__ import annotations

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from semantic_query_compressor.llm import GroqLLMClient
from semantic_query_compressor.compressor import critical_sentence_coverage
from semantic_query_compressor.pipeline import CompressionMethod, compress_and_evaluate
from semantic_query_compressor.similarity import tf_idf_cosine_similarity
from semantic_query_compressor.text_utils import word_count


EXAMPLE_FILE = PROJECT_ROOT / "example.txt"


def load_query() -> str:
    if not EXAMPLE_FILE.exists():
        raise FileNotFoundError(f"Example file not found: {EXAMPLE_FILE}")
    print(f"Using prompt file: {EXAMPLE_FILE.name}\n")
    return EXAMPLE_FILE.read_text(encoding="utf-8").strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demonstrate query compression and LLM consistency scoring.")
    parser.add_argument(
        "--compression-method",
        choices=("amr", "llm"),
        help="Choose AMR-based compression or LLM-based compression. If omitted, you will be prompted.",
    )
    parser.add_argument("--target-ratio", type=float, default=0.50, help="Approximate compression target.")
    return parser


def build_llm_client() -> object:
    return GroqLLMClient()


def get_compression_method(selected_method: str | None) -> CompressionMethod:
    if selected_method in ("amr", "llm"):
        return selected_method

    while True:
        user_choice = input("Choose compression method [amr/llm]: ").strip().lower()
        if user_choice in ("amr", "llm"):
            return user_choice
        print("Invalid choice. Type 'amr' or 'llm'.")


def print_section(title: str, value: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)
    print(value)


def main() -> int:
    args = build_parser().parse_args()
    compression_method = get_compression_method(args.compression_method)
    query = load_query()
    try:
        llm_client = build_llm_client()
        result = compress_and_evaluate(
            query,
            llm_client=llm_client,
            target_ratio=args.target_ratio,
            compression_method=compression_method,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Check that `.env` contains GROQ_API_KEY and that Groq is reachable.", file=sys.stderr)
        return 1

    original_words = word_count(query)
    compressed_words = word_count(result.compressed_query)
    retention_ratio = compressed_words / original_words if original_words else 0.0
    reduction_ratio = 1.0 - retention_ratio
    words_removed = original_words - compressed_words
    query_similarity = tf_idf_cosine_similarity(query, result.compressed_query)
    critical_coverage = critical_sentence_coverage(query, result.compressed_query)

    print_section("ORIGINAL QUERY", query)
    print_section("COMPRESSED QUERY", result.compressed_query)
    print_section("ORIGINAL LLM OUTPUT", result.original_output)
    print_section("COMPRESSED LLM OUTPUT", result.compressed_output)

    # print("\n" + "=" * 80)
    # print("SCORES AND RESULT COMPARISON")
    # print("=" * 80)
    # print(f"Original word count      : {original_words}")
    # print(f"Compressed word count    : {compressed_words}")
    # print(f"Words removed            : {words_removed}")
    # print(f"Retention ratio          : {retention_ratio:.2f}")
    # print(f"Size reduction           : {reduction_ratio:.2%}")
    # print(f"Compression method       : {result.compression_method}")
    # if result.compression_method == "amr":
    #     print(f"Critical sentence coverage: {critical_coverage:.3f}")
    print(f"Query similarity         : {query_similarity:.3f}")
    print(f"LLM response similarity  : {result.similarity_score:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
