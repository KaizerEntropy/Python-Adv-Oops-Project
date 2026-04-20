from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from semantic_query_compressor.compressor import compress_query
from semantic_query_compressor.llm import GroqLLMClient, LLMClient
from semantic_query_compressor.similarity import tf_idf_cosine_similarity


CompressionMethod = Literal["amr", "llm"]


@dataclass(frozen=True)
class CompressionResult:
    compressed_query: str
    original_output: str
    compressed_output: str
    similarity_score: float
    compression_method: str

    def as_dict(self) -> dict[str, str | float]:
        return asdict(self)


def compress_and_evaluate(
    query: str,
    *,
    llm_client: LLMClient | None = None,
    target_ratio: float = 0.35,
    compression_method: CompressionMethod = "amr",
) -> CompressionResult:
    """Compress a long query and compare LLM outputs for semantic consistency."""
    client = llm_client or GroqLLMClient()
    compressed_query = compress_with_method(
        query,
        llm_client=client,
        target_ratio=target_ratio,
        compression_method=compression_method,
    )

    original_output = client.generate(query)
    compressed_output = client.generate(compressed_query)
    similarity_score = tf_idf_cosine_similarity(original_output, compressed_output)

    return CompressionResult(
        compressed_query=compressed_query,
        original_output=original_output,
        compressed_output=compressed_output,
        similarity_score=similarity_score,
        compression_method=compression_method,
    )


def compress_with_method(
    query: str,
    *,
    llm_client: LLMClient,
    target_ratio: float = 0.35,
    compression_method: CompressionMethod = "amr",
) -> str:
    if compression_method == "amr":
        return compress_query(query, target_ratio=target_ratio)
    if compression_method == "llm":
        return compress_query_with_llm(query, llm_client=llm_client, target_ratio=target_ratio)
    raise ValueError("compression_method must be either 'amr' or 'llm'")


def compress_query_with_llm(
    query: str,
    *,
    llm_client: LLMClient,
    target_ratio: float = 0.35,
) -> str:
    prompt = f"""
Compress the following user query while preserving all user intent, required outputs,
constraints, technical terms, numbers, and formatting requirements.

Target length: about {target_ratio:.0%} of the original text.

Return only the compressed query. Do not answer the query.

Original query:
{query}
""".strip()
    compressed = llm_client.generate(prompt).strip()
    if not compressed:
        raise RuntimeError("LLM compression returned an empty string")
    return compressed
