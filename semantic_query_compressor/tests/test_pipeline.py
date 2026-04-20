from __future__ import annotations

import pytest

from semantic_query_compressor.compressor import compress_query, critical_sentence_coverage
from semantic_query_compressor.pipeline import compress_and_evaluate
from semantic_query_compressor.similarity import tf_idf_cosine_similarity
from semantic_query_compressor.text_utils import split_sentences, tokenize


class FakeLLMClient:
    def generate(self, prompt: str) -> str:
        if "Return only the compressed query" in prompt:
            return "Please propose a practical customer support automation plan with risk controls and metrics."
        if "customer support" in prompt.lower():
            return "The request asks for a structured customer support automation plan."
        return "The request asks for a structured automation plan."


LONG_QUERY = """
Our company wants to redesign the customer support intake process because the current
workflow is slow, inconsistent, and hard to measure. Agents currently receive tickets
from email, chat, social media, and a legacy form, but the information is not normalized
before it reaches the queue. Some customers include order numbers, account identifiers,
screenshots, urgency details, and product categories, while others provide only vague
descriptions. The support team wants an automated approach that reads incoming requests,
identifies the intent, extracts critical fields, assigns a priority level, routes the
ticket to the correct specialist group, and drafts a concise first response. The solution
should avoid making unsupported promises to customers, should escalate billing and security
issues carefully, and should leave a clear audit trail for supervisors. We also need a
measurement plan that compares the new workflow against the old workflow using response
time, resolution time, customer satisfaction, escalation rate, and manual correction rate.
Please propose a practical implementation plan that covers data preparation, model
selection, evaluation, monitoring, risk controls, rollout phases, and team training.
"""

def test_compress_query_reduces_long_text() -> None:
    compressed = compress_query(LONG_QUERY)
    assert len(compressed.split()) < len(LONG_QUERY.split())
    assert "customer support" in compressed.lower()


def test_compress_query_rejects_short_text() -> None:
    with pytest.raises(ValueError):
        compress_query("too short")


def test_similarity_is_between_zero_and_one() -> None:
    score = tf_idf_cosine_similarity("create a support automation plan", "support automation plan")
    assert 0.0 <= score <= 1.0
    assert score > 0.3


def test_split_sentences_handles_missing_space_after_period() -> None:
    sentences = split_sentences("Keep this sentence.Then split this one.")
    assert sentences == ["Keep this sentence.", "Then split this one."]


def test_tokenize_preserves_technical_terms() -> None:
    tokens = tokenize("Use peer-to-peer trading, n/2 limits, and DNP3.")
    assert "peer-to-peer" in tokens
    assert "n/2" in tokens
    assert "dnp3" in tokens


def test_compress_query_preserves_task_and_constraints() -> None:
    compressed = compress_query(LONG_QUERY, target_ratio=0.25)
    assert "Please propose a practical implementation plan" in compressed
    assert "should avoid making unsupported promises" in compressed
    assert critical_sentence_coverage(LONG_QUERY, compressed) == 1.0


def test_pipeline_returns_required_outputs() -> None:
    result = compress_and_evaluate(LONG_QUERY, llm_client=FakeLLMClient())
    output = result.as_dict()
    assert set(output) == {
        "compressed_query",
        "original_output",
        "compressed_output",
        "similarity_score",
        "compression_method",
    }
    assert 0.0 <= result.similarity_score <= 1.0
    assert result.compression_method == "amr"


def test_pipeline_supports_llm_compression() -> None:
    result = compress_and_evaluate(
        LONG_QUERY,
        llm_client=FakeLLMClient(),
        compression_method="llm",
    )
    assert result.compression_method == "llm"
    assert result.compressed_query.startswith("Please propose")
    assert 0.0 <= result.similarity_score <= 1.0
