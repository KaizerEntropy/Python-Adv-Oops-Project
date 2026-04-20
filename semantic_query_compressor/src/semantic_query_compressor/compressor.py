from __future__ import annotations

import math
from collections import Counter

from semantic_query_compressor.text_utils import split_sentences, tokenize, word_count


TASK_KEYWORDS = (
    "analyze",
    "compare",
    "create",
    "design",
    "evaluate",
    "explain",
    "generate",
    "implement",
    "include",
    "produce",
    "propose",
    "provide",
    "return",
    "summarize",
    "write",
    "code",
)

CONSTRAINT_KEYWORDS = (
    "at least",
    "avoid",
    "do not",
    "don't",
    "ensure",
    "exactly",
    "exclude",
    "format",
    "include",
    "input",
    "json",
    "markdown",
    "must",
    "only",
    "output",
    "python",
    "java",
    "should",
    "table",
    "without",
)


def _sentence_concepts(sentences: list[str]) -> list[list[str]]:
    return [list(dict.fromkeys(tokenize(sentence))) for sentence in sentences]


def _build_amr_concept_graph(sentence_concepts: list[list[str]]) -> dict[str, Counter[str]]:
    """A lightweight AMR-like concept graph.

    Full AMR parsing requires external parsers and model weights. This graph is
    a deterministic approximation: content words act as concepts, repeated
    co-occurrence and local adjacency act as semantic-role style relations.
    """
    graph: dict[str, Counter[str]] = {}

    for concepts in sentence_concepts:
        unique_concepts = list(dict.fromkeys(concepts))
        for concept in unique_concepts:
            graph.setdefault(concept, Counter())

        for left, right in zip(unique_concepts, unique_concepts[1:]):
            graph[left][right] += 2
            graph[right][left] += 2

        for left_index, left in enumerate(unique_concepts):
            for right in unique_concepts[left_index + 1 :]:
                graph[left][right] += 1
                graph[right][left] += 1

    return graph


def _rank_amr_concepts(graph: dict[str, Counter[str]]) -> dict[str, float]:
    if not graph:
        return {}

    weighted_degrees = {
        concept: sum(neighbors.values())
        for concept, neighbors in graph.items()
    }
    max_degree = max(weighted_degrees.values()) or 1
    return {
        concept: degree / max_degree
        for concept, degree in weighted_degrees.items()
    }


def _task_sentence_bonus(sentence: str) -> float:
    if _is_task_sentence(sentence):
        return 1.0
    if _is_constraint_sentence(sentence):
        return 0.75
    return 0.0


def _is_task_sentence(sentence: str) -> bool:
    sentence_lower = sentence.lower()
    return any(keyword in sentence_lower for keyword in TASK_KEYWORDS)


def _is_constraint_sentence(sentence: str) -> bool:
    sentence_lower = sentence.lower()
    return any(keyword in sentence_lower for keyword in CONSTRAINT_KEYWORDS)


def _critical_sentence_indexes(sentences: list[str]) -> set[int]:
    return {
        index
        for index, sentence in enumerate(sentences)
        if _is_task_sentence(sentence) or _is_constraint_sentence(sentence)
    }


def _score_sentence(
    concepts: list[str],
    concept_scores: dict[str, float],
    sentence: str,
) -> float:
    if not concepts:
        return 0.0

    concept_coverage = sum(concept_scores.get(concept, 0.0) for concept in concepts)
    normalized_coverage = concept_coverage / math.sqrt(len(concepts))
    return normalized_coverage + _task_sentence_bonus(sentence)


def _concept_similarity(left: list[str], right: list[str]) -> float:
    # Jacard similarity 
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _select_with_mmr(
    sentence_scores: list[float],
    sentence_concepts: list[list[str]],
    selected_indexes: set[int],
    *,
    target_count: int,
    importance_weight: float = 0.75,
) -> set[int]:
    while len(selected_indexes) < target_count:
        best_index: int | None = None
        best_score = -math.inf

        for index, importance in enumerate(sentence_scores):
            if index in selected_indexes or not sentence_concepts[index]:
                continue

            redundancy = 0.0
            if selected_indexes:
                redundancy = max(
                    _concept_similarity(sentence_concepts[index], sentence_concepts[selected_index])
                    for selected_index in selected_indexes
                )

            mmr_score = importance_weight * importance - (1 - importance_weight) * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_index = index

        if best_index is None:
            break

        selected_indexes.add(best_index)

    return selected_indexes


def critical_sentence_coverage(original: str, compressed: str) -> float:
    """Return how many task/constraint sentences survived compression."""
    original_sentences = split_sentences(original)
    compressed_sentences = {
        " ".join(sentence.lower().split())
        for sentence in split_sentences(compressed)
    }
    critical_sentences = [
        " ".join(sentence.lower().split())
        for index, sentence in enumerate(original_sentences)
        if index in _critical_sentence_indexes(original_sentences)
    ]
    if not critical_sentences:
        return 1.0
    preserved = sum(sentence in compressed_sentences for sentence in critical_sentences)
    return preserved / len(critical_sentences)


def compress_query(
    query: str,
    *,
    target_ratio: float = 0.35,
    min_sentences: int = 3,
    max_sentences: int | None = None,
) -> str:
    """ AMR-inspired graph compression of a long query.

    Content words become AMR-like concept nodes, co-occurrence and adjacency
    become relation edges, and sentences are scored by central concept coverage.
    Selected sentences are restored to their original order for readability.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")

    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")

    total_words = word_count(query)
    if total_words < 100:
        raise ValueError("query must contain at least 100 words")

    sentences = split_sentences(query)
    if len(sentences) <= min_sentences:
        return query

    if max_sentences is None:
        max_sentences = max(min_sentences, math.ceil(len(sentences) * target_ratio))

    max_sentences = min(max_sentences, len(sentences))
    sentence_concepts = _sentence_concepts(sentences)
    if not any(sentence_concepts):
        return " ".join(sentences[:max_sentences])

    graph = _build_amr_concept_graph(sentence_concepts)
    concept_scores = _rank_amr_concepts(graph)
    sentence_scores = [
        _score_sentence(concepts, concept_scores, sentences[index])
        for index, concepts in enumerate(sentence_concepts)
    ]
    if not any(sentence_scores):
        return " ".join(sentences[:max_sentences])

    selected_indexes = _critical_sentence_indexes(sentences)
    target_count = max(max_sentences, min(len(sentences), len(selected_indexes)))
    selected_indexes = _select_with_mmr(
        sentence_scores,
        sentence_concepts,
        selected_indexes,
        target_count=target_count,
    )
    selected_indexes = sorted(selected_indexes)
    compressed = " ".join(sentences[index] for index in selected_indexes)
    return compressed.strip()
