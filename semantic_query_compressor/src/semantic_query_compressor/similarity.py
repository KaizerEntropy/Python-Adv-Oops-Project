from __future__ import annotations

import math
from collections import Counter

from semantic_query_compressor.text_utils import cosine_similarity_from_counters, tokenize


def tf_idf_cosine_similarity(left: str, right: str) -> float:
    """Return a lexical semantic proxy score in the inclusive range [0, 1]."""
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0

    documents = [left_tokens, right_tokens]
    document_frequency: Counter[str] = Counter()
    for tokens in documents:
        document_frequency.update(set(tokens))

    def vectorize(tokens: list[str]) -> Counter[str]:
        counts = Counter(tokens)
        vector: Counter[str] = Counter()
        for term, count in counts.items():
            idf = math.log((1 + len(documents)) / (1 + document_frequency[term])) + 1
            vector[term] = count * idf
        return vector

    score = cosine_similarity_from_counters(vectorize(left_tokens), vectorize(right_tokens))
    return max(0.0, min(1.0, score))
