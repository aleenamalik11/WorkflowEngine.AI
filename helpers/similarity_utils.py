import re

import np
import numpy as np

from models import DomainNode


def _cosine_similarity(a, b):

    a = np.array(a)
    b = np.array(b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def _lexical_similarity(
    query,
    node: DomainNode,
):

    query_tokens = _token_set(
        query
    )

    if not query_tokens:
        return 0.0

    candidates = [
        node.name,
        node.id,
        node.description,
        *(node.aliases or []),
        *(node.types or []),
    ]

    best = 0.0

    for candidate in candidates:

        candidate_tokens = (
            _token_set(
                candidate
            )
        )

        if not candidate_tokens:
            continue

        intersection = (
            query_tokens
            & candidate_tokens
        )

        union = (
            query_tokens
            | candidate_tokens
        )

        if not union:
            continue

        score = (
            len(intersection)
            / len(union)
        )

        best = max(
            best,
            score,
        )

    return best

def _token_set(
    text,
):

    return set(
        _normalize_text(
            text
        ).split()
    )


def _normalize_text(
    text,
):

    text = str(
        text or ""
    ).lower()

    text = text.replace(
        "_",
        " ",
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    return " ".join(
        text.split()
    )