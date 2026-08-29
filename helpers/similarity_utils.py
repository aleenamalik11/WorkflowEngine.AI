import re

import np
import numpy as np

from helpers.ontology_semantics import semantics_for
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


def _node_text(node):
    parts = [
        node.name or "",
        node.description or "",
    ]

    parts.extend(
        node.aliases or []
    )

    return " ".join(
        p for p in parts if p
    )

def _combined_score(
    lexical_score,
    semantic_score,
    candidate_score,
):
    """
    Candidate score from DomainGraphClient is already a
    lexical/embedding blend.

    We still recompute semantic and lexical similarity here
    so the planner has transparent scores.

    The explicit semantic/lexical values dominate.
    """

    return (
        0.45 * semantic_score
        + 0.35 * lexical_score
        + 0.20 * max(
            0.0,
            min(
                1.0,
                candidate_score,
            ),
        )
    )


def _relationship_relevance(
    relation,
):
    """
    Convert ontology relationship semantics into a relevance
    score.

    IMPORTANT:
    This is NOT a path cost.

    Lower relationship weights mean stronger semantic relevance.
    """

    semantics = semantics_for(
        relation
    )

    if semantics.classification == "REQUIRED":
        return 1.0

    if semantics.classification == "POSSIBLE":
        return 0.75

    if semantics.classification == "CONTEXT":
        return 0.40

    return 0.20
