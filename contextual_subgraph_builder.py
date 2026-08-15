"""
Stages 3-6:

    Semantic steps
        |
        +--> semantic + lexical candidate matching
        |
        +--> immediate neighborhood expansion
        |
        +--> semantic re-ranking of neighboring concepts
        |
        +--> contextual domain graph
        |
        +--> prompt dependency edges

No Dijkstra.
No shortest-path routing.

Relationship weights are used only as relevance/order signals.
"""

import math
import networkx as nx

from relationship_semantics import (
    apply_relationship_semantics,
    semantics_for,
)

def _has_embedding(value):
    """
    Safely determine whether an embedding exists.

    Supports:
      - None
      - Python lists
      - tuples
      - numpy arrays
      - other array-like objects
    """
    if value is None:
        return False

    try:
        return len(value) > 0
    except TypeError:
        return False


def build_contextual_subgraph(
    interpretation,
    domain_client,
    embedding_service,
    k=5,
    neighborhood_depth=1,
):
    """
    Build the contextual planning structure.

    Important design rule:

    The NetworkX graph represents DOMAIN structure.

    Candidate mappings are kept separately because one domain node may
    correspond to multiple semantic steps.
    """

    domain_graph = nx.DiGraph()

    candidate_map = {}

    candidates_debug = []
    context_attachments = []
    constraint_edges = []

    # Used to avoid repeated relationship debug entries.
    seen_context_edges = set()

    # Used to remember the best semantic score for a domain node.
    best_node_score = {}

    # ---------------------------------------------------------
    # STAGE 3
    #
    # Match EVERY semantic step independently.
    # ---------------------------------------------------------

    for step_index, step in enumerate(
        interpretation.steps
    ):

        step_embedding = embedding_service.encode(
            step.text
        )

        direct_candidates = domain_client.candidate_nodes(
            text=step.text,
            embedding=step_embedding,
            k=k,
        )

        step_candidates = []

        for candidate in direct_candidates:

            node = candidate.node

            lexical_score = _lexical_similarity(
                step.text,
                node.name,
                node.aliases,
                node.description,
            )

            semantic_score = _semantic_similarity(
                step_embedding,
                node.embedding,
            )

            combined_score = _combined_score(
                lexical_score,
                semantic_score,
                candidate.score,
            )

            item = {
                "node_id": node.id,
                "name": node.name,
                "node_type": node.node_type,
                "score": combined_score,
                "lexical_score": lexical_score,
                "semantic_score": semantic_score,
                "explicit": bool(step.explicit),
                "inferred": not bool(step.explicit),
                "source": "direct",
                "prompt_text": step.text,
            }

            step_candidates.append(item)

            _add_domain_node(
                domain_graph,
                node,
                item,
            )

            best_node_score[node.id] = max(
                best_node_score.get(
                    node.id,
                    0.0,
                ),
                combined_score,
            )

        # -----------------------------------------------------
        # STAGE 4
        #
        # Expand the immediate neighborhood.
        #
        # Neighborhood nodes are NOT automatically accepted.
        # They are semantically re-ranked first.
        # -----------------------------------------------------

        neighborhood_seed_ids = [
            item["node_id"]
            for item in step_candidates
        ]

        for seed_id in neighborhood_seed_ids:

            relationships = domain_client.neighborhood(
                seed_id,
                depth=neighborhood_depth,
            )

            for relationship in relationships:

                source = domain_client.get_node(
                    relationship.source_id
                )

                target = domain_client.get_node(
                    relationship.target_id
                )

                if source is None or target is None:
                    continue

                _add_domain_node(
                    domain_graph,
                    source,
                    None,
                )

                _add_domain_node(
                    domain_graph,
                    target,
                    None,
                )

                # Preserve the ontology edge.
                domain_graph.add_edge(
                    source.id,
                    target.id,
                    relation=relationship.relation,
                    inferred_context=True,
                    origin="domain",
                )

                context_key = (
                    source.id,
                    relationship.relation,
                    target.id,
                )

                if context_key not in seen_context_edges:

                    seen_context_edges.add(
                        context_key
                    )

                    context_attachments.append(
                        {
                            "seed": seed_id,
                            "source": source.name,
                            "target": target.name,
                            "relation": relationship.relation,
                        }
                    )

                # -------------------------------------------------
                # Semantically evaluate BOTH endpoints as possible
                # inferred concepts.
                # -------------------------------------------------

                for neighbor in (
                    source,
                    target,
                ):

                    if neighbor.id in {
                        item["node_id"]
                        for item in step_candidates
                    }:
                        continue

                    neighbor_embedding = (
                        neighbor.embedding
                    )

                    # Some graph implementations may not store
                    # embeddings on nodes. Generate one from the
                    # semantic text if necessary.
                    if neighbor_embedding is None:

                        neighbor_embedding = (
                            embedding_service.encode(
                                _node_text(neighbor)
                            )
                        )

                    lexical_score = _lexical_similarity(
                        step.text,
                        neighbor.name,
                        neighbor.aliases,
                        neighbor.description,
                    )

                    semantic_score = _semantic_similarity(
                        step_embedding,
                        neighbor_embedding,
                    )

                    relation_score = _relationship_relevance(
                        relationship.relation
                    )

                    contextual_score = (
                        0.50 * semantic_score
                        + 0.25 * lexical_score
                        + 0.25 * relation_score
                    )

                    # Don't pull every arbitrary neighbor into
                    # the workflow.
                    #
                    # The threshold is deliberately permissive
                    # because this is contextual inference rather
                    # than final function matching.
                    if contextual_score < 0.25:
                        continue

                    inferred_item = {
                        "node_id": neighbor.id,
                        "name": neighbor.name,
                        "node_type": neighbor.node_type,
                        "score": contextual_score,
                        "lexical_score": lexical_score,
                        "semantic_score": semantic_score,
                        "explicit": False,
                        "inferred": True,
                        "source": "neighborhood",
                        "prompt_text": step.text,
                        "relation_score": relation_score,
                    }

                    step_candidates.append(
                        inferred_item
                    )

                    best_node_score[neighbor.id] = max(
                        best_node_score.get(
                            neighbor.id,
                            0.0,
                        ),
                        contextual_score,
                    )

        # -----------------------------------------------------
        # Deduplicate candidates for THIS semantic step.
        # -----------------------------------------------------

        deduped = {}

        for item in step_candidates:

            node_id = item["node_id"]

            existing = deduped.get(node_id)

            if existing is None:

                deduped[node_id] = item

            elif item["score"] > existing["score"]:

                # Preserve explicit status if either occurrence
                # was explicit.
                item["explicit"] = (
                    item["explicit"]
                    or existing["explicit"]
                )

                item["inferred"] = (
                    not item["explicit"]
                )

                deduped[node_id] = item

        step_candidates = list(
            deduped.values()
        )

        step_candidates.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        step_candidates = step_candidates[
            : max(k * 2, 10)
        ]

        candidate_map[step_index] = (
            step_candidates
        )

        candidates_debug.append(
            {
                "step_index": step_index,
                "step": step.text,
                "explicit": bool(step.explicit),
                "candidates": step_candidates,
            }
        )

    # ---------------------------------------------------------
    # STAGE 5
    #
    # LLM-derived semantic dependencies.
    #
    # Resolve dependencies AGAINST THE CANDIDATE MAP rather
    # than against prompt_text stored on shared graph nodes.
    # ---------------------------------------------------------

    for dependency in interpretation.dependencies:

        before = dependency.get(
            "before"
        )

        after = dependency.get(
            "after"
        )

        if not before or not after:
            continue

        before_node = _best_candidate_for_text(
            interpretation.steps,
            candidate_map,
            before,
        )

        after_node = _best_candidate_for_text(
            interpretation.steps,
            candidate_map,
            after,
        )

        if before_node is None or after_node is None:
            continue

        if before_node == after_node:
            continue

        domain_graph.add_edge(
            before_node,
            after_node,
            relation="PROMPT_DEPENDENCY",
            inferred_context=True,
            origin="prompt",
        )

        constraint_edges.append(
            {
                "before": before,
                "after": after,
                "source_node": before_node,
                "target_node": after_node,
            }
        )

    # ---------------------------------------------------------
    # STAGE 6 / 7
    #
    # Apply relationship semantics.
    # ---------------------------------------------------------

    apply_relationship_semantics(
        domain_graph
    )

    # ---------------------------------------------------------
    # Candidate information for nodes that are not direct
    # candidates but are useful contextual concepts.
    # ---------------------------------------------------------

    inferred_nodes = []

    direct_ids = set()

    for candidates in candidate_map.values():

        for candidate in candidates:

            if candidate["source"] == "direct":
                direct_ids.add(
                    candidate["node_id"]
                )

    for node_id, data in domain_graph.nodes(
        data=True
    ):

        if node_id in direct_ids:
            continue

        inferred_nodes.append(
            {
                "node_id": node_id,
                "name": data.get(
                    "name",
                    node_id,
                ),
                "score": best_node_score.get(
                    node_id,
                    0.0,
                ),
                "inferred": True,
            }
        )

    inferred_nodes.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    candidate_plan = {
        "domain_graph": domain_graph,

        # CRITICAL:
        # This preserves the relationship between a semantic
        # prompt step and its candidate domain nodes.
        "candidate_map": candidate_map,

        "semantic_steps": interpretation.steps,

        "intent": interpretation.intent,

        "inferred_nodes": inferred_nodes,
    }

    debug = {
        "candidates": candidates_debug,

        "context_attachments": context_attachments,

        "prompt_constraint_edges": constraint_edges,

        "inferred_nodes": inferred_nodes,
    }

    return candidate_plan, debug


# =============================================================
# Helpers
# =============================================================


def _add_domain_node(graph, node, source=None, score=None):
    """
    Add a domain node to the contextual graph.

    The node keeps its ontology identity and embedding so that later
    semantic matching can operate on the actual domain representation.
    """

    if node is None:
        return

    node_id = node.id

    data = {
        "name": node.name,
        "node_type": node.node_type,
        "description": node.description,
        "aliases": list(node.aliases or []),
        "embedding": node.embedding,
    }

    if score is not None:
        data["semantic_score"] = score

    if source is not None:
        data["source"] = source

    embedding = data.get("embedding")

    # IMPORTANT:
    # Do not use:
    #
    #     if not embedding:
    #
    # because embeddings may be numpy arrays.
    #
    # A numpy array with multiple values cannot be evaluated
    # directly as a boolean.
    if not _has_embedding(embedding):
        data["embedding"] = None

    if graph.has_node(node_id):
        existing = graph.nodes[node_id]

        # Preserve the strongest semantic score.
        if score is not None:
            existing_score = existing.get("semantic_score")

            if (
                existing_score is None
                or score > existing_score
            ):
                existing["semantic_score"] = score

        # Preserve embedding if the existing node does not have one.
        if (
            not _has_embedding(existing.get("embedding"))
            and _has_embedding(data.get("embedding"))
        ):
            existing["embedding"] = data["embedding"]

        # Preserve source information.
        if source is not None:
            sources = existing.setdefault("sources", [])

            if source not in sources:
                sources.append(source)

        return

    data["sources"] = []

    if source is not None:
        data["sources"].append(source)

    graph.add_node(node_id, **data)

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


def _semantic_similarity(
    a,
    b,
):
    if a is None or b is None:
        return 0.0

    try:
        dot = sum(
            x * y
            for x, y in zip(a, b)
        )

        norm_a = math.sqrt(
            sum(x * x for x in a)
        )

        norm_b = math.sqrt(
            sum(y * y for y in b)
        )

        if norm_a == 0 or norm_b == 0:
            return 0.0

        # Sentence embeddings can theoretically produce
        # negative cosine values. Normalize to [0, 1].
        cosine = dot / (
            norm_a * norm_b
        )

        return max(
            0.0,
            min(
                1.0,
                (cosine + 1.0) / 2.0,
            ),
        )

    except Exception:
        return 0.0


def _lexical_similarity(
    query,
    name,
    aliases=None,
    description="",
):
    query_tokens = set(
        _normalize(query).split()
    )

    if not query_tokens:
        return 0.0

    candidates = [
        name or "",
        *(aliases or []),
        description or "",
    ]

    best = 0.0

    for candidate in candidates:

        candidate_tokens = set(
            _normalize(candidate).split()
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

        score = (
            len(intersection)
            / len(union)
        )

        best = max(
            best,
            score,
        )

    return best


def _normalize(text):
    return (
        str(text)
        .lower()
        .replace("_", " ")
        .strip()
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


def _best_candidate_for_text(
    steps,
    candidate_map,
    text,
):
    """
    Resolve a semantic dependency text to the best matching
    candidate across semantic steps.
    """

    normalized = _normalize(
        text
    )

    best = None
    best_score = -1.0

    for index, step in enumerate(
        steps
    ):

        step_similarity = _lexical_similarity(
            normalized,
            step.text,
        )

        for candidate in candidate_map.get(
            index,
            [],
        ):

            score = (
                0.65 * candidate["score"]
                + 0.35 * step_similarity
            )

            if score > best_score:

                best_score = score
                best = candidate[
                    "node_id"
                ]

    return best

