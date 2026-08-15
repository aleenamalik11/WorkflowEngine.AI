"""
Stage 3-6:
    semantic concepts
        -> candidate domain nodes
        -> immediate neighborhood expansion
        -> contextual subgraph

No Dijkstra.
No shortest-path selection.

The graph is used to discover local semantic context.
"""

import networkx as nx

from relationship_semantics import apply_relationship_semantics


def build_contextual_subgraph(
    interpretation,
    domain_client,
    embedding_service,
    k=5,
    neighborhood_depth=1,
):

    domain_graph = nx.DiGraph()

    candidates_debug = []
    context_attachments = []
    constraint_edges = []

    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # Process EVERY semantic step independently.
    # ---------------------------------------------------------

    for step_index, step in enumerate(
        interpretation.steps
    ):

        embedding = embedding_service.encode(
            step.text
        )

        candidates = domain_client.candidate_nodes(
            text=step.text,
            embedding=embedding,
            k=k,
        )

        candidates_debug.append(
            {
                "step": step.text,
                "explicit": step.explicit,
                "candidates": [
                    {
                        "id": candidate.node.id,
                        "name": candidate.node.name,
                        "type": candidate.node.node_type,
                        "score": candidate.score,
                    }
                    for candidate in candidates
                ],
            }
        )

        # -----------------------------------------------------
        # Add ALL good candidates, not just candidate[0].
        # -----------------------------------------------------

        for candidate in candidates:

            node = candidate.node

            domain_graph.add_node(
                node.id,
                name=node.name,
                node_type=node.node_type,
                description=node.description,
                semantic_score=candidate.score,
                prompt_text=step.text,
                explicit=step.explicit,
            )

        # -----------------------------------------------------
        # Expand immediate neighborhood of candidates.
        # -----------------------------------------------------

        for candidate in candidates:

            relationships = domain_client.neighborhood(
                candidate.node.id,
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

                domain_graph.add_node(
                    source.id,
                    name=source.name,
                    node_type=source.node_type,
                    description=source.description,
                )

                domain_graph.add_node(
                    target.id,
                    name=target.name,
                    node_type=target.node_type,
                    description=target.description,
                )

                domain_graph.add_edge(
                    source.id,
                    target.id,
                    relation=relationship.relation,
                    inferred_context=True,
                )

                context_attachments.append(
                    {
                        "from": candidate.node.name,
                        "source": source.name,
                        "target": target.name,
                        "relation": relationship.relation,
                    }
                )

    # ---------------------------------------------------------
    # Apply relationship semantics.
    #
    # This gives relationships meaning but does NOT perform
    # shortest-path routing.
    # ---------------------------------------------------------

    apply_relationship_semantics(
        domain_graph
    )

    # ---------------------------------------------------------
    # Prompt dependencies from the LLM
    # ---------------------------------------------------------

    for dependency in interpretation.dependencies:

        before = dependency.get("before")
        after = dependency.get("after")

        if not before or not after:
            continue

        before_node = _find_semantic_node(
            domain_graph,
            before,
        )

        after_node = _find_semantic_node(
            domain_graph,
            after,
        )

        if before_node and after_node:

            domain_graph.add_edge(
                before_node,
                after_node,
                relation="PROMPT_DEPENDENCY",
                inferred_context=True,
                weight=0.0,
            )

            constraint_edges.append(
                {
                    "before": before,
                    "after": after,
                    "source_node": before_node,
                    "target_node": after_node,
                }
            )

    candidate_plan = {

        "domain_graph": domain_graph,

        "semantic_steps": interpretation.steps,

        "intent": interpretation.intent,
    }

    debug = {

        "candidates": candidates_debug,

        "context_attachments": context_attachments,

        "prompt_constraint_edges": constraint_edges,
    }

    return candidate_plan, debug


def _find_semantic_node(
    graph,
    text,
):

    normalized = text.lower().strip()

    # Exact semantic text/name match first.
    for node_id, data in graph.nodes(data=True):

        name = data.get(
            "name",
            "",
        ).lower().strip()

        prompt_text = data.get(
            "prompt_text",
            "",
        ).lower().strip()

        if normalized == name:
            return node_id

        if normalized == prompt_text:
            return node_id

    # Then substring match.
    for node_id, data in graph.nodes(data=True):

        name = data.get(
            "name",
            "",
        ).lower()

        if normalized in name or name in normalized:
            return node_id

    return None