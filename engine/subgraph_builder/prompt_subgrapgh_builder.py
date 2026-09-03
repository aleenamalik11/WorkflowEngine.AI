import networkx as nx

from helpers.similarity_utils import (
    _cosine_similarity,
    _lexical_similarity,
    _combined_score,
    _relationship_relevance,
    _node_text,
)
from helpers.utils import _add_domain_node
from models import SemanticInterpretation


COMPOSITE_RELATIONSHIP = "OPERATION_INCLUDES"


class PromptSubGraphBuilder:

    def __init__(
        self,
        embedding_service,
        domain_graph_service,
    ):
        self._embedding_service = embedding_service
        self._domain_graph_service = domain_graph_service

    # =========================================================
    # Build
    # =========================================================

    def build(
        self,
        interpretation: SemanticInterpretation,
        k=5,
        neighborhood_depth=1,
    ):

        prompt_domain_subgraph = nx.DiGraph()

        candidate_map = {}

        for step_index, step in enumerate(
            interpretation.steps
        ):

            step_embedding = (
                self._embedding_service.encode(
                    step.text
                )
            )

            # -------------------------------------------------
            # Retrieve a larger candidate set before truncation.
            # -------------------------------------------------

            candidates = (
                self._domain_graph_service.candidate_nodes(
                    step.text,
                    step_embedding,
                    k=max(k, 25),
                )
            )

            step_candidates = []

            for candidate in candidates:

                node = candidate.node

                candidate_embedding = (
                    candidate.embedding
                )

                if candidate_embedding is None:
                    candidate_embedding = (
                        self._embedding_service.encode(
                            _node_text(node)
                        )
                    )

                semantic_similarity = (
                    _cosine_similarity(
                        candidate_embedding,
                        step_embedding,
                    )
                )

                lexical_similarity = (
                    _lexical_similarity(
                        step.text,
                        node,
                    )
                )

                score = _combined_score(
                    lexical_similarity,
                    semantic_similarity,
                    candidate.score,
                )

                item = {
                    "node_id": node.id,
                    "name": node.name,
                    "node_type": node.node_type,
                    "score": score,
                    "lexical_score": lexical_similarity,
                    "semantic_score": semantic_similarity,
                    "explicit": bool(
                        step.explicit
                    ),
                    "inferred": not bool(
                        step.explicit
                    ),
                    "source": "direct",
                    "prompt_text": step.text,
                }

                self._add_candidate(
                    step_candidates,
                    item,
                )

                _add_domain_node(
                    prompt_domain_subgraph,
                    node,
                    item,
                )

            # -------------------------------------------------
            # Expand the candidate neighborhood.
            # -------------------------------------------------

            (
                prompt_domain_subgraph,
                inferred_candidates,
            ) = self.expand_neighborhood(
                prompt_domain_subgraph,
                step,
                step_embedding,
                step_candidates,
                neighborhood_depth,
            )

            step_candidates.extend(
                inferred_candidates
            )

            # -------------------------------------------------
            # IMPORTANT:
            #
            # If a composite operation is present in the graph,
            # make sure it survives candidate ranking.
            #
            # Example:
            #
            # transfer_funds
            #     |
            #     +-- retrieve_account
            #     +-- debit_account
            #     +-- credit_account
            #
            # "transfer funds" means transfer_funds, not one
            # arbitrary implementation child.
            # -------------------------------------------------

            self._ensure_composite_parents(
                prompt_domain_subgraph,
                step,
                step_embedding,
                step_candidates,
            )

            # -------------------------------------------------
            # Deduplicate.
            # -------------------------------------------------

            step_candidates = (
                self._deduplicate(
                    step_candidates
                )
            )

            # -------------------------------------------------
            # Exact/direct candidates get priority.
            # -------------------------------------------------

            step_candidates.sort(
                key=lambda candidate: (
                    candidate.get(
                        "lexical_score",
                        0.0,
                    ) >= 0.90,

                    candidate.get(
                        "source"
                    ) == "direct",

                    candidate.get(
                        "score",
                        0.0,
                    ),
                ),
                reverse=True,
            )

            step_candidates = step_candidates[
                : max(k * 2, 10)
            ]

            candidate_map[
                step_index
            ] = {
                "step_embedding": step_embedding,
                "candidates": step_candidates,
            }

        prompt_domain_subgraph = (
            self.set_execution_order(
                interpretation,
                prompt_domain_subgraph,
                candidate_map,
            )
        )

        return {
            "prompt_domain_subgraph": (
                prompt_domain_subgraph
            ),
            "candidate_map": candidate_map,
            "semantic_steps": interpretation.steps,
            "intent": interpretation.intent,
        }

    # =========================================================
    # Candidate helpers
    # =========================================================

    def _add_candidate(
        self,
        candidates,
        item,
    ):

        existing = next(
            (
                candidate
                for candidate in candidates
                if candidate["node_id"]
                == item["node_id"]
            ),
            None,
        )

        if existing is None:
            candidates.append(item)
            return

        if (
            item["score"]
            > existing["score"]
        ):
            index = candidates.index(
                existing
            )

            candidates[index] = item

    def _deduplicate(
        self,
        candidates,
    ):

        best = {}

        for candidate in candidates:

            node_id = candidate[
                "node_id"
            ]

            existing = best.get(
                node_id
            )

            if (
                existing is None
                or candidate["score"]
                > existing["score"]
            ):
                best[node_id] = candidate

        return list(
            best.values()
        )

    # =========================================================
    # Composite operation discovery
    # =========================================================

    def _ensure_composite_parents(
        self,
        graph,
        step,
        step_embedding,
        step_candidates,
    ):
        """
        Ensure that an operation that owns executable operations
        through OPERATION_INCLUDES is available as the semantic root.
        """

        candidate_ids = {
            candidate["node_id"]
            for candidate in step_candidates
        }

        # -----------------------------------------------------
        # Inspect every operation already present in the graph.
        # -----------------------------------------------------

        operation_ids = [
            node_id
            for node_id, data
            in graph.nodes(data=True)
            if data.get("node_type")
            in {
                "Operation",
                "operation",
            }
        ]

        for operation_id in operation_ids:

            children = []

            for _, child_id, edge_data in (
                graph.out_edges(
                    operation_id,
                    data=True,
                )
            ):

                if edge_data.get(
                    "relation"
                ) != COMPOSITE_RELATIONSHIP:
                    continue

                if not graph.has_node(
                    child_id
                ):
                    continue

                child_type = (
                    graph.nodes[
                        child_id
                    ].get("node_type")
                )

                if child_type in {
                    "Operation",
                    "operation",
                }:
                    children.append(
                        child_id
                    )

            if not children:
                continue

            node = (
                self._domain_graph_service.get_node(
                    operation_id
                )
            )

            if node is None:
                continue

            semantic_score = 0.0

            node_embedding = node.embedding

            if node_embedding is None:
                node_embedding = (
                    self._embedding_service.encode(
                        _node_text(node)
                    )
                )

            if node_embedding is not None:
                semantic_score = (
                    _cosine_similarity(
                        node_embedding,
                        step_embedding,
                    )
                )

            lexical_score = (
                _lexical_similarity(
                    step.text,
                    node,
                )
            )

            # -------------------------------------------------
            # Do NOT promote arbitrary composite operations.
            #
            # Exact name/alias matches are authoritative.
            # Otherwise require meaningful semantic similarity.
            # -------------------------------------------------

            if (
                lexical_score < 0.50
                and semantic_score < 0.60
            ):
                continue

            score = (
                0.55 * semantic_score
                + 0.45 * lexical_score
            )

            # Exact lexical identity should dominate.
            if lexical_score >= 0.90:
                score = max(
                    score,
                    1.0,
                )

            item = {
                "node_id": operation_id,
                "name": node.name,
                "node_type": node.node_type,
                "score": score,
                "lexical_score": lexical_score,
                "semantic_score": semantic_score,
                "explicit": True,
                "inferred": False,
                "source": "direct",
                "prompt_text": step.text,
                "composite": True,
                "composite_children": children,
            }

            existing = next(
                (
                    candidate
                    for candidate in step_candidates
                    if candidate["node_id"]
                    == operation_id
                ),
                None,
            )

            if existing is None:
                step_candidates.append(
                    item
                )
            else:
                existing["score"] = max(
                    existing["score"],
                    score,
                )

                existing[
                    "lexical_score"
                ] = max(
                    existing.get(
                        "lexical_score",
                        0.0,
                    ),
                    lexical_score,
                )

                existing[
                    "semantic_score"
                ] = max(
                    existing.get(
                        "semantic_score",
                        0.0,
                    ),
                    semantic_score,
                )

                existing[
                    "composite"
                ] = True

                existing[
                    "composite_children"
                ] = children

            # -------------------------------------------------
            # Ensure the composite node is present.
            # -------------------------------------------------

            _add_domain_node(
                graph,
                node,
                item,
            )

            # -------------------------------------------------
            # Ensure all included children and their edges exist.
            # -------------------------------------------------

            for child_id in children:

                child = (
                    self._domain_graph_service.get_node(
                        child_id
                    )
                )

                if child is None:
                    continue

                _add_domain_node(
                    graph,
                    child,
                    None,
                )

                graph.add_edge(
                    operation_id,
                    child_id,
                    relation=COMPOSITE_RELATIONSHIP,
                    inferred_context=True,
                    origin="domain",
                )

                # -------------------------------------------------
                # IMPORTANT:
                #
                # Pull the child's immediate operation relationships
                # too. If the ontology contains:
                #
                # A -> B OPERATION_PRECEDES
                #
                # we need it in the prompt subgraph for ordering.
                # -------------------------------------------------

                child_relationships = (
                    self._domain_graph_service.neighborhood(
                        child_id,
                        1,
                    )
                )

                for relationship in child_relationships:

                    source = (
                        self._domain_graph_service.get_node(
                            relationship.source_id
                        )
                    )

                    target = (
                        self._domain_graph_service.get_node(
                            relationship.target_id
                        )
                    )

                    if (
                        source is None
                        or target is None
                    ):
                        continue

                    if (
                        source.node_type
                        not in {
                            "Operation",
                            "operation",
                        }
                        or target.node_type
                        not in {
                            "Operation",
                            "operation",
                        }
                    ):
                        continue

                    _add_domain_node(
                        graph,
                        source,
                        None,
                    )

                    _add_domain_node(
                        graph,
                        target,
                        None,
                    )

                    graph.add_edge(
                        source.id,
                        target.id,
                        relation=relationship.relation,
                        inferred_context=True,
                        origin="domain",
                    )

    # =========================================================
    # Neighborhood
    # =========================================================

    def expand_neighborhood(
        self,
        prompt_subgraph,
        step,
        step_embedding,
        step_candidates,
        neighborhood_depth,
    ):

        neighborhood_seed_ids = [
            item["node_id"]
            for item in step_candidates
        ]

        inferred_candidates = []

        for seed_id in neighborhood_seed_ids:

            relationships = (
                self._domain_graph_service.neighborhood(
                    seed_id,
                    neighborhood_depth,
                )
            )

            for relationship in relationships:

                source = (
                    self._domain_graph_service.get_node(
                        relationship.source_id
                    )
                )

                target = (
                    self._domain_graph_service.get_node(
                        relationship.target_id
                    )
                )

                if (
                    source is None
                    or target is None
                ):
                    continue

                _add_domain_node(
                    prompt_subgraph,
                    source,
                    None,
                )

                _add_domain_node(
                    prompt_subgraph,
                    target,
                    None,
                )

                prompt_subgraph.add_edge(
                    source.id,
                    target.id,
                    relation=relationship.relation,
                    inferred_context=True,
                    origin="domain",
                )

                inferred_candidates.extend(
                    self.infer_candidates(
                        step,
                        step_embedding,
                        step_candidates,
                        source,
                        target,
                        relationship,
                    )
                )

        return (
            prompt_subgraph,
            inferred_candidates,
        )

    # =========================================================
    # Inferred candidates
    # =========================================================

    def infer_candidates(
        self,
        step,
        step_embedding,
        step_candidates,
        source,
        target,
        relationship,
    ):

        inferred_candidates = []

        existing_ids = {
            item["node_id"]
            for item in step_candidates
        }

        for neighbor in (
            source,
            target,
        ):

            if neighbor.id in existing_ids:
                continue

            if neighbor.node_type not in {
                "Operation",
                "operation",
            }:
                continue

            neighbor_embedding = (
                neighbor.embedding
            )

            if neighbor_embedding is None:
                neighbor_embedding = (
                    self._embedding_service.encode(
                        _node_text(neighbor)
                    )
                )

            lexical_score = (
                _lexical_similarity(
                    step.text,
                    neighbor,
                )
            )

            semantic_score = (
                _cosine_similarity(
                    step_embedding,
                    neighbor_embedding,
                )
            )

            relation_score = (
                _relationship_relevance(
                    relationship.relation
                )
            )

            contextual_score = (
                0.50 * semantic_score
                + 0.25 * lexical_score
                + 0.25 * relation_score
            )

            if contextual_score < 0.25:
                continue

            inferred_candidates.append(
                {
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
            )

        return inferred_candidates

    # =========================================================
    # Execution dependencies
    # =========================================================

    def set_execution_order(
        self,
        interpretation,
        prompt_subgraph,
        candidate_map,
    ):

        for dependency in (
            interpretation.dependencies
        ):

            before = dependency.get(
                "before"
            )

            after = dependency.get(
                "after"
            )

            relation = dependency.get(
                "relation",
                "PROMPT_DEPENDENCY",
            )

            if not before or not after:
                continue

            before_embedding = (
                self._embedding_service.encode(
                    before
                )
            )

            after_embedding = (
                self._embedding_service.encode(
                    after
                )
            )

            before_node = (
                self.best_candidate_for_embedding(
                    candidate_map,
                    before_embedding,
                )
            )

            after_node = (
                self.best_candidate_for_embedding(
                    candidate_map,
                    after_embedding,
                )
            )

            if (
                before_node is None
                or after_node is None
            ):
                continue

            if before_node == after_node:
                continue

            if relation == "PROMPT_CONDITION":

                condition = dependency.get(
                    "condition",
                    "",
                )

                if prompt_subgraph.has_node(
                    after_node
                ):

                    prompt_subgraph.nodes[
                        after_node
                    ][
                        "condition"
                    ] = condition

                    prompt_subgraph.nodes[
                        after_node
                    ][
                        "is_conditional"
                    ] = True

                if prompt_subgraph.has_node(
                    before_node
                ):

                    prompt_subgraph.nodes[
                        before_node
                    ][
                        "is_condition_check"
                    ] = True

                    prompt_subgraph.nodes[
                        before_node
                    ][
                        "condition_for"
                    ] = after_node

                continue

            prompt_subgraph.add_edge(
                before_node,
                after_node,
                relation="PROMPT_DEPENDENCY",
                inferred_context=True,
                origin="prompt",
            )

        return prompt_subgraph

    # =========================================================
    # Dependency candidate
    # =========================================================

    def best_candidate_for_embedding(
        self,
        candidate_map,
        dependency_embedding,
    ):

        best_entry = None
        best_similarity = float(
            "-inf"
        )

        for entry in (
            candidate_map.values()
        ):

            similarity = (
                _cosine_similarity(
                    dependency_embedding,
                    entry[
                        "step_embedding"
                    ],
                )
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry

        if best_entry is None:
            return None

        candidates = best_entry[
            "candidates"
        ]

        if not candidates:
            return None

        best_candidate = max(
            candidates,
            key=lambda candidate:
                candidate.get(
                    "score",
                    0.0,
                ),
        )

        if best_similarity < 0.5:
            return None

        return best_candidate.get(
            "node_id"
        )