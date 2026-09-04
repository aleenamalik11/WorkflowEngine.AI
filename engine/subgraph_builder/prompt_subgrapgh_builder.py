import networkx as nx
import re

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
            # Retrieve candidates for THIS semantic step only.
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
                    "explicit": bool(step.explicit),
                    "inferred": not bool(step.explicit),
                    "source": "direct",
                    "prompt_text": step.text,

                    # -------------------------------------------------
                    # IMPORTANT:
                    # Preserve the semantic step identity.
                    # Later stages must never guess the step by
                    # embedding the dependency text again.
                    # -------------------------------------------------
                    "step_index": step_index,
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
                step_index=step_index,
            )

            step_candidates.extend(
                inferred_candidates
            )

            # -------------------------------------------------
            # Deduplicate candidates belonging to THIS step.
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
                "step_index": step_index,
                "step_text": step.text,
                "step_embedding": step_embedding,
                "candidates": step_candidates,
            }

        # -----------------------------------------------------
        # Add prompt dependencies using semantic step indexes.
        # -----------------------------------------------------

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

        # Keep the strongest candidate while preserving
        # the semantic step identity.
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
    # Neighborhood
    # =========================================================

    def expand_neighborhood(
        self,
        prompt_subgraph,
        step,
        step_embedding,
        step_candidates,
        neighborhood_depth,
        step_index=None,
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
                        step_index=step_index,
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
        step_index=None,
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

                    # Preserve the originating semantic step.
                    "step_index": step_index,
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
        """
        Convert semantic prompt dependencies into graph
        relationships.

        IMPORTANT:

        The old implementation attempted to identify the
        `before` and `after` steps by embedding their text and
        finding the most similar semantic step.

        That is unsafe.

        Example:

            "reject the transaction"
            "record the failed transaction"
            "notify the customer"

        can all have similar embeddings. The dependency
        resolver could therefore accidentally map multiple
        prompt steps to the same domain operation.

        This implementation first resolves the dependency
        against the actual SemanticStep text/index and then
        selects the best candidate ONLY from that step's
        candidate list.
        """

        for dependency in (
            interpretation.dependencies
        ):

            before_text = dependency.get(
                "before"
            )

            after_text = dependency.get(
                "after"
            )

            relation = dependency.get(
                "relation",
                "PROMPT_DEPENDENCY",
            )

            if (
                not before_text
                or not after_text
            ):
                continue

            before_index = (
                self._find_step_index(
                    interpretation,
                    before_text,
                )
            )

            after_index = (
                self._find_step_index(
                    interpretation,
                    after_text,
                )
            )

            if (
                before_index is None
                or after_index is None
            ):
                continue

            # -------------------------------------------------
            # IMPORTANT:
            # Resolve each side ONLY against candidates
            # belonging to that semantic step.
            # -------------------------------------------------

            before_node = (
                self._best_candidate_for_step(
                    candidate_map,
                    before_index,
                )
            )

            after_node = (
                self._best_candidate_for_step(
                    candidate_map,
                    after_index,
                )
            )

            if (
                before_node is None
                or after_node is None
            ):
                continue

            if before_node == after_node:
                continue

            # -------------------------------------------------
            # Conditional dependency
            # -------------------------------------------------

            if relation == "PROMPT_CONDITION":

                condition = dependency.get(
                    "condition",
                    "",
                )

                branch = dependency.get(
                    "branch",
                    "then",
                )

                # Mark the semantic candidate nodes with
                # explicit prompt-condition metadata.
                #
                # This is intentionally metadata at this stage.
                # Stage 10/11 is responsible for translating
                # this into the actual WorkflowModel condition
                # structure.

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
                        "branch"
                    ] = branch

                    prompt_subgraph.nodes[
                        after_node
                    ][
                        "is_conditional"
                    ] = True

                    prompt_subgraph.nodes[
                        after_node
                    ][
                        "condition_source"
                    ] = before_node

                    prompt_subgraph.nodes[
                        after_node
                    ][
                        "condition_source_step_index"
                    ] = before_index

                    prompt_subgraph.nodes[
                        after_node
                    ][
                        "conditional_step_index"
                    ] = after_index

                if prompt_subgraph.has_node(
                    before_node
                ):

                    prompt_subgraph.nodes[
                        before_node
                    ][
                        "is_condition_check"
                    ] = True

                    # Keep all conditional targets rather
                    # than overwriting the previous target.
                    existing_targets = (
                        prompt_subgraph.nodes[
                            before_node
                        ].get(
                            "condition_targets",
                            [],
                        )
                    )

                    if after_node not in existing_targets:
                        existing_targets.append(
                            after_node
                        )

                    prompt_subgraph.nodes[
                        before_node
                    ][
                        "condition_targets"
                    ] = existing_targets

                    prompt_subgraph.nodes[
                        before_node
                    ][
                        "condition"
                    ] = condition

                # -------------------------------------------------
                # Add an explicit prompt edge as well.
                #
                # This preserves the dependency for Stage 9
                # ordering and allows Stage 10/11 to construct
                # the actual branch.
                # -------------------------------------------------

                prompt_subgraph.add_edge(
                    before_node,
                    after_node,
                    relation="PROMPT_CONDITION",
                    condition=condition,
                    branch=branch,
                    inferred_context=False,
                    origin="prompt",
                )

                continue

            # -------------------------------------------------
            # Normal sequential dependency
            # -------------------------------------------------

            prompt_subgraph.add_edge(
                before_node,
                after_node,
                relation="PROMPT_DEPENDENCY",
                inferred_context=True,
                origin="prompt",
            )

        return prompt_subgraph

    # =========================================================
    # Semantic step resolution
    # =========================================================

    def _find_step_index(
        self,
        interpretation,
        text,
    ):
        """
        Find the exact SemanticStep represented by a
        dependency endpoint.

        Prefer normalized exact text over embeddings.

        This is important because dependency endpoints already
        originate from the semantic parser.
        """

        normalized_target = self._normalize_text(
            text
        )

        if not normalized_target:
            return None

        # -----------------------------------------------------
        # First: exact normalized match.
        # -----------------------------------------------------

        for index, step in enumerate(
            interpretation.steps
        ):

            if (
                self._normalize_text(
                    step.text
                )
                == normalized_target
            ):
                return index

        # -----------------------------------------------------
        # Second: normalized containment.
        #
        # Useful if one stage cleaned punctuation differently
        # from another stage.
        # -----------------------------------------------------

        for index, step in enumerate(
            interpretation.steps
        ):

            normalized_step = (
                self._normalize_text(
                    step.text
                )
            )

            if not normalized_step:
                continue

            if (
                normalized_target
                in normalized_step
                or normalized_step
                in normalized_target
            ):
                return index

        return None

    def _best_candidate_for_step(
        self,
        candidate_map,
        step_index,
    ):
        """
        Select the best candidate belonging to one specific
        semantic step.

        Never compare candidates belonging to other prompt
        steps here.
        """

        entry = candidate_map.get(
            step_index
        )

        if not entry:
            return None

        candidates = entry.get(
            "candidates",
            [],
        )

        if not candidates:
            return None

        # Prefer direct candidates over inferred neighborhood
        # candidates, then use the candidate score.
        ranked = sorted(
            candidates,
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

        return ranked[0].get(
            "node_id"
        )

    # =========================================================
    # Backward-compatible helper
    # =========================================================

    def best_candidate_for_embedding(
        self,
        candidate_map,
        dependency_embedding,
    ):
        """
        Backward-compatible helper.

        This method is intentionally retained because older
        pipeline code may still call it.

        New dependency processing MUST NOT use it because an
        embedding cannot reliably identify which semantic step
        a dependency endpoint belongs to.

        If called by older code, it performs the old broad
        lookup behavior, but set_execution_order() no longer
        depends on it.
        """

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

        candidates = best_entry.get(
            "candidates",
            [],
        )

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

    # =========================================================
    # Text normalization
    # =========================================================

    @staticmethod
    def _normalize_text(
        text,
    ):
        if text is None:
            return ""

        text = str(text).lower()

        text = re.sub(
            r"[^a-z0-9\s]+",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()