from helpers.similarity_utils import (
    _lexical_similarity,
    _cosine_similarity,
)
from helpers.utils import (
    _has_embedding,
    _domain_node_text,
)
from models import DomainNode, ScoredNode


EXECUTABLE_NODE_TYPES = {
    "Operation",
    "operation",
}


class DomainGraphService:

    def __init__(
        self,
        embedding_service,
        domain_graph_accessor,
    ):
        self._embedding_service = embedding_service
        self._domain_graph_accessor = domain_graph_accessor
        self._embedding_cache = {}

    # =========================================================
    # Candidate retrieval
    # =========================================================

    def candidate_nodes(
        self,
        text,
        embedding,
        node_types=None,
        k=5,
    ) -> list[ScoredNode]:

        if not text:
            return []

        requested_types = (
            set(node_types)
            if node_types
            else None
        )

        all_nodes = (
            self._domain_graph_accessor.all_nodes()
        )

        results = []

        for node in all_nodes:

            if (
                requested_types
                and not requested_types.intersection(
                    set(node.types or [])
                    | {node.node_type}
                )
            ):
                continue

            node_embedding = (
                self._get_or_create_embedding(node)
            )

            semantic_score = (
                _cosine_similarity(
                    embedding,
                    node_embedding,
                )
                if (
                    embedding is not None
                    and node_embedding is not None
                )
                else 0.0
            )

            lexical_score = (
                _lexical_similarity(
                    text,
                    node,
                )
            )

            # -------------------------------------------------
            # Base semantic score.
            # -------------------------------------------------

            score = (
                0.70 * semantic_score
                + 0.30 * lexical_score
            )

            # -------------------------------------------------
            # IMPORTANT:
            #
            # If the user explicitly names an operation, lexical
            # identity must beat embedding similarity.
            #
            # Example:
            #
            # prompt:
            #     "transfer funds"
            #
            # node:
            #     "Transfer Funds"
            #
            # This must beat:
            #
            #     "Persist Fund Transfer"
            #
            # even if the latter has a slightly better embedding
            # similarity.
            # -------------------------------------------------

            if node.node_type in EXECUTABLE_NODE_TYPES:

                if lexical_score >= 0.90:
                    score = max(
                        score,
                        0.95,
                    )

                elif lexical_score >= 0.75:
                    score = max(
                        score,
                        0.85,
                    )

                elif lexical_score >= 0.50:
                    score = max(
                        score,
                        0.70,
                    )

            if score <= 0.0:
                continue

            results.append(
                ScoredNode(
                    node=node,
                    score=score,
                    lexical_score=lexical_score,
                    semantic_score=semantic_score,
                    embedding=node_embedding,
                )
            )

        # -----------------------------------------------------
        # Exact lexical matches first.
        #
        # This is intentionally a stable two-level ordering:
        #
        # 1. lexical identity
        # 2. overall semantic score
        #
        # The domain ontology is authoritative for operation names.
        # -----------------------------------------------------

        results.sort(
            key=lambda item: (
                item.lexical_score >= 0.90,
                item.lexical_score >= 0.75,
                item.score,
            ),
            reverse=True,
        )

        return results[:k]

    # =========================================================
    # Embeddings
    # =========================================================

    def _get_or_create_embedding(
        self,
        node: DomainNode,
    ):

        if _has_embedding(
            node.embedding
        ):
            return node.embedding

        cached = self._embedding_cache.get(
            node.id
        )

        if _has_embedding(cached):
            node.embedding = cached
            return cached

        if self._embedding_service is None:
            return None

        text = _domain_node_text(node)

        if not text:
            return None

        embedding = (
            self._embedding_service.encode(
                text
            )
        )

        self._embedding_cache[
            node.id
        ] = embedding

        node.embedding = embedding

        return embedding

    # =========================================================
    # Neighborhood
    # =========================================================

    def neighborhood(
        self,
        node_id,
        depth,
    ):
        return (
            self._domain_graph_accessor.neighborhood(
                node_id,
                depth,
            )
        )

    # =========================================================
    # Get node
    # =========================================================

    def get_node(
        self,
        node_id,
    ) -> DomainNode | None:

        return (
            self._domain_graph_accessor.get_node(
                node_id
            )
        )