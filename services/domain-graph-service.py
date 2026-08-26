from helpers.similarity_utils import _lexical_similarity, _cosine_similarity
from helpers.utils import _has_embedding, _domain_node_text
from models import DomainNode, ScoredNode


class DomainGraphService:

    def __init__(
            self,
            embedding_service,
            domain_graph_accessor
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

        # -----------------------------------------------------
        # Get ontology nodes.
        # -----------------------------------------------------

        all_nodes = self._domain_graph_accessor.all_nodes()

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

            # -------------------------------------------------
            # Make sure the domain node has an embedding.
            # -------------------------------------------------

            node_embedding = (
                self._get_or_create_embedding(
                    node
                )
            )

            semantic_score = (
                _cosine_similarity(embedding, node_embedding)
                if (
                        embedding is not None
                        and node_embedding is not None
                )
                else 0.0
            )

            # -------------------------------------------------
            # Lexical similarity.
            #
            # Compare against:
            #
            #   name
            #   aliases
            #   description
            #   id
            #   ontology types
            # -------------------------------------------------

            lexical_score = (
                _lexical_similarity(
                    text,
                    node,
                )
            )

            # -------------------------------------------------
            # Combined score.
            #
            # Semantic similarity dominates.
            # Lexical similarity helps exact terminology.
            # -------------------------------------------------

            score = (
                    0.70 * semantic_score
                    + 0.30 * lexical_score
            )

            if score <= 0.0:
                continue

            results.append(
                ScoredNode(
                    node=node,
                    score=score,
                    lexical_score=lexical_score,
                    semantic_score=semantic_score,
                )
            )

        results.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return results[:k]



    def _get_or_create_embedding(
            self,
            node: DomainNode,
    ):

        if _has_embedding(node.embedding):
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

        self._embedding_cache[node.id] = embedding

        node.embedding = embedding

        return embedding
