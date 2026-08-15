"""
Domain graph access layer.

Neo4j is the source of truth for the domain ontology.

Responsibilities:

    1. Retrieve domain nodes from Neo4j.
    2. Match prompt concepts against domain nodes using:
           - semantic embedding similarity
           - lexical similarity
    3. Retrieve immediate graph neighborhoods.
    4. Return the same contract for Neo4j and in-memory graphs.

This module does NOT:

    - perform workflow planning
    - perform Dijkstra
    - perform shortest-path routing
    - choose registered implementation functions
    - create workflow execution order

The domain graph represents semantic/domain structure.
The planner decides execution order later.
"""

import abc
import math
import re

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# =============================================================
# Domain models
# =============================================================

@dataclass
class DomainNode:
    id: str
    name: str
    node_type: str

    description: str = ""

    aliases: List[str] = field(
        default_factory=list
    )

    embedding: Optional[Any] = None

    # Preserve all ontology types from Neo4j.
    types: List[str] = field(
        default_factory=list
    )


@dataclass
class DomainRelationship:
    source_id: str
    target_id: str
    relation: str


@dataclass
class ScoredNode:
    node: DomainNode

    score: float

    lexical_score: float = 0.0

    semantic_score: float = 0.0


# =============================================================
# Interface
# =============================================================

class DomainGraphClient(abc.ABC):

    @abc.abstractmethod
    def candidate_nodes(
        self,
        text: str,
        embedding,
        node_types=None,
        k=5,
    ) -> List[ScoredNode]:
        """
        Return the top-k semantically and lexically relevant
        domain concepts.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def neighborhood(
        self,
        node_id: str,
        depth: int = 1,
    ) -> List[DomainRelationship]:
        """
        Return ontology relationships around a node.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_node(
        self,
        node_id: str,
    ) -> Optional[DomainNode]:
        raise NotImplementedError

    @abc.abstractmethod
    def all_nodes(self) -> List[DomainNode]:
        """
        Return all ontology nodes.

        Used for:

            - semantic retrieval
            - LLM domain context
        """
        raise NotImplementedError


# =============================================================
# Neo4j implementation
# =============================================================

class Neo4jDomainGraph(DomainGraphClient):

    def __init__(
        self,
        driver,
        database="neo4j",
        embedding_service=None,
        fulltext_index="domainNodeSearch",
    ):
        self.driver = driver

        self.database = database

        self.embedding_service = (
            embedding_service
        )

        # Kept for compatibility/configuration.

        self.fulltext_index = (
            fulltext_index
        )

        # -----------------------------------------------------
        # In-memory embedding cache.
        #
        # Neo4j's current importer does not store embeddings.
        # We therefore generate them once per process and cache
        # them here.
        # -----------------------------------------------------

        self._embedding_cache = {}

        # Cache complete domain nodes as well.
        self._node_cache = None

    # =========================================================
    # Retrieve all nodes
    # =========================================================

    def all_nodes(self) -> List[DomainNode]:

        if self._node_cache is not None:
            return list(
                self._node_cache.values()
            )

        # The ontology importer creates :GraphNode.
        #
        # Use the label explicitly rather than MATCH (n),
        # so unrelated Neo4j application data is not accidentally
        # treated as domain ontology.

        cypher = """
        MATCH (n:GraphNode)
        RETURN n
        """

        with self.driver.session(
            database=self.database
        ) as session:

            rows = session.run(
                cypher
            )

            nodes = {}

            for row in rows:

                node = self._to_domain_node(
                    row["n"]
                )

                if node.id:
                    nodes[node.id] = node

            self._node_cache = nodes

        return list(
            self._node_cache.values()
        )

    # =========================================================
    # Candidate retrieval
    # =========================================================

    def candidate_nodes(
        self,
        text,
        embedding,
        node_types=None,
        k=5,
    ) -> List[ScoredNode]:

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

        all_nodes = self.all_nodes()

        results = []

        # -----------------------------------------------------
        # Score every relevant domain node.
        #
        # IMPORTANT:
        #
        # We intentionally do NOT first use full-text search
        # and then calculate embeddings only on that subset.
        #
        # That architecture can miss semantically related
        # concepts whose wording differs from the prompt.
        # -----------------------------------------------------

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

    # =========================================================
    # Neighborhood
    # =========================================================

    def neighborhood(
        self,
        node_id,
        depth=1,
    ) -> List[DomainRelationship]:

        depth = max(
            1,
            int(depth),
        )

        # The ontology importer creates GraphNode nodes.
        #
        # We preserve relationship direction, while allowing
        # neighborhood traversal to discover both incoming and
        # outgoing relationships.

        cypher = f"""
        MATCH (n:GraphNode {{id: $id}})
              -[r*1..{depth}]-
              (m:GraphNode)

        UNWIND r AS rel

        RETURN DISTINCT
            startNode(rel).id AS source_id,
            type(rel) AS relation,
            endNode(rel).id AS target_id
        """

        with self.driver.session(
            database=self.database
        ) as session:

            rows = session.run(
                cypher,
                id=node_id,
            )

            relationships = []

            seen = set()

            for row in rows:

                source_id = row[
                    "source_id"
                ]

                target_id = row[
                    "target_id"
                ]

                relation = row[
                    "relation"
                ]

                key = (
                    source_id,
                    relation,
                    target_id,
                )

                if key in seen:
                    continue

                seen.add(key)

                relationships.append(
                    DomainRelationship(
                        source_id=source_id,
                        target_id=target_id,
                        relation=relation,
                    )
                )

            return relationships

    # =========================================================
    # Get one node
    # =========================================================

    def get_node(
        self,
        node_id,
    ) -> Optional[DomainNode]:

        # First check cache.

        if self._node_cache is not None:

            cached = self._node_cache.get(
                node_id
            )

            if cached is not None:
                return cached

        cypher = """
        MATCH (n:GraphNode {id: $id})
        RETURN n
        LIMIT 1
        """

        with self.driver.session(
            database=self.database
        ) as session:

            row = session.run(
                cypher,
                id=node_id,
            ).single()

            if not row:
                return None

            node = self._to_domain_node(
                row["n"]
            )

            if self._node_cache is None:
                self._node_cache = {}

            self._node_cache[
                node.id
            ] = node

            return node

    # =========================================================
    # Embedding handling
    # =========================================================

    def _get_or_create_embedding(
        self,
        node: DomainNode,
    ):
        """
        Return the node embedding.

        Priority:

            1. Existing Neo4j embedding.
            2. In-memory cache.
            3. Generate embedding from ontology text.

        The current ontology importer does not store embeddings,
        so #3 is expected for the current repository.
        """

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

        if self.embedding_service is None:
            return None

        text = _domain_node_text(
            node
        )

        if not text:
            return None

        embedding = (
            self.embedding_service.encode(
                text
            )
        )

        self._embedding_cache[
            node.id
        ] = embedding

        node.embedding = embedding

        return embedding

    # =========================================================
    # Neo4j -> DomainNode
    # =========================================================

    @staticmethod
    def _to_domain_node(
        neo4j_node,
    ) -> DomainNode:

        props = dict(
            neo4j_node
        )

        labels = list(
            neo4j_node.labels
        ) if hasattr(
            neo4j_node,
            "labels",
        ) else []

        raw_types = (
            props.get(
                "types",
                [],
            )
            or []
        )

        if isinstance(
            raw_types,
            str,
        ):
            raw_types = [
                raw_types
            ]

        types = list(
            dict.fromkeys(
                [
                    *raw_types,
                    *[
                        label
                        for label in labels
                        if label
                        != "GraphNode"
                    ],
                ]
            )
        )

        # Prefer ontology type rather than blindly using the
        # first Neo4j label.

        node_type = (
            types[0]
            if types
            else props.get(
                "type",
                "DomainEntity",
            )
        )

        aliases = (
            props.get(
                "aliases",
                [],
            )
            or []
        )

        if isinstance(
            aliases,
            str,
        ):
            aliases = [
                aliases
            ]

        embedding = props.get(
            "embedding"
        )

        return DomainNode(
            id=str(
                props.get(
                    "id"
                )
            ),
            name=str(
                props.get(
                    "name",
                    props.get(
                        "id",
                        "",
                    ),
                )
            ),
            node_type=node_type,
            description=str(
                props.get(
                    "description",
                    "",
                )
                or ""
            ),
            aliases=list(
                aliases
            ),
            embedding=embedding,
            types=types,
        )


# =============================================================
# In-memory implementation
# =============================================================

class InMemoryDomainGraph(
    DomainGraphClient
):

    def __init__(
        self,
        nodes: Dict[
            str,
            DomainNode
        ],
        relationships: List[
            DomainRelationship
        ],
        embedding_service=None,
    ):

        self.nodes = nodes

        self.relationships = (
            relationships
        )

        self.embedding_service = (
            embedding_service
        )

        self._adjacency = {}

        for relationship in (
            relationships
        ):

            self._adjacency.setdefault(
                relationship.source_id,
                [],
            ).append(
                relationship
            )

            self._adjacency.setdefault(
                relationship.target_id,
                [],
            ).append(
                relationship
            )

        # Generate missing embeddings.

        if embedding_service:

            for node in self.nodes.values():

                if not _has_embedding(
                    node.embedding
                ):

                    node.embedding = (
                        embedding_service.encode(
                            _domain_node_text(
                                node
                            )
                        )
                    )

    def all_nodes(self):

        return list(
            self.nodes.values()
        )

    def candidate_nodes(
        self,
        text,
        embedding,
        node_types=None,
        k=5,
    ):

        requested_types = (
            set(node_types)
            if node_types
            else None
        )

        results = []

        for node in (
            self.nodes.values()
        ):

            if (
                requested_types
                and not requested_types.intersection(
                    set(node.types or [])
                    | {node.node_type}
                )
            ):
                continue

            semantic_score = (
                _cosine_similarity(
                    embedding,
                    node.embedding,
                )
                if (
                    embedding is not None
                    and node.embedding is not None
                )
                else 0.0
            )

            lexical_score = (
                _lexical_similarity(
                    text,
                    node,
                )
            )

            score = (
                0.70 * semantic_score
                + 0.30 * lexical_score
            )

            if score <= 0:
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

    def neighborhood(
        self,
        node_id,
        depth=1,
    ):

        depth = max(
            1,
            int(depth),
        )

        visited = {
            node_id
        }

        frontier = {
            node_id
        }

        collected = []

        for _ in range(depth):

            next_frontier = set()

            for current in frontier:

                for relationship in (
                    self._adjacency.get(
                        current,
                        [],
                    )
                ):

                    collected.append(
                        relationship
                    )

                    other = (
                        relationship.target_id
                        if (
                            relationship.source_id
                            == current
                        )
                        else
                        relationship.source_id
                    )

                    if other not in visited:

                        visited.add(
                            other
                        )

                        next_frontier.add(
                            other
                        )

            frontier = next_frontier

        seen = set()
        result = []

        for relationship in collected:

            key = (
                relationship.source_id,
                relationship.relation,
                relationship.target_id,
            )

            if key in seen:
                continue

            seen.add(key)

            result.append(
                relationship
            )

        return result

    def get_node(
        self,
        node_id,
    ):

        return self.nodes.get(
            node_id
        )


# =============================================================
# Similarity helpers
# =============================================================

def _has_embedding(
    value,
):
    """
    Safe check for lists, tuples and numpy arrays.

    Never use:

        if embedding:

    because numpy arrays raise:

        ValueError:
        truth value of an array is ambiguous.
    """

    if value is None:
        return False

    try:
        return len(value) > 0
    except (
        TypeError,
        ValueError,
    ):
        return False


def _domain_node_text(
    node,
):

    parts = [
        node.name or "",
        node.description or "",
        node.id or "",
    ]

    parts.extend(
        node.aliases or []
    )

    parts.extend(
        node.types or []
    )

    return " ".join(
        part
        for part in parts
        if part
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

    text = text.replace(
        "-",
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


def _token_set(
    text,
):

    return set(
        _normalize_text(
            text
        ).split()
    )


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


def _cosine_similarity(
    a,
    b,
):

    if (
        a is None
        or b is None
    ):
        return 0.0

    try:

        values_a = list(a)
        values_b = list(b)

        if not values_a or not values_b:
            return 0.0

        if len(values_a) != len(
            values_b
        ):
            return 0.0

        dot = sum(
            x * y
            for x, y in zip(
                values_a,
                values_b,
            )
        )

        norm_a = math.sqrt(
            sum(
                x * x
                for x in values_a
            )
        )

        norm_b = math.sqrt(
            sum(
                y * y
                for y in values_b
            )
        )

        if (
            norm_a == 0
            or norm_b == 0
        ):
            return 0.0

        cosine = (
            dot
            / (
                norm_a
                * norm_b
            )
        )

        # Convert [-1, 1] to [0, 1].
        return max(
            0.0,
            min(
                1.0,
                (cosine + 1.0)
                / 2.0,
            ),
        )

    except (
        TypeError,
        ValueError,
        ZeroDivisionError,
    ):
        return 0.0