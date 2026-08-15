"""
Domain graph access layer.

The domain graph is the source of truth for domain concepts and their
relationships.

This module deliberately does NOT perform path finding or shortest-path
routing.

Candidate retrieval uses:

    lexical similarity
        +
    embedding similarity

The two implementations expose the same contract:

    Neo4jDomainGraph
    InMemoryDomainGraph

Nothing downstream needs to know whether Neo4j is being used.
"""

import abc
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DomainNode:
    id: str
    name: str
    node_type: str
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    embedding: Optional[list] = None


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
        Return top-k domain concepts using lexical + semantic similarity.
        """

    @abc.abstractmethod
    def neighborhood(
        self,
        node_id: str,
        depth: int = 1,
    ) -> List[DomainRelationship]:
        """
        Return relationships around a node.
        """

    @abc.abstractmethod
    def get_node(
        self,
        node_id: str,
    ) -> Optional[DomainNode]:
        ...

    @abc.abstractmethod
    def all_nodes(self) -> List[DomainNode]:
        """
        Return all domain nodes.

        Used to provide domain context to the semantic LLM.
        """
        ...


###############################################################
# Neo4j
###############################################################

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
        self.embedding_service = embedding_service
        self.fulltext_index = fulltext_index

    def all_nodes(self) -> List[DomainNode]:

        cypher = """
        MATCH (n)
        RETURN n
        """

        with self.driver.session(database=self.database) as session:
            rows = session.run(cypher)

            return [
                self._to_domain_node(row["n"])
                for row in rows
            ]

    def candidate_nodes(
        self,
        text,
        embedding,
        node_types=None,
        k=5,
    ) -> List[ScoredNode]:

        # ---------------------------------------------------------
        # Lexical candidates
        # ---------------------------------------------------------

        lexical_nodes = {}

        try:

            cypher = f"""
            CALL db.index.fulltext.queryNodes(
                $index,
                $text
            )
            YIELD node, score

            WHERE
                $types IS NULL
                OR any(
                    label IN labels(node)
                    WHERE label IN $types
                )

            RETURN node, score
            ORDER BY score DESC
            LIMIT $limit
            """

            with self.driver.session(
                database=self.database
            ) as session:

                rows = session.run(
                    cypher,
                    index=self.fulltext_index,
                    text=text,
                    types=list(node_types)
                    if node_types
                    else None,
                    limit=max(k * 3, 15),
                )

                for row in rows:

                    node = self._to_domain_node(
                        row["node"]
                    )

                    lexical_nodes[node.id] = (
                        node,
                        float(row["score"])
                    )

        except Exception:
            # If the full-text index is unavailable, semantic retrieval
            # below can still operate if embeddings exist.
            lexical_nodes = {}

        # ---------------------------------------------------------
        # Semantic candidates
        #
        # IMPORTANT:
        #
        # We do not restrict semantic retrieval to lexical results.
        # That was one of the previous architectural bugs.
        # ---------------------------------------------------------

        semantic_candidates = []

        if embedding is not None:

            all_nodes = self.all_nodes()

            for node in all_nodes:

                if (
                    node_types
                    and node.node_type not in node_types
                ):
                    continue

                if node.embedding is None:
                    continue

                semantic_score = _cosine(
                    embedding,
                    node.embedding,
                )

                semantic_candidates.append(
                    (
                        node,
                        semantic_score,
                    )
                )

            semantic_candidates.sort(
                key=lambda x: x[1],
                reverse=True,
            )

            semantic_candidates = semantic_candidates[
                :max(k * 3, 15)
            ]

        # ---------------------------------------------------------
        # Union candidates
        # ---------------------------------------------------------

        merged = {}

        for node, lex_raw in lexical_nodes.values():

            lexical_score = _normalize_lexical_score(
                lex_raw
            )

            semantic_score = 0.0

            if embedding is not None and node.embedding is not None:
                semantic_score = _cosine(
                    embedding,
                    node.embedding,
                )

            merged[node.id] = ScoredNode(
                node=node,
                score=_combined_score(
                    lexical_score,
                    semantic_score,
                ),
                lexical_score=lexical_score,
                semantic_score=semantic_score,
            )

        for node, semantic_score in semantic_candidates:

            existing = merged.get(node.id)

            lexical_score = (
                existing.lexical_score
                if existing
                else _lexical_similarity(
                    text,
                    node.name,
                    node.aliases,
                    node.description,
                )
            )

            combined = _combined_score(
                lexical_score,
                semantic_score,
            )

            if existing is None or combined > existing.score:

                merged[node.id] = ScoredNode(
                    node=node,
                    score=combined,
                    lexical_score=lexical_score,
                    semantic_score=semantic_score,
                )

        results = list(merged.values())

        results.sort(
            key=lambda x: x.score,
            reverse=True,
        )

        return results[:k]

    def neighborhood(
        self,
        node_id,
        depth=1,
    ) -> List[DomainRelationship]:

        depth = max(1, int(depth))

        cypher = f"""
        MATCH (n {{id: $id}})-[r*1..{depth}]-(m)
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

            return [
                DomainRelationship(
                    row["source_id"],
                    row["target_id"],
                    row["relation"],
                )
                for row in rows
            ]

    def get_node(
        self,
        node_id,
    ) -> Optional[DomainNode]:

        cypher = """
        MATCH (n {id: $id})
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

            return self._to_domain_node(
                row["n"]
            )

    @staticmethod
    def _to_domain_node(neo4j_node):

        props = dict(neo4j_node)

        labels = (
            list(neo4j_node.labels)
            if hasattr(neo4j_node, "labels")
            else []
        )

        return DomainNode(
            id=props.get("id"),
            name=props.get(
                "name",
                props.get("id"),
            ),
            node_type=(
                labels[0]
                if labels
                else props.get(
                    "type",
                    "Operation",
                )
            ),
            description=props.get(
                "description",
                "",
            ),
            aliases=props.get(
                "aliases",
                [],
            ) or [],
            embedding=props.get(
                "embedding"
            ),
        )


###############################################################
# In-memory implementation
###############################################################

class InMemoryDomainGraph(DomainGraphClient):

    def __init__(
        self,
        nodes: Dict[str, DomainNode],
        relationships: List[DomainRelationship],
        embedding_service=None,
    ):

        self.nodes = nodes
        self.relationships = relationships
        self.embedding_service = embedding_service

        self._adjacency = {}

        for relationship in relationships:

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

        # -----------------------------------------------------
        # Generate embeddings for demo/local nodes.
        #
        # This is critical. Previously the demo nodes had no
        # embeddings, which meant semantic matching silently
        # degraded to lexical matching.
        # -----------------------------------------------------

        if embedding_service is not None:

            for node in self.nodes.values():

                if node.embedding is None:

                    node_text = _node_text(
                        node
                    )

                    node.embedding = embedding_service.encode(
                        node_text
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

        results = []

        for node in self.nodes.values():

            if (
                node_types
                and node.node_type not in node_types
            ):
                continue

            lexical_score = _lexical_similarity(
                text,
                node.name,
                node.aliases,
                node.description,
            )

            semantic_score = 0.0

            if (
                embedding is not None
                and node.embedding is not None
            ):

                semantic_score = _cosine(
                    embedding,
                    node.embedding,
                )

            score = _combined_score(
                lexical_score,
                semantic_score,
            )

            if score > 0:

                results.append(
                    ScoredNode(
                        node=node,
                        score=score,
                        lexical_score=lexical_score,
                        semantic_score=semantic_score,
                    )
                )

        results.sort(
            key=lambda x: x.score,
            reverse=True,
        )

        return results[:k]

    def neighborhood(
        self,
        node_id,
        depth=1,
    ):

        depth = max(1, int(depth))

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

                for relationship in self._adjacency.get(
                    current,
                    [],
                ):

                    collected.append(
                        relationship
                    )

                    other = (
                        relationship.target_id
                        if relationship.source_id == current
                        else relationship.source_id
                    )

                    if other not in visited:

                        visited.add(other)

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

            if key not in seen:

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


###############################################################
# Similarity helpers
###############################################################

def _node_text(node):

    parts = [
        node.name,
        node.description,
    ]

    parts.extend(
        node.aliases or []
    )

    return " ".join(
        p for p in parts
        if p
    )


def _cosine(a, b):

    if a is None or b is None:
        return 0.0

    try:

        dot = sum(
            x * y
            for x, y in zip(a, b)
        )

        na = math.sqrt(
            sum(x * x for x in a)
        )

        nb = math.sqrt(
            sum(x * x for x in b)
        )

        if na == 0 or nb == 0:
            return 0.0

        return dot / (
            na * nb
        )

    except Exception:

        return 0.0


def _lexical_similarity(
    query,
    name,
    aliases=None,
    description="",
):

    aliases = aliases or []

    query_tokens = set(
        _tokens(query)
    )

    if not query_tokens:
        return 0.0

    candidates = [
        name,
        description,
        *aliases,
    ]

    best = 0.0

    for candidate in candidates:

        if not candidate:
            continue

        candidate_tokens = set(
            _tokens(candidate)
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


def _tokens(text):

    return [
        token
        for token in (
            text.lower()
            .replace("-", " ")
            .replace("_", " ")
            .split()
        )
        if token
    ]


def _normalize_lexical_score(score):

    # Neo4j full-text scores are not guaranteed to be
    # normalized. Keep them in a useful 0..1 range.
    return min(
        max(float(score), 0.0) / 10.0,
        1.0,
    )


def _combined_score(
    lexical_score,
    semantic_score,
):

    # Embeddings are primary.
    # Lexical matching rescues exact terminology.
    return (
        0.70 * max(0.0, semantic_score)
        + 0.30 * max(0.0, lexical_score)
    )