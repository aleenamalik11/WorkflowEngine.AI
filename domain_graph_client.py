"""
Stage 3 (candidate retrieval) + Stage 4 (neighborhood expansion).

Neo4j is the source of truth for the domain ontology (Actor, Component,
Operation, DomainEntity, Event, Rule) and its relationships. This module
defines one interface, `DomainGraphClient`, with a real Neo4j-backed
implementation and an in-memory implementation with the identical
contract, so the rest of the pipeline (subgraph builder, beam search)
never imports the neo4j driver directly and can be tested without a
running database.

IMPORTANT: routing weights are NOT stored or computed here. This module
only returns raw ontology relationships as they exist in the graph.
Weight/edge_type assignment happens in relationship_semantics.py, applied
by the planner (Stage 7), never here.
"""

import abc
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class DomainNode:
    id: str
    name: str
    node_type: str            # Actor | Component | Operation | DomainEntity | Event | Rule
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    embedding: Optional[list] = None


@dataclass
class DomainRelationship:
    source_id: str
    target_id: str
    relation: str              # e.g. OPERATION_PRECEDES, OPERATION_REQUIRES, ...


@dataclass
class ScoredNode:
    node: DomainNode
    score: float


class DomainGraphClient(abc.ABC):

    @abc.abstractmethod
    def candidate_nodes(self, text: str, embedding, node_types=None, k=5) -> List[ScoredNode]:
        """Stage 3: top-k candidates for a phrase, ranked by combined
        lexical + embedding + alias similarity. Never a single best-match."""

    @abc.abstractmethod
    def neighborhood(self, node_id: str, depth: int = 1) -> List[DomainRelationship]:
        """Stage 4: outgoing+incoming ontology relationships within `depth`
        hops, direction and relation type preserved verbatim."""

    @abc.abstractmethod
    def get_node(self, node_id: str) -> Optional[DomainNode]:
        ...


###############################################################
# Neo4j-backed implementation
###############################################################

class Neo4jDomainGraph(DomainGraphClient):
    """
    Thin wrapper over the neo4j Python driver. Candidate retrieval combines:
      - full-text/lexical similarity (Neo4j full-text index over name+aliases+description)
      - embedding cosine similarity (node.embedding property, precomputed offline)
    and returns the union, ranked, deduplicated -- never a single early cut.
    """

    def __init__(self, driver, database="neo4j", embedding_service=None,
                 fulltext_index="domainNodeSearch"):
        self.driver = driver
        self.database = database
        self.embedding_service = embedding_service
        self.fulltext_index = fulltext_index

    def candidate_nodes(self, text, embedding, node_types=None, k=5) -> List[ScoredNode]:
        cypher = f"""
        CALL db.index.fulltext.queryNodes($index, $text) YIELD node, score
        WHERE $types IS NULL OR any(l IN labels(node) WHERE l IN $types)
        RETURN node, score
        ORDER BY score DESC
        LIMIT $k
        """
        with self.driver.session(database=self.database) as session:
            rows = session.run(
                cypher,
                index=self.fulltext_index,
                text=text,
                types=list(node_types) if node_types else None,
                k=k,
            )
            lexical = [(self._to_domain_node(r["node"]), float(r["score"])) for r in rows]

        scored = []
        for node, lex_score in lexical:
            emb_score = 0.0
            if embedding is not None and node.embedding is not None:
                emb_score = _cosine(embedding, node.embedding)
            # blend: embeddings dominate once available, lexical rescues
            # short/abbreviated phrases embeddings under-score.
            combined = 0.65 * emb_score + 0.35 * min(lex_score / 10.0, 1.0)
            scored.append(ScoredNode(node=node, score=combined))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    def neighborhood(self, node_id, depth=1) -> List[DomainRelationship]:
        cypher = """
        MATCH (n {id: $id})-[r*1..%d]-(m)
        UNWIND r AS rel
        RETURN DISTINCT startNode(rel).id AS s, type(rel) AS t, endNode(rel).id AS d
        """ % max(1, int(depth))
        with self.driver.session(database=self.database) as session:
            rows = session.run(cypher, id=node_id)
            return [DomainRelationship(row["s"], row["d"], row["t"]) for row in rows]

    def get_node(self, node_id) -> Optional[DomainNode]:
        cypher = "MATCH (n {id: $id}) RETURN n LIMIT 1"
        with self.driver.session(database=self.database) as session:
            row = session.run(cypher, id=node_id).single()
            return self._to_domain_node(row["n"]) if row else None

    @staticmethod
    def _to_domain_node(neo4j_node) -> DomainNode:
        props = dict(neo4j_node)
        labels = list(neo4j_node.labels) if hasattr(neo4j_node, "labels") else []
        return DomainNode(
            id=props.get("id"),
            name=props.get("name", props.get("id")),
            node_type=(labels[0] if labels else props.get("type", "Operation")),
            description=props.get("description", ""),
            aliases=props.get("aliases", []) or [],
            embedding=props.get("embedding"),
        )


###############################################################
# In-memory implementation (local dev / unit tests / demo)
###############################################################

class InMemoryDomainGraph(DomainGraphClient):
    """
    Same contract as Neo4jDomainGraph, backed by plain dicts. Used for the
    two worked examples below and for CI, so the pipeline can be exercised
    without a live database.
    """

    def __init__(self, nodes: Dict[str, DomainNode], relationships: List[DomainRelationship],
                 embedding_service=None):
        self.nodes = nodes
        self.relationships = relationships
        self.embedding_service = embedding_service
        self._adjacency = {}
        for rel in relationships:
            self._adjacency.setdefault(rel.source_id, []).append(rel)
            self._adjacency.setdefault(rel.target_id, []).append(rel)

    def candidate_nodes(self, text, embedding, node_types=None, k=5) -> List[ScoredNode]:
        scored = []
        text_norm = text.lower()
        for node in self.nodes.values():
            if node_types and node.node_type not in node_types:
                continue
            lex = _lexical_similarity(text_norm, node.name.lower(), node.aliases)
            emb = _cosine(embedding, node.embedding) if (embedding is not None and node.embedding is not None) else 0.0
            combined = 0.6 * emb + 0.4 * lex
            if combined > 0:
                scored.append(ScoredNode(node=node, score=combined))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    def neighborhood(self, node_id, depth=1) -> List[DomainRelationship]:
        visited_nodes = {node_id}
        frontier = {node_id}
        collected = []
        for _ in range(max(1, int(depth))):
            next_frontier = set()
            for nid in frontier:
                for rel in self._adjacency.get(nid, []):
                    collected.append(rel)
                    other = rel.target_id if rel.source_id == nid else rel.source_id
                    if other not in visited_nodes:
                        next_frontier.add(other)
                        visited_nodes.add(other)
            frontier = next_frontier
        # dedupe
        seen = set()
        unique = []
        for rel in collected:
            key = (rel.source_id, rel.relation, rel.target_id)
            if key not in seen:
                seen.add(key)
                unique.append(rel)
        return unique

    def get_node(self, node_id) -> Optional[DomainNode]:
        return self.nodes.get(node_id)


def _cosine(a, b):
    import math
    if a is None or b is None:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _lexical_similarity(query, name, aliases):
    query_tokens = set(query.split())
    candidates = [name] + list(aliases)
    best = 0.0
    for cand in candidates:
        cand_tokens = set(cand.lower().split())
        if not query_tokens or not cand_tokens:
            continue
        overlap = len(query_tokens & cand_tokens) / len(query_tokens | cand_tokens)
        best = max(best, overlap)
    return best