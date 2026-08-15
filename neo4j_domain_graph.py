"""Fetch the domain knowledge graph from Neo4j.

``domain-knowledge-graph`` maps a raw BRS extraction onto the core ontology
and imports it into Neo4j as ``(:GraphNode:<OntologyType>...)`` nodes joined by
semantic relationships (``ACTOR_PERFORMS``, ``OPERATION_REQUIRES``, ...).

This module reads that graph back and turns it into the structure the planning
pipeline already understands: a ``{"nodes": {...}, "edges": [...]}`` mapping
with node embeddings and relationship weights.  The trained pickle graph is no
longer required.
"""

import json
import os

###############################################################
# Ontology semantics
###############################################################

# Nodes that can become workflow steps.  Everything else (actors, components,
# entities, events, rules) is *context*: it explains how operations connect,
# but it is never executed.
EXECUTABLE_TYPES = {"Operation"}

CONSTRAINT_TYPES = {"Rule"}

# Lower weights are preferred by weighted shortest-path routing: they express
# how *required* / relevant a relationship is for reaching the next operation.
RELATIONSHIP_SEMANTICS = {
    "OPERATION_REQUIRES": ("mandatory", 1),
    "OPERATION_PRECEDES": ("mandatory", 1),
    "OPERATION_INCLUDES": ("mandatory", 2),
    "EVENT_TRIGGERS": ("mandatory", 2),
    "OPERATION_PRODUCES_EVENT": ("mandatory", 2),
    "OPERATION_CREATES": ("alternative", 3),
    "OPERATION_MODIFIES": ("alternative", 3),
    "OPERATION_VALIDATES": ("alternative", 3),
    "OPERATION_PRODUCES": ("alternative", 3),
    "OPERATION_ACCEPTS": ("alternative", 4),
    "COMPONENT_EXECUTES": ("alternative", 4),
    "ACTOR_PERFORMS": ("alternative", 4),
    "ACTOR_REQUESTS": ("alternative", 4),
    "RULE_CONSTRAINS": ("optional", 5),
    "ENTITY_OWNS": ("optional", 6),
    "ENTITY_LINKED_TO": ("optional", 6),
    "EVENT_RELATES_TO": ("optional", 6),
}

DEFAULT_RELATIONSHIP_SEMANTICS = ("optional", 5)

# Relationships that only describe *who* or *what* is involved.  They are still
# traversable, but a workflow must not be built out of them alone.
CONTEXT_RELATIONSHIPS = {
    "ACTOR_PERFORMS",
    "ACTOR_REQUESTS",
    "COMPONENT_EXECUTES",
    "ENTITY_OWNS",
    "ENTITY_LINKED_TO",
    "EVENT_RELATES_TO",
    "RULE_CONSTRAINS",
}

NODE_QUERY = """
MATCH (node:GraphNode)
RETURN node.id AS id,
       node.name AS name,
       node.types AS types,
       labels(node) AS labels,
       node.raw_records_json AS raw_records_json
"""

EDGE_QUERY = """
MATCH (source:GraphNode)-[relationship]->(target:GraphNode)
RETURN relationship.id AS id,
       source.id AS source,
       target.id AS target,
       type(relationship) AS type,
       relationship.condition AS condition,
       relationship.raw_type AS raw_type
"""


def relationship_semantics(relationship_type):
    """Return the ``(edge_type, weight)`` pair for an ontology relationship."""
    return RELATIONSHIP_SEMANTICS.get(
        str(relationship_type).strip().upper(),
        DEFAULT_RELATIONSHIP_SEMANTICS,
    )


def node_text(record):
    """Build the sentence used to embed a domain node.

    The ontology labels and the original BRS statements carry most of the
    meaning, so they are embedded together with the node name.
    """
    parts = [record.get("name") or record.get("id") or ""]

    types = record.get("types") or []
    if types:
        parts.append(" ".join(types))

    raw_records_json = record.get("raw_records_json")
    if raw_records_json:
        try:
            raw_records = json.loads(raw_records_json)
        except (TypeError, ValueError):
            raw_records = []

        for raw_record in raw_records if isinstance(raw_records, list) else []:
            if not isinstance(raw_record, dict):
                continue
            for key in ("statement", "description", "summary"):
                value = raw_record.get(key)
                if value:
                    parts.append(str(value))

    return " ".join(part for part in parts if part).strip()


###############################################################
# Neo4j Domain Graph
###############################################################

class Neo4jDomainGraph:
    """Load the persisted ontology graph from Neo4j.

    The result is intentionally shaped like the previously trained graph so
    that :class:`GraphMatcher` and the planners keep working unchanged.
    """

    def __init__(
            self,
            embedding_service,
            driver=None,
            database="neo4j",
            fulltext_index="domainNodeSearch",
            uri=None,
            user=None,
            password=None,
    ):
        self.embedding_service = embedding_service

        self.driver = driver

        self.database = database

        self.fulltext_index = fulltext_index

        self.uri = uri or os.environ.get(
            "NEO4J_URI",
            "bolt://localhost:7687",
        )

        self.user = user or os.environ.get(
            "NEO4J_USER",
            "neo4j",
        )

        self.password = password

    ###############################################################
    # Fetch
    ###############################################################

    def fetch(self):
        """Read every node and relationship from Neo4j."""

        try:
            from neo4j import GraphDatabase
        except ImportError as error:
            raise ImportError(
                "Install the Neo4j Python driver first: "
                "python -m pip install neo4j"
            ) from error

        if self.driver is None:

            if not self.password:
                raise RuntimeError(
                    "Neo4j driver was not supplied and "
                    "no Neo4j password was provided."
                )

            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(
                    self.user,
                    self.password,
                ),
            )

            owns_driver = True

        else:

            owns_driver = False

        try:

            self.driver.verify_connectivity()

            with self.driver.session(
                    database=self.database
            ) as session:

                nodes = [
                    dict(record)
                    for record in session.run(NODE_QUERY)
                ]

                edges = [
                    dict(record)
                    for record in session.run(EDGE_QUERY)
                ]

        finally:

            if owns_driver:
                self.driver.close()

        return self.build(
            nodes,
            edges,
        )

    ###############################################################
    # Build the planner-facing graph
    ###############################################################

    def build(self, node_records, edge_records):
        """Convert raw Neo4j records into the domain graph structure."""
        nodes = {}

        for record in node_records:
            node_id = record.get("id")
            if not node_id:
                continue

            types = list(record.get("types") or [])
            if not types:
                types = [
                    label for label in (record.get("labels") or [])
                    if label != "GraphNode"
                ]

            text = node_text({**record, "types": types})
            name = record.get("name") or node_id

            nodes[node_id] = {
                "id": node_id,
                "name": name,
                "type": types[0] if types else "Custom",
                "types": types,
                "executable": bool(set(types) & EXECUTABLE_TYPES),
                "constraint": bool(set(types) & CONSTRAINT_TYPES),
                "text": text,
                "embedding": self.embedding_service.encode(text).tolist(),
                # The BRS statements folded into ``text`` describe the node in
                # detail, but they also dilute short prompts such as
                # "Transfer funds".  The name is embedded separately so the
                # matcher can compare against the concept itself as well.
                "name_embedding": self.embedding_service.encode(name).tolist(),
            }

        edges = []
        adjacency = {}
        reverse_adjacency = {}

        for record in edge_records:
            source = record.get("source")
            target = record.get("target")

            if source not in nodes or target not in nodes:
                continue

            relationship_type = record.get("type") or "RELATED_TO"
            edge_type, weight = relationship_semantics(relationship_type)

            edge = {
                "id": record.get("id"),
                "source": source,
                "target": target,
                "relation": relationship_type,
                "transition": record.get("raw_type") or relationship_type,
                "semantic_type": relationship_type,
                "edge_type": edge_type,
                "weight": weight,
                "condition": record.get("condition"),
                "context": relationship_type in CONTEXT_RELATIONSHIPS,
                "confidence": 1.0,
            }

            edges.append(edge)
            adjacency.setdefault(source, []).append({**edge})
            reverse_adjacency.setdefault(target, []).append({**edge})

        return {
            "source": "neo4j",
            "nodes": nodes,
            "edges": edges,
            "adjacency": adjacency,
            "reverse_adjacency": reverse_adjacency,
        }

    ###############################################################
    # Pretty Print
    ###############################################################

    @staticmethod
    def print_summary(domain_graph):
        nodes = domain_graph.get("nodes", {})
        edges = domain_graph.get("edges", [])

        executable = [
            node for node in nodes.values() if node.get("executable")
        ]

        print()
        print("=" * 60)
        print("Domain Graph (Neo4j)")
        print("=" * 60)
        print("Nodes      :", len(nodes))
        print("Operations :", len(executable))
        print("Edges      :", len(edges))
