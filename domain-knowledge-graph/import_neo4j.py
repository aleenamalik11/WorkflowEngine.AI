#!/usr/bin/env python3
"""Load a mapped domain graph into Neo4j.

The mapped graph is the source of truth for semantic relationship types. A
node may carry more than one ontology type: those types become Neo4j labels.
For example, ``customer`` becomes ``(:GraphNode:Actor:DomainEntity)``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def cypher_identifier(value: str) -> str:
    """Return a safe Cypher label/relationship identifier from ontology data."""
    if not IDENTIFIER.fullmatch(value):
        raise ValueError(f"Unsafe Cypher identifier: {value!r}")
    return f"`{value}`"


def node_properties(node: dict[str, Any]) -> dict[str, Any]:
    """Convert nested provenance to Neo4j-supported scalar/list properties."""
    provenance = node.get("provenance", {})
    return {
        "id": node["id"],
        "name": node.get("name", node["id"]),
        "types": node["types"],
        "raw_categories": provenance.get("raw_categories", []),
        "raw_records_json": json.dumps(provenance.get("raw_records", []), sort_keys=True),
    }


def edge_properties(edge: dict[str, Any]) -> dict[str, Any]:
    provenance = edge.get("provenance", {})
    return {
        "id": edge["id"],
        "condition": edge.get("condition"),
        "raw_type": provenance.get("raw_type"),
        "relationship_index": provenance.get("relationship_index"),
    }


def group_nodes_by_labels(nodes: Iterable[dict[str, Any]]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    """Group only for query construction; every type remains a node label."""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        labels = tuple(sorted(set(node["types"])))
        if not labels:
            raise ValueError(f"Node {node['id']!r} has no ontology types.")
        for label in labels:
            cypher_identifier(label)
        groups[labels].append(node_properties(node))
    return groups


def group_edges_by_type(edges: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        relationship_type = edge["type"]
        cypher_identifier(relationship_type)
        groups[relationship_type].append({
            "source": edge["source"],
            "target": edge["target"],
            "properties": edge_properties(edge),
        })
    return groups


def chunks(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def import_graph(driver: Any, graph: dict[str, Any], database: str, batch_size: int) -> None:
    """Create/update nodes and semantic edges in a single Neo4j database."""
    with driver.session(database=database) as session:
        session.run(
            "CREATE CONSTRAINT graph_node_id IF NOT EXISTS "
            "FOR (node:GraphNode) REQUIRE node.id IS UNIQUE"
        ).consume()

        for labels, rows in group_nodes_by_labels(graph.get("nodes", [])).items():
            label_clause = ":".join([cypher_identifier("GraphNode"), *(cypher_identifier(label) for label in labels)])
            query = f"UNWIND $rows AS row MERGE (node:{label_clause} {{id: row.id}}) SET node += row"
            for batch in chunks(rows, batch_size):
                session.run(query, rows=batch).consume()

        for relationship_type, rows in group_edges_by_type(graph.get("edges", [])).items():
            query = (
                "UNWIND $rows AS row "
                "MATCH (source:GraphNode {id: row.source}) "
                "MATCH (target:GraphNode {id: row.target}) "
                f"MERGE (source)-[relationship:{cypher_identifier(relationship_type)} {{id: row.properties.id}}]->(target) "
                "SET relationship += row.properties"
            )
            for batch in chunks(rows, batch_size):
                session.run(query, rows=batch).consume()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a mapped ontology graph into Neo4j.")
    parser.add_argument("input", type=Path, help="Mapped graph JSON, e.g. data/mapped/banking_brs_raw_mapped.json")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j Bolt URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j user")
    parser.add_argument("--password", required=True, help="Neo4j password")
    parser.add_argument("--database", default="neo4j", help="Neo4j database")
    parser.add_argument("--batch-size", type=int, default=500, help="Rows per Cypher query")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    try:
        from neo4j import GraphDatabase
    except ImportError as error:
        raise SystemExit("Install the Neo4j Python driver first: python3 -m pip install neo4j") from error

    graph = json.loads(args.input.read_text(encoding="utf-8"))
    with GraphDatabase.driver(args.uri, auth=(args.user, args.password)) as driver:
        driver.verify_connectivity()
        import_graph(driver, graph, args.database, args.batch_size)
    print(f"Imported {len(graph.get('nodes', []))} nodes and {len(graph.get('edges', []))} edges into {args.database!r}.")


if __name__ == "__main__":
    main()
