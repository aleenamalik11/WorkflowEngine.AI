"""Map an extracted BRS JSON graph to a controlled ontology graph.

The mapper is intentionally deterministic. An LLM/semantic resolver can be
introduced later only for relationships reported as unmapped.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def read_first_json(path: str | Path) -> dict[str, Any]:
    """Read the first JSON value, tolerating prose appended after it."""
    text = Path(path).read_text(encoding="utf-8")
    value, _ = json.JSONDecoder().raw_decode(text.lstrip())
    if not isinstance(value, dict):
        raise ValueError("The BRS input must begin with a JSON object.")
    return value


def normalise_term(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


@dataclass(frozen=True)
class RelationshipDefinition:
    type: str
    raw_terms: tuple[str, ...]
    source_types: frozenset[str]
    target_types: frozenset[str]

    def accepts(self, source_types: set[str], target_types: set[str]) -> bool:
        return bool(self.source_types & source_types) and bool(self.target_types & target_types)


class OntologyMapper:
    def __init__(self, ontology_path: str | Path):
        ontology = json.loads(Path(ontology_path).read_text(encoding="utf-8"))
        self.version = ontology["version"]
        self.node_types: dict[str, str] = ontology["node_types"]
        self.relationships = [
            RelationshipDefinition(
                type=item["type"],
                raw_terms=tuple(normalise_term(term) for term in item["raw_terms"]),
                source_types=frozenset(item["source_types"]),
                target_types=frozenset(item["target_types"]),
            )
            for item in ontology["relationships"]
        ]

    def map_graph(self, raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        nodes, node_index, collisions = self._map_nodes(raw)
        edges: list[dict[str, Any]] = []
        unresolved_references: list[dict[str, Any]] = []
        unmapped_relationships: list[dict[str, Any]] = []
        invalid_relationships: list[dict[str, Any]] = []

        for index, relationship in enumerate(raw.get("relationships", [])):
            source_id = relationship.get("source")
            target_id = relationship.get("target")
            source = node_index.get(source_id)
            target = node_index.get(target_id)
            if source is None or target is None:
                unresolved_references.append({
                    "relationship_index": index,
                    "relationship": relationship,
                    "missing_node_ids": [
                        node_id for node_id, node in ((source_id, source), (target_id, target)) if node is None
                    ],
                })
                continue

            # Accept both extractor formats. Earlier extractions use
            # ``raw_type``; the current generic extractor uses ``type``.
            raw_type = relationship.get("raw_type") or relationship.get("type", "")
            candidates = [
                definition for definition in self.relationships
                if normalise_term(raw_type) in definition.raw_terms
            ]
            if not candidates:
                unmapped_relationships.append({"relationship_index": index, "relationship": relationship})
                continue

            valid = [
                definition for definition in candidates
                if definition.accepts(set(source["types"]), set(target["types"]))
            ]
            if not valid:
                invalid_relationships.append({
                    "relationship_index": index,
                    "relationship": relationship,
                    "source_types": source["types"],
                    "target_types": target["types"],
                    "candidate_types": [definition.type for definition in candidates],
                })
                continue

            definition = valid[0]
            edges.append({
                "id": f"edge:{index}",
                "source": source_id,
                "target": target_id,
                "type": definition.type,
                "condition": relationship.get("condition"),
                "provenance": {
                    "raw_type": raw_type,
                    "relationship_index": index,
                },
            })

        mapped = {
            "ontology_version": self.version,
            "nodes": nodes,
            "edges": edges,
        }
        report = {
            "summary": {
                "raw_node_count": sum(len(items) for items in raw.values() if isinstance(items, list)) - len(raw.get("relationships", [])),
                "mapped_node_count": len(nodes),
                "raw_relationship_count": len(raw.get("relationships", [])),
                "mapped_relationship_count": len(edges),
            },
            "duplicate_raw_ids": collisions,
            "unresolved_references": unresolved_references,
            "unmapped_relationships": unmapped_relationships,
            "invalid_relationships": invalid_relationships,
        }
        return mapped, report

    def _map_nodes(self, raw: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
        grouped: dict[str, list[tuple[str, str, dict[str, Any]]]] = defaultdict(list)
        for category, canonical_type in self.node_types.items():
            for node in raw.get(category, []):
                if "id" not in node:
                    continue
                grouped[node["id"]].append((category, canonical_type, node))

        nodes: list[dict[str, Any]] = []
        index: dict[str, dict[str, Any]] = {}
        collisions: list[dict[str, Any]] = []
        for node_id, entries in grouped.items():
            types = sorted({canonical_type for _, canonical_type, _ in entries})
            first = entries[0][2]
            node = {
                "id": node_id,
                "name": first.get("name", node_id),
                "types": types,
                "provenance": {
                    "raw_categories": sorted({category for category, _, _ in entries}),
                    "raw_records": [record for _, _, record in entries],
                },
            }
            nodes.append(node)
            index[node_id] = node
            if len(entries) > 1:
                collisions.append({
                    "id": node_id,
                    "types": types,
                    "raw_record_count": len(entries),
                    "action": "retained as a multi-label node; review during normalization",
                })
        return nodes, index, collisions
