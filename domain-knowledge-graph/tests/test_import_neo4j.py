import unittest
from pathlib import Path

from import_neo4j import edge_properties, group_edges_by_type, group_nodes_by_labels
from mapping.ontology_mapper import read_first_json


ROOT = Path(__file__).resolve().parents[1]


class Neo4jImportTests(unittest.TestCase):
    def test_multi_type_node_keeps_every_ontology_label(self):
        graph = read_first_json(ROOT / "data" / "mapped" / "banking_brs_raw_mapped.json")
        groups = group_nodes_by_labels(graph["nodes"])
        customer = next(row for row in groups[("Actor", "DomainEntity")] if row["id"] == "customer")
        self.assertEqual(customer["types"], ["Actor", "DomainEntity"])

    def test_edges_keep_the_mapped_semantic_relationship_type(self):
        graph = read_first_json(ROOT / "data" / "mapped" / "banking_brs_raw_mapped.json")
        edges = group_edges_by_type(graph["edges"])
        self.assertEqual(edges["ACTOR_PERFORMS"][0]["properties"]["raw_type"], "performs")
        self.assertIn("ENTITY_OWNS", edges)

    def test_edge_provenance_is_flattened_for_neo4j_properties(self):
        properties = edge_properties({"id": "edge:1", "condition": None, "provenance": {"raw_type": "owns", "relationship_index": 4}})
        self.assertEqual(properties["relationship_index"], 4)


if __name__ == "__main__":
    unittest.main()
