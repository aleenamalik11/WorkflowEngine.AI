import unittest
from pathlib import Path

from mapping.ontology_mapper import OntologyMapper, read_first_json


ROOT = Path(__file__).resolve().parents[1]


class OntologyMapperTests(unittest.TestCase):
    def setUp(self):
        self.mapper = OntologyMapper(ROOT / "ontology" / "core_ontology.json")

    def test_maps_relationships_by_term_and_endpoint_types(self):
        raw = {
            "actors": [{"id": "customer", "name": "Customer"}],
            "operations": [{"id": "transfer", "name": "Transfer"}],
            "entities": [{"id": "response", "name": "Response"}],
            "events": [{"id": "completed", "name": "Completed"}],
            "rules": [],
            "relationships": [
                {"source": "customer", "target": "transfer", "raw_type": "performs"},
                {"source": "transfer", "target": "response", "raw_type": "produces"},
                {"source": "transfer", "target": "completed", "raw_type": "produces"},
            ],
        }
        mapped, report = self.mapper.map_graph(raw)
        self.assertEqual([edge["type"] for edge in mapped["edges"]], [
            "ACTOR_PERFORMS", "OPERATION_PRODUCES", "OPERATION_PRODUCES_EVENT",
        ])
        self.assertFalse(report["unresolved_references"])

    def test_reports_missing_references_without_creating_nodes(self):
        raw = {
            "actors": [], "operations": [{"id": "register", "name": "Register"}],
            "entities": [], "events": [], "rules": [],
            "relationships": [{"source": "administrator", "target": "register", "raw_type": "performs"}],
        }
        mapped, report = self.mapper.map_graph(raw)
        self.assertEqual(mapped["nodes"], [{
            "id": "register", "name": "Register", "types": ["Operation"],
            "provenance": {"raw_categories": ["operations"], "raw_records": [{"id": "register", "name": "Register"}]},
        }])
        self.assertEqual(report["unresolved_references"][0]["missing_node_ids"], ["administrator"])

    def test_reads_json_when_prose_follows(self):
        fixture = ROOT / "tests" / "fixtures" / "with_trailing_text.txt"
        self.assertEqual(read_first_json(fixture), {"actors": []})

    def test_accepts_type_as_the_raw_relationship_field(self):
        raw = {
            "actors": [{"id": "customer", "name": "Customer"}],
            "operations": [{"id": "transfer", "name": "Transfer"}],
            "entities": [], "events": [], "rules": [],
            "relationships": [{"source": "customer", "target": "transfer", "type": "performs"}],
        }
        mapped, report = self.mapper.map_graph(raw)
        self.assertEqual(mapped["edges"][0]["type"], "ACTOR_PERFORMS")
        self.assertFalse(report["unmapped_relationships"])


if __name__ == "__main__":
    unittest.main()
