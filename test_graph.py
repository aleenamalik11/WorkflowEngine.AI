import unittest
import sys
import types

import networkx as nx

# GraphMatcher only needs this utility. Stubbing it keeps these structural
# tests independent of loading the sentence-transformer runtime.
utils_stub = types.ModuleType("utils")
utils_stub.cosine_similarity = lambda a, b: sum(x * y for x, y in zip(a, b))
sys.modules.setdefault("utils", utils_stub)

from graph_matcher import GraphMatcher
from graph_planner import GraphPlanner
from migrate_domain_graph_weights import apply_edge_weights


class FakeEmbeddingService:
    def encode(self, text):
        return {"register": [1.0, 0.0], "assign": [0.0, 1.0]}[text]


class GraphMatcherPathTests(unittest.TestCase):
    def test_expands_prompt_order_using_the_domain_shortest_path(self):
        domain_graph = {
            "nodes": {
                "create": {"name": "Create Student", "embedding": [1.0, 0.0]},
                "validate": {"name": "Validate Documents", "embedding": [0.5, 0.5]},
                "assign": {"name": "Assign Classroom", "embedding": [0.0, 1.0]},
            },
            "edges": [
                {"source": "create", "target": "validate", "transition": "success"},
                {"source": "validate", "target": "assign", "transition": "success"},
            ],
        }
        prompt_graph = nx.DiGraph()
        prompt_graph.add_node("P1", name="register", description="register", type="Action")
        prompt_graph.add_node("P2", name="assign", description="assign", type="Action")
        prompt_graph.add_edge("P1", "P2", relation="sequence")

        result = GraphMatcher(FakeEmbeddingService(), domain_graph).match(
            prompt_graph, threshold=0.9
        )

        self.assertEqual(list(result.nodes), ["create", "assign", "validate"])
        self.assertEqual(list(result.edges), [("create", "validate"), ("validate", "assign")])
        self.assertFalse(result.nodes["create"]["inferred"])
        self.assertTrue(result.nodes["validate"]["inferred"])
        self.assertEqual(result.edges["create", "validate"]["relation"], "success")

    def test_does_not_create_a_prompt_edge_when_domain_nodes_are_disconnected(self):
        domain_graph = {
            "nodes": {
                "create": {"name": "Create Student", "embedding": [1.0, 0.0]},
                "assign": {"name": "Assign Classroom", "embedding": [0.0, 1.0]},
            },
            "edges": [],
        }
        prompt_graph = nx.DiGraph()
        prompt_graph.add_node("P1", name="register", description="register", type="Action")
        prompt_graph.add_node("P2", name="assign", description="assign", type="Action")
        prompt_graph.add_edge("P1", "P2", relation="sequence")

        result = GraphMatcher(FakeEmbeddingService(), domain_graph).match(
            prompt_graph, threshold=0.9
        )

        self.assertEqual(list(result.edges), [])
        self.assertEqual(len(result.graph["unreachable_prompt_pairs"]), 1)

    def test_prefers_lower_weight_domain_path(self):
        domain_graph = {
            "nodes": {
                "create": {"name": "Create Student", "embedding": [1.0, 0.0]},
                "validate": {"name": "Validate Documents", "embedding": [0.5, 0.5]},
                "assign": {"name": "Assign Classroom", "embedding": [0.0, 1.0]},
            },
            "edges": [
                {"source": "create", "target": "assign", "weight": 100},
                {"source": "create", "target": "validate", "weight": 1},
                {"source": "validate", "target": "assign", "weight": 1},
            ],
        }
        prompt_graph = nx.DiGraph()
        prompt_graph.add_node("P1", name="register", description="register", type="Action")
        prompt_graph.add_node("P2", name="assign", description="assign", type="Action")
        prompt_graph.add_edge("P1", "P2", relation="sequence")

        result = GraphMatcher(FakeEmbeddingService(), domain_graph).match(
            prompt_graph, threshold=0.9
        )

        self.assertEqual(list(result.edges), [("create", "validate"), ("validate", "assign")])
        self.assertEqual(result.edges["create", "validate"]["weight"], 1)

    def test_weight_migration_marks_optional_and_deprecated_edges(self):
        domain_graph = {
            "edges": [
                {"transition": "success"},
                {"transition": "optional"},
                {"transition": "deprecated"},
            ],
            "adjacency": {},
            "reverse_adjacency": {},
        }

        apply_edge_weights(domain_graph)

        self.assertEqual(
            [(edge["edge_category"], edge["weight"]) for edge in domain_graph["edges"]],
            [("mandatory", 1), ("optional", 5), ("deprecated", 100)],
        )

    def test_planner_preserves_domain_path_nodes_as_inferred(self):
        matched_graph = nx.DiGraph()
        matched_graph.add_node("create", name="Create Student", inferred=False)
        matched_graph.add_node("validate", name="Validate Documents", inferred=True)
        matched_graph.add_edge("create", "validate", relation="success")

        plan = GraphPlanner(candidate_threshold=1).plan(
            matched_graph, nx.DiGraph(), []
        )

        self.assertFalse(plan["graph"].nodes["create"]["inferred"])
        self.assertTrue(plan["graph"].nodes["validate"]["inferred"])


if __name__ == "__main__":
    unittest.main()
