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
from beam_search_planner import BeamSearchPlanner
from graph_planner import GraphPlanner
from migrate_domain_graph_weights import apply_edge_weights


class FakeEmbeddingService:
    def encode(self, text):
        return {
            "register": [1.0, 0.0],
            "assign": [0.0, 1.0],
            "notify": [0.0, 1.0],
        }[text]


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

        self.assertEqual(set(result.nodes), {"create", "assign"})
        self.assertEqual(list(result.edges), [])
        self.assertEqual(len(result.graph["unreachable_prompt_pairs"]), 1)

    def test_records_rejected_actions_in_match_diagnostics(self):
        domain_graph = {
            "nodes": {
                "create": {"name": "Create Student", "embedding": [1.0, 0.0]},
            },
            "edges": [],
        }
        prompt_graph = nx.DiGraph()
        prompt_graph.add_node("P1", name="assign", description="assign", type="Action")

        result = GraphMatcher(FakeEmbeddingService(), domain_graph).match(
            prompt_graph, threshold=0.9
        )

        self.assertEqual(list(result.nodes), [])
        self.assertEqual(len(result.graph["unmatched_prompt_actions"]), 1)
        self.assertFalse(result.graph["match_diagnostics"][0]["accepted"])

    def test_reports_multiple_actions_that_map_to_the_same_domain_node(self):
        domain_graph = {
            "nodes": {
                "notify": {"name": "Notify Teacher", "embedding": [0.0, 1.0]},
            },
            "edges": [],
        }
        prompt_graph = nx.DiGraph()
        prompt_graph.add_node("P1", name="assign", description="assign", type="Action")
        prompt_graph.add_node("P2", name="notify", description="notify", type="Action")

        result = GraphMatcher(FakeEmbeddingService(), domain_graph).match(
            prompt_graph, threshold=0.0
        )

        self.assertEqual(len(result.graph["duplicate_domain_matches"]), 1)
        self.assertEqual(result.nodes["notify"]["matched_prompt_texts"], ["assign", "notify"])

    def test_prefers_lower_weight_domain_path(self):
        domain_graph = {
            "nodes": {
                "create": {"name": "Create Student", "embedding": [1.0, 0.0]},
                "validate": {"name": "Validate Documents", "embedding": [0.5, 0.5]},
                "assign": {"name": "Assign Classroom", "embedding": [0.0, 1.0]},
            },
            "edges": [
                {"source": "create", "target": "assign", "edge_type": "deprecated", "weight": 100},
                {"source": "create", "target": "validate", "edge_type": "mandatory", "weight": 1},
                {"source": "validate", "target": "assign", "edge_type": "mandatory", "weight": 1},
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
        self.assertEqual(result.edges["create", "validate"]["edge_type"], "mandatory")
        self.assertEqual(result.edges["create", "validate"]["confidence"], 1.0)

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
            [(edge["edge_type"], edge["weight"]) for edge in domain_graph["edges"]],
            [("mandatory", 1), ("optional", 5), ("deprecated", 100)],
        )

    def test_planner_preserves_domain_path_nodes_as_inferred(self):
        matched_graph = nx.DiGraph()
        matched_graph.add_node("create", name="Create Student", inferred=False)
        matched_graph.add_node("validate", name="Validate Documents", inferred=True)
        matched_graph.add_edge("create", "validate", relation="success")

        plan = GraphPlanner().plan(matched_graph)

        self.assertFalse(plan["graph"].nodes["create"]["inferred"])
        self.assertTrue(plan["graph"].nodes["validate"]["inferred"])


class BeamSearchPlannerTests(unittest.TestCase):

    def _candidate_domain_graph(self):
        return {
            "nodes": {
                "enroll": {"name": "Enroll Student", "embedding": [1.0, 0.0]},
                "create": {"name": "Create Student", "embedding": [0.95, 0.0]},
                "validate": {"name": "Validate Documents", "embedding": [0.4, 0.4]},
                "assign_teacher": {"name": "Assign Teacher", "embedding": [0.0, 1.0]},
                "assign_room": {"name": "Assign Classroom", "embedding": [0.0, 0.95]},
            },
            "edges": [
                {"source": "create", "target": "validate", "transition": "success"},
                {"source": "validate", "target": "assign_teacher", "transition": "success"},
            ],
        }

    def _prompt_graph(self):
        prompt_graph = nx.DiGraph()
        prompt_graph.add_node("P1", name="register", description="register", type="Action")
        prompt_graph.add_node("P2", name="assign", description="assign", type="Action")
        prompt_graph.add_edge("P1", "P2", relation="sequence")
        return prompt_graph

    def test_candidates_returns_top_k_domain_nodes_per_action(self):
        matcher = GraphMatcher(FakeEmbeddingService(), self._candidate_domain_graph())

        candidate_plan = matcher.candidates(self._prompt_graph(), k=2, threshold=0.5)

        names = [
            [candidate["domain_node_name"] for candidate in action["candidates"]]
            for action in candidate_plan["actions"]
        ]
        self.assertEqual(
            names,
            [["Enroll Student", "Create Student"],
             ["Assign Teacher", "Assign Classroom"]],
        )

    def test_candidates_drops_matches_below_threshold(self):
        matcher = GraphMatcher(FakeEmbeddingService(), self._candidate_domain_graph())

        candidate_plan = matcher.candidates(self._prompt_graph(), k=5, threshold=0.99)

        for action in candidate_plan["actions"]:
            self.assertEqual(len(action["candidates"]), 1)

    def test_beam_search_prefers_a_connected_sequence_over_the_best_local_match(self):
        matcher = GraphMatcher(FakeEmbeddingService(), self._candidate_domain_graph())
        candidate_plan = matcher.candidates(self._prompt_graph(), k=2, threshold=0.5)

        search_result = BeamSearchPlanner(beam_width=2).search(candidate_plan)

        self.assertEqual(
            [step["domain_node_id"] for step in search_result["selection"]],
            ["create", "assign_teacher"],
        )

    def test_beam_keeps_only_beam_width_partial_workflows(self):
        matcher = GraphMatcher(FakeEmbeddingService(), self._candidate_domain_graph())
        candidate_plan = matcher.candidates(self._prompt_graph(), k=2, threshold=0.5)

        search_result = BeamSearchPlanner(beam_width=2).search(candidate_plan)

        self.assertEqual(len(search_result["beam"]), 2)

    def test_expand_inserts_domain_path_nodes_as_inferred(self):
        matcher = GraphMatcher(FakeEmbeddingService(), self._candidate_domain_graph())
        candidate_plan = matcher.candidates(self._prompt_graph(), k=2, threshold=0.5)
        planner = BeamSearchPlanner(beam_width=2)

        workflow_graph = planner.plan(candidate_plan)

        self.assertEqual(
            list(workflow_graph.edges),
            [("create", "validate"), ("validate", "assign_teacher")],
        )
        self.assertTrue(workflow_graph.nodes["validate"]["inferred"])
        self.assertFalse(workflow_graph.nodes["create"]["inferred"])

    def test_expand_reports_transitions_without_a_domain_path(self):
        domain_graph = {
            "nodes": {
                "create": {"name": "Create Student", "embedding": [1.0, 0.0]},
                "assign_room": {"name": "Assign Classroom", "embedding": [0.0, 1.0]},
            },
            "edges": [],
        }
        matcher = GraphMatcher(FakeEmbeddingService(), domain_graph)
        candidate_plan = matcher.candidates(self._prompt_graph(), k=5, threshold=0.5)

        workflow_graph = BeamSearchPlanner(beam_width=2).plan(candidate_plan)

        self.assertEqual(set(workflow_graph.nodes), {"create", "assign_room"})
        self.assertEqual(list(workflow_graph.edges), [])
        self.assertEqual(len(workflow_graph.graph["unreachable_prompt_pairs"]), 1)

    def test_beam_search_handles_actions_without_candidates(self):
        domain_graph = {
            "nodes": {
                "create": {"name": "Create Student", "embedding": [1.0, 0.0]},
            },
            "edges": [],
        }
        matcher = GraphMatcher(FakeEmbeddingService(), domain_graph)
        candidate_plan = matcher.candidates(self._prompt_graph(), k=5, threshold=0.9)

        search_result = BeamSearchPlanner().search(candidate_plan)

        self.assertEqual(
            [step["domain_node_id"] for step in search_result["selection"]],
            ["create"],
        )
        self.assertEqual(len(search_result["skipped_actions"]), 1)

    def test_deprecated_transitions_are_penalised(self):
        domain_graph = {
            "nodes": {
                "create": {"name": "Create Student", "embedding": [1.0, 0.0]},
                "assign_teacher": {"name": "Assign Teacher", "embedding": [0.0, 1.0]},
                "assign_room": {"name": "Assign Classroom", "embedding": [0.0, 0.99]},
            },
            "edges": [
                {
                    "source": "create",
                    "target": "assign_teacher",
                    "edge_type": "deprecated",
                    "weight": 100,
                },
                {
                    "source": "create",
                    "target": "assign_room",
                    "edge_type": "mandatory",
                    "weight": 1,
                },
            ],
        }
        matcher = GraphMatcher(FakeEmbeddingService(), domain_graph)
        candidate_plan = matcher.candidates(self._prompt_graph(), k=5, threshold=0.5)

        search_result = BeamSearchPlanner(beam_width=2).search(candidate_plan)

        self.assertEqual(
            [step["domain_node_id"] for step in search_result["selection"]],
            ["create", "assign_room"],
        )


if __name__ == "__main__":
    unittest.main()
