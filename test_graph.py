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
from contextual_subgraph import ContextualSubgraphBuilder
from graph_planner import GraphPlanner
from migrate_domain_graph_weights import apply_edge_weights
from neo4j_domain_graph import Neo4jDomainGraph


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


class FakeNeo4jEmbeddingService:
    def encode(self, text):
        class _Vector(list):
            def tolist(self):
                return list(self)

        return _Vector([float(len(text)), 1.0])


class Neo4jDomainGraphTests(unittest.TestCase):

    def _graph(self):
        nodes = [
            {"id": "customer", "name": "Customer", "types": ["Actor", "DomainEntity"]},
            {"id": "register_user", "name": "Register User", "types": ["Operation"]},
            {"id": "validate_user", "name": "Validate User", "types": ["Operation"]},
            {"id": "BR-001", "name": "Unique email", "types": ["Rule"]},
        ]
        edges = [
            {
                "id": "edge:0",
                "source": "register_user",
                "target": "validate_user",
                "type": "OPERATION_INCLUDES",
                "condition": None,
                "raw_type": "includes",
            },
            {
                "id": "edge:1",
                "source": "customer",
                "target": "register_user",
                "type": "ACTOR_PERFORMS",
                "condition": "customer exists",
                "raw_type": "performs",
            },
            {
                "id": "edge:2",
                "source": "BR-001",
                "target": "register_user",
                "type": "RULE_CONSTRAINS",
                "condition": None,
                "raw_type": "constrains",
            },
        ]
        return Neo4jDomainGraph(FakeNeo4jEmbeddingService()).build(nodes, edges)

    def test_only_operations_are_marked_executable(self):
        graph = self._graph()

        self.assertTrue(graph["nodes"]["register_user"]["executable"])
        self.assertFalse(graph["nodes"]["customer"]["executable"])
        self.assertTrue(graph["nodes"]["BR-001"]["constraint"])

    def test_relationship_types_become_routing_weights(self):
        edges = {edge["id"]: edge for edge in self._graph()["edges"]}

        self.assertEqual(edges["edge:0"]["edge_type"], "mandatory")
        self.assertEqual(edges["edge:0"]["weight"], 2)
        self.assertTrue(edges["edge:1"]["context"])
        self.assertEqual(edges["edge:1"]["condition"], "customer exists")

    def test_nodes_are_embedded_and_keep_the_ontology_relation_name(self):
        graph = self._graph()

        self.assertEqual(len(graph["nodes"]["register_user"]["embedding"]), 2)
        self.assertEqual(graph["edges"][0]["relation"], "OPERATION_INCLUDES")

    def test_edges_pointing_at_unknown_nodes_are_dropped(self):
        graph = Neo4jDomainGraph(FakeNeo4jEmbeddingService()).build(
            [{"id": "register_user", "name": "Register User", "types": ["Operation"]}],
            [{"id": "edge:0", "source": "register_user", "target": "missing", "type": "OPERATION_INCLUDES"}],
        )

        self.assertEqual(graph["edges"], [])


class ContextualSubgraphTests(unittest.TestCase):

    def _domain_graph(self):
        graph = nx.DiGraph()
        graph.add_node("register", name="Register User", executable=True)
        graph.add_node("validate", name="Validate User", executable=True)
        graph.add_node("notify", name="Notify Customer", executable=True)
        graph.add_node("customer", name="Customer", executable=False, types=["Actor"])
        graph.add_edge("register", "validate", relation="OPERATION_INCLUDES", edge_type="mandatory", weight=2)
        graph.add_edge("validate", "notify", relation="OPERATION_PRECEDES", edge_type="mandatory", weight=1)
        graph.add_edge("customer", "register", relation="ACTOR_PERFORMS", edge_type="alternative", weight=4, context=True)
        return graph

    def _plan(self, first, second, graph=None):
        return {
            "actions": [
                {
                    "prompt_node_id": "P1",
                    "prompt_text": "register",
                    "candidates": [{"domain_node_id": first, "domain_node_name": first, "similarity": 0.9}],
                },
                {
                    "prompt_node_id": "P2",
                    "prompt_text": "notify",
                    "candidates": [{"domain_node_id": second, "domain_node_name": second, "similarity": 0.8}],
                },
            ],
            "domain_graph": self._domain_graph() if graph is None else graph,
        }

    def test_context_expansion_keeps_the_relationships_around_matches(self):
        contextual = ContextualSubgraphBuilder().build(
            self._plan("register", "notify")
        )
        subgraph = contextual["domain_graph"]

        self.assertEqual(set(subgraph.nodes), {"register", "validate", "notify", "customer"})
        self.assertIn(("customer", "register"), subgraph.edges)

    def test_non_executable_matches_are_reported_as_unsupported(self):
        contextual = ContextualSubgraphBuilder().build(
            self._plan("customer", "notify")
        )

        self.assertEqual(len(contextual["invalid_actions"]), 1)
        self.assertEqual(contextual["actions"][0]["candidates"], [])

    def test_weighted_relationships_select_dijkstra(self):
        contextual = ContextualSubgraphBuilder().build(
            self._plan("register", "notify")
        )

        self.assertEqual(contextual["search_strategy"], "dijkstra")

    def test_equal_cost_relationships_select_bfs(self):
        graph = nx.DiGraph()
        graph.add_node("register", name="Register", executable=True)
        graph.add_node("notify", name="Notify", executable=True)
        graph.add_edge("register", "notify", relation="OPERATION_PRECEDES", weight=1)

        contextual = ContextualSubgraphBuilder().build(
            self._plan("register", "notify", graph=graph)
        )

        self.assertEqual(contextual["search_strategy"], "bfs")

    def test_disconnected_matches_are_connected_through_the_domain_graph(self):
        graph = self._domain_graph()
        graph.add_node("archive", name="Archive", executable=True)
        graph.add_node("report", name="Report", executable=True)
        graph.add_edge("notify", "archive", relation="OPERATION_PRECEDES", edge_type="mandatory", weight=1)
        graph.add_edge("archive", "report", relation="OPERATION_PRECEDES", edge_type="mandatory", weight=1)

        contextual = ContextualSubgraphBuilder().build(
            self._plan("register", "report", graph=graph)
        )
        subgraph = contextual["domain_graph"]

        self.assertTrue(nx.has_path(subgraph, "register", "report"))
        self.assertEqual(
            subgraph.graph["connections"][0]["path"],
            ["register", "validate", "notify", "archive", "report"],
        )

    def test_paths_that_no_longer_describe_the_prompt_are_rejected(self):
        graph = self._domain_graph()
        graph.add_node("archive", name="Archive", executable=True)
        graph.add_node("report", name="Report", executable=True)
        graph.add_edge("notify", "archive", relation="OPERATION_PRECEDES", edge_type="mandatory", weight=1)
        graph.add_edge("archive", "report", relation="OPERATION_PRECEDES", edge_type="mandatory", weight=1)

        contextual = ContextualSubgraphBuilder(max_path_length=2).build(
            self._plan("register", "report", graph=graph)
        )
        subgraph = contextual["domain_graph"]

        self.assertEqual(subgraph.graph["connections"], [])
        self.assertIn("too long", subgraph.graph["disconnected_pairs"][0]["reason"])

    def test_deprecated_relationships_are_never_used_to_reconnect(self):
        graph = nx.DiGraph()
        graph.add_node("register", name="Register", executable=True)
        graph.add_node("notify", name="Notify", executable=True)
        graph.add_edge("register", "notify", relation="OPERATION_PRECEDES", edge_type="deprecated", weight=100)
        graph.add_node("other", name="Other", executable=True)
        graph.add_edge("other", "register", relation="OPERATION_PRECEDES", edge_type="mandatory", weight=1)

        contextual = ContextualSubgraphBuilder(radius=0).build(
            self._plan("register", "notify", graph=graph)
        )
        subgraph = contextual["domain_graph"]

        self.assertEqual(subgraph.graph["connections"], [])
        self.assertIn("deprecated", subgraph.graph["disconnected_pairs"][0]["reason"])


class BeamSearchScoringTests(unittest.TestCase):

    def _contextual_plan(self):
        graph = nx.DiGraph()
        graph.add_node("register", name="Register User", executable=True)
        graph.add_node("validate", name="Validate User", executable=True)
        graph.add_node("notify", name="Notify Customer", executable=True)
        graph.add_node("customer", name="Customer", executable=False)
        graph.add_edge("register", "validate", relation="OPERATION_INCLUDES", edge_type="mandatory", weight=2)
        graph.add_edge("validate", "notify", relation="OPERATION_PRECEDES", edge_type="mandatory", weight=1, condition="user is valid")
        graph.graph["search_strategy"] = "dijkstra"

        return {
            "actions": [
                {
                    "prompt_node_id": "P1",
                    "prompt_text": "register",
                    "candidates": [{"domain_node_id": "register", "domain_node_name": "Register User", "similarity": 0.9}],
                },
                {
                    "prompt_node_id": "P2",
                    "prompt_text": "notify",
                    "candidates": [
                        {"domain_node_id": "customer", "domain_node_name": "Customer", "similarity": 0.95},
                        {"domain_node_id": "notify", "domain_node_name": "Notify Customer", "similarity": 0.8},
                    ],
                },
            ],
            "domain_graph": graph,
        }

    def test_constraint_score_rejects_a_non_executable_best_match(self):
        search_result = BeamSearchPlanner(beam_width=2).search(self._contextual_plan())

        self.assertEqual(
            [step["domain_node_id"] for step in search_result["selection"]],
            ["register", "notify"],
        )
        self.assertLess(search_result["selection"][1]["constraint_score"], 0.0001)

    def test_every_score_axis_is_reported_per_step(self):
        search_result = BeamSearchPlanner(beam_width=2).search(self._contextual_plan())
        step = search_result["selection"][1]

        for key in ("semantic_score", "relationship_score", "connectivity_score", "constraint_score"):
            self.assertIn(key, step)

    def test_relationship_conditions_are_surfaced_as_workflow_constraints(self):
        workflow_graph = BeamSearchPlanner(beam_width=2).plan(self._contextual_plan())

        self.assertEqual(
            [item["condition"] for item in workflow_graph.graph["constraints"]],
            ["user is valid"],
        )

    def test_bfs_strategy_is_used_when_the_subgraph_asks_for_it(self):
        plan = self._contextual_plan()
        plan["domain_graph"].graph["search_strategy"] = "bfs"

        search_result = BeamSearchPlanner(beam_width=2).search(plan)

        self.assertEqual(
            search_result["selection"][1]["transition"]["path"],
            ["register", "validate", "notify"],
        )


if __name__ == "__main__":
    unittest.main()
