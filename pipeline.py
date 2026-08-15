"""
End-to-end orchestrator, replacing inference.py's linear script.

    user prompt
        -> HybridSemanticParser            (Stage 1/2/11)
        -> build_contextual_subgraph        (Stage 3/4/5/6)
        -> BeamSearchPlanner                (Stage 7 weights already applied, Stage 8 search)
        -> GraphPlanner                     (Stage 9, raises PlanningError on cycles)
        -> FunctionMatcher                  (Stage 10, unchanged)
        -> WorkflowGenerator                (Stage 10/11 -> workflow JSON)
"""

from semantic_parser import HybridSemanticParser
from contextual_subgraph_builder import build_contextual_subgraph
from beam_search_planner import BeamSearchPlanner
from graph_planner import GraphPlanner, PlanningError
from workflow_generator import WorkflowGenerator


class WorkflowPipeline:

    def __init__(
            self,
            domain_client,
            embedding_service,
            function_matcher,
            parser=None,
            llm_service=None,
            beam_width=3,
            top_k=5,
            neighborhood_depth=1,
            verbose=True,
    ):

        self.domain_client = domain_client
        self.embedding_service = embedding_service
        self.function_matcher = function_matcher

        self.llm_service = llm_service

        self.parser = parser or HybridSemanticParser(
            llm_service=llm_service,
            enable_llm=llm_service is not None,
        )

        self.beam_planner = BeamSearchPlanner(
            beam_width=beam_width
        )

        self.graph_planner = GraphPlanner()

        self.generator = WorkflowGenerator(
            function_matcher
        )

        self.top_k = top_k
        self.neighborhood_depth = neighborhood_depth
        self.verbose = verbose

    def run(self, prompt, workflow_name="Generated Workflow"):
        debug = {}

        # Stage 1 / 2
        interpretation = self.parser.parse(prompt)
        debug["semantic_interpretation"] = interpretation.as_debug_dict()
        self._log("STAGE 1/2 - Semantic Interpretation", debug["semantic_interpretation"])

        # Stage 3 / 4 / 5 / 6
        candidate_plan, subgraph_debug = build_contextual_subgraph(
            interpretation, self.domain_client, self.embedding_service,
            k=self.top_k, neighborhood_depth=self.neighborhood_depth,
        )
        debug.update(subgraph_debug)
        self._log("STAGE 3 - Candidate domain nodes", subgraph_debug["candidates"])
        self._log("STAGE 4/6 - Context attachments (non-routable)", subgraph_debug["context_attachments"])
        self._log("STAGE 5 - Prompt constraint edges added", subgraph_debug["prompt_constraint_edges"])

        # Stage 7 already baked into candidate_plan["domain_graph"] edges
        self._log("STAGE 7 - Weighted domain graph edges", [
            {"source": s, "target": t, **d}
            for s, t, d in candidate_plan["domain_graph"].edges(data=True)
        ])

        # Stage 8
        search_result = self.beam_planner.search(candidate_plan)
        debug["beam_search"] = search_result["beam"]
        self._log("STAGE 8 - Beam candidates", search_result["beam"])
        self._log("STAGE 8 - Selected path", [
            (s["prompt_text"], "->", s["domain_node_name"]) for s in search_result["selection"]
        ])

        workflow_graph = self.beam_planner.expand(search_result)

        # Stage 9
        try:
            plan = self.graph_planner.plan(workflow_graph)
        except PlanningError as e:
            self._log("STAGE 9 - PLANNING ERROR", str(e))
            raise
        debug["execution_order"] = [
            plan["graph"].nodes[n].get("name", n) for n in plan["execution_order"]
        ]
        self._log("STAGE 9 - Execution order", debug["execution_order"])

        # Stage 10 / 11
        workflow = self.generator.generate(plan, workflow_name=workflow_name)
        self._log("STAGE 10/11 - Workflow JSON", workflow)

        return workflow, debug

    def _log(self, title, payload):
        if not self.verbose:
            return
        import json
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
        try:
            print(json.dumps(payload, indent=2, default=str))
        except TypeError:
            print(payload)