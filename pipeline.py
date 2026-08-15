"""
End-to-end workflow pipeline.

    user prompt
        |
        v
    HybridSemanticParser
        |
        | Stage 1/2/11
        v
    Contextual Subgraph Builder
        |
        | Stage 3-7
        v
    Beam Search
        |
        | Stage 8
        v
    Graph Planner
        |
        | Stage 9
        v
    Workflow Generator
        |
        | Stage 10/11
        v
    Workflow JSON
"""

from semantic_parser import HybridSemanticParser
from contextual_subgraph_builder import (
    build_contextual_subgraph,
)
from beam_search_planner import (
    BeamSearchPlanner,
)
from graph_planner import (
    GraphPlanner,
    PlanningError,
)
from workflow_generator import (
    WorkflowGenerator,
)


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

        self.domain_client = (
            domain_client
        )

        self.embedding_service = (
            embedding_service
        )

        self.function_matcher = (
            function_matcher
        )

        self.llm_service = (
            llm_service
        )

        self.parser = (
            parser
            or HybridSemanticParser(
                llm_service=llm_service,
                enable_llm=(
                    llm_service
                    is not None
                ),
            )
        )

        self.beam_planner = (
            BeamSearchPlanner(
                beam_width=beam_width
            )
        )

        self.graph_planner = (
            GraphPlanner()
        )

        self.generator = (
            WorkflowGenerator(
                function_matcher
            )
        )

        self.top_k = top_k

        self.neighborhood_depth = (
            neighborhood_depth
        )

        self.verbose = verbose

    # =========================================================
    # Pipeline
    # =========================================================

    def run(
        self,
        prompt,
        workflow_name="Generated Workflow",
    ):

        debug = {}

        # -----------------------------------------------------
        # STAGE 1 / 2
        # Semantic interpretation + LLM enrichment
        # -----------------------------------------------------

        interpretation = (
            self.parser.parse(
                prompt
            )
        )

        debug[
            "semantic_interpretation"
        ] = interpretation.as_debug_dict()

        self._log(
            "STAGE 1/2 - Semantic Interpretation",
            debug[
                "semantic_interpretation"
            ],
        )

        # -----------------------------------------------------
        # STAGES 3-7
        # Candidate matching + contextual neighborhood
        # -----------------------------------------------------

        (
            candidate_plan,
            subgraph_debug,
        ) = build_contextual_subgraph(
            interpretation,
            self.domain_client,
            self.embedding_service,
            k=self.top_k,
            neighborhood_depth=(
                self.neighborhood_depth
            ),
        )

        debug.update(
            subgraph_debug
        )

        self._log(
            "STAGE 3 - Candidate domain nodes",
            subgraph_debug[
                "candidates"
            ],
        )

        self._log(
            "STAGE 4/6 - Context attachments",
            subgraph_debug[
                "context_attachments"
            ],
        )

        self._log(
            "STAGE 5 - Prompt constraints",
            subgraph_debug[
                "prompt_constraint_edges"
            ],
        )

        self._log(
            "STAGE 6 - Inferred neighborhood concepts",
            subgraph_debug.get(
                "inferred_nodes",
                [],
            ),
        )

        # -----------------------------------------------------
        # STAGE 7
        # Relationship semantics
        # -----------------------------------------------------

        weighted_edges = []

        for source, target, data in (
            candidate_plan[
                "domain_graph"
            ].edges(
                data=True
            )
        ):

            weighted_edges.append(
                {
                    "source": source,
                    "target": target,
                    **data,
                }
            )

        self._log(
            "STAGE 7 - Weighted domain graph",
            weighted_edges,
        )

        # -----------------------------------------------------
        # STAGE 8
        # Beam Search
        # -----------------------------------------------------

        search_result = (
            self.beam_planner.search(
                candidate_plan
            )
        )

        debug[
            "beam_search"
        ] = search_result[
            "beam"
        ]

        self._log(
            "STAGE 8 - Beam candidates",
            search_result[
                "beam"
            ],
        )

        self._log(
            "STAGE 8 - Selected concepts",
            [
                (
                    item[
                        "prompt_text"
                    ],
                    "->",
                    item[
                        "domain_node_name"
                    ],
                    "source=",
                    item.get(
                        "source"
                    ),
                )
                for item in search_result[
                    "selection"
                ]
            ],
        )

        # -----------------------------------------------------
        # Stage 8 -> Stage 9
        # -----------------------------------------------------

        workflow_graph = (
            self.beam_planner.expand(
                search_result
            )
        )

        # -----------------------------------------------------
        # STAGE 9
        # Topological planning
        # -----------------------------------------------------

        try:

            plan = (
                self.graph_planner.plan(
                    workflow_graph
                )
            )

        except PlanningError as error:

            self._log(
                "STAGE 9 - PLANNING ERROR",
                str(error),
            )

            raise

        debug[
            "execution_order"
        ] = [
            plan[
                "graph"
            ].nodes[node].get(
                "name",
                node,
            )
            for node in plan[
                "execution_order"
            ]
        ]

        self._log(
            "STAGE 9 - Execution order",
            debug[
                "execution_order"
            ],
        )

        # -----------------------------------------------------
        # STAGE 10 / 11
        # Function matching + workflow generation
        # -----------------------------------------------------

        workflow = (
            self.generator.generate(
                plan,
                workflow_name=(
                    workflow_name
                ),
            )
        )

        self._log(
            "STAGE 10/11 - Workflow JSON",
            workflow,
        )

        return (
            workflow,
            debug,
        )

    # =========================================================
    # Debug logging
    # =========================================================

    def _log(
        self,
        title,
        payload,
    ):

        if not self.verbose:
            return

        import json

        print(
            "\n"
            + "=" * 70
        )

        print(title)

        print(
            "=" * 70
        )

        try:

            print(
                json.dumps(
                    payload,
                    indent=2,
                    default=str,
                )
            )

        except TypeError:

            print(payload)