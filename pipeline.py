"""
End-to-end workflow pipeline.

Architecture:

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

The domain graph is the semantic source of truth.

No Dijkstra.
No shortest-path routing.
"""

import json

from semantic_parser import (
    HybridSemanticParser,
)

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
        # STAGE 0
        #
        # Load domain context for the LLM.
        #
        # IMPORTANT:
        #
        # The LLM can infer semantic concepts, but it should
        # not hallucinate arbitrary domain concepts.
        #
        # The ontology gives it the vocabulary of the domain.
        # -----------------------------------------------------

        domain_context = (
            self._build_domain_context()
        )

        # -----------------------------------------------------
        # STAGE 1 / 2 / 11
        # Semantic interpretation
        # -----------------------------------------------------

        interpretation = (
            self.parser.parse(
                prompt,
                domain_context=(
                    domain_context
                ),
            )
        )

        debug[
            "semantic_interpretation"
        ] = (
            interpretation
            .as_debug_dict()
        )

        self._log(
            "STAGE 1/2 - Semantic Interpretation",
            debug[
                "semantic_interpretation"
            ],
        )

        # -----------------------------------------------------
        # STAGES 3-7
        # Semantic + lexical matching
        # + neighborhood expansion
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

        for (
            source,
            target,
            data,
        ) in (
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

        # self._log(
        #     "STAGE 7 - Weighted domain graph",
        #     weighted_edges,
        # )

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
        ] = search_result.get(
            "beam",
            [],
        )

        self._log(
            "STAGE 8 - Beam candidates",
            search_result.get(
                "beam",
                [],
            ),
        )

        self._log(
            "STAGE 8 - Selected concepts",
            [
                (
                    item.get(
                        "prompt_text",
                        "",
                    ),
                    "->",
                    item.get(
                        "domain_node_name",
                        "",
                    ),
                    "source=",
                    item.get(
                        "source",
                        "unknown",
                    ),
                )
                for item in search_result.get(
                    "selection",
                    [],
                )
            ],
        )

        # -----------------------------------------------------
        # Fail explicitly rather than generating an empty
        # workflow.
        # -----------------------------------------------------

        if not search_result.get(
            "selection"
        ):

            raise RuntimeError(
                "Beam search produced no selected workflow nodes. "
                "Stage 3 returned no usable domain candidates."
            )

        # -----------------------------------------------------
        # STAGE 8 -> STAGE 9
        # -----------------------------------------------------

        workflow_graph = (
            self.beam_planner.expand(
                search_result,
                candidate_plan=(
                    candidate_plan
                ),
            )
        )

        # -----------------------------------------------------
        # STAGE 9
        # Graph planning
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
            ].nodes[
                node
            ].get(
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
        # Function matching + generation
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
    # Domain context for LLM
    # =========================================================

    def _build_domain_context(
        self,
    ):

        try:

            nodes = (
                self.domain_client.all_nodes()
            )

        except Exception:

            return []

        context = []

        for node in nodes:

            context.append(
                {
                    "id": node.id,
                    "name": node.name,
                    "type": node.node_type,
                    "types": list(
                        node.types or []
                    ),
                    "description": (
                        node.description
                        or ""
                    ),
                    "aliases": list(
                        node.aliases
                        or []
                    ),
                }
            )

        return context

    # =========================================================
    # Logging
    # =========================================================

    def _log(
        self,
        title,
        payload,
    ):

        if not self.verbose:
            return

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