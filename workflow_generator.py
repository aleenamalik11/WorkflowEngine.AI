import uuid

from models import (
    FunctionMatch,
    WorkflowFunctionDetails,
)


class WorkflowGenerator:

    def __init__(
        self,
        function_matcher,
    ):

        self.function_matcher = (
            function_matcher
        )

    # ==========================================================
    # Helpers
    # ==========================================================

    @staticmethod
    def _get_node_data(
        graph,
        node_id,
    ):
        """
        Safely retrieve graph node metadata.
        """

        if graph is None:
            return {}

        if hasattr(
            graph,
            "nodes",
        ):

            if node_id in graph.nodes:

                return dict(
                    graph.nodes[node_id]
                )

            return {}

        if (
            isinstance(
                graph,
                dict,
            )
            and "nodes" in graph
        ):

            nodes = graph["nodes"]

            if isinstance(
                nodes,
                dict,
            ):

                return dict(
                    nodes.get(
                        node_id,
                        {},
                    )
                )

            for node in nodes:

                if (
                    node.get("id")
                    == node_id
                ):

                    return dict(node)

        return {}

    # ==========================================================
    # Extract graph edges
    # ==========================================================

    @staticmethod
    def _get_edges(
        graph,
    ):
        """
        Normalize graph edges.
        """

        if graph is None:
            return []

        if hasattr(
            graph,
            "edges",
        ):

            return list(
                graph.edges(
                    data=True
                )
            )

        if (
            isinstance(
                graph,
                dict,
            )
            and "edges" in graph
        ):

            edges = []

            for edge in graph[
                "edges"
            ]:

                if (
                    isinstance(
                        edge,
                        (tuple, list),
                    )
                    and len(edge) >= 2
                ):

                    source = edge[0]

                    target = edge[1]

                    data = (
                        edge[2]
                        if (
                            len(edge) > 2
                            and isinstance(
                                edge[2],
                                dict,
                            )
                        )
                        else {}
                    )

                    edges.append(
                        (
                            source,
                            target,
                            data,
                        )
                    )

                elif isinstance(
                    edge,
                    dict,
                ):

                    edges.append(
                        (
                            edge.get(
                                "source"
                            ),
                            edge.get(
                                "target"
                            ),
                            edge,
                        )
                    )

            return edges

        return []

    # ==========================================================
    # Resolve function
    # ==========================================================

    def _match_semantic_node(
        self,
        node_data,
    ):
        """
        Match the already-selected semantic/domain node against
        the registered function catalogue.

        IMPORTANT:

        This method never changes the semantic node.

        The semantic node is the source of truth.

        Function matching only determines whether there is an
        implementation attached to that node.
        """

        semantic_name = (
            node_data.get(
                "name"
            )
            or node_data.get(
                "domain_node_name"
            )
            or node_data.get(
                "prompt_text"
            )
            or ""
        )

        semantic_description = (
            node_data.get(
                "description"
            )
            or ""
        )

        # ------------------------------------------------------
        # Prefer the semantic/domain description when available.
        #
        # This gives the function matcher richer meaning without
        # changing the identity of the node.
        # ------------------------------------------------------

        match_text = (
            semantic_name
        )

        if semantic_description:

            match_text = (
                f"{semantic_name}. "
                f"{semantic_description}"
            )

        match = (
            self.function_matcher.match(
                match_text
            )
        )

        return (
            semantic_name,
            match_text,
            match,
        )

    # ==========================================================
    # Generate Workflow JSON
    # ==========================================================

    def generate(
        self,
        plan,
        workflow_name="Generated Workflow",
    ):
        """
        Stage 10/11.

        Converts the Stage 9 semantic execution plan into the
        final workflow model.

        Architectural rule:

            Stage 8/9 decides WHAT the workflow means.

            Stage 10 decides WHETHER each semantic node has an
            executable registered function.

        Stage 10 must never replace a semantic node with another
        node merely because a function happens to be similar.
        """

        # ------------------------------------------------------
        # Extract graph
        # ------------------------------------------------------

        if isinstance(
            plan,
            dict,
        ):

            graph = plan.get(
                "graph"
            )

            execution_order = (
                plan.get(
                    "execution_order"
                )
                or plan.get(
                    "order"
                )
                or plan.get(
                    "path"
                )
                or []
            )

        else:

            graph = plan

            execution_order = []

        # ------------------------------------------------------
        # Fallback only when the planner did not provide an
        # execution order.
        # ------------------------------------------------------

        if (
            not execution_order
            and graph is not None
        ):

            if hasattr(
                graph,
                "nodes",
            ):

                execution_order = list(
                    graph.nodes()
                )

            elif (
                isinstance(
                    graph,
                    dict,
                )
                and "nodes" in graph
            ):

                nodes = graph[
                    "nodes"
                ]

                if isinstance(
                    nodes,
                    dict,
                ):

                    execution_order = list(
                        nodes.keys()
                    )

                else:

                    execution_order = [
                        node.get(
                            "id",
                            index,
                        )
                        for index, node
                        in enumerate(nodes)
                    ]

        # ------------------------------------------------------
        # Workflow root
        # ------------------------------------------------------

        workflow = {

            "Id": str(
                uuid.uuid4()
            ),

            "Name": workflow_name,

            "Version": "1.0",

            "StartNodeId": None,

            "Inputs": [],

            "Nodes": [],

            "Connections": {},
        }

        workflow_node_lookup = {}

        if graph is None:

            return workflow

        # ======================================================
        # STAGE 10
        # Semantic node -> registered function
        # ======================================================

        for node_id in execution_order:

            node_data = (
                self._get_node_data(
                    graph,
                    node_id,
                )
            )

            # --------------------------------------------------
            # The semantic node name is authoritative.
            # --------------------------------------------------

            (
                semantic_name,
                match_text,
                function_match,
            ) = self._match_semantic_node(
                node_data
            )

            # --------------------------------------------------
            # Preserve semantic node identity.
            # --------------------------------------------------

            workflow_node_id = str(
                uuid.uuid4()
            )

            workflow_node_lookup[
                node_id
            ] = workflow_node_id

            # --------------------------------------------------
            # Function details are metadata attached to the
            # semantic node.
            # --------------------------------------------------

            function_details = (
                WorkflowFunctionDetails.from_match(
                    function_match,
                    semantic_text=match_text,
                )
            )

            # --------------------------------------------------
            # Node inputs/outputs:
            #
            # If a function exists, its signature can describe
            # the executable implementation.
            #
            # If it does not exist, we DO NOT invent inputs or
            # outputs from another function.
            # --------------------------------------------------

            node_inputs = []

            node_outputs = []

            if function_details.found:

                node_inputs = list(
                    function_details.inputs
                )

                node_outputs = list(
                    function_details.outputs
                )

            # --------------------------------------------------
            # Construct workflow node.
            #
            # IMPORTANT:
            #
            # Name = semantic/domain concept
            #
            # NOT:
            #
            # Name = function name
            # --------------------------------------------------

            workflow_node = {

                "Id": workflow_node_id,

                "Name": semantic_name,

                "Type": "CustomNode",

                "Description": (
                    node_data.get(
                        "description",
                        "",
                    )
                ),

                "Inputs": node_inputs,

                "Outputs": node_outputs,

                "Inferred": node_data.get(
                    "inferred",
                    False,
                ),

                "FunctionDetails": (
                    function_details.to_dict()
                ),
            }

            workflow[
                "Nodes"
            ].append(
                workflow_node
            )

        # ======================================================
        # STAGE 11
        # Start node
        # ======================================================

        if execution_order:

            first = (
                execution_order[0]
            )

            if (
                first
                in workflow_node_lookup
            ):

                workflow[
                    "StartNodeId"
                ] = workflow_node_lookup[
                    first
                ]

        elif workflow[
            "Nodes"
        ]:

            workflow[
                "StartNodeId"
            ] = workflow[
                "Nodes"
            ][0]["Id"]

        # ======================================================
        # STAGE 11
        # Connections
        # ======================================================

        edges = self._get_edges(
            graph
        )

        for (
            source,
            target,
            edge_data,
        ) in edges:

            if (
                source
                not in workflow_node_lookup
            ):

                continue

            if (
                target
                not in workflow_node_lookup
            ):

                continue

            source_id = (
                workflow_node_lookup[
                    source
                ]
            )

            target_id = (
                workflow_node_lookup[
                    target
                ]
            )

            relation = (
                edge_data.get(
                    "relation",
                    "success",
                )
                if isinstance(
                    edge_data,
                    dict,
                )
                else "success"
            )

            workflow[
                "Connections"
            ].setdefault(
                source_id,
                {},
            )

            workflow[
                "Connections"
            ][source_id][
                relation
            ] = target_id

        # ======================================================
        # Sequential fallback
        # ======================================================
        #
        # Only used when the planner supplied no usable edges.
        #
        # We do NOT create semantic/function substitutions here.
        # ======================================================

        if (
            not workflow[
                "Connections"
            ]
            and len(
                workflow[
                    "Nodes"
                ]
            ) > 1
        ):

            for index in range(
                len(
                    workflow[
                        "Nodes"
                    ]
                )
                - 1
            ):

                source_id = (
                    workflow[
                        "Nodes"
                    ][index]["Id"]
                )

                target_id = (
                    workflow[
                        "Nodes"
                    ][index + 1]["Id"]
                )

                workflow[
                    "Connections"
                ].setdefault(
                    source_id,
                    {},
                )

                workflow[
                    "Connections"
                ][source_id][
                    "success"
                ] = target_id

        # ======================================================
        # Terminal nodes
        # ======================================================

        for node in workflow[
            "Nodes"
        ]:

            node_id = node[
                "Id"
            ]

            if (
                node_id
                not in workflow[
                    "Connections"
                ]
            ):

                workflow[
                    "Connections"
                ][node_id] = {
                    "success": "Done"
                }

        # ======================================================
        # Workflow inputs
        # ======================================================
        #
        # Only collect inputs from functions that actually exist.
        #
        # An unmatched semantic node must not cause an unrelated
        # function's inputs to appear in the workflow.
        # ======================================================

        inputs = []

        for node in workflow[
            "Nodes"
        ]:

            function_details = (
                node.get(
                    "FunctionDetails",
                    {},
                )
            )

            if not function_details.get(
                "Found",
                False,
            ):

                continue

            for input_name in (
                function_details.get(
                    "Inputs",
                    [],
                )
            ):

                if (
                    input_name
                    not in inputs
                ):

                    inputs.append(
                        input_name
                    )

        workflow[
            "Inputs"
        ] = inputs

        return workflow

    # ==========================================================
    # Pretty Print
    # ==========================================================

    @staticmethod
    def print(
        workflow,
    ):

        print()

        print(
            "=" * 70
        )

        print(
            workflow.get(
                "Name",
                "Generated Workflow",
            )
        )

        print(
            "=" * 70
        )

        print()

        print(
            "Start Node"
        )

        print(
            workflow.get(
                "StartNodeId"
            )
        )

        print()

        print(
            "Nodes"
        )

        for node in workflow.get(
            "Nodes",
            [],
        ):

            details = node.get(
                "FunctionDetails",
                {},
            )

            if details.get(
                "Found",
                False,
            ):

                status = (
                    "function="
                    + str(
                        details.get(
                            "FunctionName"
                        )
                    )
                )

            else:

                status = (
                    "matching function not found"
                )

            inferred = (
                "Inferred"
                if node.get(
                    "Inferred"
                )
                else "Matched"
            )

            print(
                f"{node.get('Name')} "
                f"[{inferred}] "
                f"[{status}]"
            )

        print()

        print(
            "Connections"
        )

        for (
            source,
            transitions,
        ) in workflow.get(
            "Connections",
            {},
        ).items():

            print(
                source
            )

            for (
                key,
                target,
            ) in transitions.items():

                print(
                    "   ",
                    key,
                    "->",
                    target,
                )