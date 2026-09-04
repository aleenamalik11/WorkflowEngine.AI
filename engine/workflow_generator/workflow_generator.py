import uuid

from models import (
    WorkflowFunctionDetails,
)

from helpers.beam_search_utils import (
    INFERRED_OPERATION_RELATIONSHIPS,
)


COMPOSITE_RELATIONSHIP = "OPERATION_INCLUDES"

PROMPT_DEPENDENCY = "PROMPT_DEPENDENCY"
PROMPT_CONDITION = "PROMPT_CONDITION"


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
                    or node.get("Id")
                    == node_id
                ):

                    return dict(
                        node
                    )

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
                                "source",
                                edge.get("Source"),
                            ),
                            edge.get(
                                "target",
                                edge.get("Target"),
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

        Function matching NEVER changes the semantic identity
        of the node.

        Example:

            semantic node:
                Notify Customer

            function matcher:
                send_notification

        The workflow node remains:

            Name = Notify Customer

        and send_notification is stored only inside
        FunctionDetails.
        """

        semantic_name = (
            node_data.get(
                "name"
            )
            or node_data.get(
                "Name"
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
            or node_data.get(
                "Description"
            )
            or ""
        )

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
    # Generate Workflow
    # ==========================================================

    def generate(
        self,
        plan,
        workflow_name="Generated Workflow",
    ):
        """
        Converts the Stage 9 execution graph into the final
        workflow representation.

        Stages 8/9 determine WHAT the workflow means.

        Stage 10 determines whether each semantic node has a
        registered implementation.

        Stage 11 creates workflow transitions.

        Function matching NEVER replaces a semantic/domain node.
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

        if graph is None:

            return {
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

        # ------------------------------------------------------
        # Fallback execution order.
        #
        # Normally Stage 9 always provides this.
        # ------------------------------------------------------

        if not execution_order:

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
                            node.get(
                                "Id",
                                index,
                            ),
                        )
                        for index, node
                        in enumerate(nodes)
                    ]

        # ------------------------------------------------------
        # Workflow root.
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

            if not node_data:
                continue

            # --------------------------------------------------
            # Semantic node is authoritative.
            # --------------------------------------------------

            (
                semantic_name,
                match_text,
                function_match,
            ) = self._match_semantic_node(
                node_data
            )

            if not semantic_name:

                semantic_name = str(
                    node_id
                )

            # --------------------------------------------------
            # Create runtime workflow ID.
            # --------------------------------------------------

            workflow_node_id = str(
                uuid.uuid4()
            )

            workflow_node_lookup[
                node_id
            ] = workflow_node_id

            # --------------------------------------------------
            # Attach function metadata.
            # --------------------------------------------------

            function_details = (
                WorkflowFunctionDetails.from_match(
                    function_match,
                    semantic_text=match_text,
                )
            )

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
            # Condition.
            #
            # Stage 9 may provide condition metadata either
            # directly on the node or through conditional edges.
            # --------------------------------------------------

            condition = (
                node_data.get(
                    "condition"
                )
                or ""
            )

            is_conditional = bool(
                node_data.get(
                    "is_conditional",
                    False,
                )
                or condition
            )

            # --------------------------------------------------
            # Workflow node.
            # --------------------------------------------------

            workflow_node = {

                "Id": workflow_node_id,

                # IMPORTANT:
                # semantic name, NOT function name.
                "Name": semantic_name,

                "Type": "CustomNode",

                "Description": (
                    node_data.get(
                        "description",
                        node_data.get(
                            "Description",
                            "",
                        ),
                    )
                ),

                "Inputs": node_inputs,

                "Outputs": node_outputs,

                "Inferred": bool(
                    node_data.get(
                        "inferred",
                        False,
                    )
                ),

                "FunctionDetails": (
                    function_details.to_dict()
                ),

                "Condition": condition,

                "IsConditional": (
                    is_conditional
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

            for node_id in execution_order:

                if node_id in (
                    workflow_node_lookup
                ):

                    workflow[
                        "StartNodeId"
                    ] = workflow_node_lookup[
                        node_id
                    ]

                    break

        if (
            workflow[
                "StartNodeId"
            ] is None
            and workflow[
                "Nodes"
            ]
        ):

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

        # ------------------------------------------------------
        # We process explicit prompt relationships first.
        #
        # This ensures domain relationships cannot override
        # semantic prompt ordering.
        # ------------------------------------------------------

        prompt_edges = []

        inferred_edges = []

        for source, target, edge_data in edges:

            if (
                source not in workflow_node_lookup
                or target not in workflow_node_lookup
            ):
                continue

            relation = edge_data.get(
                "relation"
            )

            if relation == COMPOSITE_RELATIONSHIP:
                continue

            if relation in {
                PROMPT_DEPENDENCY,
                PROMPT_CONDITION,
            }:

                prompt_edges.append(
                    (
                        source,
                        target,
                        edge_data,
                    )
                )

            elif relation in (
                INFERRED_OPERATION_RELATIONSHIPS
            ):

                inferred_edges.append(
                    (
                        source,
                        target,
                        edge_data,
                    )
                )

        # ------------------------------------------------------
        # First add sequential prompt dependencies.
        # ------------------------------------------------------

        for source, target, edge_data in (
            prompt_edges
        ):

            if edge_data.get(
                "relation"
            ) != PROMPT_DEPENDENCY:
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

            self._add_connection(
                workflow[
                    "Connections"
                ],
                source_id,
                "success",
                target_id,
            )

        # ------------------------------------------------------
        # Add conditional prompt transitions.
        #
        # Important:
        #
        # A condition may appear on several semantic actions:
        #
        #     check balance -> reject
        #     check balance -> record
        #     check balance -> notify
        #
        # These do NOT mean that "check balance" has three
        # executable success transitions.
        #
        # The actual execution chain is:
        #
        #     check -> reject -> record -> notify
        #
        # Therefore only the FIRST conditional target from a
        # source is used as its branch transition.
        # ------------------------------------------------------

        conditional_by_source = {}

        for source, target, edge_data in (
            prompt_edges
        ):

            if edge_data.get(
                "relation"
            ) != PROMPT_CONDITION:
                continue

            conditional_by_source.setdefault(
                source,
                [],
            ).append(
                (
                    target,
                    edge_data,
                )
            )

        for source, candidates in (
            conditional_by_source.items()
        ):

            if source not in workflow_node_lookup:
                continue

            # --------------------------------------------------
            # Prefer the target appearing earliest in semantic
            # execution order.
            # --------------------------------------------------

            target, edge_data = (
                self._first_conditional_target(
                    candidates,
                    execution_order,
                )
            )

            if target is None:
                continue

            if target not in workflow_node_lookup:
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

            # --------------------------------------------------
            # The condition is represented on the target node.
            # The connection itself represents the true/success
            # branch.
            # --------------------------------------------------

            self._add_connection(
                workflow[
                    "Connections"
                ],
                source_id,
                "success",
                target_id,
            )

            # --------------------------------------------------
            # False condition terminates the minimal workflow.
            # --------------------------------------------------

            self._add_connection(
                workflow[
                    "Connections"
                ],
                source_id,
                "failure",
                "Done",
            )

            # --------------------------------------------------
            # Ensure target contains the condition metadata.
            # --------------------------------------------------

            self._set_workflow_node_condition(
                workflow,
                target_id,
                edge_data,
            )

        # ------------------------------------------------------
        # Add inferred domain relationships only when there is
        # no explicit prompt transition for the same pair.
        # ------------------------------------------------------

        for source, target, edge_data in (
            inferred_edges
        ):

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

            relation = edge_data.get(
                "relation"
            )

            if relation == COMPOSITE_RELATIONSHIP:
                continue

            # --------------------------------------------------
            # Explicit prompt edge wins.
            # --------------------------------------------------

            if self._has_graph_pair_edge(
                graph,
                source,
                target,
            ):

                if self._pair_has_prompt_relation(
                    prompt_edges,
                    source,
                    target,
                ):

                    continue

            # --------------------------------------------------
            # Do not overwrite a condition transition.
            # --------------------------------------------------

            existing = workflow[
                "Connections"
            ].get(
                source_id,
                {}
            )

            if existing.get(
                "success"
            ) not in (
                None,
                "Done",
            ):

                continue

            self._add_connection(
                workflow[
                    "Connections"
                ],
                source_id,
                "success",
                target_id,
            )

        # ======================================================
        # Sequential fallback
        # ======================================================
        #
        # Only when the graph supplied no usable transitions.
        # ======================================================

        if not self._has_executable_connections(
            workflow
        ):

            workflow_nodes = workflow[
                "Nodes"
            ]

            for index in range(
                len(
                    workflow_nodes
                ) - 1
            ):

                source_id = (
                    workflow_nodes[
                        index
                    ]["Id"]
                )

                target_id = (
                    workflow_nodes[
                        index + 1
                    ]["Id"]
                )

                self._add_connection(
                    workflow[
                        "Connections"
                    ],
                    source_id,
                    "success",
                    target_id,
                )

        # ======================================================
        # Terminal nodes
        # ======================================================

        for node in workflow[
            "Nodes"
        ]:

            node_id = node[
                "Id"
            ]

            connections = workflow[
                "Connections"
            ].setdefault(
                node_id,
                {},
            )

            if not connections:

                connections[
                    "success"
                ] = "Done"

            elif (
                "success" not in connections
                and "failure" not in connections
            ):

                connections[
                    "success"
                ] = "Done"

        # ======================================================
        # Workflow inputs
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
    # Conditional target ordering
    # ==========================================================

    @staticmethod
    def _first_conditional_target(
        candidates,
        execution_order,
    ):
        """
        Select the earliest conditional target according to the
        semantic execution order.
        """

        position = {
            node_id: index
            for index, node_id
            in enumerate(
                execution_order
            )
        }

        candidates = sorted(
            candidates,
            key=lambda pair: position.get(
                pair[0],
                float("inf"),
            ),
        )

        if not candidates:
            return (
                None,
                {},
            )

        return candidates[0]

    # ==========================================================
    # Connection helpers
    # ==========================================================

    @staticmethod
    def _add_connection(
        connections,
        source_id,
        transition,
        target_id,
    ):
        """
        Add a transition without accidentally replacing a
        previously established explicit transition.

        Explicit prompt transitions have priority.
        """

        source_connections = (
            connections.setdefault(
                source_id,
                {},
            )
        )

        if transition not in (
            source_connections
        ):

            source_connections[
                transition
            ] = target_id

    @staticmethod
    def _has_executable_connections(
        workflow,
    ):
        """
        Return True if at least one actual workflow transition
        exists.
        """

        for transitions in workflow.get(
            "Connections",
            {},
        ).values():

            for transition, target in (
                transitions.items()
            ):

                if target != "Done":

                    return True

        return False

    @staticmethod
    def _has_graph_pair_edge(
        graph,
        source,
        target,
    ):

        if hasattr(
            graph,
            "has_edge",
        ):

            return graph.has_edge(
                source,
                target,
            )

        return False

    @staticmethod
    def _pair_has_prompt_relation(
        prompt_edges,
        source,
        target,
    ):

        for edge_source, edge_target, edge_data in (
            prompt_edges
        ):

            if (
                edge_source == source
                and edge_target == target
            ):

                return True

        return False

    # ==========================================================
    # Condition metadata
    # ==========================================================

    @staticmethod
    def _set_workflow_node_condition(
        workflow,
        workflow_node_id,
        edge_data,
    ):

        condition = edge_data.get(
            "condition",
            "",
        )

        if not condition:
            return

        for node in workflow[
            "Nodes"
        ]:

            if node.get(
                "Id"
            ) != workflow_node_id:
                continue

            node[
                "Condition"
            ] = condition

            node[
                "IsConditional"
            ] = True

            return

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
                transition,
                target,
            ) in transitions.items():

                print(
                    "   ",
                    transition,
                    "->",
                    target,
                )