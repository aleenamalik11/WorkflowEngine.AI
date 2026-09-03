import networkx as nx

from helpers.beam_search_utils import (
    _is_executable_selection_item,
    INFERRED_OPERATION_RELATIONSHIPS,
)


COMPOSITE_RELATIONSHIP = "OPERATION_INCLUDES"


class WorkflowGraphBuilder:

    def build(
        self,
        search_result,
        candidate_plan=None,
    ):

        if not search_result:
            raise RuntimeError(
                "BeamSearchPlanner.expand() received "
                "no search result."
            )

        selection = search_result.get(
            "selection",
            [],
        )

        if not selection:
            raise RuntimeError(
                "Beam search produced no "
                "selected workflow nodes."
            )

        prompt_subgraph = None

        if candidate_plan:
            prompt_subgraph = candidate_plan.get(
                "prompt_domain_subgraph"
            )

        # ---------------------------------------------------------
        # Expand composite operations before creating runtime nodes.
        # ---------------------------------------------------------

        if prompt_subgraph is not None:

            selection = (
                self._expand_composites(
                    selection,
                    prompt_subgraph,
                )
            )

        selected_graph = nx.DiGraph()

        selected_ids = []

        # ---------------------------------------------------------
        # Add executable nodes.
        # ---------------------------------------------------------

        for item in selection:

            node_id = item[
                "domain_node_id"
            ]

            if not _is_executable_selection_item(
                item
            ):
                continue

            if node_id not in selected_ids:
                selected_ids.append(
                    node_id
                )

            if selected_graph.has_node(
                node_id
            ):
                continue

            selected_graph.add_node(
                node_id,
                **item,
            )

        if not selected_ids:
            raise RuntimeError(
                "Beam search produced no "
                "executable workflow operations."
            )

        # ---------------------------------------------------------
        # No contextual graph.
        # ---------------------------------------------------------

        if prompt_subgraph is None:

            workflow_graph = (
                selected_graph.copy()
            )

            execution_order = (
                self._execution_order(
                    workflow_graph
                )
            )

            return {
                "graph": workflow_graph,
                "execution_order": execution_order,
                "conditional_branches": [],
            }

        # ---------------------------------------------------------
        # Prompt text.
        # ---------------------------------------------------------

        prompt = ""

        for item in selection:

            if item.get(
                "prompt_text"
            ):

                prompt = item[
                    "prompt_text"
                ]

                break

        # ---------------------------------------------------------
        # Connect operations.
        # ---------------------------------------------------------

        for source_id in selected_ids:

            for target_id in selected_ids:

                if source_id == target_id:
                    continue

                # -------------------------------------------------
                # Direct domain relationship.
                # -------------------------------------------------

                if prompt_subgraph.has_edge(
                    source_id,
                    target_id,
                ):

                    edge_data = (
                        prompt_subgraph.get_edge_data(
                            source_id,
                            target_id,
                        )
                    )

                    relation = edge_data.get(
                        "relation"
                    )

                    # OPERATION_INCLUDES does not mean:
                    #
                    #     parent executes before child
                    #
                    # It only represents composition.
                    if (
                        relation
                        == COMPOSITE_RELATIONSHIP
                    ):
                        continue

                    path = [
                        source_id,
                        target_id,
                    ]

                else:

                    path = (
                        self._most_contextual_path(
                            prompt_subgraph,
                            source_id,
                            target_id,
                            prompt,
                        )
                    )

                if not path:
                    continue

                # -------------------------------------------------
                # Add intermediate executable operations.
                # -------------------------------------------------

                for node_id in path:

                    if selected_graph.has_node(
                        node_id
                    ):
                        continue

                    if not prompt_subgraph.has_node(
                        node_id
                    ):
                        continue

                    node_data = (
                        prompt_subgraph.nodes[
                            node_id
                        ]
                    )

                    if node_data.get(
                        "node_type"
                    ) not in {
                        "Operation",
                        "operation",
                    }:
                        continue

                    selected_graph.add_node(
                        node_id,
                        **node_data,
                    )

                # -------------------------------------------------
                # Add path edges.
                # -------------------------------------------------

                for source, target in zip(
                    path,
                    path[1:],
                ):

                    edge_data = (
                        prompt_subgraph.get_edge_data(
                            source,
                            target,
                        )
                    )

                    if edge_data is None:
                        continue

                    if (
                        edge_data.get(
                            "relation"
                        )
                        == COMPOSITE_RELATIONSHIP
                    ):
                        continue

                    if selected_graph.has_edge(
                        source,
                        target,
                    ):
                        continue

                    selected_graph.add_edge(
                        source,
                        target,
                        **edge_data,
                    )

        workflow_graph = (
            selected_graph.copy()
        )

        for _, node in (
            workflow_graph.nodes(
                data=True
            )
        ):

            node.setdefault(
                "inferred",
                False,
            )

        execution_order = (
            self._execution_order(
                workflow_graph
            )
        )

        conditional_branches = [
            {
                "source": source,
                "target": target,
                "condition": data.get(
                    "condition",
                    "",
                ),
            }
            for source, target, data
            in workflow_graph.edges(
                data=True
            )
            if data.get(
                "relation"
            ) == "PROMPT_CONDITION"
        ]

        return {
            "graph": workflow_graph,
            "execution_order": execution_order,
            "conditional_branches": (
                conditional_branches
            ),
        }

    # =========================================================
    # Composite expansion
    # =========================================================

    def _expand_composites(
        self,
        selection,
        graph,
    ):

        expanded = []

        for item in selection:

            node_id = item.get(
                "domain_node_id"
            )

            if not graph.has_node(
                node_id
            ):
                expanded.append(
                    item
                )
                continue

            node_data = graph.nodes[
                node_id
            ]

            if node_data.get(
                "node_type"
            ) not in {
                "Operation",
                "operation",
            }:
                continue

            children = []

            for _, child_id, edge_data in (
                graph.out_edges(
                    node_id,
                    data=True,
                )
            ):

                if edge_data.get(
                    "relation"
                ) != COMPOSITE_RELATIONSHIP:
                    continue

                if not graph.has_node(
                    child_id
                ):
                    continue

                child_data = graph.nodes[
                    child_id
                ]

                if child_data.get(
                    "node_type"
                ) not in {
                    "Operation",
                    "operation",
                }:
                    continue

                children.append(
                    child_id
                )

            # -----------------------------------------------------
            # Ordinary operation.
            # -----------------------------------------------------

            if not children:
                expanded.append(
                    item
                )
                continue

            # -----------------------------------------------------
            # Composite operation.
            #
            # Do not create a runtime node for:
            #
            #     transfer_funds
            #
            # Create runtime nodes for its executable children.
            # -----------------------------------------------------

            for child_id in children:

                child_data = graph.nodes[
                    child_id
                ]

                expanded.append(
                    {
                        "prompt_text": item.get(
                            "prompt_text",
                            "",
                        ),
                        "domain_node_id": child_id,
                        "domain_node_name": child_data.get(
                            "name",
                            child_id,
                        ),
                        "domain_node_type": child_data.get(
                            "node_type",
                            "Operation",
                        ),
                        "explicit": False,
                        "inferred": True,
                        "source": "composite_expansion",
                        "semantic_score": child_data.get(
                            "semantic_score",
                            0.0,
                        ),
                        "lexical_score": child_data.get(
                            "lexical_score",
                            0.0,
                        ),
                        "candidate_score": child_data.get(
                            "score",
                            0.0,
                        ),
                        "relationship_score": 0.0,
                        "connectivity_score": 0.0,
                        "constraint_penalty": 0.0,
                        "constraint_violations": [],
                        "condition": item.get(
                            "condition",
                            "",
                        ),
                        "composite_parent": node_id,
                    }
                )

        # ---------------------------------------------------------
        # Deduplicate.
        # ---------------------------------------------------------

        result = []

        seen = set()

        for item in expanded:

            node_id = item.get(
                "domain_node_id"
            )

            if node_id in seen:
                continue

            seen.add(
                node_id
            )

            result.append(
                item
            )

        return result

    # =========================================================
    # Execution order
    # =========================================================

    def _execution_order(
        self,
        graph,
    ):

        try:

            return list(
                nx.topological_sort(
                    graph
                )
            )

        except nx.NetworkXUnfeasible:

            cycle = (
                self._describe_cycle(
                    graph
                )
            )

            raise RuntimeError(
                "Selected workflow contains "
                "a cycle and cannot be ordered: "
                + cycle
            )

    # =========================================================
    # Cycle diagnostics
    # =========================================================

    def _describe_cycle(
        self,
        graph,
    ):

        try:

            cycle_edges = (
                nx.find_cycle(
                    graph
                )
            )

        except nx.NetworkXNoCycle:

            return (
                "(cycle detected by "
                "topological_sort but not "
                "reproducible via find_cycle)"
            )

        names = []

        for source, target in cycle_edges:

            edge_data = graph.edges[
                source,
                target,
            ]

            names.append(
                f"{graph.nodes[source].get('name', source)} "
                f"--{edge_data.get('relation', '?')}--> "
                f"{graph.nodes[target].get('name', target)}"
            )

        return " ; ".join(
            names
        )

    # =========================================================
    # Contextual path
    # =========================================================

    def _most_contextual_path(
        self,
        graph,
        source,
        target,
        prompt,
        max_path_length=4,
    ):

        if source == target:
            return [
                source
            ]

        if (
            source not in graph
            or target not in graph
        ):
            return None

        # -----------------------------------------------------
        # Direct relationship.
        # -----------------------------------------------------

        if graph.has_edge(
            source,
            target,
        ):

            edge_data = (
                graph.get_edge_data(
                    source,
                    target,
                )
            )

            relation = edge_data.get(
                "relation"
            )

            if (
                relation
                in INFERRED_OPERATION_RELATIONSHIPS
                and relation
                != COMPOSITE_RELATIONSHIP
            ):

                return [
                    source,
                    target,
                ]

        best_path = None

        best_score = float(
            "-inf"
        )

        try:

            paths = nx.all_simple_paths(
                graph,
                source,
                target,
                cutoff=max_path_length,
            )

            for path in paths:

                if not self._is_contextual_operation_path(
                    graph,
                    path,
                ):
                    continue

                score = (
                    self._contextual_path_score(
                        graph,
                        path,
                        prompt,
                    )
                )

                if score > best_score:

                    best_score = score

                    best_path = path

        except nx.NetworkXNoPath:

            return None

        return best_path

    # =========================================================
    # Valid contextual path
    # =========================================================

    def _is_contextual_operation_path(
        self,
        graph,
        path,
    ):

        for node_id in path:

            node_type = (
                graph.nodes[
                    node_id
                ].get(
                    "node_type"
                )
            )

            if node_type not in {
                "Operation",
                "operation",
            }:
                return False

        for source, target in zip(
            path,
            path[1:],
        ):

            relation = (
                graph.edges[
                    source,
                    target,
                ].get(
                    "relation"
                )
            )

            if (
                relation
                not in INFERRED_OPERATION_RELATIONSHIPS
            ):
                return False

            if (
                relation
                == COMPOSITE_RELATIONSHIP
            ):
                return False

        return True

    # =========================================================
    # Path scoring
    # =========================================================

    def _contextual_path_score(
        self,
        graph,
        path,
        prompt,
    ):

        if not path:
            return float(
                "-inf"
            )

        if len(path) == 2:
            return 1.0

        node_scores = []

        for node_id in path:

            node_data = graph.nodes.get(
                node_id,
                {},
            )

            semantic_score = (
                node_data.get(
                    "semantic_score",
                    0.0,
                )
            )

            lexical_score = (
                node_data.get(
                    "lexical_score",
                    0.0,
                )
            )

            node_scores.append(
                0.7 * semantic_score
                + 0.3 * lexical_score
            )

        intermediate_scores = (
            node_scores[1:-1]
        )

        contextual_score = (
            sum(
                intermediate_scores
            )
            / len(
                intermediate_scores
            )
            if intermediate_scores
            else 0.0
        )

        path_penalty = (
            0.1
            * (len(path) - 2)
        )

        return (
            contextual_score
            - path_penalty
        )