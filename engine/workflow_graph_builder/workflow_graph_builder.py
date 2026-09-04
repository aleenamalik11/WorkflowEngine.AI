import networkx as nx

from helpers.beam_search_utils import (
    _is_executable_selection_item,
    INFERRED_OPERATION_RELATIONSHIPS,
)


COMPOSITE_RELATIONSHIP = "OPERATION_INCLUDES"

PROMPT_DEPENDENCY = "PROMPT_DEPENDENCY"
PROMPT_CONDITION = "PROMPT_CONDITION"


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
            selection = self._expand_composites(
                selection,
                prompt_subgraph,
            )

        # ---------------------------------------------------------
        # Keep only executable selections.
        # ---------------------------------------------------------

        executable_selection = [
            item
            for item in selection
            if _is_executable_selection_item(item)
        ]

        if not executable_selection:
            raise RuntimeError(
                "Beam search produced no "
                "executable workflow operations."
            )

        # ---------------------------------------------------------
        # Sort by semantic prompt order.
        # ---------------------------------------------------------

        executable_selection = sorted(
            executable_selection,
            key=self._selection_order_key,
        )

        selected_graph = nx.DiGraph()

        selected_ids = []

        # ---------------------------------------------------------
        # Add executable nodes.
        # ---------------------------------------------------------

        for item in executable_selection:

            node_id = item.get(
                "domain_node_id"
            )

            if not node_id:
                continue

            if node_id not in selected_ids:
                selected_ids.append(
                    node_id
                )

            if selected_graph.has_node(
                node_id
            ):
                self._merge_node_metadata(
                    selected_graph.nodes[node_id],
                    item,
                )
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

        for item in executable_selection:

            if item.get("prompt_text"):

                prompt = item[
                    "prompt_text"
                ]

                break

        # ---------------------------------------------------------
        # First establish explicit prompt ordering.
        # ---------------------------------------------------------

        self._add_prompt_structure(
            selected_graph,
            executable_selection,
        )

        # ---------------------------------------------------------
        # Infer domain relationships between consecutive
        # selected operations.
        # ---------------------------------------------------------

        self._add_contextual_relationships(
            selected_graph,
            selected_ids,
            prompt_subgraph,
            prompt,
        )

        # ---------------------------------------------------------
        # Copy relevant node metadata from the prompt subgraph.
        # ---------------------------------------------------------

        for node_id in list(
            selected_graph.nodes
        ):

            if not prompt_subgraph.has_node(
                node_id
            ):
                continue

            prompt_node_data = (
                prompt_subgraph.nodes[
                    node_id
                ]
            )

            runtime_node_data = (
                selected_graph.nodes[
                    node_id
                ]
            )

            for key, value in (
                prompt_node_data.items()
            ):

                runtime_node_data.setdefault(
                    key,
                    value,
                )

        # ---------------------------------------------------------
        # Mark inferred nodes.
        # ---------------------------------------------------------

        for _, node_data in (
            selected_graph.nodes(
                data=True
            )
        ):

            node_data.setdefault(
                "inferred",
                False,
            )

        workflow_graph = (
            selected_graph.copy()
        )

        # ---------------------------------------------------------
        # Execution order.
        # ---------------------------------------------------------

        execution_order = (
            self._execution_order(
                workflow_graph
            )
        )

        # ---------------------------------------------------------
        # Conditional branches.
        # ---------------------------------------------------------

        conditional_branches = []

        for source, target, data in (
            workflow_graph.edges(
                data=True
            )
        ):

            if data.get(
                "relation"
            ) != PROMPT_CONDITION:
                continue

            conditional_branches.append(
                {
                    "source": source,
                    "target": target,
                    "condition": data.get(
                        "condition",
                        "",
                    ),
                    "branch": data.get(
                        "branch",
                        "then",
                    ),
                    "condition_negated": data.get(
                        "condition_negated",
                        False,
                    ),
                }
            )

        return {
            "graph": workflow_graph,
            "execution_order": execution_order,
            "conditional_branches": (
                conditional_branches
            ),
        }

    # =========================================================
    # Selection ordering
    # =========================================================

    @staticmethod
    def _selection_order_key(
        item,
    ):
        step_index = item.get(
            "step_index"
        )

        if isinstance(
            step_index,
            int,
        ):
            return (
                0,
                step_index,
            )

        return (
            1,
            float("inf"),
        )

    # =========================================================
    # Prompt structure
    # =========================================================

    def _add_prompt_structure(
        self,
        graph,
        selection,
    ):

        by_step = {}

        for item in selection:

            step_index = item.get(
                "step_index"
            )

            node_id = item.get(
                "domain_node_id"
            )

            if not isinstance(
                step_index,
                int,
            ):
                continue

            if not node_id:
                continue

            by_step.setdefault(
                step_index,
                node_id,
            )

        if not by_step:
            return

        ordered_steps = sorted(
            by_step.keys()
        )

        # ---------------------------------------------------------
        # Sequential prompt dependencies.
        # ---------------------------------------------------------

        for previous_step, current_step in zip(
            ordered_steps,
            ordered_steps[1:],
        ):

            source_id = by_step[
                previous_step
            ]

            target_id = by_step[
                current_step
            ]

            if source_id == target_id:
                continue

            existing = (
                graph.get_edge_data(
                    source_id,
                    target_id,
                )
            )

            if existing is not None:
                if existing.get(
                    "relation"
                ) == PROMPT_CONDITION:

                    existing.setdefault(
                        "prompt_dependency",
                        True,
                    )

                continue

            graph.add_edge(
                source_id,
                target_id,
                relation=PROMPT_DEPENDENCY,
                inferred_context=False,
                origin="prompt",
            )

        # ---------------------------------------------------------
        # Conditional structure.
        # ---------------------------------------------------------

        for item in selection:

            target_step = item.get(
                "step_index"
            )

            target_id = item.get(
                "domain_node_id"
            )

            if not isinstance(
                target_step,
                int,
            ):
                continue

            if not target_id:
                continue

            condition = item.get(
                "condition",
                "",
            )

            condition_source_step = item.get(
                "condition_source_step_index"
            )

            if not condition:
                continue

            if not isinstance(
                condition_source_step,
                int,
            ):
                continue

            source_id = by_step.get(
                condition_source_step
            )

            if not source_id:
                continue

            if source_id == target_id:
                continue

            branch = item.get(
                "branch",
                "then",
            )

            condition_negated = item.get(
                "condition_negated",
                False,
            )

            existing = (
                graph.get_edge_data(
                    source_id,
                    target_id,
                )
            )

            if existing is not None:

                if existing.get(
                    "relation"
                ) == PROMPT_CONDITION:

                    existing.update(
                        {
                            "condition": condition,
                            "branch": branch,
                            "condition_negated": (
                                condition_negated
                            ),
                            "prompt_dependency": existing.get(
                                "prompt_dependency",
                                False,
                            ),
                        }
                    )

                else:

                    graph.remove_edge(
                        source_id,
                        target_id,
                    )

                    graph.add_edge(
                        source_id,
                        target_id,
                        relation=PROMPT_CONDITION,
                        condition=condition,
                        branch=branch,
                        condition_negated=(
                            condition_negated
                        ),
                        inferred_context=False,
                        origin="prompt",
                    )

                continue

            graph.add_edge(
                source_id,
                target_id,
                relation=PROMPT_CONDITION,
                condition=condition,
                branch=branch,
                condition_negated=(
                    condition_negated
                ),
                inferred_context=False,
                origin="prompt",
            )

    # =========================================================
    # Contextual relationships
    # =========================================================

    def _add_contextual_relationships(
        self,
        selected_graph,
        selected_ids,
        prompt_subgraph,
        prompt,
    ):
        """
        Infer domain relationships between CONSECUTIVE selected
        operations.

        The selected operation sequence comes from Beam Search.

        For each consecutive pair:

            A -> B

        we first look for a direct domain relationship.

        If none exists, we search the domain graph for a contextual
        path:

            A -> X -> Y -> B

        X and Y are then inferred into the runtime graph together
        with their domain relationships.

        Explicit prompt relationships always take precedence.

        OPERATION_INCLUDES is never treated as execution flow.
        """

        for source_id, target_id in zip(
            selected_ids,
            selected_ids[1:],
        ):

            if source_id == target_id:
                continue

            # -----------------------------------------------------
            # Explicit prompt relationship already exists.
            #
            # Do not overwrite it with a domain relationship.
            # -----------------------------------------------------

            existing = (
                selected_graph.get_edge_data(
                    source_id,
                    target_id,
                )
            )

            if existing is not None:

                if existing.get(
                    "relation"
                ) in {
                    PROMPT_DEPENDENCY,
                    PROMPT_CONDITION,
                }:
                    continue

            # -----------------------------------------------------
            # Direct domain relationship.
            # -----------------------------------------------------

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

                if (
                    relation
                    in INFERRED_OPERATION_RELATIONSHIPS
                    and relation
                    != COMPOSITE_RELATIONSHIP
                ):

                    selected_graph.add_edge(
                        source_id,
                        target_id,
                        **edge_data,
                        inferred_context=True,
                        origin="domain",
                    )

                    continue

            # -----------------------------------------------------
            # No direct relationship.
            #
            # Find a contextual path in the domain graph.
            # -----------------------------------------------------

            path = self._most_contextual_path(
                prompt_subgraph,
                source_id,
                target_id,
                prompt,
            )

            if not path or len(path) < 2:
                continue

            if not self._is_contextual_operation_path(
                prompt_subgraph,
                path,
            ):
                continue

            # -----------------------------------------------------
            # Add every node in the contextual path.
            #
            # This is the important part:
            #
            # selected A -> inferred X -> inferred Y -> selected B
            # -----------------------------------------------------

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
                    inferred=True,
                    source="contextual_domain",
                )

            # -----------------------------------------------------
            # Add the actual domain relationships along the path.
            # -----------------------------------------------------

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

                relation = edge_data.get(
                    "relation"
                )

                if (
                    relation
                    not in INFERRED_OPERATION_RELATIONSHIPS
                ):
                    continue

                if (
                    relation
                    == COMPOSITE_RELATIONSHIP
                ):
                    continue

                # Explicit prompt structure wins.
                if selected_graph.has_edge(
                    source,
                    target,
                ):
                    continue

                selected_graph.add_edge(
                    source,
                    target,
                    **edge_data,
                    inferred_context=True,
                    origin="domain",
                )

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
                expanded.append(
                    item
                )
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

            if not children:
                expanded.append(
                    item
                )
                continue

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
                        "step_index": item.get(
                            "step_index"
                        ),
                        "step_text": item.get(
                            "step_text",
                            "",
                        ),
                        "explicit": item.get(
                            "explicit",
                            False,
                        ),
                        "inferred": True,
                        "source": "composite_expansion",
                        "semantic_score": child_data.get(
                            "semantic_score",
                            item.get(
                                "semantic_score",
                                0.0,
                            ),
                        ),
                        "lexical_score": child_data.get(
                            "lexical_score",
                            item.get(
                                "lexical_score",
                                0.0,
                            ),
                        ),
                        "candidate_score": child_data.get(
                            "score",
                            item.get(
                                "candidate_score",
                                0.0,
                            ),
                        ),
                        "relationship_score": 0.0,
                        "connectivity_score": 0.0,
                        "constraint_penalty": 0.0,
                        "constraint_violations": [],
                        "condition": item.get(
                            "condition",
                            child_data.get(
                                "condition",
                                "",
                            ),
                        ),
                        "branch": item.get(
                            "branch",
                            child_data.get(
                                "branch",
                                "",
                            ),
                        ),
                        "condition_negated": item.get(
                            "condition_negated",
                            child_data.get(
                                "condition_negated",
                                False,
                            ),
                        ),
                        "condition_source_step_index": (
                            item.get(
                                "condition_source_step_index"
                            )
                        ),
                        "composite_parent": node_id,
                    }
                )

        result = []
        seen = {}

        for item in expanded:

            node_id = item.get(
                "domain_node_id"
            )

            if not node_id:
                continue

            if node_id not in seen:

                seen[node_id] = len(
                    result
                )

                result.append(
                    item
                )

                continue

            existing_index = seen[
                node_id
            ]

            existing = result[
                existing_index
            ]

            if (
                item.get(
                    "explicit",
                    False,
                )
                and not existing.get(
                    "explicit",
                    False,
                )
            ):

                result[
                    existing_index
                ] = item

        return result

    # =========================================================
    # Node metadata merging
    # =========================================================

    @staticmethod
    def _merge_node_metadata(
        target,
        source,
    ):

        for key, value in source.items():

            if key not in target:
                target[key] = value
                continue

            if target[key] in (
                None,
                "",
                [],
            ) and value not in (
                None,
                "",
                [],
            ):

                target[key] = value

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
        prompt="",
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
        # Direct inferred relationship.
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

    # =========================================================
    # Consecutive selected path expansion
    # =========================================================

    def _expand_selected_paths(
        self,
        selected_ids,
        domain_graph,
    ):

        final_ids = set(
            selected_ids
        )

        final_edges = []

        for source_id, target_id in zip(
            selected_ids,
            selected_ids[1:],
        ):

            if source_id == target_id:
                continue

            # -------------------------------------------------
            # Direct domain relationship.
            # -------------------------------------------------

            if domain_graph.has_edge(
                source_id,
                target_id,
            ):

                relation = (
                    domain_graph.edges[
                        source_id,
                        target_id,
                    ].get(
                        "relation"
                    )
                )

                if (
                    relation
                    != COMPOSITE_RELATIONSHIP
                ):

                    final_edges.append(
                        (
                            source_id,
                            target_id,
                            relation,
                        )
                    )

                    continue

            # -------------------------------------------------
            # Contextual domain path.
            # -------------------------------------------------

            path = self._most_contextual_path(
                domain_graph,
                source_id,
                target_id,
            )

            if not path:
                continue

            # -------------------------------------------------
            # Add inferred nodes.
            # -------------------------------------------------

            for node_id in path:
                final_ids.add(
                    node_id
                )

            # -------------------------------------------------
            # Add inferred relationships.
            # -------------------------------------------------

            for source, target in zip(
                path,
                path[1:],
            ):

                relation = (
                    domain_graph.edges[
                        source,
                        target,
                    ].get(
                        "relation"
                    )
                )

                if (
                    relation
                    == COMPOSITE_RELATIONSHIP
                ):
                    continue

                if (
                    relation
                    not in INFERRED_OPERATION_RELATIONSHIPS
                ):
                    continue

                final_edges.append(
                    (
                        source,
                        target,
                        relation,
                    )
                )

        return (
            final_ids,
            final_edges,
        )