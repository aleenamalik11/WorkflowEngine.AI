import networkx as nx

from helpers.beam_search_utils import _is_executable_selection_item, INFERRED_OPERATION_RELATIONSHIPS


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

        selected_graph = nx.DiGraph()
        selected_ids = []

        # ---------------------------------------------------------
        # 1. Add beam-selected executable nodes
        # ---------------------------------------------------------

        for item in selection:

            node_id = item[
                "domain_node_id"
            ]

            if not _is_executable_selection_item(item):
                continue

            if node_id not in selected_ids:
                selected_ids.append(node_id)

            if selected_graph.has_node(node_id):
                continue

            selected_graph.add_node(
                node_id,
                **item,
            )

        # ---------------------------------------------------------
        # 2. Get prompt contextual subgraph
        # ---------------------------------------------------------

        prompt_subgraph = None

        if candidate_plan:
            prompt_subgraph = candidate_plan.get(
                "prompt_domain_subgraph"
            )

        if prompt_subgraph is None:
            return selected_graph

        # Use the prompt associated with the selected items.
        prompt = ""

        for item in selection:
            if item.get("prompt_text"):
                prompt = item["prompt_text"]
                break

        # ---------------------------------------------------------
        # 3. Traverse prompt subgraph
        # ---------------------------------------------------------

        for source_id in selected_ids:

            for target_id in selected_ids:

                if source_id == target_id:
                    continue

                # -------------------------------------------------
                # Direct connection
                # -------------------------------------------------

                if prompt_subgraph.has_edge(
                        source_id,
                        target_id,
                ):
                    path = [
                        source_id,
                        target_id,
                    ]

                # -------------------------------------------------
                # Indirect connection
                # -------------------------------------------------

                else:
                    path = self._most_contextual_path(
                        prompt_subgraph,
                        source_id,
                        target_id,
                        prompt,
                    )

                if not path:
                    continue

                # -------------------------------------------------
                # Add intermediate nodes
                # -------------------------------------------------

                for node_id in path:

                    if selected_graph.has_node(
                            node_id
                    ):
                        continue

                    node_data = prompt_subgraph.nodes[
                        node_id
                    ]

                    selected_graph.add_node(
                        node_id,
                        **node_data,
                    )

                # -------------------------------------------------
                # Add edges from the path
                # -------------------------------------------------

                for source, target in zip(
                        path,
                        path[1:],
                ):

                    if selected_graph.has_edge(
                            source,
                            target,
                    ):
                        continue

                    edge_data = (
                        prompt_subgraph
                        .get_edge_data(
                            source,
                            target,
                        )
                    )

                    if edge_data is None:
                        continue

                    selected_graph.add_edge(
                        source,
                        target,
                        **edge_data,
                    )

        workflow_graph = selected_graph.copy()
        for _, node in workflow_graph.nodes(data=True):
            node.setdefault("inferred", False)

        try:
            execution_order = list(nx.topological_sort(workflow_graph))
        except nx.NetworkXUnfeasible:
            cycle = self._describe_cycle(workflow_graph)
            raise RuntimeError(
                "Selected workflow contains a cycle and cannot be ordered: "
                + cycle
            )

        conditional_branches = [
            {"source": source, "target": target, "condition": data.get("condition", "")}
            for source, target, data in workflow_graph.edges(data=True)
            if data.get("relation") == "PROMPT_CONDITION"
        ]

        return {
            "graph": workflow_graph,
            "execution_order": execution_order,
            # A topological order is for serialization; runtime follows the
            # conditional transitions instead of treating it as one path.
            "conditional_branches": conditional_branches,
        }

    def _describe_cycle(graph):
        try:
            cycle_edges = nx.find_cycle(graph)
        except nx.NetworkXNoCycle:
            return "(cycle detected by topological_sort but not reproducible via find_cycle)"

        names = []
        for s, t in cycle_edges:
            edge_data = graph.edges[s, t]
            names.append(
                f"{graph.nodes[s].get('name', s)} --{edge_data.get('relation', '?')}--> "
                f"{graph.nodes[t].get('name', t)}"
            )
        return " ; ".join(names)

    def _most_contextual_path(
            self,
            graph,
            source,
            target,
            prompt,
            max_path_length=4,
    ):
        if source == target:
            return [source]

        if (
                source not in graph
                or target not in graph
        ):
            return None

        # Direct connection is always preferred.
        if graph.has_edge(source, target):
            edge_data = graph.get_edge_data(
                source,
                target,
            )

            relationship = edge_data.get(
                "relationship"
            )

            if (
                    relationship
                    in INFERRED_OPERATION_RELATIONSHIPS
            ):
                return [source, target]

        best_path = None
        best_score = float("-inf")

        try:
            paths = nx.all_simple_paths(
                graph,
                source,
                target,
                cutoff=max_path_length,
            )

            for path in paths:

                # Make sure this is a valid contextual
                # operation path.
                if not self._is_contextual_operation_path(
                        graph,
                        path,
                ):
                    continue

                score = self._contextual_path_score(
                    graph,
                    path,
                    prompt,
                )

                if score > best_score:
                    best_score = score
                    best_path = path

        except nx.NetworkXNoPath:
            return None

        return best_path

    def _contextual_path_score(
            self,
            graph,
            path,
            prompt,
    ):
        if not path:
            return float("-inf")

        # Direct connection gets the highest preference.
        if len(path) == 2:
            return 1.0

        node_scores = []

        for node_id in path:
            node_data = graph.nodes.get(
                node_id,
                {},
            )

            semantic_score = node_data.get(
                "semantic_score",
                0.0,
            )

            lexical_score = node_data.get(
                "lexical_score",
                0.0,
            )

            node_score = (
                    0.7 * semantic_score
                    + 0.3 * lexical_score
            )

            node_scores.append(node_score)

        # Only intermediate nodes contribute to
        # contextual relevance.
        intermediate_scores = node_scores[1:-1]

        if intermediate_scores:
            contextual_score = sum(
                intermediate_scores
            ) / len(intermediate_scores)
        else:
            contextual_score = 0.0

        # Prefer shorter paths, but don't let path length
        # completely dominate semantic relevance.
        path_penalty = 0.1 * (len(path) - 2)

        return (
                contextual_score
                - path_penalty
        )
