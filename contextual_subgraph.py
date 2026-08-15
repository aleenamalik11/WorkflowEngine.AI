"""Build the contextual subgraph handed to beam search.

The matcher only proposes *nodes*.  A node on its own says nothing about the
domain: the meaning lives in the relationships around it.  This module walks
those relationships, keeps the part of the domain graph that is relevant to the
prompt, detects unsupported actions early, and reconnects concepts that ended
up in separate components.

Search strategy for reconnection:

* **BFS**   when every relationship costs the same - the only useful notion of
  "closest" left is the hop count.
* **Dijkstra** when relationship weights differ, because then the weights
  encode relevance / requiredness (``OPERATION_REQUIRES`` is cheaper than
  ``ENTITY_LINKED_TO``).
"""

import networkx as nx


class ContextualSubgraphBuilder:
    """Turn candidate matches into a connected, prompt-aligned subgraph."""

    ###############################################################
    # Configuration
    ###############################################################

    DEFAULT_RADIUS = 1

    # Longer detours stop describing the prompt and start inventing workflows.
    DEFAULT_MAX_PATH_LENGTH = 4

    REJECTED_EDGE_TYPES = {"deprecated"}

    def __init__(self,
                 radius=DEFAULT_RADIUS,
                 max_path_length=DEFAULT_MAX_PATH_LENGTH,
                 require_executable=True):

        self.radius = max(0, int(radius))
        self.max_path_length = max(1, int(max_path_length))
        self.require_executable = require_executable

    ###############################################################
    # Entry point
    ###############################################################

    def build(self, candidate_plan):
        """Return ``candidate_plan`` with a contextual subgraph attached."""
        domain_graph = candidate_plan.get("domain_graph") or nx.DiGraph()
        actions = list(candidate_plan.get("actions", []))

        ###########################################################
        # 1. Validate the matched concepts against the ontology
        ###########################################################

        actions, invalid_actions = self._validate_actions(
            domain_graph, actions
        )

        seeds = [
            candidate["domain_node_id"]
            for action in actions
            for candidate in action["candidates"]
            if candidate["domain_node_id"] in domain_graph
        ]

        ###########################################################
        # 2. Semantic context around every matched concept
        ###########################################################

        subgraph = self._expand_context(domain_graph, seeds)

        ###########################################################
        # 3. Reconnect concepts that fell into separate components
        ###########################################################

        strategy = self._strategy(domain_graph)

        connections, disconnected = self._connect(
            domain_graph,
            subgraph,
            actions,
            strategy,
        )

        subgraph.graph["seed_nodes"] = sorted(set(seeds))
        subgraph.graph["search_strategy"] = strategy
        subgraph.graph["connections"] = connections
        subgraph.graph["disconnected_pairs"] = disconnected
        subgraph.graph["invalid_actions"] = invalid_actions

        contextual_plan = dict(candidate_plan)
        contextual_plan["actions"] = actions
        contextual_plan["domain_graph"] = subgraph
        contextual_plan["invalid_actions"] = invalid_actions
        contextual_plan["search_strategy"] = strategy

        return contextual_plan

    ###############################################################
    # 1. Early detection of invalid / unsupported actions
    ###############################################################

    def _validate_actions(self, domain_graph, actions):
        """Drop candidates the domain graph cannot execute.

        A prompt step is only supported when it resolves to an ``Operation``.
        Matching an actor, an entity or a business rule means the prompt asked
        for something the domain does not offer as an action.
        """
        validated = []
        invalid = []

        for action in actions:
            supported = []
            unsupported = []

            for candidate in action.get("candidates", []):
                node_id = candidate["domain_node_id"]
                node = (
                    dict(domain_graph.nodes[node_id])
                    if node_id in domain_graph else {}
                )

                if self._is_executable(node):
                    supported.append(candidate)
                else:
                    unsupported.append({
                        **candidate,
                        "reason": "not an executable domain operation",
                        "domain_node_types": node.get("types", []),
                    })

            action = dict(action)

            if supported or not unsupported:
                action["candidates"] = supported
                action["unsupported_candidates"] = unsupported
            else:
                # Every candidate is context-only: the prompt asked for an
                # action the domain does not support.
                action["candidates"] = []
                action["unsupported_candidates"] = unsupported
                invalid.append({
                    "prompt_node_id": action.get("prompt_node_id"),
                    "prompt_text": action.get("prompt_text"),
                    "reason": "no executable domain operation matches this step",
                    "rejected": unsupported,
                })

            validated.append(action)

        return validated, invalid

    def _is_executable(self, node):
        if not self.require_executable:
            return True

        if "executable" in node:
            return bool(node["executable"])

        # Graphs without ontology metadata (e.g. the legacy trained graph)
        # cannot classify their nodes, so nothing is rejected.
        return True

    ###############################################################
    # 2. Relationship expansion
    ###############################################################

    def _expand_context(self, domain_graph, seeds):
        """Collect matched nodes plus their relationship neighbourhood."""
        selected = set()

        for seed in seeds:
            if seed not in domain_graph:
                continue

            frontier = {seed}
            selected.add(seed)

            for _ in range(self.radius):
                neighbours = set()
                for node_id in frontier:
                    neighbours.update(domain_graph.successors(node_id))
                    neighbours.update(domain_graph.predecessors(node_id))
                neighbours -= selected
                selected.update(neighbours)
                frontier = neighbours

        subgraph = nx.DiGraph()

        for node_id in selected:
            subgraph.add_node(node_id, **dict(domain_graph.nodes[node_id]))

        for source, target, edge in domain_graph.edges(data=True):
            if source not in selected or target not in selected:
                continue

            # Deprecated relationships are not part of the supported domain.
            if str(edge.get("edge_type", "")).lower() in self.REJECTED_EDGE_TYPES:
                continue

            subgraph.add_edge(source, target, **dict(edge))

        return subgraph

    ###############################################################
    # 3. Connectivity
    ###############################################################

    @staticmethod
    def _strategy(domain_graph):
        """BFS for equal-cost relationships, Dijkstra for weighted ones."""
        weights = {
            float(edge.get("weight", 1))
            for _, _, edge in domain_graph.edges(data=True)
        }
        return "bfs" if len(weights) <= 1 else "dijkstra"

    def _shortest_path(self, domain_graph, source, target, strategy):
        try:
            if strategy == "bfs":
                return nx.shortest_path(
                    domain_graph, source=source, target=target
                )

            return nx.dijkstra_path(
                domain_graph,
                source=source,
                target=target,
                weight="weight",
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _connect(self, domain_graph, subgraph, actions, strategy):
        """Link consecutive prompt steps that the context did not connect."""
        connections = []
        disconnected = []

        ordered = [
            action for action in actions if action.get("candidates")
        ]

        for previous, current in zip(ordered, ordered[1:]):
            pairs = [
                (source["domain_node_id"], target["domain_node_id"])
                for source in previous["candidates"]
                for target in current["candidates"]
                if source["domain_node_id"] != target["domain_node_id"]
            ]

            for source, target in pairs:
                if self._shortest_path(subgraph, source, target, strategy):
                    # Already connected through the semantic context.
                    continue

                path = self._shortest_path(
                    domain_graph, source, target, strategy
                )

                rejection = self._rejection_reason(domain_graph, path)

                if rejection:
                    disconnected.append({
                        "source": source,
                        "target": target,
                        "reason": rejection,
                    })
                    continue

                self._merge_path(domain_graph, subgraph, path)

                connections.append({
                    "source": source,
                    "target": target,
                    "strategy": strategy,
                    "path": path,
                    "cost": self._path_cost(domain_graph, path),
                })

        return connections, disconnected

    def _rejection_reason(self, domain_graph, path):
        """Verify that a connecting path still aligns with the prompt."""
        if not path:
            return "no domain path between the matched concepts"

        if len(path) - 1 > self.max_path_length:
            return "domain path is too long to still describe the prompt"

        for source, target in zip(path, path[1:]):
            edge = domain_graph.edges[source, target]
            if str(edge.get("edge_type", "")).lower() in self.REJECTED_EDGE_TYPES:
                return "domain path relies on a deprecated relationship"

        return None

    @staticmethod
    def _path_cost(domain_graph, path):
        return sum(
            float(domain_graph.edges[source, target].get("weight", 1))
            for source, target in zip(path, path[1:])
        )

    @staticmethod
    def _merge_path(domain_graph, subgraph, path):
        for node_id in path:
            if node_id not in subgraph:
                subgraph.add_node(node_id, **dict(domain_graph.nodes[node_id]))

        for source, target in zip(path, path[1:]):
            subgraph.add_edge(
                source,
                target,
                **dict(domain_graph.edges[source, target]),
            )

    ###############################################################
    # Pretty Print
    ###############################################################

    @staticmethod
    def print_subgraph(contextual_plan):
        subgraph = contextual_plan.get("domain_graph") or nx.DiGraph()

        print()
        print("=" * 60)
        print("Contextual Subgraph")
        print("=" * 60)
        print()

        print(
            f"Nodes: {subgraph.number_of_nodes()}  "
            f"Edges: {subgraph.number_of_edges()}  "
            f"Search: {subgraph.graph.get('search_strategy', 'n/a')}"
        )

        invalid = contextual_plan.get("invalid_actions", [])
        if invalid:
            print()
            print("Unsupported actions")
            for item in invalid:
                print(f"  {item['prompt_text']} -> {item['reason']}")

        connections = subgraph.graph.get("connections", [])
        if connections:
            print()
            print("Connected paths")
            for item in connections:
                names = " -> ".join(
                    subgraph.nodes[node_id].get("name", node_id)
                    for node_id in item["path"]
                )
                print(f"  [{item['strategy']}] {names} (cost={item['cost']:.1f})")

        for item in subgraph.graph.get("disconnected_pairs", []):
            print(f"  disconnected {item['source']} -> {item['target']}: {item['reason']}")
