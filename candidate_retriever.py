import networkx as nx
from collections import deque


class CandidateRetriever:

    def __init__(self, domain_graph):
        self.domain_graph = domain_graph

    ############################################################
    # Helpers for dual storage formats (NetworkX DiGraph or Dict)
    ############################################################

    def _get_successors(self, node):
        """Returns outgoing neighbors/successors for a given node."""
        if hasattr(self.domain_graph, "successors"):
            return list(self.domain_graph.successors(node))
        elif isinstance(self.domain_graph, dict):
            adjacency = self.domain_graph.get("adjacency", {})
            return adjacency.get(node, [])
        return []

    def _get_predecessors(self, node):
        """Returns incoming neighbors/predecessors for a given node."""
        if hasattr(self.domain_graph, "predecessors"):
            return list(self.domain_graph.predecessors(node))
        elif isinstance(self.domain_graph, dict):
            reverse_adjacency = self.domain_graph.get("reverse_adjacency", {})
            return reverse_adjacency.get(node, [])
        return []

    def _get_node_data(self, node):
        """Retrieves node attributes dictionary safely."""
        if hasattr(self.domain_graph, "nodes"):
            return self.domain_graph.nodes[node]
        elif isinstance(self.domain_graph, dict):
            nodes = self.domain_graph.get("nodes", {})
            if isinstance(nodes, dict):
                return nodes.get(node, {})
            elif isinstance(nodes, list):
                for item in nodes:
                    if isinstance(item, dict) and item.get("id") == node:
                        return item
        return {}

    def _get_edges_for_nodes(self, nodes_set):
        """Retrieves directed edges between a set of candidate nodes safely."""
        edges = []

        if hasattr(self.domain_graph, "subgraph"):
            subgraph = self.domain_graph.subgraph(nodes_set)
            return list(subgraph.edges(data=True))

        elif isinstance(self.domain_graph, dict):
            raw_edges = self.domain_graph.get("edges", [])

            for e in raw_edges:
                u, v, data = None, None, {}

                # Format 1: e is a dict (e.g. {"source": "A", "target": "B"})
                if isinstance(e, dict):
                    u = e.get("source") or e.get("from") or e.get("u")
                    v = e.get("target") or e.get("to") or e.get("v")
                    data = e

                # Format 2: e is a tuple or list (e.g. ("A", "B") or ["A", "B", {...}])
                elif isinstance(e, (tuple, list)) and len(e) >= 2:
                    u, v = e[0], e[1]
                    if len(e) > 2 and isinstance(e[2], dict):
                        data = e[2]

                if u in nodes_set and v in nodes_set:
                    edges.append((u, v, data))

            # Fallback to adjacency mapping if no edges matched from edge list
            if not edges:
                adjacency = self.domain_graph.get("adjacency", {})
                for u in nodes_set:
                    for v in adjacency.get(u, []):
                        if v in nodes_set:
                            edges.append((u, v, {}))

        return edges

    ############################################################
    # BFS Expansion
    ############################################################

    def _expand(self, start_node, max_depth):
        visited = set()
        queue = deque()
        queue.append((start_node, 0))
        results = []

        while queue:
            node, depth = queue.popleft()

            if node in visited:
                continue

            visited.add(node)
            results.append(node)

            if depth >= max_depth:
                continue

            # Successors
            for neighbour in self._get_successors(node):
                queue.append((neighbour, depth + 1))

            # Predecessors
            for neighbour in self._get_predecessors(node):
                queue.append((neighbour, depth + 1))

        return results

    ############################################################
    # Retrieve candidates
    ############################################################

    def retrieve(self, matched_graph, depth=2):
        candidate_graph = nx.DiGraph()

        # Expand every matched node
        for node_id in matched_graph.nodes():
            nodes = self._expand(node_id, depth)

            # Copy Nodes
            for n in nodes:
                node_data = self._get_node_data(n)
                candidate_graph.add_node(n, **node_data)

            # Copy Edges
            subgraph_edges = self._get_edges_for_nodes(set(nodes))
            for source, target, edge_data in subgraph_edges:
                candidate_graph.add_edge(source, target, **edge_data)

        return candidate_graph

    ############################################################
    # Score Missing Concepts
    ############################################################

    def rank_missing_nodes(self, matched_graph, candidate_graph):
        scores = []
        matched = set(matched_graph.nodes())

        for node in candidate_graph.nodes():
            if node in matched:
                continue

            score = 0

            # Number of matched outgoing neighbors
            for neighbour in candidate_graph.neighbors(node):
                if neighbour in matched:
                    score += 1

            # Number of matched incoming neighbors
            for neighbour in candidate_graph.predecessors(node):
                if neighbour in matched:
                    score += 1

            scores.append({
                "node": node,
                "score": score,
                "data": candidate_graph.nodes[node]
            })

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores

    ############################################################
    # Print
    ############################################################

    @staticmethod
    def print_candidates(scores):
        print()
        print("=" * 60)
        print("Candidate Concepts")
        print("=" * 60)

        for item in scores:
            name = item["data"].get("name", item["node"])
            print(f"{name}   score={item['score']}")