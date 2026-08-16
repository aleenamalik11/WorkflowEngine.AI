import networkx as nx


class PlanningError(Exception):
    """Raised when the selected workflow subgraph cannot be safely ordered."""


class GraphPlanner:
    """
    Stage 9. Topological sort now happens strictly AFTER beam search has
    already committed to a final DAG (domain relationships + prompt
    constraints + planner scoring). This class's only job is to convert
    that DAG into an execution order -- it is not responsible for
    discovering order, and it must never hide a planning failure.

    Root cause fixed: the old implementation caught
    nx.NetworkXUnfeasible and fell back to
        execution_order = list(workflow_graph.nodes())
    which silently turns a cycle (a genuine upstream planning bug --
    contradictory OPERATION_PRECEDES / PROMPT_PRECEDES edges) into an
    arbitrary, unordered node list. That looks like a working plan and
    ships broken workflow JSON. A cycle must be a visible error.
    """

    def plan(self, workflow_graph):
        workflow_graph = workflow_graph.copy()
        for _, node in workflow_graph.nodes(data=True):
            node.setdefault("inferred", False)

        try:
            execution_order = list(nx.topological_sort(workflow_graph))
        except nx.NetworkXUnfeasible:
            cycle = self._describe_cycle(workflow_graph)
            raise PlanningError(
                "Selected workflow contains a cycle and cannot be ordered: "
                + cycle
            )

        return {
            "graph": workflow_graph,
            "execution_order": execution_order,
        }

    @staticmethod
    def _describe_cycle(graph):
        try:
            cycle_edges = nx.find_cycle(graph)
        except nx.NetworkXNoCycle:
            return "(cycle detected by topological_sort but not reproducible via find_cycle)"
        # nx.find_cycle() on a DiGraph returns 2-tuples (u, v), not
        # (u, v, data) -- edge data has to be looked up separately. The
        # previous version unpacked these as 3-tuples, which raised
        # "ValueError: not enough values to unpack (expected 3, got 2)"
        # for every prompt whose workflow graph actually contained a
        # cycle, masking the real PlanningError.
        names = []
        for s, t in cycle_edges:
            edge_data = graph.edges[s, t]
            names.append(
                f"{graph.nodes[s].get('name', s)} --{edge_data.get('relation', '?')}--> "
                f"{graph.nodes[t].get('name', t)}"
            )
        return " ; ".join(names)

    @staticmethod
    def print_plan(plan):
        graph = plan["graph"]
        print()
        print("=" * 60)
        print("Execution Plan")
        print("=" * 60)
        print()
        print("Execution Order")
        for node in plan["execution_order"]:
            info = graph.nodes[node]
            flag = "(Inferred)" if info.get("inferred") else "(Matched)"
            print(info.get("name", node), flag)
        print()
        print("Edges")
        for source, target, edge in graph.edges(data=True):
            print(
                graph.nodes[source].get("name", source), "--",
                edge.get("relation", "?"), "-->",
                graph.nodes[target].get("name", target),
                f"[{edge.get('origin', 'domain')}]",
            )