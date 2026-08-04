import networkx as nx


class GraphPlanner:

    def __init__(self,
                 similarity_threshold=0.75,
                 candidate_threshold=2):

        self.similarity_threshold = similarity_threshold
        self.candidate_threshold = candidate_threshold

    ###############################################################
    # Merge matched graph + inferred concepts
    ###############################################################

    def plan(self,
             matched_graph,
             candidate_graph,
             ranked_candidates):

        ###########################################################
        # Start with matched graph
        ###########################################################

        workflow_graph = nx.DiGraph()

        ###########################################################
        # Copy matched nodes
        ###########################################################

        for node_id, node in matched_graph.nodes(data=True):

            workflow_graph.add_node(

                node_id,

                **node,

                inferred=False

            )

        ###########################################################
        # Copy matched edges
        ###########################################################

        for source, target, edge in matched_graph.edges(data=True):

            workflow_graph.add_edge(

                source,

                target,

                **edge

            )

        ###########################################################
        # Infer missing concepts
        ###########################################################

        for candidate in ranked_candidates:

            if candidate["score"] < self.candidate_threshold:
                continue

            node = candidate["data"]

            workflow_graph.add_node(

                candidate["node"],

                **node,

                inferred=True

            )

        ###########################################################
        # Restore connections
        ###########################################################

        for source, target, edge in candidate_graph.edges(data=True):

            if source not in workflow_graph:
                continue

            if target not in workflow_graph:
                continue

            workflow_graph.add_edge(

                source,

                target,

                **edge

            )

        ###########################################################
        # Remove isolated nodes
        ###########################################################

        isolated = list(

            nx.isolates(

                workflow_graph

            )

        )

        workflow_graph.remove_nodes_from(

            isolated

        )

        ###########################################################
        # Topological Sort
        ###########################################################

        try:

            execution_order = list(

                nx.topological_sort(

                    workflow_graph

                )

            )

        except:

            execution_order = list(

                workflow_graph.nodes()

            )

        return {

            "graph": workflow_graph,

            "execution_order": execution_order

        }

    ###############################################################
    # Pretty Print
    ###############################################################

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

            flag = "(Inferred)" if info["inferred"] else "(Matched)"

            print(

                info["name"],

                flag

            )

        print()

        print("Edges")

        for source, target, edge in graph.edges(data=True):

            print(

                graph.nodes[source]["name"],

                "--",

                edge["relation"],

                "-->",

                graph.nodes[target]["name"]

            )