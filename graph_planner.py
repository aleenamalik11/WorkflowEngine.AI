import networkx as nx


class GraphPlanner:

    ###############################################################
    # Order the complete workflow graph discovered from domain paths
    ###############################################################

    def plan(self, workflow_graph):

        ###########################################################
        # Preserve matched and inferred path nodes
        ###########################################################

        workflow_graph = workflow_graph.copy()
        for _, node in workflow_graph.nodes(data=True):
            node.setdefault("inferred", False)

        ###########################################################
        # Topological Sort
        ###########################################################

        try:

            execution_order = list(

                nx.topological_sort(

                    workflow_graph

                )

            )

        except nx.NetworkXUnfeasible:

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
