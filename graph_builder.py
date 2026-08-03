import json

from models import (
    GraphNode,
    GraphEdge
)

from utils import (
    create_graph,
    add_node,
    add_edge,
    save_graph,
    new_id,
    load_dataset
)

###############################################################
# CONFIG
###############################################################

DATASET = "School_Workflow_Dataset_100.xlsx"
GRAPH_FILE = "models/workflow_graph.gpickle"

###############################################################
# LOAD DATASET
###############################################################

print("=" * 60)
print("Loading dataset...")
print("=" * 60)

df = load_dataset(DATASET)

###############################################################
# CREATE GRAPH
###############################################################

graph = create_graph()

# Keeps track of graph nodes already created
graph_nodes = {}

###############################################################
# BUILD GRAPH
###############################################################

for _, row in df.iterrows():

    workflow = json.loads(row["WorkflowJson"])

    workflow_nodes = workflow["Nodes"]

    ###########################################################
    # Workflow lookup (Id -> Node)
    ###########################################################

    workflow_lookup = {
        node["Id"]: node
        for node in workflow_nodes
    }

    ###########################################################
    # Add graph nodes
    ###########################################################

    for node in workflow_nodes:

        name = node["Name"]

        if name not in graph_nodes:

            graph_node = GraphNode(
                id=new_id(),
                name=name,
                node_type=node.get("Type", "Action"),
                metadata={
                    "workflow": workflow["Name"]
                }
            )

            add_node(graph, graph_node)

            graph_nodes[name] = graph_node

    ###########################################################
    # Add graph edges
    ###########################################################

    for source_id, transitions in workflow["Connections"].items():

        source = workflow_lookup.get(source_id)

        if source is None:
            print(f"Source node not found: {source_id}")
            continue

        source_name = source["Name"]

        for transition, destination_id in transitions.items():

            # Terminal state
            if destination_id in ("Done", "End", "Stop"):
                continue

            destination = workflow_lookup.get(destination_id)

            if destination is None:
                print(f"Destination node not found: {destination_id}")
                continue

            destination_name = destination["Name"]

            edge = GraphEdge(
                source=source_name,
                target=destination_name,
                relation=transition
            )

            add_edge(graph, edge)

###############################################################
# SAVE
###############################################################

save_graph(
    graph,
    GRAPH_FILE
)

###############################################################
# PRINT
###############################################################

print()
print("=" * 60)
print("Knowledge Graph")
print("=" * 60)

print("\nNodes")

for node in graph.nodes(data=True):
    print(node)

print("\nEdges")

for edge in graph.edges(data=True):
    print(edge)

print()
print("=" * 60)
print("Finished")
print("=" * 60)