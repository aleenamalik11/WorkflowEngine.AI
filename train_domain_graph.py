import json
import pickle
import uuid
from collections import defaultdict

import pandas as pd

from utils import EmbeddingService
from utils import default_path_info


###############################################################
# CONFIG
###############################################################

DATASET = "School_Workflow_Dataset_100.xlsx"

# In train_domain_graph.py or your config file
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

OUTPUT = "models/domain_graph.pkl"

###############################################################
# Load embedding model
###############################################################

embedding_service = EmbeddingService(MODEL)

###############################################################
# Load dataset
###############################################################

print("=" * 60)
print("Loading dataset...")
print("=" * 60)

df = pd.read_excel(DATASET)

###############################################################
# Graph
###############################################################

# Define a top-level function instead of an inline lambda


# Updated graph structure
graph = {
    "nodes": {},
    "edges": [],
    "adjacency": defaultdict(list),
    "reverse_adjacency": defaultdict(list),
    "paths": defaultdict(default_path_info),
    "workflow_lookup": {},
}

###############################################################
# Helpers
###############################################################

def get_node_embedding(name):

    return embedding_service.encode(name).tolist()


###############################################################
# Build graph
###############################################################

for _, row in df.iterrows():

    workflow_json = row["WorkflowJson"]

    if isinstance(workflow_json, str):

        workflow = json.loads(workflow_json)

    else:

        workflow = workflow_json

    workflow_name = workflow["Name"]

    graph["workflow_lookup"][workflow_name] = workflow

    ###########################################################
    # Nodes
    ###########################################################

    for node in workflow["Nodes"]:

        name = node["Name"]

        if name not in graph["nodes"]:

            graph["nodes"][name] = {

                "id": str(uuid.uuid4()),

                "name": name,

                "type": node.get("Type", "Custom"),

                "embedding": get_node_embedding(name),

                "count": 1

            }

        else:

            graph["nodes"][name]["count"] += 1

    ###########################################################
    # Connections
    ###########################################################

    id_to_name = {

        n["Id"]: n["Name"]

        for n in workflow["Nodes"]

    }

    ###########################################################
    # Edges
    ###########################################################

    for source_id, transitions in workflow["Connections"].items():

        if source_id not in id_to_name:
            continue

        source = id_to_name[source_id]

        for transition, target_id in transitions.items():

            if target_id in ("Done", "End", "Stop"):
                continue

            if target_id not in id_to_name:
                continue

            target = id_to_name[target_id]

            edge = {

                "source": source,

                "target": target,

                "transition": transition

            }

            graph["edges"].append(edge)

            ###################################################
            # Adjacency
            ###################################################

            graph["adjacency"][source].append(

                {

                    "target": target,

                    "transition": transition

                }

            )

            graph["reverse_adjacency"][target].append(

                {

                    "source": source,

                    "transition": transition

                }

            )

    ###########################################################
    # Extract workflow paths
    ###########################################################

    start = workflow.get("StartNodeId")

    if not start or start not in id_to_name:
        continue

    current = start

    visited = set()

    path = []

    while True:

        if current in visited:
            break

        visited.add(current)

        path.append(id_to_name[current])

        transitions = workflow["Connections"].get(current)

        if not transitions:
            break

        next_node = None

        for _, target in transitions.items():

            if target in ("Done", "End", "Stop"):
                next_node = None
                break

            next_node = target
            break

        if next_node is None:
            break

        current = next_node

    ###########################################################
    # Save path
    ###########################################################

    path = tuple(path)

    graph["paths"][path]["frequency"] += 1

###############################################################
# Compute path embeddings
###############################################################

print("Generating path embeddings...")

for path in graph["paths"]:

    sentence = " -> ".join(path)

    graph["paths"][path]["embedding"] = (

        embedding_service.encode(sentence).tolist()

    )

###############################################################
# Save graph
###############################################################

print("Saving graph...")

with open(OUTPUT, "wb") as f:

    pickle.dump(graph, f)

###############################################################
# Summary
###############################################################

print()

print("=" * 60)
print("Training Complete")
print("=" * 60)

print("Nodes :", len(graph["nodes"]))
print("Edges :", len(graph["edges"]))
print("Paths :", len(graph["paths"]))