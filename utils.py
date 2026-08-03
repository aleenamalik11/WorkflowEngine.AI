import json
import pickle
import uuid
from typing import List

import networkx as nx
import numpy as np

from sentence_transformers import SentenceTransformer

from models import GraphNode, GraphEdge


##############################################################
# JSON
##############################################################

def load_json(path: str):

    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def save_json(data, path: str):

    with open(path, "w", encoding="utf8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )


##############################################################
# Pickle
##############################################################

def save_pickle(obj, path):

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):

    with open(path, "rb") as f:
        return pickle.load(f)


##############################################################
# Embeddings
##############################################################

class EmbeddingService:

    def __init__(
            self,
            model_name="sentence-transformers/all-MiniLM-L6-v2"):

        self.model = SentenceTransformer(model_name)

    def encode(self, text):

        return self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def encode_batch(self, texts):

        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )


##############################################################
# Similarity
##############################################################

def cosine_similarity(a, b):

    a = np.array(a)

    b = np.array(b)

    return np.dot(a, b)


##############################################################
# Workflow JSON
##############################################################

def workflow_to_text(workflow_json):

    if isinstance(workflow_json, str):
        workflow = json.loads(workflow_json)
    else:
        workflow = workflow_json

    lines = []

    lines.append(
        f"Workflow {workflow['Name']}"
    )

    lines.append("Nodes")

    for node in workflow["Nodes"]:

        if isinstance(node, dict):
            lines.append(node["Name"])
        else:
            lines.append(str(node))

    lines.append("Connections")

    for source, transitions in workflow["Connections"].items():

        for key, destination in transitions.items():

            lines.append(
                f"{source} --{key}--> {destination}"
            )

    return "\n".join(lines)


##############################################################
# Graph
##############################################################

def create_graph():

    return nx.MultiDiGraph()


def add_node(
        graph,
        node: GraphNode):

    graph.add_node(

        node.id,

        name=node.name,

        node_type=node.node_type,

        embedding=node.embedding,

        metadata=node.metadata
    )


def add_edge(
        graph,
        edge: GraphEdge):

    graph.add_edge(

        edge.source,

        edge.target,

        relation=edge.relation,

        weight=edge.weight
    )


##############################################################
# Graph Serialization
##############################################################

def save_graph(
        graph,
        path):

    nx.write_gpickle(
        graph,
        path
    )


def load_graph(
        path):

    return nx.read_gpickle(path)


##############################################################
# UUID
##############################################################

def new_id():

    return str(uuid.uuid4())


##############################################################
# Pretty Printing
##############################################################

def print_graph(graph):

    print()

    print("=" * 70)

    print("Nodes")

    print("=" * 70)

    for node in graph.nodes(data=True):

        print(node)

    print()

    print("=" * 70)

    print("Edges")

    print("=" * 70)

    for edge in graph.edges(data=True):

        print(edge)


##############################################################
# Graph Search Helpers
##############################################################

def find_node_by_name(
        graph,
        name):

    name = name.lower()

    for node_id, data in graph.nodes(data=True):

        if data["name"].lower() == name:
            return node_id

    return None


def neighbors(
        graph,
        node_id):

    return list(
        graph.successors(node_id)
    )


##############################################################
# Dataset
##############################################################

def load_dataset(path):

    import pandas as pd

    df = pd.read_excel(path)

    df["WorkflowText"] = df["WorkflowJson"].apply(
        workflow_to_text
    )

    return df


##############################################################
# Registered Functions
##############################################################

def functions_to_text(function):

    text = []

    text.append(function.name)

    text.append(function.description)

    if function.inputs:

        text.append("Inputs")

        text.extend(function.inputs)

    if function.outputs:

        text.append("Outputs")

        text.extend(function.outputs)

    return "\n".join(text)