import os
import json
import pickle

import numpy as np
import pandas as pd
import networkx as nx

from sentence_transformers import SentenceTransformer


###############################################################
# Embedding Service
###############################################################

class EmbeddingService:

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):

        self.model = SentenceTransformer(model_name)


    def encode(self, text):

        if not text:
            text = ""

        return self.model.encode(
            text,
            normalize_embeddings=True
        )



###############################################################
# Similarity
###############################################################

def cosine_similarity(a, b):

    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )



###############################################################
# Function text conversion
###############################################################

def functions_to_text(function):

    text = function.name

    text += "\n"

    text += function.description


    if function.inputs:

        text += "\nInputs: "

        text += ", ".join(
            function.inputs
        )


    if function.outputs:

        text += "\nOutputs: "

        text += ", ".join(
            function.outputs
        )


    return text



###############################################################
# Dataset Loading
###############################################################

def load_dataset(path):

    if not os.path.exists(path):
        raise FileNotFoundError(path)


    if path.endswith(".xlsx"):

        return pd.read_excel(path)


    if path.endswith(".csv"):

        return pd.read_csv(path)


    raise Exception(
        "Unsupported dataset format"
    )



###############################################################
# Graph Creation
###############################################################

def create_graph():

    return nx.DiGraph()



def add_node(graph, node):

    graph.add_node(
        node.id,
        name=node.name,
        type=node.node_type,
        metadata=node.metadata
    )



def add_edge(graph, edge):

    graph.add_edge(

        edge.source,

        edge.target,

        relation=edge.relation,

        metadata=edge.metadata

    )



###############################################################
# Graph Serialization
###############################################################

def save_graph(graph, path):

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )


    with open(path, "wb") as f:

        pickle.dump(
            graph,
            f
        )



def load_graph(path):

    with open(path, "rb") as f:

        return pickle.load(f)



###############################################################
# Metadata Serialization
###############################################################

def save_metadata(data, path):

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )


    with open(path, "wb") as f:

        pickle.dump(
            data,
            f
        )



def load_metadata(path):

    with open(path, "rb") as f:

        return pickle.load(f)



###############################################################
# FAISS helpers
###############################################################

def save_json(data, path):

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )


    with open(
        path,
        "w",
        encoding="utf8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )

def default_path_info():
    return {"frequency": 0, "embedding": None}

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf8"
    ) as f:

        return json.load(f)



###############################################################
# ID Generator
###############################################################

def new_id():

    import uuid

    return str(uuid.uuid4())