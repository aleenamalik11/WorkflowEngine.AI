import re

import numpy as np
import pandas as pd
import networkx as nx

from models import DomainNode

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
# ID Generator
###############################################################

def new_id():

    import uuid

    return str(uuid.uuid4())


def _to_domain_node(
    neo4j_node,
) -> DomainNode:

    props = dict(neo4j_node)

    labels = list(neo4j_node.labels) if hasattr(
        neo4j_node,
        "labels",
    ) else []

    raw_types = (
        props.get(
            "types",
            [],
        )
        or []
    )

    if isinstance(
        raw_types,
        str,
    ):
        raw_types = [
            raw_types
        ]

    types = list(
        dict.fromkeys(
            [
                *raw_types,
                *[
                    label
                    for label in labels
                    if label
                    != "GraphNode"
                ],
            ]
        )
    )

    # Prefer ontology type rather than blindly using the
    # first Neo4j label.

    node_type = (
        types[0]
        if types
        else props.get(
            "type",
            "DomainEntity",
        )
    )

    aliases = (
        props.get(
            "aliases",
            [],
        )
        or []
    )

    if isinstance(
        aliases,
        str,
    ):
        aliases = [
            aliases
        ]

    embedding = props.get(
        "embedding"
    )

    return DomainNode(
        id=str(
            props.get(
                "id"
            )
        ),
        name=str(
            props.get(
                "name",
                props.get(
                    "id",
                    "",
                ),
            )
        ),
        node_type=node_type,
        description=str(
            props.get(
                "description",
                "",
            )
            or ""
        ),
        aliases=list(
            aliases
        ),
        embedding=embedding,
        types=types,
    )

def _domain_node_text(
        node,
):

    parts = [
        node.name or "",
        node.description or "",
        node.id or "",
    ]

    parts.extend(
        node.aliases or []
    )

    parts.extend(
        node.types or []
    )

    return " ".join(
        part
        for part in parts
        if part
    )

def _has_embedding(
        value,
):

    if value is None:
        return False

    try:
        return len(value) > 0
    except (
            TypeError,
            ValueError,
    ):
        return False

def _add_domain_node(graph, node, source=None, score=None):
    """
    Add a domain node to the contextual graph.

    The node keeps its ontology identity and embedding so that later
    semantic matching can operate on the actual domain representation.
    """

    if node is None:
        return

    node_id = node.id

    data = {
        "name": node.name,
        "node_type": node.node_type,
        "description": node.description,
        "aliases": list(node.aliases or []),
        "embedding": node.embedding,
    }

    if score is not None:
        data["semantic_score"] = score

    if source is not None:
        data["source"] = source

    embedding = data.get("embedding")

    # IMPORTANT:
    # Do not use:
    #
    #     if not embedding:
    #
    # because embeddings may be numpy arrays.
    #
    # A numpy array with multiple values cannot be evaluated
    # directly as a boolean.
    if not _has_embedding(embedding):
        data["embedding"] = None

    if graph.has_node(node_id):
        existing = graph.nodes[node_id]

        # Preserve the strongest semantic score.
        if score is not None:
            existing_score = existing.get("semantic_score")

            if (
                existing_score is None
                or score > existing_score
            ):
                existing["semantic_score"] = score

        # Preserve embedding if the existing node does not have one.
        if (
            not _has_embedding(existing.get("embedding"))
            and _has_embedding(data.get("embedding"))
        ):
            existing["embedding"] = data["embedding"]

        # Preserve source information.
        if source is not None:
            sources = existing.setdefault("sources", [])

            if source not in sources:
                sources.append(source)

        return

    data["sources"] = []

    if source is not None:
        data["sources"].append(source)

    graph.add_node(node_id, **data)

    def _build_inferred_selection_item(
            cls,
            operation_id,
            node_data,
            prompt_text,
            source,
    ):
        return {
            "prompt_text": prompt_text,
            "domain_node_id": operation_id,
            "domain_node_name": node_data.get("name", operation_id),
            "domain_node_type": node_data.get("node_type", "Operation"),
            "explicit": False,
            "inferred": True,
            "source": source,
            "semantic_score": 0.0,
            "lexical_score": 0.0,
            "candidate_score": 0.0,
            "relationship_score": 0.0,
            "connectivity_score": 0.0,
            "constraint_penalty": 0.0,
            "constraint_violations": [],
            "condition": "",
        }

def _tokenize(
    text,
):
    if not text:
        return set()

    return set(
        re.findall(
            r"[a-z0-9]+",
            str(text).lower(),
        )
    )
