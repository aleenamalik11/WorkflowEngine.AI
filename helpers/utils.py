import numpy as np
import pandas as pd
import networkx as nx

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
