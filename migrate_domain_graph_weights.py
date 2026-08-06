"""Add routing weights to a previously trained domain graph."""

import pickle
from pathlib import Path


EDGE_WEIGHTS = {
    "mandatory": 1,
    "optional": 5,
    "deprecated": 100,
}


def apply_edge_weights(graph):
    """Annotate persisted edges and adjacency records in place."""
    collections = (
        graph.get("edges", []),
        (
            item
            for neighbours in graph.get("adjacency", {}).values()
            for item in neighbours
        ),
        (
            item
            for neighbours in graph.get("reverse_adjacency", {}).values()
            for item in neighbours
        ),
    )

    for records in collections:
        for edge in records:
            if not isinstance(edge, dict):
                continue
            category = edge.get("edge_category", edge.get("transition", "mandatory"))
            category = str(category).strip().lower()
            if category not in EDGE_WEIGHTS:
                category = "mandatory"
            edge["edge_category"] = category
            edge["weight"] = EDGE_WEIGHTS[category]


if __name__ == "__main__":
    graph_path = Path("models/domain_graph.pkl")
    with graph_path.open("rb") as graph_file:
        domain_graph = pickle.load(graph_file)

    apply_edge_weights(domain_graph)

    with graph_path.open("wb") as graph_file:
        pickle.dump(domain_graph, graph_file)

    print(f"Added edge weights to {graph_path}")
