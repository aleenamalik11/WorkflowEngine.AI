"""Add routing weights to a previously trained domain graph."""

import pickle
from pathlib import Path


EDGE_WEIGHTS = {
    "mandatory": 1,
    "alternative": 3,
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
            edge_type = edge.get(
                "edge_type", edge.get("edge_category", edge.get("transition", "mandatory"))
            )
            edge_type = str(edge_type).strip().lower()
            if edge_type not in EDGE_WEIGHTS:
                edge_type = "mandatory"
            edge["relation"] = edge.get("relation", edge.get("transition", "success"))
            edge["edge_type"] = edge_type
            edge.setdefault("weight", EDGE_WEIGHTS[edge_type])
            edge.setdefault("confidence", 1.0)


if __name__ == "__main__":
    graph_path = Path("models/domain_graph.pkl")
    with graph_path.open("rb") as graph_file:
        domain_graph = pickle.load(graph_file)

    apply_edge_weights(domain_graph)

    with graph_path.open("wb") as graph_file:
        pickle.dump(domain_graph, graph_file)

    print(f"Added edge weights to {graph_path}")
