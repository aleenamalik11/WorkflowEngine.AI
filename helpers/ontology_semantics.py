"""
Relationship semantics.

The domain graph stores ontology relationships only.

Weights are NOT stored in Neo4j.

This module translates a relationship type into semantic meaning that can
be used by the contextual subgraph builder and beam-search planner.

IMPORTANT:

These weights are NOT shortest-path/Dijkstra costs.

They are relevance/order signals used when ranking neighboring concepts.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RelationSemantics:
    edge_type: str
    weight: float
    classification: str
    traverses_for_ordering: bool


RELATIONSHIP_TABLE = {
    # Strong workflow/order relationships
    "OPERATION_REQUIRES":
        RelationSemantics("mandatory", 1.0, "REQUIRED", True),

    "OPERATION_PRECEDES":
        RelationSemantics("mandatory", 1.0, "REQUIRED", True),

    "OPERATION_INCLUDES":
        RelationSemantics("mandatory", 2.0, "REQUIRED", True),

    # Useful contextual relationships
    "OPERATION_CREATES":
        RelationSemantics("alternative", 3.0, "POSSIBLE", True),

    "OPERATION_MODIFIES":
        RelationSemantics("alternative", 3.0, "POSSIBLE", True),

    "OPERATION_VALIDATES":
        RelationSemantics("alternative", 3.0, "POSSIBLE", True),

    "OPERATION_ACCEPTS":
        RelationSemantics("alternative", 4.0, "POSSIBLE", True),

    "OPERATION_PRODUCES":
        RelationSemantics("alternative", 3.0, "POSSIBLE", True),

    # Context-only relationships
    "ACTOR_PERFORMS":
        RelationSemantics("optional", 6.0, "CONTEXT", False),

    "ACTOR_REQUESTS":
        RelationSemantics("optional", 6.0, "CONTEXT", False),

    "ENTITY_OWNS":
        RelationSemantics("optional", 7.0, "CONTEXT", False),

    "ENTITY_LINKED_TO":
        RelationSemantics("optional", 7.0, "CONTEXT", False),

    "EVENT_RELATES_TO":
        RelationSemantics("optional", 7.0, "CONTEXT", False),

    "RULE_CONSTRAINS":
        RelationSemantics("optional", 5.0, "CONTEXT", False),

    "COMPONENT_EXECUTES":
        RelationSemantics("optional", 6.0, "CONTEXT", False),

    "OPERATION_PRODUCES_EVENT":
        RelationSemantics("optional", 6.0, "OPTIONAL", False),

    "EVENT_TRIGGERS":
        RelationSemantics("optional", 5.0, "OPTIONAL", False),
}


DEFAULT_SEMANTICS = RelationSemantics(
    "optional",
    8.0,
    "OPTIONAL",
    False,
)


PROMPT_RELATION_TABLE = {
    "PROMPT_PRECEDES":
        RelationSemantics("mandatory", 0.0, "REQUIRED", True),

    "PROMPT_AND":
        RelationSemantics("alternative", 2.0, "POSSIBLE", True),

    "PROMPT_OR":
        RelationSemantics("alternative", 2.0, "POSSIBLE", False),

    "PROMPT_CONDITION":
        RelationSemantics("optional", 4.0, "CONTEXT", False),

    "PROMPT_UNLESS":
        RelationSemantics("optional", 4.0, "CONTEXT", False),

    "PROMPT_DEPENDENCY":
        RelationSemantics("mandatory", 0.0, "REQUIRED", True),
}


def semantics_for(relation: str) -> RelationSemantics:
    """
    Return semantic information for a domain relationship.
    """
    return RELATIONSHIP_TABLE.get(
        relation,
        DEFAULT_SEMANTICS,
    )


def semantics_for_prompt_relation(
    relation: str,
) -> RelationSemantics:
    """
    Return semantic information for a prompt-derived relationship.
    """
    return PROMPT_RELATION_TABLE.get(
        relation,
        RelationSemantics(
            "optional",
            5.0,
            "OPTIONAL",
            False,
        ),
    )


def apply_relationship_semantics(graph):
    """
    Apply relationship semantics to every graph edge.

    This does NOT perform graph traversal.

    It simply annotates each edge with:
        edge_type
        weight
        classification
        traverses_for_ordering
    """

    for source, target, data in graph.edges(data=True):

        relation = data.get(
            "relation",
            "",
        )

        if relation.startswith("PROMPT_"):
            semantics = semantics_for_prompt_relation(
                relation
            )
        else:
            semantics = semantics_for(
                relation
            )

        data["edge_type"] = semantics.edge_type
        data["weight"] = semantics.weight
        data["classification"] = semantics.classification
        data["traverses_for_ordering"] = (
            semantics.traverses_for_ordering
        )