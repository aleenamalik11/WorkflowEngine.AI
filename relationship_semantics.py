"""
Stage 6 + Stage 7.

Ontology relationships never carry weights in Neo4j (per architectural
constraint). This module is the single place that maps a relationship
*type* to:
  - a routing edge_type/weight pair (Stage 7, consumed by beam search)
  - a semantic classification used before routing even starts (Stage 6):
    REQUIRED / POSSIBLE / CONTEXT / OPTIONAL

Nothing here is learned or LLM-assigned; it is a static table, same spirit
as the old EDGE_WEIGHTS dict in migrate_domain_graph_weights.py, just
keyed by the real ontology relation names instead of a generic
"transition" string.
"""

from dataclasses import dataclass


@dataclass
class RelationSemantics:
    edge_type: str          # mandatory | alternative | optional
    weight: float
    classification: str      # REQUIRED | POSSIBLE | CONTEXT | OPTIONAL
    traverses_for_ordering: bool  # does this relation contribute to execution order?


RELATIONSHIP_TABLE = {
    "OPERATION_REQUIRES":       RelationSemantics("mandatory",   1, "REQUIRED", True),
    "OPERATION_PRECEDES":       RelationSemantics("mandatory",   1, "REQUIRED", True),
    "OPERATION_INCLUDES":       RelationSemantics("mandatory",   2, "REQUIRED", True),
    "OPERATION_CREATES":        RelationSemantics("alternative", 3, "POSSIBLE", True),
    "OPERATION_MODIFIES":       RelationSemantics("alternative", 3, "POSSIBLE", True),
    "OPERATION_VALIDATES":      RelationSemantics("alternative", 3, "POSSIBLE", True),
    "OPERATION_ACCEPTS":        RelationSemantics("alternative", 4, "POSSIBLE", True),
    "ACTOR_PERFORMS":           RelationSemantics("alternative", 4, "CONTEXT",  False),
    "ENTITY_OWNS":              RelationSemantics("optional",    6, "CONTEXT",  False),

    # relations mentioned in the ontology but not given explicit weights in
    # the spec -- given conservative defaults so they can't silently pull
    # unrelated nodes into a "required" path.
    "COMPONENT_EXECUTES":       RelationSemantics("optional",    5, "CONTEXT",  False),
    "ACTOR_REQUESTS":           RelationSemantics("optional",    5, "CONTEXT",  False),
    "OPERATION_PRODUCES":       RelationSemantics("alternative", 4, "POSSIBLE", True),
    "OPERATION_PRODUCES_EVENT": RelationSemantics("optional",    5, "OPTIONAL", False),
    "ENTITY_LINKED_TO":         RelationSemantics("optional",    6, "CONTEXT",  False),
    "EVENT_TRIGGERS":           RelationSemantics("optional",    5, "OPTIONAL", False),
    "EVENT_RELATES_TO":         RelationSemantics("optional",    6, "CONTEXT",  False),
    "RULE_CONSTRAINS":          RelationSemantics("optional",    5, "CONTEXT",  False),
}

DEFAULT_SEMANTICS = RelationSemantics("optional", 8, "OPTIONAL", False)


def semantics_for(relation: str) -> RelationSemantics:
    return RELATIONSHIP_TABLE.get(relation, DEFAULT_SEMANTICS)


# Prompt-derived constraints (Stage 5) get their own, separate weighting so
# they can never be confused with permanent ontology edges downstream.
PROMPT_RELATION_TABLE = {
    "PROMPT_PRECEDES": RelationSemantics("mandatory", 1, "REQUIRED", True),
    "PROMPT_AND":       RelationSemantics("alternative", 2, "POSSIBLE", True),
    "PROMPT_OR":        RelationSemantics("alternative", 2, "POSSIBLE", False),
    "PROMPT_CONDITION": RelationSemantics("optional", 4, "CONTEXT", False),
    "PROMPT_UNLESS":    RelationSemantics("optional", 4, "CONTEXT", False),
}


def semantics_for_prompt_relation(relation: str) -> RelationSemantics:
    return PROMPT_RELATION_TABLE.get(relation, RelationSemantics("optional", 5, "OPTIONAL", False))