###############################################################
# Ontology semantics
###############################################################

# Nodes that can become workflow steps.  Everything else (actors, components,
# entities, events, rules) is *context*: it explains how operations connect,
# but it is never executed.
EXECUTABLE_TYPES = {"Operation"}

CONSTRAINT_TYPES = {"Rule"}

# Lower weights are preferred by weighted shortest-path routing: they express
# how *required* / relevant a relationship is for reaching the next operation.
RELATIONSHIP_SEMANTICS = {
    "OPERATION_REQUIRES": ("mandatory", 1),
    "OPERATION_PRECEDES": ("mandatory", 1),
    "OPERATION_INCLUDES": ("mandatory", 2),
    "EVENT_TRIGGERS": ("mandatory", 2),
    "OPERATION_PRODUCES_EVENT": ("mandatory", 2),
    "OPERATION_CREATES": ("alternative", 3),
    "OPERATION_MODIFIES": ("alternative", 3),
    "OPERATION_VALIDATES": ("alternative", 3),
    "OPERATION_PRODUCES": ("alternative", 3),
    "OPERATION_ACCEPTS": ("alternative", 4),
    "COMPONENT_EXECUTES": ("alternative", 4),
    "ACTOR_PERFORMS": ("alternative", 4),
    "ACTOR_REQUESTS": ("alternative", 4),
    "RULE_CONSTRAINS": ("optional", 5),
    "ENTITY_OWNS": ("optional", 6),
    "ENTITY_LINKED_TO": ("optional", 6),
    "EVENT_RELATES_TO": ("optional", 6),
}

DEFAULT_RELATIONSHIP_SEMANTICS = ("optional", 5)

# Relationships that only describe *who* or *what* is involved.  They are still
# traversable, but a workflow must not be built out of them alone.
CONTEXT_RELATIONSHIPS = {
    "ACTOR_PERFORMS",
    "ACTOR_REQUESTS",
    "COMPONENT_EXECUTES",
    "ENTITY_OWNS",
    "ENTITY_LINKED_TO",
    "EVENT_RELATES_TO",
    "RULE_CONSTRAINS",
}
