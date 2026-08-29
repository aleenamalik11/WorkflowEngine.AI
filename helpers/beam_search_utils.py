# Relationships that can participate in actual execution ordering.
#
# IMPORTANT:
#
# OPERATION_INCLUDES is intentionally not used to order two
# prompt-selected candidates. During expansion, however, it exposes
# inferred child operations whose domain edges are retained.
ORDERING_RELATIONSHIPS = {
    "OPERATION_PRECEDES",
    "PROMPT_DEPENDENCY",
    "PROMPT_PRECEDES",
}

# Relationships that identify executable neighbors during domain
# expansion. OPERATION_INCLUDES is the common parent-to-child
# workflow structure.
STRUCTURAL_RELATIONSHIPS = {
    "OPERATION_INCLUDES",
}

# Domain relationships that may expose another executable operation
# required to carry out a selected operation.  These relationships are
# copied into the execution graph so GraphPlanner can order the inferred
# steps from the ontology.
INFERRED_OPERATION_RELATIONSHIPS = {
    "OPERATION_INCLUDES",
    "OPERATION_REQUIRES",
    "OPERATION_PRECEDES",
    "OPERATION_VALIDATES",
    "OPERATION_PRODUCES",
    "OPERATION_MODIFIES",
    "OPERATION_ACCEPTS",
}

# Relationships that describe domain context.
#
# They are useful when evaluating whether an operation is
# semantically compatible with a prompt, but their endpoint
# entities are NOT workflow execution steps.
CONTEXT_RELATIONSHIPS = {
    "ENTITY_OWNS",
    "OPERATION_REQUIRES",
    "OPERATION_ACCEPTS",
    "OPERATION_CREATES",
    "OPERATION_MODIFIES",
    "OPERATION_VALIDATES",
    "OPERATION_PRODUCES",
    "OPERATION_PRODUCES_EVENT",
    "EVENT_RELATES_TO",
    "ENTITY_LINKED_TO",
}

# Rules are deliberately separate from ordinary context.
#
# A Rule is not executable, but RULE_CONSTRAINS is important
# because it can invalidate a candidate operation.
CONSTRAINT_RELATIONSHIPS = {
    "RULE_CONSTRAINS",
}

# Only these node types are allowed to become workflow steps.
#
# The ontology defines Operation as the executable semantic unit.
EXECUTABLE_NODE_TYPES = {
    "Operation",
    "operation",
}

# Nodes of these types are explicitly contextual/non-executable.
NON_EXECUTABLE_NODE_TYPES = {
    "DomainEntity",
    "Entity",
    "Actor",
    "Component",
    "Event",
    "Rule",
    "domainentity",
    "entity",
    "actor",
    "component",
    "event",
    "rule",
}

# ============================================================
# Beam scoring weights
# ============================================================

CONNECTIVITY_BONUS = 0.20

REQUIRED_RELATION_BONUS = 0.25

POSSIBLE_RELATION_BONUS = 0.08

CONTEXT_RELATION_BONUS = 0.03

# Structural relationships such as OPERATION_INCLUDES are useful
# evidence that two operations belong to the same domain process,
# but they are weaker than explicit execution-order relationships.
STRUCTURAL_RELATION_BONUS = 0.12

# Strong penalty for a candidate that conflicts with a domain rule.
RULE_CONTRADICTION_PENALTY = 0.75

# Penalty for choosing a candidate that has no relationship to
# anything already selected.
DISCONNECTED_PENALTY = 0.15

# Small penalty for inferred/neighborhood candidates.
INFERRED_PENALTY = 0.03

# Explicit direct matches receive a small preference.
EXPLICIT_DIRECT_BONUS = 0.15

# Prefer a direct operation whose name starts with the requested action.
# This prevents a related neighborhood operation from winning on a small
# embedding-score difference when the user explicitly names the action.
DIRECT_NAME_ALIGNMENT_BONUS = 0.12


def _is_executable_selection_item(
        cls,
        item,
):
    node_type = item.get(
        "domain_node_type"
    )

    return cls._is_executable_node_type(
        node_type
    )


def _is_executable_node_type(
        cls,
        node_type,
):
    return node_type in (
        cls.EXECUTABLE_NODE_TYPES
    )

def _is_contextual_operation_path(cls, graph, path):
    for node_id in path:
        if not cls._is_executable_node_type(
                graph.nodes[node_id].get("node_type")
        ):
            return False

    for source, target in zip(path, path[1:]):
        if graph.edges[source, target].get(
                "relation"
        ) not in cls.INFERRED_OPERATION_RELATIONSHIPS:
            return False

    return True