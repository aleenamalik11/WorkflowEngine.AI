# Relationships that can participate in actual execution ordering.
#
# IMPORTANT:
#
# OPERATION_INCLUDES is intentionally not used to order two
# prompt-selected candidates. During expansion, however, it exposes
# inferred child operations whose domain edges are retained.
from helpers.similarity_utils import _node_text
from helpers.utils import _tokenize

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
        item,
):
    node_type = item.get(
        "domain_node_type"
    )

    return _is_executable_node_type(
        node_type
    )


def _is_executable_node_type(
        node_type,
):
    return node_type in (
        EXECUTABLE_NODE_TYPES
    )

def _is_contextual_operation_path(graph, path):
    for node_id in path:
        if not _is_executable_node_type(
                graph.nodes[node_id].get("node_type")
        ):
            return False

    for source, target in zip(path, path[1:]):
        if graph.edges[source, target].get(
                "relation"
        ) not in INFERRED_OPERATION_RELATIONSHIPS:
            return False

    return True

def _is_executable_candidate(
        graph,
        candidate,
):
    """
    Return True only for actual Operation nodes.
    """

    node_id = candidate.get(
        "node_id"
    )

    if graph is None:
        return False

    if not graph.has_node(node_id):
        return False

    node_data = graph.nodes[node_id]

    node_type = (
            candidate.get("node_type")
            or node_data.get("node_type")
    )

    if node_type in EXECUTABLE_NODE_TYPES:
        return True

    if node_type in NON_EXECUTABLE_NODE_TYPES:
        return False

    return False

def _relationship_compatibility(
        graph,
        selected_ids,
        candidate_id,
    ):

        if not selected_ids:
            return 0.0

        if not graph.has_node(candidate_id):
            return 0.0

        best_score = 0.0

        for selected_id in selected_ids:

            if not graph.has_node(selected_id):
                continue

            # ----------------------------------------------------
            # selected -> candidate
            # ----------------------------------------------------

            if graph.has_edge(
                selected_id,
                candidate_id,
            ):
                edge = graph.edges[
                    selected_id,
                    candidate_id,
                ]

                best_score = max(
                    best_score,
                    _edge_compatibility_score(
                        edge
                    ),
                )

            # ----------------------------------------------------
            # candidate -> selected
            # ----------------------------------------------------

            if graph.has_edge(
                candidate_id,
                selected_id,
            ):
                edge = graph.edges[
                    candidate_id,
                    selected_id,
                ]

                best_score = max(
                    best_score,
                    _edge_compatibility_score(
                        edge
                    ),
                )

        return best_score

def _edge_compatibility_score(
    edge,
):
    relation = edge.get(
        "relation",
        "",
    )

    classification = edge.get(
        "classification"
    )

    # Explicit execution/dependency relationships are strongest.
    if relation in ORDERING_RELATIONSHIPS:
        return REQUIRED_RELATION_BONUS

    # OPERATION_INCLUDES is useful structural evidence,
    # but it must not be interpreted as execution order.
    if relation in STRUCTURAL_RELATIONSHIPS:
        return STRUCTURAL_RELATION_BONUS

    if classification == "REQUIRED":
        return REQUIRED_RELATION_BONUS

    if classification == "POSSIBLE":
        return POSSIBLE_RELATION_BONUS

    if classification == "CONTEXT":
        return CONTEXT_RELATION_BONUS

    return 0.0

# ============================================================
# Graph connectivity
# ============================================================

def _connectivity_score(
    graph,
    selected_ids,
    candidate_id,
):

    if not selected_ids:
        return 0.0

    if not graph.has_node(candidate_id):
        return 0.0

    best = 0.0

    for selected_id in selected_ids:

        if not graph.has_node(selected_id):
            continue

        if graph.has_edge(
            selected_id,
            candidate_id,
        ):
            edge = graph.edges[
                selected_id,
                candidate_id,
            ]

            best = max(
                best,
                _connectivity_edge_score(
                    graph,
                    selected_id,
                    candidate_id,
                    edge,
                ),
            )

        if graph.has_edge(
            candidate_id,
            selected_id,
        ):
            edge = graph.edges[
                candidate_id,
                selected_id,
            ]

            best = max(
                best,
                _connectivity_edge_score(
                    graph,
                    candidate_id,
                    selected_id,
                    edge,
                ),
            )

    return best

def _connectivity_edge_score(
    graph,
    source,
    target,
    edge,
):
    relation = edge.get(
        "relation",
        "",
    )

    source_type = graph.nodes[
        source
    ].get(
        "node_type"
    )

    target_type = graph.nodes[
        target
    ].get(
        "node_type"
    )

    # --------------------------------------------------------
    # Operation -> Operation
    # --------------------------------------------------------

    if (
        source_type in EXECUTABLE_NODE_TYPES
        and target_type in EXECUTABLE_NODE_TYPES
    ):

        # Explicit ordering relationship.
        if relation in ORDERING_RELATIONSHIPS:
            return CONNECTIVITY_BONUS

        if relation in STRUCTURAL_RELATIONSHIPS:
            return (
                CONNECTIVITY_BONUS
                * 0.60
            )

        return (
            CONNECTIVITY_BONUS
            * 0.50
        )

    # --------------------------------------------------------
    # Contextual relationship.
    # --------------------------------------------------------

    if relation in CONTEXT_RELATIONSHIPS:
        return CONTEXT_RELATION_BONUS

    # --------------------------------------------------------
    # Rule relationships do not create workflow connectivity.
    # They are handled by _rule_constraint_penalty().
    # --------------------------------------------------------

    if relation in CONSTRAINT_RELATIONSHIPS:
        return 0.0

    return 0.0

def _rule_constraint_penalty(
    graph,
    prompt_text,
    operation_id,
):

    if graph is None:
        return 0.0, []

    if not graph.has_node(operation_id):
        return 0.0, []

    prompt_tokens = _tokenize(
        prompt_text
    )

    violations = []

    # --------------------------------------------------------
    # Find Rule -> Operation constraints.
    # --------------------------------------------------------

    for rule_id, _, edge in graph.in_edges(
        operation_id,
        data=True,
    ):

        relation = edge.get(
            "relation",
            "",
        )

        if relation not in CONSTRAINT_RELATIONSHIPS:
            continue

        if not graph.has_node(rule_id):
            continue

        rule_data = graph.nodes[
            rule_id
        ]

        rule_text = _node_text(
            rule_data
        )

        rule_tokens = _tokenize(
            rule_text
        )

        if not rule_tokens:
            continue

        overlap = (
            prompt_tokens
            & rule_tokens
        )

        if not overlap:
            continue

        if _prompt_contradicts_rule(
            prompt_text,
            rule_text,
        ):

            violations.append(
                {
                    "rule_node_id": rule_id,

                    "rule_name": rule_data.get(
                        "name",
                        rule_id,
                    ),

                    "operation_id": operation_id,

                    "relation": relation,

                    "overlap_terms": sorted(
                        overlap
                    ),

                    "rule_text": rule_text,

                    "reason": (
                        "Prompt appears to "
                        "contradict a domain "
                        "rule constraining "
                        "this operation."
                    ),
                }
            )

    if not violations:
        return 0.0, []

    penalty = (
        RULE_CONTRADICTION_PENALTY
        * len(violations)
    )

    return penalty, violations


def _prompt_contradicts_rule(prompt_text, rule_text):
    prompt = prompt_text.lower().strip()
    rule = rule_text.lower().strip()

    # A shared domain term is NOT a contradiction.
    # Contradiction requires an explicit opposite state.

    opposite_pairs = (
        ("active", "inactive"),
        ("enabled", "disabled"),
        ("valid", "invalid"),
        ("verified", "unverified"),
        ("approved", "rejected"),
        ("allowed", "forbidden"),
        ("authorized", "unauthorized"),
        ("successful", "failed"),
        ("success", "failure"),
        ("completed", "failed"),
    )

    for positive, negative in opposite_pairs:
        if positive in prompt and negative in rule:
            return True

        if negative in prompt and positive in rule:
            return True

    return False

def _infer_operations_from_rules(
        graph,
        selected_ids,
        prompt,
):
    if graph is None:
        return []

    inferred = []
    seen = set(selected_ids)

    for operation_id in list(selected_ids):
        if not graph.has_node(operation_id):
            continue

        for rule_id, _, edge in graph.in_edges(operation_id, data=True):
            if edge.get("relation", "") not in CONSTRAINT_RELATIONSHIPS:
                continue
            if not graph.has_node(rule_id):
                continue

            rule_text = _node_text(graph.nodes[rule_id])
            if _prompt_contradicts_rule(prompt, rule_text):
                continue

            for candidate_id in _get_rule_required_operations(
                    graph, rule_id, operation_id
            ):
                if candidate_id in seen:
                    continue
                seen.add(candidate_id)
                inferred.append(candidate_id)

    return inferred

def _get_rule_required_operations(
        graph,
        rule_id,
        selected_operation_id,
):
    """Return executable operations constrained by the same applicable rule."""
    operations = []
    for _, candidate_id, edge in graph.out_edges(rule_id, data=True):
        if edge.get("relation", "") not in CONSTRAINT_RELATIONSHIPS:
            continue
        if candidate_id == selected_operation_id:
            continue
        if not graph.has_node(candidate_id):
            continue
        if not _is_executable_node_type(
                graph.nodes[candidate_id].get("node_type")
        ):
            continue
        operations.append(candidate_id)
    return operations