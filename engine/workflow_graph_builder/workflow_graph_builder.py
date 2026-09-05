"""
Stage 9 - Workflow Graph Builder.

Responsibilities
----------------
1. Keep the executable operations selected by beam search.
2. Expand selected composite operations using OPERATION_INCLUDES.
3. Treat OPERATION_INCLUDES as structural composition, NOT execution order.
4. Preserve explicit prompt relationships.
5. When a required connection between selected prompt steps is missing:
       - start from the selected source operation
       - lazily fetch its domain neighborhood
       - follow valid directed domain-operation relationships
       - add discovered operations directly into selected_graph
       - continue traversal until the selected target operation is reached
6. Resume prompt execution from the LAST inferred domain operation.
7. Never delete selected or expanded operations because they are disconnected.
8. Never invent execution order.
9. Never propagate a prompt-level condition to every child of a composite.
10. Never build a temporary/full local copy of the domain graph.

Important
---------
Beam selection order != execution order.

Execution order comes from:
    - explicit PROMPT_DEPENDENCY
    - explicit PROMPT_PRECEDES
    - actual domain ordering relationships
    - domain relationships discovered lazily while repairing a missing road

OPERATION_INCLUDES only expands a composite operation. It does not
establish execution order.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Optional

import networkx as nx

from helpers.beam_search_utils import (
    _is_executable_selection_item,
    INFERRED_OPERATION_RELATIONSHIPS,
)


# ================================================================
# Relationship constants
# ================================================================

COMPOSITE_RELATIONSHIP = "OPERATION_INCLUDES"

PROMPT_DEPENDENCY = "PROMPT_DEPENDENCY"
PROMPT_CONDITION = "PROMPT_CONDITION"
PROMPT_PRECEDES = "PROMPT_PRECEDES"


# ================================================================
# Node types
# ================================================================

OPERATION_NODE_TYPES = {
    "Operation",
    "operation",
}


# ================================================================
# Relationships which can actually establish execution order.
#
# IMPORTANT:
#
# OPERATION_INCLUDES is deliberately excluded.
#
# OPERATION_REQUIRES, OPERATION_VALIDATES, OPERATION_PRODUCES,
# OPERATION_MODIFIES and OPERATION_ACCEPTS may exist in the ontology
# and can be useful while traversing the domain, but they should only
# establish runtime order when the ontology explicitly uses them as
# operation-to-operation execution relationships.
# ================================================================

DOMAIN_EXECUTION_RELATIONSHIPS = {
    "OPERATION_PRECEDES",
    "OPERATION_REQUIRES",
    "OPERATION_VALIDATES",
    "OPERATION_PRODUCES",
    "OPERATION_MODIFIES",
    "OPERATION_ACCEPTS",
}


# ================================================================
# WorkflowGraphBuilder
# ================================================================


class WorkflowGraphBuilder:
    """
    Build the executable workflow graph from the beam-search result.

    The builder owns the graph being constructed.

    It does NOT create a second/local copy of the domain graph.

    When a connection is missing, the builder asks the domain graph
    service for the neighborhood of the operation it is currently
    traversing and adds useful discovered operations directly to
    selected_graph.
    """

    def __init__(
        self,
        domain_graph_service=None,
        max_domain_traversal_depth: int = 8,
    ):
        self.domain_graph_service = domain_graph_service
        self.max_domain_traversal_depth = max(
            1,
            int(max_domain_traversal_depth),
        )

    # ============================================================
    # Public entry point
    # ============================================================

    def build(
        self,
        search_result,
        candidate_plan=None,
    ):
        if not search_result:
            raise RuntimeError(
                "WorkflowGraphBuilder.build() received no search result."
            )

        selection = list(
            search_result.get(
                "selection",
                [],
            )
        )

        if not selection:
            raise RuntimeError(
                "Beam search produced no selected workflow nodes."
            )

        prompt_subgraph = None

        if candidate_plan:
            prompt_subgraph = candidate_plan.get(
                "prompt_domain_subgraph"
            )

        # --------------------------------------------------------
        # Composite expansion
        # --------------------------------------------------------
        #
        # A composite is replaced by its executable children.
        #
        # OPERATION_INCLUDES itself is never converted into an
        # execution edge.
        # --------------------------------------------------------

        if prompt_subgraph is not None:
            selection = self._expand_composites(
                selection,
                prompt_subgraph,
            )

        # --------------------------------------------------------
        # Keep executable operations only.
        # --------------------------------------------------------

        executable_selection = [
            item
            for item in selection
            if _is_executable_selection_item(item)
        ]

        if not executable_selection:
            raise RuntimeError(
                "Beam search produced no executable workflow operations."
            )

        # --------------------------------------------------------
        # Semantic step order is retained as metadata.
        #
        # It is used to identify prompt roads/groups.
        #
        # It is NOT itself converted into execution edges.
        # --------------------------------------------------------

        executable_selection = sorted(
            executable_selection,
            key=self._selection_order_key,
        )

        # --------------------------------------------------------
        # Construct the selected workflow graph.
        # --------------------------------------------------------

        selected_graph = nx.DiGraph()

        for item in executable_selection:
            node_id = item.get(
                "domain_node_id"
            )

            if not node_id:
                continue

            if selected_graph.has_node(node_id):
                self._merge_node_metadata(
                    selected_graph.nodes[node_id],
                    item,
                )
                continue

            selected_graph.add_node(
                node_id,
                **item,
            )

        if selected_graph.number_of_nodes() == 0:
            raise RuntimeError(
                "No executable workflow operations remained after filtering."
            )

        # --------------------------------------------------------
        # No prompt subgraph means there is nothing to repair.
        # --------------------------------------------------------

        if prompt_subgraph is None:
            self._mark_inferred_nodes(
                selected_graph
            )

            return {
                "graph": selected_graph,
                "execution_order": self._execution_order(
                    selected_graph
                ),
                "conditional_branches": [],
            }

        # --------------------------------------------------------
        # Copy metadata from prompt/domain subgraph for nodes that
        # already exist in the selected graph.
        # --------------------------------------------------------

        self._copy_prompt_metadata(
            selected_graph,
            prompt_subgraph,
        )

        # --------------------------------------------------------
        # Build prompt roads.
        #
        # Explicit prompt relationships are preferred.
        # We do not blindly create:
        #
        #     selection[i] -> selection[i + 1]
        #
        # because beam order is not execution order.
        # --------------------------------------------------------

        prompt_roads = self._build_prompt_roads(
            executable_selection,
            prompt_subgraph,
        )

        # --------------------------------------------------------
        # Repair missing roads lazily.
        #
        # This is the central strategy.
        #
        # Existing selected graph first.
        # Prompt/domain subgraph second.
        # DomainGraphService.neighborhood() only when necessary.
        # --------------------------------------------------------

        for road in prompt_roads:
            self._process_prompt_road(
                selected_graph=selected_graph,
                source_nodes=road["source_nodes"],
                target_nodes=road["target_nodes"],
                prompt_subgraph=prompt_subgraph,
                prompt=road.get("prompt", ""),
            )

        # --------------------------------------------------------
        # Explicit prompt conditions are added after road repair.
        #
        # Conditions are control-flow edges. They are not used as
        # ordinary domain traversal relationships.
        # --------------------------------------------------------

        self._add_prompt_conditions(
            selected_graph,
            executable_selection,
        )

        # --------------------------------------------------------
        # Never allow composite expansion metadata to leak a
        # parent-level prompt condition onto every child.
        # --------------------------------------------------------

        self._normalise_composite_conditions(
            selected_graph
        )

        # --------------------------------------------------------
        # Mark inferred nodes.
        # --------------------------------------------------------

        self._mark_inferred_nodes(
            selected_graph
        )

        # --------------------------------------------------------
        # Final workflow graph.
        # --------------------------------------------------------

        workflow_graph = selected_graph.copy()

        execution_order = self._execution_order(
            workflow_graph
        )

        conditional_branches = (
            self._collect_conditional_branches(
                workflow_graph
            )
        )

        return {
            "graph": workflow_graph,
            "execution_order": execution_order,
            "conditional_branches": conditional_branches,
        }

    # ============================================================
    # Selection ordering
    # ============================================================

    @staticmethod
    def _selection_order_key(item):
        step_index = item.get(
            "step_index"
        )

        if isinstance(step_index, int):
            return (
                0,
                step_index,
            )

        return (
            1,
            float("inf"),
        )

    # ============================================================
    # Composite expansion
    # ============================================================

    def _expand_composites(
        self,
        selection,
        graph,
    ):
        """
        Expand selected composite operations.

        Example:

            transfer_funds
                 |
                 +-- retrieve_account
                 +-- verify_account_status
                 +-- validate_transfer_request
                 +-- check_sufficient_balance
                 +-- debit_account
                 +-- credit_account
                 +-- persist_fund_transfer
                 +-- generate_transfer_response
                 +-- create_transaction

        OPERATION_INCLUDES tells us that the children belong to
        the composite.

        It does NOT tell us the execution order of those children.

        The children therefore remain separate operations and their
        actual order is established later from real domain edges.
        """

        expanded = []

        for item in selection:
            node_id = item.get(
                "domain_node_id"
            )

            if not node_id:
                expanded.append(item)
                continue

            if not graph.has_node(node_id):
                expanded.append(item)
                continue

            node_data = graph.nodes[node_id]

            if not self._is_operation_node(
                node_data
            ):
                expanded.append(item)
                continue

            children = []

            for _, child_id, edge_data in graph.out_edges(
                node_id,
                data=True,
            ):
                relation = edge_data.get(
                    "relation"
                )

                if relation != COMPOSITE_RELATIONSHIP:
                    continue

                if not graph.has_node(child_id):
                    continue

                child_data = graph.nodes[child_id]

                if not self._is_operation_node(
                    child_data
                ):
                    continue

                children.append(child_id)

            if not children:
                expanded.append(item)
                continue

            # ----------------------------------------------------
            # Replace composite with its children.
            #
            # IMPORTANT:
            #
            # Do NOT copy the parent's prompt condition here.
            #
            # The condition belongs to the prompt-level action,
            # not automatically to every implementation child.
            # ----------------------------------------------------

            for child_id in children:
                child_data = graph.nodes[child_id]

                child_item = {
                    "prompt_text": item.get(
                        "prompt_text",
                        "",
                    ),
                    "domain_node_id": child_id,
                    "domain_node_name": child_data.get(
                        "name",
                        child_id,
                    ),
                    "domain_node_type": child_data.get(
                        "node_type",
                        "Operation",
                    ),
                    "step_index": item.get(
                        "step_index"
                    ),
                    "step_text": item.get(
                        "step_text",
                        "",
                    ),
                    "explicit": False,
                    "inferred": True,
                    "source": "composite_expansion",
                    "semantic_score": child_data.get(
                        "semantic_score",
                        item.get(
                            "semantic_score",
                            0.0,
                        ),
                    ),
                    "lexical_score": child_data.get(
                        "lexical_score",
                        item.get(
                            "lexical_score",
                            0.0,
                        ),
                    ),
                    "candidate_score": child_data.get(
                        "score",
                        item.get(
                            "candidate_score",
                            0.0,
                        ),
                    ),
                    "relationship_score": 0.0,
                    "connectivity_score": 0.0,
                    "constraint_penalty": 0.0,
                    "constraint_violations": [],
                    "condition": "",
                    "branch": "",
                    "condition_negated": False,
                    "condition_source_step_index": None,
                    "composite_parent": node_id,
                }

                expanded.append(
                    child_item
                )

        # --------------------------------------------------------
        # De-duplicate operations.
        #
        # Never delete an operation merely because it is
        # disconnected.
        # --------------------------------------------------------

        result = []
        seen = {}

        for item in expanded:
            node_id = item.get(
                "domain_node_id"
            )

            if not node_id:
                continue

            if node_id not in seen:
                seen[node_id] = len(result)
                result.append(item)
                continue

            existing_index = seen[node_id]
            existing = result[existing_index]

            # Preserve the explicit version when an operation was
            # both directly selected and reached through expansion.
            if (
                item.get("explicit", False)
                and not existing.get("explicit", False)
            ):
                result[existing_index] = item

        return result

    # ============================================================
    # Prompt road construction
    # ============================================================

    def _build_prompt_roads(
        self,
        selection,
        prompt_subgraph,
    ):
        """
        Build the logical roads between prompt steps.

        Priority:

        1. Explicit prompt/domain dependency edges.
        2. Prompt PRECEDES edges.
        3. Semantic step ordering only when no explicit road exists.

        The final fallback creates a road between consecutive semantic
        steps, but the road itself is NOT immediately inserted into
        the workflow graph.

        It is merely a request to the domain traversal layer:

            connect step A to step B

        This distinction is important.
        """

        by_step = {}

        for item in selection:
            step_index = item.get(
                "step_index"
            )

            node_id = item.get(
                "domain_node_id"
            )

            if not isinstance(step_index, int):
                continue

            if not node_id:
                continue

            by_step.setdefault(
                step_index,
                [],
            )

            if node_id not in by_step[step_index]:
                by_step[step_index].append(
                    node_id
                )

        if not by_step:
            return []

        roads = []

        # --------------------------------------------------------
        # First use explicit prompt/domain roads.
        # --------------------------------------------------------

        explicit_roads = self._extract_explicit_prompt_roads(
            prompt_subgraph,
            by_step,
        )

        roads.extend(
            explicit_roads
        )

        # --------------------------------------------------------
        # Determine which semantic step boundaries already have an
        # explicit road.
        # --------------------------------------------------------

        covered_boundaries = {
            (
                road["source_step"],
                road["target_step"],
            )
            for road in explicit_roads
            if road.get("source_step") is not None
            and road.get("target_step") is not None
        }

        # --------------------------------------------------------
        # For semantic step boundaries without an explicit road,
        # create a logical road.
        #
        # Again: this does NOT add a graph edge.
        #
        # It asks the domain traversal to determine whether a real
        # domain road exists.
        # --------------------------------------------------------

        ordered_steps = sorted(
            by_step.keys()
        )

        for previous_step, current_step in zip(
            ordered_steps,
            ordered_steps[1:],
        ):
            boundary = (
                previous_step,
                current_step,
            )

            if boundary in covered_boundaries:
                continue

            roads.append(
                {
                    "source_step": previous_step,
                    "target_step": current_step,
                    "source_nodes": list(
                        by_step[previous_step]
                    ),
                    "target_nodes": list(
                        by_step[current_step]
                    ),
                    "prompt": self._prompt_for_step(
                        selection,
                        current_step,
                    ),
                    "explicit": False,
                }
            )

        return roads

    # ============================================================
    # Explicit prompt road extraction
    # ============================================================

    def _extract_explicit_prompt_roads(
        self,
        prompt_subgraph,
        by_step,
    ):
        roads = []

        if prompt_subgraph is None:
            return roads

        for source, target, edge_data in prompt_subgraph.edges(
            data=True
        ):
            relation = edge_data.get(
                "relation"
            )

            if relation not in {
                PROMPT_DEPENDENCY,
                PROMPT_PRECEDES,
            }:
                continue

            source_nodes = (
                list(by_step.get(source, []))
            )

            target_nodes = (
                list(by_step.get(target, []))
            )

            # ----------------------------------------------------
            # Some prompt subgraphs may use domain node IDs rather
            # than semantic step indexes.
            # ----------------------------------------------------

            if not source_nodes and source in prompt_subgraph:
                if self._is_operation_node(
                    prompt_subgraph.nodes[source]
                ):
                    source_nodes = [source]

            if not target_nodes and target in prompt_subgraph:
                if self._is_operation_node(
                    prompt_subgraph.nodes[target]
                ):
                    target_nodes = [target]

            if not source_nodes or not target_nodes:
                continue

            roads.append(
                {
                    "source_step": (
                        self._step_index_for_node(
                            source,
                            by_step,
                        )
                    ),
                    "target_step": (
                        self._step_index_for_node(
                            target,
                            by_step,
                        )
                    ),
                    "source_nodes": source_nodes,
                    "target_nodes": target_nodes,
                    "prompt": edge_data.get(
                        "prompt",
                        "",
                    ),
                    "explicit": True,
                }
            )

        return roads

    # ============================================================
    # Process one prompt road
    # ============================================================

    def _process_prompt_road(
        self,
        selected_graph,
        source_nodes,
        target_nodes,
        prompt_subgraph,
        prompt="",
    ):
        """
        Connect one logical prompt road.

        Strategy:

            A
            |
            | already in selected graph?
            v
            inspect selected/prompt graph
            |
            | missing
            v
            neighborhood(A)
            |
            v
            X
            |
            neighborhood(X)
            |
            v
            Y
            |
            v
            B

        The traversal stops when a selected target operation is
        reached.

        The important property is that the traversal directly mutates
        selected_graph. No temporary domain graph is constructed.
        """

        source_nodes = [
            node_id
            for node_id in source_nodes
            if node_id in selected_graph
        ]

        target_nodes = set(
            node_id
            for node_id in target_nodes
            if node_id in selected_graph
        )

        if not source_nodes or not target_nodes:
            return False

        # --------------------------------------------------------
        # If the road already exists, do absolutely nothing.
        # --------------------------------------------------------

        if self._has_existing_execution_connection(
            selected_graph,
            source_nodes,
            target_nodes,
        ):
            return True

        # --------------------------------------------------------
        # First inspect the prompt subgraph.
        #
        # We do not query Neo4j if the needed road is already
        # present there.
        # --------------------------------------------------------

        if self._repair_from_prompt_subgraph(
            selected_graph,
            source_nodes,
            target_nodes,
            prompt_subgraph,
        ):
            return True

        # --------------------------------------------------------
        # Missing road.
        #
        # Now and only now use the domain graph service.
        # --------------------------------------------------------

        return self._repair_connection_by_traversal(
            selected_graph=selected_graph,
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            prompt=prompt,
        )

    # ============================================================
    # Existing selected-graph connection
    # ============================================================

    def _has_existing_execution_connection(
        self,
        graph,
        source_nodes,
        target_nodes,
    ):
        target_nodes = set(target_nodes)

        for source in source_nodes:
            for target in target_nodes:
                if source == target:
                    return True

                if not nx.has_path(
                    graph,
                    source,
                    target,
                ):
                    continue

                path = nx.shortest_path(
                    graph,
                    source,
                    target,
                )

                if self._path_is_execution_path(
                    graph,
                    path,
                ):
                    return True

        return False

    # ============================================================
    # Prompt-subgraph road repair
    # ============================================================

    def _repair_from_prompt_subgraph(
        self,
        selected_graph,
        source_nodes,
        target_nodes,
        prompt_subgraph,
    ):
        if prompt_subgraph is None:
            return False

        target_nodes = set(
            target_nodes
        )

        # --------------------------------------------------------
        # We only traverse relationships already copied into the
        # prompt subgraph.
        #
        # No OPERATION_INCLUDES traversal here.
        # --------------------------------------------------------

        queue = deque()

        for source in source_nodes:
            queue.append(
                (
                    source,
                    0,
                )
            )

        visited = set()

        while queue:
            current, depth = queue.popleft()

            if current in visited:
                continue

            visited.add(current)

            if current in target_nodes:
                return True

            if depth >= self.max_domain_traversal_depth:
                continue

            if current not in prompt_subgraph:
                continue

            for _, neighbour, edge_data in prompt_subgraph.out_edges(
                current,
                data=True,
            ):
                relation = edge_data.get(
                    "relation"
                )

                if not self._is_domain_execution_relationship(
                    relation
                ):
                    continue

                if neighbour not in prompt_subgraph:
                    continue

                neighbour_data = (
                    prompt_subgraph.nodes[neighbour]
                )

                if not self._is_operation_node(
                    neighbour_data
                ):
                    continue

                self._add_inferred_operation(
                    selected_graph,
                    neighbour,
                    neighbour_data,
                    source="prompt_subgraph",
                )

                self._add_domain_edge(
                    selected_graph,
                    current,
                    neighbour,
                    edge_data,
                    source="prompt_subgraph",
                )

                queue.append(
                    (
                        neighbour,
                        depth + 1,
                    )
                )

        return False

    # ============================================================
    # Lazy domain traversal
    # ============================================================

    def _repair_connection_by_traversal(
        self,
        selected_graph,
        source_nodes,
        target_nodes,
        prompt="",
    ):
        """
        Lazily traverse the domain graph.

        NO local domain graph is created.

        For each encountered operation:

            neighborhood(operation_id)

        is called.

        Returned relationships are inspected immediately.

        Valid discovered operation nodes and edges are inserted
        directly into selected_graph.

        When the traversal reaches a selected target, the road is
        complete.

        A parent map is maintained so that we know exactly which
        inferred operation was the last operation before the target.
        """

        service = self._get_domain_graph_service()

        if service is None:
            return False

        target_nodes = set(
            target_nodes
        )

        queue = deque()

        visited = set()

        parent = {}

        # --------------------------------------------------------
        # Start from the selected source boundary nodes.
        # --------------------------------------------------------

        for source in source_nodes:
            queue.append(
                (
                    source,
                    0,
                )
            )

        while queue:
            current, depth = queue.popleft()

            if current in visited:
                continue

            visited.add(current)

            # ----------------------------------------------------
            # Target reached.
            #
            # Because the traversal adds each edge immediately,
            # selected_graph already contains:
            #
            #     source -> inferred1 -> inferred2 -> target
            #
            # The last inferred node therefore naturally becomes
            # the predecessor of the resumed prompt node.
            # ----------------------------------------------------

            if current in target_nodes:
                return True

            if depth >= self.max_domain_traversal_depth:
                continue

            try:
                relationships = service.neighborhood(
                    current,
                    depth=1,
                )
            except Exception:
                # A failed neighborhood query must not destroy the
                # rest of the workflow.
                continue

            if not relationships:
                continue

            for relationship in relationships:
                source_id = self._relationship_value(
                    relationship,
                    "source_id",
                )

                target_id = self._relationship_value(
                    relationship,
                    "target_id",
                )

                relation = self._relationship_value(
                    relationship,
                    "relation",
                )

                if not source_id or not target_id:
                    continue

                # ------------------------------------------------
                # neighborhood() returns relationships around the
                # node and therefore may contain both incoming and
                # outgoing edges.
                #
                # Execution traversal is directed.
                #
                # Only follow:
                #
                #     current -> target
                #
                # Never:
                #
                #     target -> current
                # ------------------------------------------------

                if source_id != current:
                    continue

                if not self._is_domain_execution_relationship(
                    relation
                ):
                    continue

                # ------------------------------------------------
                # OPERATION_INCLUDES is intentionally rejected by
                # _is_domain_execution_relationship().
                #
                # It is structural composition, not execution.
                # ------------------------------------------------

                node_data = self._get_domain_node_data(
                    target_id
                )

                if node_data is None:
                    # If the target is already in selected_graph,
                    # use its existing metadata.
                    if target_id in selected_graph:
                        node_data = dict(
                            selected_graph.nodes[target_id]
                        )
                    else:
                        continue

                if not self._is_operation_node(
                    node_data
                ):
                    # Contextual entities/events/components/rules
                    # are traversal information, not workflow nodes.
                    continue

                # ------------------------------------------------
                # Add discovered operation directly into the
                # selected workflow graph.
                # ------------------------------------------------

                self._add_inferred_operation(
                    selected_graph,
                    target_id,
                    node_data,
                    source="domain_neighborhood",
                )

                # ------------------------------------------------
                # Add the actual ontology relationship.
                # ------------------------------------------------

                edge_data = self._normalise_relationship(
                    relationship
                )

                self._add_domain_edge(
                    selected_graph,
                    current,
                    target_id,
                    edge_data,
                    source="domain_neighborhood",
                )

                # ------------------------------------------------
                # Remember traversal parent.
                # ------------------------------------------------

                parent[target_id] = current

                # ------------------------------------------------
                # Continue lazily from the newly encountered
                # operation.
                # ------------------------------------------------

                if target_id not in visited:
                    queue.append(
                        (
                            target_id,
                            depth + 1,
                        )
                    )

                # ------------------------------------------------
                # Stop as soon as the selected target is reached.
                # ------------------------------------------------

                if target_id in target_nodes:
                    return True

        return False

    # ============================================================
    # Domain graph service
    # ============================================================

    def _get_domain_graph_service(self):
        """
        Resolve the domain graph service.

        Normally this is injected through __init__.

        The fallback allows callers that attach the service to the
        builder after construction to continue working.
        """

        return self.domain_graph_service

    # ============================================================
    # Domain node lookup
    # ============================================================

    def _get_domain_node_data(
        self,
        node_id,
    ):
        service = self._get_domain_graph_service()

        if service is None:
            return None

        try:
            node = service.get_node(
                node_id
            )
        except Exception:
            return None

        if node is None:
            return None

        if isinstance(node, dict):
            data = dict(node)

            data.setdefault(
                "id",
                node_id,
            )

            data.setdefault(
                "node_type",
                data.get(
                    "type",
                    "Operation",
                ),
            )

            return data

        # --------------------------------------------------------
        # Support DomainNode dataclass objects.
        # --------------------------------------------------------

        data = {
            "id": getattr(
                node,
                "id",
                node_id,
            ),
            "name": getattr(
                node,
                "name",
                node_id,
            ),
            "node_type": getattr(
                node,
                "node_type",
                "Operation",
            ),
            "description": getattr(
                node,
                "description",
                "",
            ),
            "aliases": list(
                getattr(
                    node,
                    "aliases",
                    [],
                )
                or []
            ),
            "types": list(
                getattr(
                    node,
                    "types",
                    [],
                )
                or []
            ),
        }

        return data

    # ============================================================
    # Add inferred operation
    # ============================================================

    def _add_inferred_operation(
        self,
        graph,
        node_id,
        node_data,
        source,
    ):
        if graph.has_node(node_id):
            existing = graph.nodes[node_id]

            # Do not overwrite prompt metadata.
            self._merge_node_metadata(
                existing,
                node_data,
            )

            existing.setdefault(
                "inferred",
                False,
            )

            return

        data = dict(
            node_data
        )

        data.setdefault(
            "domain_node_id",
            node_id,
        )

        data.setdefault(
            "domain_node_name",
            data.get(
                "name",
                node_id,
            ),
        )

        data.setdefault(
            "domain_node_type",
            data.get(
                "node_type",
                "Operation",
            ),
        )

        data.setdefault(
            "prompt_text",
            data.get(
                "name",
                node_id,
            ),
        )

        data["inferred"] = True
        data["source"] = source

        graph.add_node(
            node_id,
            **data,
        )

    # ============================================================
    # Add domain edge
    # ============================================================

    def _add_domain_edge(
        self,
        graph,
        source,
        target,
        edge_data,
        source_label,
    ):
        if source == target:
            return

        relation = edge_data.get(
            "relation"
        )

        if not self._is_domain_execution_relationship(
            relation
        ):
            return

        # --------------------------------------------------------
        # Never overwrite an explicit prompt relationship.
        # --------------------------------------------------------

        existing = graph.get_edge_data(
            source,
            target,
        )

        if existing is not None:
            existing_relation = existing.get(
                "relation"
            )

            if existing_relation in {
                PROMPT_DEPENDENCY,
                PROMPT_PRECEDES,
                PROMPT_CONDITION,
            }:
                return

        data = dict(
            edge_data
        )

        data["relation"] = relation
        data["inferred_context"] = True
        data["origin"] = source_label

        graph.add_edge(
            source,
            target,
            **data,
        )

    # ============================================================
    # Relationship helpers
    # ============================================================

    @classmethod
    def _is_domain_execution_relationship(
        cls,
        relation,
    ):
        if not relation:
            return False

        relation = str(
            relation
        ).strip()

        if relation == COMPOSITE_RELATIONSHIP:
            return False

        return (
            relation
            in DOMAIN_EXECUTION_RELATIONSHIPS
        )

    @staticmethod
    def _relationship_value(
        relationship,
        field,
    ):
        if isinstance(
            relationship,
            dict,
        ):
            return relationship.get(
                field
            )

        return getattr(
            relationship,
            field,
            None,
        )

    @classmethod
    def _normalise_relationship(
        cls,
        relationship,
    ):
        if isinstance(
            relationship,
            dict,
        ):
            data = dict(
                relationship
            )
        else:
            data = {
                "source_id": getattr(
                    relationship,
                    "source_id",
                    None,
                ),
                "target_id": getattr(
                    relationship,
                    "target_id",
                    None,
                ),
                "relation": getattr(
                    relationship,
                    "relation",
                    None,
                ),
            }

        data["relation"] = data.get(
            "relation"
        )

        return data

    # ============================================================
    # Operation helpers
    # ============================================================

    @staticmethod
    def _is_operation_node(
        node_data,
    ):
        if not node_data:
            return False

        node_type = node_data.get(
            "node_type"
        )

        if node_type is None:
            node_type = node_data.get(
                "type"
            )

        if node_type is None:
            node_type = node_data.get(
                "domain_node_type"
            )

        return node_type in OPERATION_NODE_TYPES

    # ============================================================
    # Prompt conditions
    # ============================================================

    def _add_prompt_conditions(
        self,
        graph,
        selection,
    ):
        """
        Add prompt-level conditional edges.

        Conditions are attached to the actual selected prompt
        operation, not blindly copied to composite children.
        """

        by_step = {}

        for item in selection:
            step_index = item.get(
                "step_index"
            )

            node_id = item.get(
                "domain_node_id"
            )

            if not isinstance(
                step_index,
                int,
            ):
                continue

            if not node_id:
                continue

            # For a step containing multiple composite children,
            # retain all members.
            by_step.setdefault(
                step_index,
                [],
            )

            if node_id not in by_step[step_index]:
                by_step[step_index].append(
                    node_id
                )

        for item in selection:
            condition = item.get(
                "condition",
                "",
            )

            if not condition:
                continue

            target_step = item.get(
                "step_index"
            )

            source_step = item.get(
                "condition_source_step_index"
            )

            if not isinstance(
                target_step,
                int,
            ):
                continue

            if not isinstance(
                source_step,
                int,
            ):
                continue

            source_nodes = by_step.get(
                source_step,
                [],
            )

            target_nodes = by_step.get(
                target_step,
                [],
            )

            if not source_nodes or not target_nodes:
                continue

            branch = item.get(
                "branch",
                "then",
            )

            condition_negated = item.get(
                "condition_negated",
                False,
            )

            # ----------------------------------------------------
            # For composite children, the prompt condition is
            # represented at the step boundary.
            #
            # We connect the condition source to the first
            # executable member of the target step.
            # ----------------------------------------------------

            target_id = target_nodes[0]

            source_id = source_nodes[-1]

            if source_id == target_id:
                continue

            existing = graph.get_edge_data(
                source_id,
                target_id,
            )

            if existing is not None:
                existing_relation = existing.get(
                    "relation"
                )

                if existing_relation in {
                    PROMPT_DEPENDENCY,
                    PROMPT_PRECEDES,
                }:
                    graph.remove_edge(
                        source_id,
                        target_id,
                    )

                elif existing_relation == PROMPT_CONDITION:
                    existing.update(
                        {
                            "condition": condition,
                            "branch": branch,
                            "condition_negated": (
                                condition_negated
                            ),
                        }
                    )
                    continue

            graph.add_edge(
                source_id,
                target_id,
                relation=PROMPT_CONDITION,
                condition=condition,
                branch=branch,
                condition_negated=condition_negated,
                inferred_context=False,
                origin="prompt",
            )

    # ============================================================
    # Composite condition cleanup
    # ============================================================

    @staticmethod
    def _normalise_composite_conditions(
        graph,
    ):
        """
        Composite children must not inherit a prompt condition merely
        because their parent composite was conditional.

        A condition is retained only when it belongs to an actual
        prompt-condition edge.
        """

        conditional_targets = set()

        for source, target, data in graph.edges(
            data=True
        ):
            if data.get(
                "relation"
            ) != PROMPT_CONDITION:
                continue

            conditional_targets.add(
                target
            )

        for node_id, node_data in graph.nodes(
            data=True
        ):
            if not node_data.get(
                "composite_parent"
            ):
                continue

            if node_id in conditional_targets:
                continue

            # ----------------------------------------------------
            # This operation is an implementation child.
            #
            # It should not carry the parent's prompt condition.
            # ----------------------------------------------------

            node_data["condition"] = ""
            node_data["branch"] = ""
            node_data["condition_negated"] = False
            node_data["condition_source_step_index"] = None

    # ============================================================
    # Metadata
    # ============================================================

    @staticmethod
    def _merge_node_metadata(
        target,
        source,
    ):
        for key, value in source.items():
            if key not in target:
                target[key] = value
                continue

            if target[key] in (
                None,
                "",
                [],
            ) and value not in (
                None,
                "",
                [],
            ):
                target[key] = value

    def _copy_prompt_metadata(
        self,
        selected_graph,
        prompt_subgraph,
    ):
        for node_id in list(
            selected_graph.nodes
        ):
            if not prompt_subgraph.has_node(
                node_id
            ):
                continue

            prompt_data = prompt_subgraph.nodes[
                node_id
            ]

            runtime_data = selected_graph.nodes[
                node_id
            ]

            self._merge_node_metadata(
                runtime_data,
                prompt_data,
            )

    # ============================================================
    # Inferred marking
    # ============================================================

    @staticmethod
    def _mark_inferred_nodes(
        graph,
    ):
        for _, node_data in graph.nodes(
            data=True
        ):
            node_data.setdefault(
                "inferred",
                False,
            )

    # ============================================================
    # Prompt helpers
    # ============================================================

    @staticmethod
    def _prompt_for_step(
        selection,
        step_index,
    ):
        for item in selection:
            if item.get(
                "step_index"
            ) != step_index:
                continue

            prompt = item.get(
                "prompt_text"
            )

            if prompt:
                return prompt

            prompt = item.get(
                "step_text"
            )

            if prompt:
                return prompt

        return ""

    @staticmethod
    def _step_index_for_node(
        node_id,
        by_step,
    ):
        for step_index, nodes in by_step.items():
            if node_id in nodes:
                return step_index

        return None

    # ============================================================
    # Execution-path validation
    # ============================================================

    @classmethod
    def _path_is_execution_path(
        cls,
        graph,
        path,
    ):
        if not path:
            return False

        for source, target in zip(
            path,
            path[1:],
        ):
            edge_data = graph.get_edge_data(
                source,
                target,
            )

            if edge_data is None:
                return False

            relation = edge_data.get(
                "relation"
            )

            if relation == PROMPT_CONDITION:
                continue

            if relation in {
                PROMPT_DEPENDENCY,
                PROMPT_PRECEDES,
            }:
                continue

            if cls._is_domain_execution_relationship(
                relation
            ):
                continue

            return False

        return True

    # ============================================================
    # Execution order
    # ============================================================

    @staticmethod
    def _execution_order(
        graph,
    ):
        try:
            return list(
                nx.topological_sort(
                    graph
                )
            )

        except nx.NetworkXUnfeasible:
            cycle = WorkflowGraphBuilder._describe_cycle(
                graph
            )

            raise RuntimeError(
                "Selected workflow contains a cycle and cannot be "
                "ordered: "
                + cycle
            )

    # ============================================================
    # Cycle diagnostics
    # ============================================================

    @staticmethod
    def _describe_cycle(
        graph,
    ):
        try:
            cycle_edges = nx.find_cycle(
                graph
            )

        except nx.NetworkXNoCycle:
            return (
                "(cycle detected by topological_sort but not "
                "reproducible via find_cycle)"
            )

        names = []

        for source, target in cycle_edges:
            edge_data = graph.edges[
                source,
                target,
            ]

            source_name = graph.nodes[
                source
            ].get(
                "name",
                source,
            )

            target_name = graph.nodes[
                target
            ].get(
                "name",
                target,
            )

            relation = edge_data.get(
                "relation",
                "?",
            )

            names.append(
                f"{source_name} "
                f"--{relation}--> "
                f"{target_name}"
            )

        return " ; ".join(
            names
        )

    # ============================================================
    # Conditional branch collection
    # ============================================================

    @staticmethod
    def _collect_conditional_branches(
        graph,
    ):
        branches = []

        for source, target, data in graph.edges(
            data=True
        ):
            if data.get(
                "relation"
            ) != PROMPT_CONDITION:
                continue

            branches.append(
                {
                    "source": source,
                    "target": target,
                    "condition": data.get(
                        "condition",
                        "",
                    ),
                    "branch": data.get(
                        "branch",
                        "then",
                    ),
                    "condition_negated": data.get(
                        "condition_negated",
                        False,
                    ),
                }
            )

        return branches