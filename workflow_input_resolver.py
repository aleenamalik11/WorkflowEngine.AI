"""
Workflow Input Resolver
=======================

Determines the external inputs required to START a generated workflow.

Architectural responsibility
----------------------------

The domain graph is the primary source of truth for determining what
conceptual information the workflow requires.

For example:

    Update Account
        |
        | OPERATION_REQUIRES
        v
    Account

means:

    "The Update Account operation requires an Account concept."

The Account node is NOT added to the workflow as an executable step.

Instead, Account becomes an external workflow input requirement when
Update Account is the first executable operation.

Optional function information can then refine the semantic requirement.

For example, the domain graph may say:

    Update Account
        --OPERATION_REQUIRES--> Account

while the actual first function may be:

    UpdateAccount(Guid accountId, UpdateAccountRequest request)

In that case the resolver can expose:

    account -> accountId

The function metadata is therefore an implementation-level refinement,
not the source of semantic workflow planning.

Important architectural rules
-----------------------------

1. Domain graph determines semantic requirements.
2. Only executable Operation nodes are workflow steps.
3. DomainEntity / Entity / Rule / Event nodes are never workflow steps.
4. OPERATION_REQUIRES and OPERATION_ACCEPTS are input-related
   relationships.
5. OPERATION_CREATES / OPERATION_PRODUCES describe outputs and are
   therefore not initial external inputs.
6. Requirements satisfied by an earlier workflow operation are NOT
   external inputs.
7. Only requirements of the workflow's starting operation(s) are
   considered initial inputs.
8. If a function is supplied, it may refine the semantic requirements
   into concrete function parameters.
9. Function metadata is optional. The resolver works entirely from
   the domain graph.
10. No function is invented when function information is unavailable.

Example
-------

Domain graph:

    Create Account
        --OPERATION_CREATES--> Account

    Update Account
        --OPERATION_REQUIRES--> Account

Workflow:

    Create Account
        ->
    Update Account

Initial inputs:

    []

because Account is produced by Create Account.

For:

    Update Account

as the first operation:

    Update Account
        --OPERATION_REQUIRES--> Account

Initial inputs:

    [
        {
            "name": "Account",
            "source": "domain_graph",
            "relationship": "OPERATION_REQUIRES"
        }
    ]

If the first function is known:

    UpdateAccount(Guid accountId, UpdateAccountRequest request)

the resolver may additionally expose the concrete implementation input:

    account -> accountId

but it does not use the function to decide whether Account is
semantically required.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


class WorkflowInputResolver:
    """
    Resolve the external inputs required to start a workflow.

    The resolver operates on the already-created workflow/domain graph.

    Parameters
    ----------
    executable_node_types:
        Node types that represent executable workflow operations.

    requirement_relationships:
        Domain relationships that can introduce input requirements.

    produced_relationships:
        Relationships that indicate that a concept is produced by
        an operation and therefore may be satisfied internally.
    """

    DEFAULT_EXECUTABLE_NODE_TYPES = {
        "Operation",
        "operation",
    }

    DEFAULT_REQUIREMENT_RELATIONSHIPS = {
        "OPERATION_REQUIRES",
        "OPERATION_ACCEPTS",
    }

    DEFAULT_PRODUCED_RELATIONSHIPS = {
        "OPERATION_CREATES",
        "OPERATION_PRODUCES",
        "OPERATION_PRODUCES_EVENT",
    }

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

    def __init__(
        self,
        executable_node_types: Optional[Set[str]] = None,
        requirement_relationships: Optional[Set[str]] = None,
        produced_relationships: Optional[Set[str]] = None,
    ):
        self.executable_node_types = (
            executable_node_types
            or set(self.DEFAULT_EXECUTABLE_NODE_TYPES)
        )

        self.requirement_relationships = (
            requirement_relationships
            or set(self.DEFAULT_REQUIREMENT_RELATIONSHIPS)
        )

        self.produced_relationships = (
            produced_relationships
            or set(self.DEFAULT_PRODUCED_RELATIONSHIPS)
        )

    # ============================================================
    # Public API
    # ============================================================

    def resolve(
        self,
        workflow_graph,
        execution_order: Optional[Sequence[Any]] = None,
        start_function: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Resolve the external inputs required to start the workflow.

        Parameters
        ----------
        workflow_graph:
            NetworkX directed graph containing the selected workflow.

        execution_order:
            Execution order returned by GraphPlanner.

            If omitted, the resolver derives the first executable
            operation from the graph.

        start_function:
            Optional RegisteredFunction / function-like object for
            the first executable operation.

            This is OPTIONAL.

            The domain graph remains the semantic source of truth.

        Returns
        -------
        dict

        Example:

            {
                "inputs": [
                    {
                        "name": "Account",
                        "type": "DomainEntity",
                        "relationship": "OPERATION_REQUIRES",
                        "source": "domain_graph"
                    }
                ],
                "start_nodes": [
                    "update-account-id"
                ]
            }
        """

        if workflow_graph is None:
            return {
                "inputs": [],
                "start_nodes": [],
                "status": "no workflow graph",
            }

        start_nodes = self._find_start_nodes(
            workflow_graph,
            execution_order,
        )

        if not start_nodes:
            return {
                "inputs": [],
                "start_nodes": [],
                "status": "no executable start node",
            }

        semantic_inputs = []

        seen = set()

        for start_node in start_nodes:

            requirements = self._get_node_requirements(
                workflow_graph,
                start_node,
            )

            for requirement in requirements:

                key = (
                    requirement["node_id"],
                    requirement["relationship"],
                    start_node,
                )

                if key in seen:
                    continue

                seen.add(key)

                requirement["required_by"] = (
                    self._node_name(
                        workflow_graph,
                        start_node,
                    )
                )

                semantic_inputs.append(
                    requirement
                )

        # --------------------------------------------------------
        # If the first function is available, refine the semantic
        # requirements using its concrete parameters.
        #
        # This does NOT replace the domain-graph requirements.
        # --------------------------------------------------------

        function_inputs = []

        if start_function is not None:
            function_inputs = (
                self._resolve_function_inputs(
                    start_function,
                    semantic_inputs,
                )
            )

        return {
            "inputs": semantic_inputs,
            "function_inputs": function_inputs,
            "start_nodes": [
                self._node_name(
                    workflow_graph,
                    node_id,
                )
                for node_id in start_nodes
            ],
            "status": "resolved",
        }

    # ============================================================
    # Find workflow start
    # ============================================================

    def _find_start_nodes(
        self,
        graph,
        execution_order: Optional[Sequence[Any]],
    ) -> List[Any]:
        """
        Determine the executable operation(s) that start the workflow.

        Preferred source:

            GraphPlanner.execution_order

        This is important because node insertion order must never be
        interpreted as execution order.
        """

        if execution_order:
            start_nodes = []

            for node_id in execution_order:

                if not graph.has_node(node_id):
                    continue

                if not self._is_executable_node(
                    graph,
                    node_id,
                ):
                    continue

                start_nodes.append(node_id)

            if start_nodes:
                first = start_nodes[0]

                # There can theoretically be multiple roots.
                # Preserve only roots that are executable and have no
                # executable predecessor in the workflow.
                roots = []

                for node_id in start_nodes:

                    if self._has_executable_predecessor(
                        graph,
                        node_id,
                    ):
                        continue

                    roots.append(node_id)

                return roots or [first]

        # --------------------------------------------------------
        # Fallback: derive roots from graph structure.
        #
        # This is only used when GraphPlanner's execution order is
        # unavailable.
        # --------------------------------------------------------

        roots = []

        for node_id in graph.nodes:

            if not self._is_executable_node(
                graph,
                node_id,
            ):
                continue

            if not self._has_executable_predecessor(
                graph,
                node_id,
            ):
                roots.append(node_id)

        return roots

    def _has_executable_predecessor(
        self,
        graph,
        node_id,
    ) -> bool:
        """
        Return True when another executable Operation points to
        this operation.

        Context entities do not count as workflow predecessors.
        """

        for predecessor in graph.predecessors(node_id):

            if self._is_executable_node(
                graph,
                predecessor,
            ):
                return True

        return False

    # ============================================================
    # Domain requirements
    # ============================================================

    def _get_node_requirements(
        self,
        graph,
        operation_id,
    ) -> List[Dict[str, Any]]:
        """
        Extract input requirements from the domain graph.

        Only incoming:

            OPERATION_REQUIRES
            OPERATION_ACCEPTS

        relationships are considered.

        The target entity remains metadata.

        It is NOT added to the workflow graph as an executable
        workflow step.
        """

        requirements = []

        if not graph.has_node(operation_id):
            return requirements

        for predecessor, _, edge in graph.in_edges(
            operation_id,
            data=True,
        ):

            # ----------------------------------------------------
            # Normally requirements are outgoing edges:
            #
            # Operation --REQUIRES--> Entity
            #
            # Therefore inspect outgoing edges below as well.
            # ----------------------------------------------------

            relation = self._relation(edge)

            if relation not in self.requirement_relationships:
                continue

            requirements.append(
                self._build_requirement(
                    graph,
                    operation_id,
                    predecessor,
                    relation,
                )
            )

        for _, target, edge in graph.out_edges(
            operation_id,
            data=True,
        ):

            relation = self._relation(edge)

            if relation not in self.requirement_relationships:
                continue

            requirements.append(
                self._build_requirement(
                    graph,
                    operation_id,
                    target,
                    relation,
                )
            )

        return self._deduplicate_requirements(
            requirements
        )

    def _build_requirement(
        self,
        graph,
        operation_id,
        target_id,
        relation,
    ) -> Dict[str, Any]:
        """
        Convert a graph requirement into an external-input
        description.
        """

        node_data = graph.nodes.get(
            target_id,
            {},
        )

        node_type = node_data.get(
            "node_type",
            node_data.get(
                "type",
                "unknown",
            ),
        )

        return {
            "name": node_data.get(
                "name",
                target_id,
            ),
            "node_id": target_id,
            "type": node_type,
            "relationship": relation,
            "source": "domain_graph",
            "required_by": self._node_name(
                graph,
                operation_id,
            ),
        }

    # ============================================================
    # Function refinement
    # ============================================================

    def _resolve_function_inputs(
        self,
        function,
        semantic_inputs: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Refine semantic domain requirements with the concrete inputs
        of the FIRST workflow function.

        The function may be:

            RegisteredFunction
            WorkflowFunctionDetails
            dict
            or another object exposing `inputs`.

        The method intentionally does NOT attempt to semantically
        match arbitrary function parameters to arbitrary domain
        entities using another ML model.

        It performs conservative name/type compatibility checks.

        If no reliable mapping can be established, the function
        parameter is still returned as an implementation input, but
        its semantic source remains unknown.
        """

        raw_inputs = self._extract_function_inputs(
            function
        )

        if not raw_inputs:
            return []

        results = []

        for raw_input in raw_inputs:

            parameter = self._normalize_function_input(
                raw_input
            )

            matched_requirement = (
                self._find_semantic_requirement_for_parameter(
                    parameter,
                    semantic_inputs,
                )
            )

            result = {
                "name": parameter["name"],
                "type": parameter["type"],
                "source": "start_function",
                "required": True,
            }

            if matched_requirement is not None:

                result["domain_concept"] = (
                    matched_requirement["name"]
                )

                result["domain_node_id"] = (
                    matched_requirement["node_id"]
                )

                result["relationship"] = (
                    matched_requirement["relationship"]
                )

            else:

                result["domain_concept"] = None
                result["domain_node_id"] = None
                result["relationship"] = None

            results.append(result)

        return results

    def _extract_function_inputs(
        self,
        function,
    ) -> List[Any]:
        """
        Extract function parameters from the supported function
        representations.
        """

        if function is None:
            return []

        if isinstance(function, dict):
            return function.get(
                "inputs",
                function.get(
                    "Inputs",
                    [],
                ),
            ) or []

        inputs = getattr(
            function,
            "inputs",
            None,
        )

        if inputs is not None:
            return list(inputs)

        return []

    @staticmethod
    def _normalize_function_input(
        value,
    ) -> Dict[str, str]:
        """
        Normalize common parameter representations.

        Supported examples:

            "accountId"

            "Guid accountId"

            {"name": "accountId", "type": "Guid"}
        """

        if isinstance(value, dict):

            return {
                "name": str(
                    value.get(
                        "name",
                        value.get(
                            "Name",
                            "",
                        ),
                    )
                ),
                "type": str(
                    value.get(
                        "type",
                        value.get(
                            "Type",
                            "",
                        ),
                    )
                ),
            }

        text = str(
            value or ""
        ).strip()

        if not text:
            return {
                "name": "",
                "type": "",
            }

        parts = text.split()

        if len(parts) >= 2:
            return {
                "type": " ".join(
                    parts[:-1]
                ),
                "name": parts[-1],
            }

        return {
            "name": text,
            "type": "",
        }

    def _find_semantic_requirement_for_parameter(
        self,
        parameter: Dict[str, str],
        requirements: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Conservatively associate a concrete function parameter with
        a semantic domain requirement.

        This deliberately avoids embeddings or an LLM because this
        resolver is not responsible for semantic planning.

        Example:

            parameter:
                accountId

            requirement:
                Account

        matches because the normalized tokens overlap.

        If no safe association exists, None is returned.
        """

        parameter_tokens = self._tokens(
            parameter.get(
                "name",
                "",
            )
        )

        parameter_type_tokens = self._tokens(
            parameter.get(
                "type",
                "",
            )
        )

        parameter_tokens.update(
            parameter_type_tokens
        )

        if not parameter_tokens:
            return None

        best = None
        best_score = 0

        for requirement in requirements:

            concept_tokens = self._tokens(
                requirement.get(
                    "name",
                    "",
                )
            )

            if not concept_tokens:
                continue

            overlap = len(
                parameter_tokens
                & concept_tokens
            )

            if overlap > best_score:

                best_score = overlap
                best = requirement

        if best_score <= 0:
            return None

        return best

    # ============================================================
    # Helpers
    # ============================================================

    @staticmethod
    def _is_executable_node(
        graph,
        node_id,
    ) -> bool:

        if not graph.has_node(node_id):
            return False

        data = graph.nodes[node_id]

        node_type = data.get(
            "node_type",
            data.get(
                "type"
            ),
        )

        return node_type in (
            "Operation",
            "operation",
        )

    @staticmethod
    def _relation(edge: Dict[str, Any]) -> str:
        """
        Support the repository's current edge representation while
        tolerating a few common alternatives.
        """

        return str(
            edge.get(
                "relation",
                edge.get(
                    "type",
                    edge.get(
                        "relationship",
                        "",
                    ),
                ),
            )
        )

    @staticmethod
    def _node_name(
        graph,
        node_id,
    ) -> str:

        data = graph.nodes.get(
            node_id,
            {},
        )

        return str(
            data.get(
                "name",
                node_id,
            )
        )

    @staticmethod
    def _tokens(
        value: str,
    ) -> Set[str]:
        """
        Normalize a name for conservative parameter/concept matching.

        Examples:

            accountId
            account_id
            Account

        all produce useful overlapping tokens.
        """

        import re

        value = str(
            value or ""
        )

        value = re.sub(
            r"([a-z0-9])([A-Z])",
            r"\1 \2",
            value,
        )

        value = value.replace(
            "_",
            " ",
        )

        value = value.replace(
            "-",
            " ",
        )

        tokens = re.findall(
            r"[a-zA-Z0-9]+",
            value.lower(),
        )

        return {
            token
            for token in tokens
            if token
        }

    @staticmethod
    def _deduplicate_requirements(
        requirements: Iterable[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        result = []
        seen = set()

        for requirement in requirements:

            key = (
                requirement.get(
                    "node_id"
                ),
                requirement.get(
                    "relationship"
                ),
                requirement.get(
                    "required_by"
                ),
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(requirement)

        return result


# ================================================================
# Convenience function
# ================================================================

def resolve_workflow_inputs(
    workflow_graph,
    execution_order=None,
    start_function=None,
):
    """
    Convenience wrapper around WorkflowInputResolver.
    """

    resolver = WorkflowInputResolver()

    return resolver.resolve(
        workflow_graph=workflow_graph,
        execution_order=execution_order,
        start_function=start_function,
    )