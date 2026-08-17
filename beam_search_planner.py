"""
Stage 8 - Beam Search Planner.

Responsibilities:

    1. Select executable domain operations for semantic prompt steps.
    2. Evaluate candidate combinations rather than selecting each
       candidate independently.
    3. Score candidates using:
         - semantic similarity
         - lexical similarity
         - graph connectivity
         - relationship compatibility
         - contradiction / constraint penalties
    4. Prefer combinations that form a coherent domain workflow.
    5. Never turn contextual entities into workflow execution nodes.
    6. Respect RULE_CONSTRAINS as domain constraints.
    7. Preserve real directed domain relationships.
    8. Preserve PROMPT_DEPENDENCY relationships created upstream.
    9. Never interpret beam-selection order as execution order.
    10. Expand only selected executable operations into the workflow graph.

Important architectural distinction:

    Beam selection order
        !=
    Execution order

The order of ``selection`` is the order in which semantic prompt steps
were processed by beam search. It is NOT an execution-order claim.

Execution order comes from actual directed graph relationships:

    - OPERATION_PRECEDES
    - OPERATION_INCLUDES
    - PROMPT_DEPENDENCY
    - other explicitly ordering relationships

Context relationships such as:

    - OPERATION_REQUIRES
    - OPERATION_CREATES
    - OPERATION_MODIFIES
    - OPERATION_VALIDATES
    - OPERATION_ACCEPTS
    - OPERATION_PRODUCES
    - ENTITY_OWNS

may help determine whether a candidate is semantically compatible, but
their endpoint entities must NOT become workflow execution nodes.

RULE_CONSTRAINS is special:

    Rule nodes are not executable workflow nodes.

    However, rules constrain operations and therefore MUST participate
    in candidate evaluation. A candidate operation that conflicts with
    an explicitly referenced domain rule receives a strong penalty and
    may be rejected.

No Dijkstra.
No shortest-path search.
No synthetic ordering from beam-selection adjacency.
"""

import re

import networkx as nx


class BeamSearchPlanner:

    # ============================================================
    # Relationship categories
    # ============================================================

    # Relationships that directly connect executable operations or
    # describe execution-related structure.
    #
    # These are allowed to contribute strongly to candidate
    # compatibility and workflow connectivity.
    EXECUTION_RELATIONSHIPS = {
        "COMPONENT_EXECUTES",
        "ACTOR_PERFORMS",
        "ACTOR_REQUESTS",
        "OPERATION_INCLUDES",
        "OPERATION_PRECEDES",
        "EVENT_TRIGGERS",
        "PROMPT_DEPENDENCY",
        "PROMPT_PRECEDES",
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

    # Beam scoring weights.
    #
    # Candidate score is already a semantic/lexical score produced
    # upstream. These values determine how much graph structure
    # influences the global combination.
    CONNECTIVITY_BONUS = 0.20
    REQUIRED_RELATION_BONUS = 0.25
    POSSIBLE_RELATION_BONUS = 0.08
    CONTEXT_RELATION_BONUS = 0.03

    # Strong penalty for a candidate that conflicts with a domain rule.
    RULE_CONTRADICTION_PENALTY = 0.75

    # Penalty for choosing a candidate that has no relationship to
    # anything already selected.
    DISCONNECTED_PENALTY = 0.15

    # Small penalty for inferred/neighborhood candidates.
    INFERRED_PENALTY = 0.03

    # Explicit direct prompt matches receive a small preference.
    EXPLICIT_DIRECT_BONUS = 0.15

    def __init__(
        self,
        beam_width=3,
        max_candidates_per_step=10,
    ):
        self.beam_width = beam_width
        self.max_candidates_per_step = (
            max_candidates_per_step
        )

    # ============================================================
    # Stage 8 - Beam Search
    # ============================================================

    def search(
        self,
        candidate_plan,
    ):
        """
        Perform global beam search over semantic-step candidates.

        The important point is that beam search does NOT merely ask:

            "What is the best node for this prompt step?"

        Instead it asks:

            "What combination of candidate operations gives the
             strongest overall workflow?"

        Each candidate therefore receives:

            semantic/lexical score
                +
            graph connectivity
                +
            relationship compatibility
                -
            disconnectedness
                -
            rule contradictions
                +
            explicit/direct-match preference

        Beam search retains the best globally coherent combinations.
        """

        semantic_steps = candidate_plan[
            "semantic_steps"
        ]

        candidate_map = candidate_plan[
            "candidate_map"
        ]

        domain_graph = candidate_plan[
            "domain_graph"
        ]

        beams = [
            {
                "selection": [],
                "selected_ids": [],
                "score": 0.0,
                "constraint_violations": [],
            }
        ]

        # --------------------------------------------------------
        # Process semantic steps.
        #
        # This order is MATCHING/PROCESSING order only.
        # It is never used as execution order.
        # --------------------------------------------------------

        for step_index, step in enumerate(
            semantic_steps
        ):
            candidates = candidate_map.get(
                step_index,
                [],
            )

            # ----------------------------------------------------
            # Critical restriction:
            #
            # Only executable Operation nodes are eligible for
            # workflow selection.
            #
            # Context entities such as Account, Customer, Balance,
            # Rule, etc. may remain in the contextual domain graph,
            # but they cannot become workflow steps.
            # ----------------------------------------------------

            candidates = [
                candidate
                for candidate in candidates
                if self._is_executable_candidate(
                    domain_graph,
                    candidate,
                )
            ]

            if not candidates:
                # Do not invent a function/node just to keep the
                # beam alive.
                #
                # The semantic step remains unresolved and should
                # ultimately be surfaced by downstream validation.
                continue

            candidates = candidates[
                : self.max_candidates_per_step
            ]

            new_beams = []

            for beam in beams:

                for candidate in candidates:

                    node_id = candidate[
                        "node_id"
                    ]

                    # ------------------------------------------------
                    # Never select the same operation twice for two
                    # semantic steps.
                    # ------------------------------------------------

                    if node_id in beam[
                        "selected_ids"
                    ]:
                        continue

                    candidate_score = float(
                        candidate.get(
                            "score",
                            0.0,
                        )
                    )

                    # ------------------------------------------------
                    # Graph compatibility.
                    # ------------------------------------------------

                    relationship_score = (
                        self._relationship_compatibility(
                            domain_graph,
                            beam[
                                "selected_ids"
                            ],
                            node_id,
                        )
                    )

                    # ------------------------------------------------
                    # Connectivity.
                    #
                    # This is different from relationship scoring.
                    #
                    # We want the beam to prefer combinations where
                    # selected operations actually participate in the
                    # same domain structure.
                    # ------------------------------------------------

                    connectivity_score = (
                        self._connectivity_score(
                            domain_graph,
                            beam[
                                "selected_ids"
                            ],
                            node_id,
                        )
                    )

                    # ------------------------------------------------
                    # Rule constraints.
                    #
                    # RULE_CONSTRAINS does NOT add a Rule node to the
                    # workflow.
                    #
                    # Instead, the rule is inspected to determine
                    # whether this operation is compatible with the
                    # prompt.
                    # ------------------------------------------------

                    constraint_penalty, violations = (
                        self._rule_constraint_penalty(
                            domain_graph,
                            step.text,
                            node_id,
                        )
                    )

                    # ------------------------------------------------
                    # Explicit direct match preference.
                    # ------------------------------------------------

                    explicit_bonus = 0.0

                    if (
                        bool(step.explicit)
                        and candidate.get(
                            "source"
                        )
                        == "direct"
                    ):
                        explicit_bonus = (
                            self.EXPLICIT_DIRECT_BONUS
                        )

                    # ------------------------------------------------
                    # Neighborhood inference penalty.
                    # ------------------------------------------------

                    inferred_penalty = 0.0

                    if candidate.get(
                        "source"
                    ) == "neighborhood":
                        inferred_penalty = (
                            self.INFERRED_PENALTY
                        )

                    # ------------------------------------------------
                    # Disconnected candidate penalty.
                    #
                    # The candidate is not automatically rejected,
                    # because an operation may legitimately be an
                    # independent branch. But an unrelated operation
                    # should lose against a connected alternative.
                    # ------------------------------------------------

                    disconnected_penalty = 0.0

                    if (
                        beam["selected_ids"]
                        and connectivity_score
                        <= 0.0
                    ):
                        disconnected_penalty = (
                            self.DISCONNECTED_PENALTY
                        )

                    total_increment = (
                        candidate_score
                        + relationship_score
                        + connectivity_score
                        + explicit_bonus
                        - inferred_penalty
                        - disconnected_penalty
                        - constraint_penalty
                    )

                    selection_item = {
                        "prompt_text": step.text,

                        "domain_node_id": node_id,

                        "domain_node_name":
                            candidate.get(
                                "name",
                                node_id,
                            ),

                        "domain_node_type":
                            candidate.get(
                                "node_type"
                            ),

                        "explicit":
                            bool(
                                step.explicit
                            ),

                        "inferred":
                            bool(
                                candidate.get(
                                    "inferred",
                                    not step.explicit,
                                )
                            ),

                        "source":
                            candidate.get(
                                "source",
                                "direct",
                            ),

                        "semantic_score":
                            candidate.get(
                                "semantic_score",
                                0.0,
                            ),

                        "lexical_score":
                            candidate.get(
                                "lexical_score",
                                0.0,
                            ),

                        "candidate_score":
                            candidate_score,

                        "relationship_score":
                            relationship_score,

                        "connectivity_score":
                            connectivity_score,

                        "constraint_penalty":
                            constraint_penalty,

                        "constraint_violations":
                            violations,
                    }

                    new_beams.append(
                        {
                            "selection":
                                beam[
                                    "selection"
                                ]
                                + [
                                    selection_item
                                ],

                            "selected_ids":
                                beam[
                                    "selected_ids"
                                ]
                                + [
                                    node_id
                                ],

                            "score":
                                beam[
                                    "score"
                                ]
                                + total_increment,

                            "constraint_violations":
                                (
                                    beam[
                                        "constraint_violations"
                                    ]
                                    + violations
                                ),
                        }
                    )

            # ----------------------------------------------------
            # If every candidate was already selected, do not
            # silently substitute another operation.
            #
            # The semantic step is unresolved rather than being
            # assigned an arbitrary operation.
            # ----------------------------------------------------

            if not new_beams:
                continue

            # ----------------------------------------------------
            # Deduplicate equivalent beam states.
            # ----------------------------------------------------

            new_beams = (
                self._deduplicate_beams(
                    new_beams
                )
            )

            new_beams.sort(
                key=lambda beam:
                    beam["score"],
                reverse=True,
            )

            beams = new_beams[
                : self.beam_width
            ]

        if not beams:
            return {
                "beam": [],
                "selection": [],
            }

        return {
            "beam": beams,
            "selection": beams[0][
                "selection"
            ],
        }

    # ============================================================
    # Candidate filtering
    # ============================================================

    @classmethod
    def _is_executable_candidate(
        cls,
        graph,
        candidate,
    ):
        """
        Return True only for actual Operation nodes.

        This prevents contextual concepts such as:

            Account
            Customer
            Balance
            Rule
            Event

        from becoming workflow execution steps.
        """

        node_id = candidate.get(
            "node_id"
        )

        if graph is None:
            return False

        if not graph.has_node(node_id):
            return False

        node_data = graph.nodes[
            node_id
        ]

        node_type = (
            candidate.get(
                "node_type"
            )
            or node_data.get(
                "node_type"
            )
        )

        if node_type in cls.EXECUTABLE_NODE_TYPES:
            return True

        if node_type in cls.NON_EXECUTABLE_NODE_TYPES:
            return False

        # Conservative fallback:
        #
        # If the node type is unknown, do not turn it into a workflow
        # step merely because it scored well.
        return False

    # ============================================================
    # Relationship compatibility
    # ============================================================

    @classmethod
    def _relationship_compatibility(
        cls,
        graph,
        selected_ids,
        candidate_id,
    ):
        """
        Score how well a candidate operation relates to already
        selected operations.

        Direction is preserved.

        We do NOT pretend that every relationship means execution
        order.

        For example:

            Update Account
                |
                | OPERATION_REQUIRES
                v
              Account

        provides contextual compatibility, but Account is not an
        executable workflow step.

        Conversely:

            Validate User
                |
                | OPERATION_PRECEDES
                v
            Create User

        is a genuine execution relationship.
        """

        if not selected_ids:
            return 0.0

        if not graph.has_node(
            candidate_id
        ):
            return 0.0

        best_score = 0.0

        for selected_id in selected_ids:

            if not graph.has_node(
                selected_id
            ):
                continue

            # selected -> candidate
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
                    cls._edge_compatibility_score(
                        edge
                    ),
                )

            # candidate -> selected
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
                    cls._edge_compatibility_score(
                        edge
                    ),
                )

        return best_score

    @classmethod
    def _edge_compatibility_score(
        cls,
        edge,
    ):
        relation = edge.get(
            "relation",
            "",
        )

        classification = edge.get(
            "classification"
        )

        if relation in {
            "OPERATION_PRECEDES",
            "PROMPT_DEPENDENCY",
            "PROMPT_PRECEDES",
        }:
            return cls.REQUIRED_RELATION_BONUS

        if relation == "OPERATION_INCLUDES":
            return cls.REQUIRED_RELATION_BONUS

        if classification == "REQUIRED":
            return cls.REQUIRED_RELATION_BONUS

        if classification == "POSSIBLE":
            return cls.POSSIBLE_RELATION_BONUS

        if classification == "CONTEXT":
            return cls.CONTEXT_RELATION_BONUS

        return 0.0

    # ============================================================
    # Graph connectivity
    # ============================================================

    @classmethod
    def _connectivity_score(
        cls,
        graph,
        selected_ids,
        candidate_id,
    ):
        """
        Measure whether the candidate participates in the same
        domain structure as selected operations.

        This is deliberately local.

        Beam search is NOT performing shortest-path routing.

        A candidate receives a positive score when there is a direct
        domain relationship to another selected operation.

        Context relationships can provide a small compatibility
        signal, but direct operation-to-operation relationships are
        preferred.
        """

        if not selected_ids:
            return 0.0

        if not graph.has_node(
            candidate_id
        ):
            return 0.0

        best = 0.0

        for selected_id in selected_ids:

            if not graph.has_node(
                selected_id
            ):
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
                    cls._connectivity_edge_score(
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
                    cls._connectivity_edge_score(
                        graph,
                        candidate_id,
                        selected_id,
                        edge,
                    ),
                )

        return best

    @classmethod
    def _connectivity_edge_score(
        cls,
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

        # Strongest case:
        #
        # Operation -> Operation
        #
        # This means the candidate actually participates in a
        # relationship with another executable operation.
        if (
            source_type in cls.EXECUTABLE_NODE_TYPES
            and target_type in cls.EXECUTABLE_NODE_TYPES
        ):
            if relation in {
                "OPERATION_PRECEDES",
                "OPERATION_INCLUDES",
                "PROMPT_DEPENDENCY",
                "PROMPT_PRECEDES",
            }:
                return cls.CONNECTIVITY_BONUS

            return (
                cls.CONNECTIVITY_BONUS
                * 0.50
            )

        # Operation -> Entity or Rule -> Operation is contextual,
        # not direct workflow connectivity.
        #
        # It can still provide a very small compatibility signal.
        if relation in cls.CONTEXT_RELATIONSHIPS:
            return (
                cls.CONTEXT_RELATION_BONUS
            )

        if relation in cls.CONSTRAINT_RELATIONSHIPS:
            return 0.0

        return 0.0

    # ============================================================
    # RULE_CONSTRAINS
    # ============================================================

    @classmethod
    def _rule_constraint_penalty(
        cls,
        graph,
        prompt_text,
        operation_id,
    ):
        """
        Check domain rules that constrain the candidate operation.

        Important:

            Rule != workflow step.

        A Rule node is only used as validation/context.

        Expected graph structure:

            Rule
              |
              | RULE_CONSTRAINS
              v
            Operation

        The method performs two levels of checking:

        1. Structural check:
           Does the operation have domain rules attached to it?

        2. Prompt-rule compatibility check:
           Does the prompt explicitly reference terminology that
           conflicts with the rule?

        The implementation is intentionally conservative. A rule
        should not invalidate a candidate merely because a rule
        exists. It should only penalize a candidate when there is
        evidence that the prompt is asking for a state/action that
        conflicts with the rule.
        """

        if graph is None:
            return 0.0, []

        if not graph.has_node(
            operation_id
        ):
            return 0.0, []

        prompt_tokens = cls._tokenize(
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

            if relation not in (
                cls.CONSTRAINT_RELATIONSHIPS
            ):
                continue

            if not graph.has_node(
                rule_id
            ):
                continue

            rule_data = graph.nodes[
                rule_id
            ]

            rule_text = cls._node_text(
                rule_data
            )

            rule_tokens = cls._tokenize(
                rule_text
            )

            if not rule_tokens:
                continue

            overlap = (
                prompt_tokens
                & rule_tokens
            )

            # ----------------------------------------------------
            # If the prompt and rule share no meaningful terms,
            # there is not enough evidence to call it a violation.
            # ----------------------------------------------------

            if not overlap:
                continue

            # ----------------------------------------------------
            # Detect explicit negation / contradiction language.
            #
            # Examples:
            #
            #   "without active account"
            #   "inactive account"
            #   "do not validate"
            #   "ignore the account status"
            #
            # This is deliberately conservative.
            # ----------------------------------------------------

            if cls._prompt_contradicts_rule(
                prompt_text,
                rule_text,
            ):
                violations.append(
                    {
                        "rule_node_id":
                            rule_id,

                        "rule_name":
                            rule_data.get(
                                "name",
                                rule_id,
                            ),

                        "operation_id":
                            operation_id,

                        "relation":
                            relation,

                        "overlap_terms":
                            sorted(
                                overlap
                            ),

                        "rule_text":
                            rule_text,

                        "reason":
                            "Prompt appears to "
                            "contradict a domain "
                            "rule constraining "
                            "this operation.",
                    }
                )

        if not violations:
            return 0.0, []

        # --------------------------------------------------------
        # Strong penalty.
        #
        # The candidate remains visible to beam search so debugging
        # can show WHY it lost. But a contradictory candidate should
        # normally lose against a valid alternative.
        # --------------------------------------------------------

        penalty = (
            cls.RULE_CONTRADICTION_PENALTY
            * len(violations)
        )

        return penalty, violations

    @staticmethod
    def _node_text(
        node_data,
    ):
        parts = [
            node_data.get(
                "name",
                "",
            ),
            node_data.get(
                "description",
                "",
            ),
        ]

        aliases = node_data.get(
            "aliases",
            [],
        )

        if aliases:
            parts.extend(
                aliases
            )

        return " ".join(
            str(part)
            for part in parts
            if part
        )

    @staticmethod
    def _tokenize(
        text,
    ):
        if not text:
            return set()

        return set(
            re.findall(
                r"[a-z0-9]+",
                str(text).lower(),
            )
        )

    @staticmethod
    def _prompt_contradicts_rule(
        prompt,
        rule,
    ):
        """
        Conservative contradiction detector.

        This is not intended to replace a dedicated semantic
        contradiction model.

        It catches explicit negation patterns around shared concepts.
        """

        prompt_normalized = (
            str(prompt)
            .lower()
            .strip()
        )

        rule_normalized = (
            str(rule)
            .lower()
            .strip()
        )

        # --------------------------------------------------------
        # Explicit negation markers.
        # --------------------------------------------------------

        negation_markers = (
            "not ",
            "do not ",
            "don't ",
            "without ",
            "never ",
            "cannot ",
            "can't ",
            "must not ",
            "should not ",
            "shouldn't ",
            "ignore ",
            "bypass ",
            "skip ",
        )

        has_negation = any(
            marker in prompt_normalized
            for marker in negation_markers
        )

        # --------------------------------------------------------
        # Common opposing state patterns.
        #
        # These are useful for rules such as:
        #
        #   "account must be active"
        #
        # against:
        #
        #   "transfer from inactive account"
        #
        # This is intentionally small and explicit rather than
        # pretending arbitrary natural-language contradiction can
        # be solved reliably with string matching.
        # --------------------------------------------------------

        opposites = {
            "active": {
                "inactive",
                "disabled",
                "blocked",
                "deactivated",
            },
            "inactive": {
                "active",
                "enabled",
                "activated",
            },
            "enabled": {
                "disabled",
                "inactive",
                "blocked",
            },
            "disabled": {
                "enabled",
                "active",
            },
            "valid": {
                "invalid",
                "unverified",
            },
            "invalid": {
                "valid",
                "verified",
            },
            "verified": {
                "unverified",
                "invalid",
            },
            "approved": {
                "rejected",
                "denied",
            },
            "rejected": {
                "approved",
            },
            "allowed": {
                "forbidden",
                "blocked",
                "disallowed",
            },
            "forbidden": {
                "allowed",
                "permitted",
            },
        }

        prompt_tokens = BeamSearchPlanner._tokenize(
            prompt_normalized
        )

        rule_tokens = BeamSearchPlanner._tokenize(
            rule_normalized
        )

        # --------------------------------------------------------
        # Check explicit opposite states.
        # --------------------------------------------------------

        for rule_token, opposite_tokens in (
            opposites.items()
        ):
            if rule_token not in rule_tokens:
                continue

            if prompt_tokens & opposite_tokens:
                return True

        # --------------------------------------------------------
        # If the prompt explicitly negates something that appears
        # in the rule, treat it as a contradiction.
        # --------------------------------------------------------

        if has_negation:
            shared = (
                prompt_tokens
                & rule_tokens
            )

            # Remove extremely generic words.
            generic = {
                "the",
                "a",
                "an",
                "must",
                "should",
                "be",
                "is",
                "are",
                "to",
                "for",
                "and",
                "or",
                "of",
                "on",
                "in",
            }

            meaningful_shared = (
                shared - generic
            )

            if meaningful_shared:
                return True

        return False

    # ============================================================
    # Beam deduplication
    # ============================================================

    @staticmethod
    def _deduplicate_beams(
        beams,
    ):
        """
        Keep only the highest-scoring beam for an identical selected
        operation combination.

        The order here is the semantic/matching order, NOT execution
        order.
        """

        best = {}

        for beam in beams:

            key = tuple(
                beam[
                    "selected_ids"
                ]
            )

            existing = best.get(
                key
            )

            if (
                existing is None
                or beam[
                    "score"
                ]
                > existing[
                    "score"
                ]
            ):
                best[key] = beam

        return list(
            best.values()
        )

    # ============================================================
    # Stage 8 -> Stage 9
    # ============================================================

    def expand(
        self,
        search_result,
        candidate_plan=None,
    ):
        """
        Convert the selected beam into the workflow graph.

        Only selected executable Operation nodes are copied into the
        workflow graph.

        Contextual entities such as:

            Account
            Customer
            Balance
            Rule

        are NOT copied as workflow nodes merely because they are
        connected to an operation.

        Their relationships can still be used during candidate
        evaluation.

        The resulting graph contains only relationships between
        selected executable operations.

        No shortest-path search is performed.

        No selection-order edges are generated.
        """

        if not search_result:
            raise RuntimeError(
                "BeamSearchPlanner.expand() received "
                "no search result."
            )

        selection = search_result.get(
            "selection",
            [],
        )

        if not selection:
            raise RuntimeError(
                "Beam search produced no "
                "selected workflow nodes."
            )

        graph = nx.DiGraph()

        selected_ids = []

        # --------------------------------------------------------
        # Add selected executable nodes only.
        # --------------------------------------------------------

        for item in selection:

            node_id = item[
                "domain_node_id"
            ]

            if not self._is_executable_selection_item(
                item
            ):
                continue

            selected_ids.append(
                node_id
            )

            if graph.has_node(
                node_id
            ):
                continue

            graph.add_node(
                node_id,
                name=item.get(
                    "domain_node_name",
                    node_id,
                ),
                node_type=item.get(
                    "domain_node_type",
                    "Operation",
                ),
                prompt_text=item.get(
                    "prompt_text",
                    "",
                ),
                explicit=item.get(
                    "explicit",
                    False,
                ),
                inferred=item.get(
                    "inferred",
                    False,
                ),
                source=item.get(
                    "source",
                    "direct",
                ),
                semantic_score=item.get(
                    "semantic_score",
                    0.0,
                ),
                lexical_score=item.get(
                    "lexical_score",
                    0.0,
                ),
                candidate_score=item.get(
                    "candidate_score",
                    0.0,
                ),
                relationship_score=item.get(
                    "relationship_score",
                    0.0,
                ),
                connectivity_score=item.get(
                    "connectivity_score",
                    0.0,
                ),
                constraint_penalty=item.get(
                    "constraint_penalty",
                    0.0,
                ),
                constraint_violations=item.get(
                    "constraint_violations",
                    [],
                ),
            )

        # --------------------------------------------------------
        # Copy ONLY direct relationships between selected
        # executable operations.
        #
        # This is crucial.
        #
        # Example:
        #
        #     Update Account
        #          |
        #          | OPERATION_REQUIRES
        #          v
        #        Account
        #
        # Account is NOT copied into the workflow graph.
        #
        # But:
        #
        #     Validate User
        #          |
        #          | OPERATION_PRECEDES
        #          v
        #     Create User
        #
        # is copied because both endpoints are executable
        # operations.
        # --------------------------------------------------------

        domain_graph = None

        if candidate_plan:
            domain_graph = candidate_plan.get(
                "domain_graph"
            )

        if domain_graph is not None:

            for (
                source,
                target,
                data,
            ) in domain_graph.edges(
                data=True
            ):

                if (
                    source not in selected_ids
                    or target not in selected_ids
                ):
                    continue

                source_type = (
                    domain_graph.nodes[
                        source
                    ].get(
                        "node_type"
                    )
                )

                target_type = (
                    domain_graph.nodes[
                        target
                    ].get(
                        "node_type"
                    )
                )

                # Only Operation -> Operation relationships can
                # become workflow edges.
                if not (
                    self._is_executable_node_type(
                        source_type
                    )
                    and
                    self._is_executable_node_type(
                        target_type
                    )
                ):
                    continue

                relation = data.get(
                    "relation",
                    "",
                )

                # Context and constraint relationships do not become
                # workflow edges.
                #
                # RULE_CONSTRAINS is intentionally excluded here.
                # It was already used during candidate validation.
                if relation in (
                    self.CONTEXT_RELATIONSHIPS
                    | self.CONSTRAINT_RELATIONSHIPS
                ):
                    continue

                graph.add_edge(
                    source,
                    target,
                    **dict(data),
                )

        # --------------------------------------------------------
        # IMPORTANT:
        #
        # There is deliberately NO:
        #
        #     selection[i] -> selection[i + 1]
        #
        # fallback.
        #
        # Beam selection order represents matching/processing order.
        # It does NOT represent execution order.
        #
        # Therefore, if:
        #
        #     selection =
        #         [
        #             Create Account,
        #             Validate User,
        #             Create User
        #         ]
        #
        # we must NOT infer:
        #
        #     Create Account -> Validate User
        #
        # just because Create Account happened to appear first.
        #
        # If the actual directed graph says:
        #
        #     Validate User -> Create User
        #     Create User  -> Create Account
        #
        # those are the relationships that determine execution.
        #
        # A disconnected selected operation remains a node without
        # an invented ordering edge.
        # --------------------------------------------------------

        return graph

    # ============================================================
    # Helpers
    # ============================================================

    @classmethod
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

    @classmethod
    def _is_executable_node_type(
        cls,
        node_type,
    ):
        return node_type in (
            cls.EXECUTABLE_NODE_TYPES
        )