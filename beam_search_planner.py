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
    10. Expand selected operations with connected executable domain
        neighbors, marking those neighbors as inferred.

Important architectural distinction:

    Beam selection order
        !=
    Execution order

The order of ``selection`` is the order in which semantic prompt steps
were processed by beam search. It is NOT an execution-order claim.

Execution order comes from actual ordering relationships:

    - OPERATION_PRECEDES
    - PROMPT_DEPENDENCY
    - PROMPT_PRECEDES

OPERATION_INCLUDES is intentionally NOT treated as an execution-order
relationship.

It represents structural/domain composition:

    Register User
        |
        | OPERATION_INCLUDES
        v
    Verify User Exists

This means that Verify User Exists is part of Register User's domain
structure. It does NOT by itself establish:

    Register User -> Verify User Exists

as an execution sequence.

This distinction is important because prompt dependencies can establish
an explicit execution order that is different from the structural
"includes" relationship.

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
    in candidate evaluation.

No Dijkstra.
No shortest-path search.
No synthetic ordering from beam-selection adjacency.
"""


import re

import networkx as nx


class BeamSearchPlanner:

    _alignment_nlp = None

    # ============================================================
    # Relationship categories
    # ============================================================

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

    def __init__(
        self,
        beam_width=3,
        max_candidates_per_step=10,
    ):
        self.beam_width = beam_width
        self.max_candidates_per_step = max_candidates_per_step

    # ============================================================
    # Stage 8 - Beam Search
    # ============================================================

    def search(
        self,
        candidate_plan,
    ):
        """
        Perform global beam search over semantic-step candidates.

        Beam search does NOT simply select the highest-scoring
        candidate independently for every prompt step.

        Instead it evaluates combinations of candidates using:

            semantic similarity
            lexical similarity
            graph connectivity
            relationship compatibility
            rule constraints
            contradiction penalties

        The purpose is to find a globally coherent combination
        of executable domain operations.

        IMPORTANT:

        The order of ``selection`` is ONLY matching/processing order.

        It is never interpreted as execution order.
        """

        semantic_steps = candidate_plan["semantic_steps"]

        candidate_map = candidate_plan["candidate_map"]

        domain_graph = candidate_plan["domain_graph"]

        beams = [
            {
                "selection": [],
                "selected_ids": [],
                "score": 0.0,
                "constraint_violations": [],
            }
        ]
        unsupported_steps = []

        # --------------------------------------------------------
        # Process semantic steps.
        #
        # This is matching/processing order only.
        # It is NOT execution order.
        # --------------------------------------------------------

        for step_index, step in enumerate(semantic_steps):

            candidates = candidate_map.get(
                step_index,
                [],
            )

            # ----------------------------------------------------
            # Only executable Operation nodes are eligible.
            #
            # Account, Customer, Balance, Rule, Event, etc.
            # cannot become workflow execution steps.
            # ----------------------------------------------------

            candidates = [
                candidate
                for candidate in candidates
                if self._is_executable_candidate(
                    domain_graph,
                    candidate,
                )
            ]

            aligned_candidates = [
                candidate
                for candidate in candidates
                if self._name_aligns_with_prompt(
                    candidate.get("name", ""),
                    step.text,
                )
            ]

            candidates = aligned_candidates

            if not candidates:
                unsupported_steps.append({
                    "step_index": step_index,
                    "prompt_text": step.text,
                    "reason": "No executable domain operation matches the requested action.",
                })
                # Do not invent a replacement operation.
                continue

            candidates = candidates[
                : self.max_candidates_per_step
            ]

            new_beams = []

            for beam in beams:

                for candidate in candidates:

                    node_id = candidate["node_id"]

                    # ------------------------------------------------
                    # Do not select the same operation twice.
                    # ------------------------------------------------

                    if node_id in beam["selected_ids"]:
                        continue

                    candidate_score = float(
                        candidate.get(
                            "score",
                            0.0,
                        )
                    )

                    # ------------------------------------------------
                    # Relationship compatibility.
                    # ------------------------------------------------

                    relationship_score = (
                        self._relationship_compatibility(
                            domain_graph,
                            beam["selected_ids"],
                            node_id,
                        )
                    )

                    # ------------------------------------------------
                    # Graph connectivity.
                    # ------------------------------------------------

                    connectivity_score = (
                        self._connectivity_score(
                            domain_graph,
                            beam["selected_ids"],
                            node_id,
                        )
                    )

                    # ------------------------------------------------
                    # Rule constraints.
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
                        and candidate.get("source") == "direct"
                    ):
                        explicit_bonus = (
                            self.EXPLICIT_DIRECT_BONUS
                        )

                    name_alignment_bonus = 0.0
                    if (
                        candidate.get("source") == "direct"
                        and self._name_aligns_with_prompt(
                            candidate.get("name", ""),
                            step.text,
                        )
                    ):
                        name_alignment_bonus = (
                            self.DIRECT_NAME_ALIGNMENT_BONUS
                        )

                    # ------------------------------------------------
                    # Neighborhood inference penalty.
                    # ------------------------------------------------

                    inferred_penalty = 0.0

                    if candidate.get("source") == "neighborhood":
                        inferred_penalty = (
                            self.INFERRED_PENALTY
                        )

                    # ------------------------------------------------
                    # Disconnected candidate penalty.
                    # ------------------------------------------------

                    disconnected_penalty = 0.0

                    if (
                        beam["selected_ids"]
                        and connectivity_score <= 0.0
                    ):
                        disconnected_penalty = (
                            self.DISCONNECTED_PENALTY
                        )

                    # ------------------------------------------------
                    # Final candidate contribution.
                    # ------------------------------------------------

                    total_increment = (
                        candidate_score
                        + relationship_score
                        + connectivity_score
                        + explicit_bonus
                        + name_alignment_bonus
                        - inferred_penalty
                        - disconnected_penalty
                        - constraint_penalty
                    )

                    selection_item = {
                        "prompt_text": step.text,

                        "domain_node_id": node_id,

                        "domain_node_name": candidate.get(
                            "name",
                            node_id,
                        ),

                        "domain_node_type": candidate.get(
                            "node_type"
                        ),

                        "explicit": bool(
                            step.explicit
                        ),

                        "inferred": bool(
                            candidate.get(
                                "inferred",
                                not step.explicit,
                            )
                        ),

                        "source": candidate.get(
                            "source",
                            "direct",
                        ),

                        "semantic_score": candidate.get(
                            "semantic_score",
                            0.0,
                        ),

                        "lexical_score": candidate.get(
                            "lexical_score",
                            0.0,
                        ),

                        "candidate_score": candidate_score,

                        "relationship_score": relationship_score,

                        "connectivity_score": connectivity_score,

                        "constraint_penalty": constraint_penalty,

                        "constraint_violations": violations,
                    }

                    new_beams.append(
                        {
                            "selection": (
                                beam["selection"]
                                + [selection_item]
                            ),

                            "selected_ids": (
                                beam["selected_ids"]
                                + [node_id]
                            ),

                            "score": (
                                beam["score"]
                                + total_increment
                            ),

                            "constraint_violations": (
                                beam[
                                    "constraint_violations"
                                ]
                                + violations
                            ),
                        }
                    )

            if not new_beams:
                continue

            # ----------------------------------------------------
            # Deduplicate equivalent beam states.
            # ----------------------------------------------------

            new_beams = self._deduplicate_beams(
                new_beams
            )

            new_beams.sort(
                key=lambda beam: beam["score"],
                reverse=True,
            )

            beams = new_beams[
                : self.beam_width
            ]

        if not beams:
            return {
                "beam": [],
                "selection": [],
                "unsupported_steps": unsupported_steps,
            }

        return {
            "beam": beams,
            "selection": beams[0]["selection"],
            "unsupported_steps": unsupported_steps,
        }

    # ============================================================
    # Candidate filtering
    # ============================================================

    @classmethod
    def _name_aligns_with_prompt(cls, name, prompt):
        """Compare action lemmas using the NLP model, not a synonym table."""
        nlp = cls._load_alignment_nlp()
        if nlp is not None:
            name_doc, prompt_doc = nlp.pipe([str(name), str(prompt)])
            name_tokens = [
                token.lemma_.lower()
                for token in name_doc
                if not token.is_punct
            ]
            prompt_actions = {
                token.lemma_.lower()
                for token in prompt_doc
                if token.pos_ in {"VERB", "AUX"}
            }
            if name_tokens and prompt_actions:
                return name_tokens[0] in prompt_actions

        name_tokens = re.findall(r"[a-z0-9]+", str(name).lower())
        return bool(
            name_tokens
            and name_tokens[0] in cls._tokenize(prompt)
        )

    @classmethod
    def _load_alignment_nlp(cls):
        if cls._alignment_nlp is not None:
            return cls._alignment_nlp or None
        try:
            import spacy
            cls._alignment_nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            cls._alignment_nlp = False
        return cls._alignment_nlp or None

    @classmethod
    def _is_executable_candidate(
        cls,
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

        if node_type in cls.EXECUTABLE_NODE_TYPES:
            return True

        if node_type in cls.NON_EXECUTABLE_NODE_TYPES:
            return False

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

        IMPORTANT:

        Relationship compatibility does NOT mean that every
        relationship represents execution order.

        For example:

            Register User
                |
                | OPERATION_INCLUDES
                v
            Verify User Exists

        means that Verify User Exists is part of the structure
        of Register User.

        It does NOT necessarily mean:

            Register User -> Verify User Exists

        Therefore OPERATION_INCLUDES contributes a structural
        compatibility score but is not treated as an execution
        dependency.
        """

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
                    cls._edge_compatibility_score(
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

        # Explicit execution/dependency relationships are strongest.
        if relation in cls.ORDERING_RELATIONSHIPS:
            return cls.REQUIRED_RELATION_BONUS

        # OPERATION_INCLUDES is useful structural evidence,
        # but it must not be interpreted as execution order.
        if relation in cls.STRUCTURAL_RELATIONSHIPS:
            return cls.STRUCTURAL_RELATION_BONUS

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
        domain structure as already-selected operations.

        This is NOT execution ordering.

        Beam search uses graph connectivity to prefer coherent
        combinations.

        Execution order is determined later from actual ordering
        relationships.
        """

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

        # --------------------------------------------------------
        # Operation -> Operation
        # --------------------------------------------------------

        if (
            source_type in cls.EXECUTABLE_NODE_TYPES
            and target_type in cls.EXECUTABLE_NODE_TYPES
        ):

            # Explicit ordering relationship.
            if relation in cls.ORDERING_RELATIONSHIPS:
                return cls.CONNECTIVITY_BONUS

            # Structural operation relationship.
            #
            # This still tells beam search that the operations are
            # related, but it is deliberately weaker because it
            # does not establish execution order.
            if relation in cls.STRUCTURAL_RELATIONSHIPS:
                return (
                    cls.CONNECTIVITY_BONUS
                    * 0.60
                )

            return (
                cls.CONNECTIVITY_BONUS
                * 0.50
            )

        # --------------------------------------------------------
        # Contextual relationship.
        # --------------------------------------------------------

        if relation in cls.CONTEXT_RELATIONSHIPS:
            return cls.CONTEXT_RELATION_BONUS

        # --------------------------------------------------------
        # Rule relationships do not create workflow connectivity.
        # They are handled by _rule_constraint_penalty().
        # --------------------------------------------------------

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

        Rule nodes never become workflow steps.

        They are used only as constraints during candidate evaluation.

        Expected structure:

            Rule
              |
              | RULE_CONSTRAINS
              v
            Operation
        """

        if graph is None:
            return 0.0, []

        if not graph.has_node(operation_id):
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

            if relation not in cls.CONSTRAINT_RELATIONSHIPS:
                continue

            if not graph.has_node(rule_id):
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

            if not overlap:
                continue

            if cls._prompt_contradicts_rule(
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

        This is intentionally limited to explicit contradictions
        rather than pretending that arbitrary natural-language
        contradiction can be reliably detected by string matching.
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
        # Opposite state detection.
        # --------------------------------------------------------

        for (
            rule_token,
            opposite_tokens,
        ) in opposites.items():

            if rule_token not in rule_tokens:
                continue

            if prompt_tokens & opposite_tokens:
                return True

        # --------------------------------------------------------
        # Explicit negation detection.
        # --------------------------------------------------------

        if has_negation:

            shared = (
                prompt_tokens
                & rule_tokens
            )

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

        The ordering here is semantic/matching order.

        It is NOT execution order.
        """

        best = {}

        for beam in beams:

            key = tuple(
                beam["selected_ids"]
            )

            existing = best.get(
                key
            )

            if (
                existing is None
                or beam["score"]
                > existing["score"]
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

        Only selected executable Operation nodes are copied.

        Contextual entities such as:

            Account
            Customer
            Balance
            Rule
            Event

        are NOT copied.

        IMPORTANT:

        Not every relationship between two Operation nodes is an
        execution relationship.

        Only explicit ordering/dependency relationships are copied
        into the final workflow:

            OPERATION_PRECEDES
            PROMPT_DEPENDENCY
            PROMPT_PRECEDES

        Structural relationships such as:

            OPERATION_INCLUDES

        remain useful to beam search but are deliberately excluded
        from the execution graph.

        This prevents contradictions such as:

            Register User
                |
                | OPERATION_INCLUDES
                v
            Verify User Exists

        together with:

            Verify User Exists
                |
                | PROMPT_DEPENDENCY
                v
            Register User

        from being interpreted as a cycle.

        The first relationship describes domain composition.

        The second describes actual execution dependency.

        Beam-selection order is never used to create edges.
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

            if node_id not in selected_ids:
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
        # Expand the selected operations with executable neighbors from
        # the domain graph.  Candidate matching answers which operation
        # represents the prompt; the ontology answers which operations are
        # needed to perform it.
        # --------------------------------------------------------

        domain_graph = None

        if candidate_plan:
            domain_graph = candidate_plan.get(
                "domain_graph"
            )

        if domain_graph is not None:

            prompt_selected_ids = list(selected_ids)
            semantic_steps = (
                candidate_plan.get("semantic_steps", [])
                if candidate_plan
                else []
            )
            user_prompt = " ".join(
                step.text
                for step in semantic_steps
                if getattr(step, "text", "")
            )

            selected_prompt_text = {
                node_id: graph.nodes[node_id].get(
                    "prompt_text",
                    "",
                )
                for node_id in selected_ids
            }

            inferred_ids = set()
            expansion_queue = list(selected_ids)
            expanded_ids = set()

            while expansion_queue:
                selected_id = expansion_queue.pop(0)
                if selected_id in expanded_ids:
                    continue
                expanded_ids.add(selected_id)

                adjacent_edges = domain_graph.out_edges(
                    selected_id,
                    data=True,
                )

                for source, target, data in adjacent_edges:
                    relation = data.get("relation", "")
                    if relation not in self.INFERRED_OPERATION_RELATIONSHIPS:
                        continue

                    neighbor_id = (
                        target if source == selected_id else source
                    )
                    neighbor_data = domain_graph.nodes[neighbor_id]

                    if not self._is_executable_node_type(
                        neighbor_data.get("node_type")
                    ):
                        continue

                    if neighbor_id in selected_ids:
                        continue

                    if neighbor_id in inferred_ids:
                        continue

                    inferred_ids.add(neighbor_id)
                    expansion_queue.append(neighbor_id)
                    inferred_data = dict(neighbor_data)
                    inferred_data.update(
                        {
                            "prompt_text": selected_prompt_text.get(
                                selected_id,
                                "",
                            ),
                            "explicit": False,
                            "inferred": True,
                            "source": "domain_neighbor",
                        }
                    )
                    graph.add_node(
                        neighbor_id,
                        **inferred_data,
                    )

            # Complete only directed domain paths between prompt-selected
            # operations. Incoming edges are used here only as part of a
            # candidate path, never as generic parent expansion.
            for source_id, target_id in zip(
                prompt_selected_ids,
                prompt_selected_ids[1:],
            ):
                prompt = user_prompt or " ".join(
                    [
                        selected_prompt_text.get(source_id, source_id),
                        selected_prompt_text.get(target_id, target_id),
                    ]
                )
                path = self._most_contextual_path(
                    domain_graph,
                    source_id,
                    target_id,
                    prompt,
                )
                if path is None:
                    continue

                if len(path) <= 2:
                    continue

                path_edges = list(zip(path, path[1:]))
                if any(
                    domain_graph.edges[edge].get("relation")
                    not in self.INFERRED_OPERATION_RELATIONSHIPS
                    for edge in path_edges
                ):
                    continue

                for node_id in path[1:-1]:
                    node_data = domain_graph.nodes[node_id]
                    if not self._is_executable_node_type(
                        node_data.get("node_type")
                    ):
                        continue
                    if node_id in selected_ids:
                        continue

                    inferred_ids.add(node_id)
                    inferred_data = dict(node_data)
                    inferred_data.update(
                        {
                            "prompt_text": " -> ".join(
                                [
                                    selected_prompt_text.get(
                                        source_id,
                                        source_id,
                                    ),
                                    selected_prompt_text.get(
                                        target_id,
                                        target_id,
                                    ),
                                ]
                            ),
                            "explicit": False,
                            "inferred": True,
                            "source": "domain_path",
                        }
                    )
                    graph.add_node(
                        node_id,
                        **inferred_data,
                    )

            selected_ids.extend(
                node_id for node_id in inferred_ids
                if node_id not in selected_ids
            )

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

                # ------------------------------------------------
                # Only Operation -> Operation edges can ever become
                # workflow edges.
                # ------------------------------------------------

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

                if relation not in self.INFERRED_OPERATION_RELATIONSHIPS:
                    continue

                graph.add_edge(
                    source,
                    target,
                    **dict(data),
                )

        # --------------------------------------------------------
        # NO selection-order fallback.
        #
        # Never do:
        #
        #     selection[i] -> selection[i + 1]
        #
        # because selection order represents semantic matching
        # order, not execution order.
        #
        # If there is no explicit ordering relationship between two
        # selected operations, they remain unrelated in the
        # execution graph.
        # --------------------------------------------------------

        return graph

    @classmethod
    def _most_contextual_path(
        cls,
        graph,
        source,
        target,
        prompt,
        max_path_length=4,
    ):
        """Choose the domain path whose nodes best match the prompt."""

        try:
            paths = nx.all_simple_paths(
                graph,
                source=source,
                target=target,
                cutoff=max_path_length,
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        best_path = None
        best_score = float("-inf")

        for path in paths:
            if not cls._is_contextual_operation_path(graph, path):
                continue

            score = cls._contextual_path_score(
                graph,
                path,
                prompt,
            )

            if score > best_score:
                best_score = score
                best_path = path

        return best_path

    @classmethod
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

    @classmethod
    def _contextual_path_score(cls, graph, path, prompt):
        prompt_tokens = cls._tokenize(prompt)
        score = 0.0

        for node_id in path[1:-1]:
            node_data = graph.nodes[node_id]
            node_tokens = cls._tokenize(
                cls._node_text(node_data)
            )

            lexical_score = (
                len(prompt_tokens & node_tokens)
                / len(prompt_tokens | node_tokens)
                if prompt_tokens and node_tokens
                else 0.0
            )

            score += (
                0.70 * lexical_score
                + 0.30 * float(
                    node_data.get("semantic_score", 0.0)
                )
            )

        for source, target in zip(path, path[1:]):
            score += cls._edge_compatibility_score(
                graph.edges[source, target]
            )

        return score

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