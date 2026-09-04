from helpers.beam_search_utils import (
    _is_executable_candidate,
    EXPLICIT_DIRECT_BONUS,
    INFERRED_PENALTY,
    DISCONNECTED_PENALTY,
    _relationship_compatibility,
    _connectivity_score,
    _rule_constraint_penalty,
    _is_executable_node_type,
    _infer_operations_from_rules,
)


class BeamSearchWorkflowSelector:
    """
    Select the best domain operation for each semantic prompt step.

    Important design rule:

        Semantic step N -> candidates belonging to semantic step N

    Candidates from another semantic step must never be used to
    satisfy the current step.

    This prevents errors such as:

        "notify the customer"
            ->
        "update_user"

    merely because update_user happened to have the highest
    embedding similarity among unrelated operations.
    """

    # ---------------------------------------------------------
    # Minimum evidence required before accepting a candidate.
    #
    # These are deliberately conservative. A workflow operation
    # should not be selected merely because it is the closest
    # available operation.
    # ---------------------------------------------------------

    MIN_DIRECT_SCORE = 0.18
    MIN_INFERRED_SCORE = 0.30

    def __init__(
        self,
        beam_width=3,
        max_candidates_per_step=10,
    ):
        self.beam_width = beam_width
        self.max_candidates_per_step = (
            max_candidates_per_step
        )

    # =========================================================
    # Search
    # =========================================================

    def search(
        self,
        candidate_plan,
    ):
        semantic_steps = candidate_plan[
            "semantic_steps"
        ]

        candidate_map = candidate_plan[
            "candidate_map"
        ]

        prompt_domain_subgraph = candidate_plan[
            "prompt_domain_subgraph"
        ]

        beams = [
            {
                "selection": [],
                "selected_ids": [],
                "score": 0.0,
                "constraint_violations": [],
            }
        ]

        unsupported_steps = []

        # =====================================================
        # Process every semantic step independently.
        # =====================================================

        for step_index, step in enumerate(
            semantic_steps
        ):

            entry = candidate_map.get(
                step_index
            )

            # -------------------------------------------------
            # IMPORTANT:
            #
            # PromptSubGraphBuilder stores:
            #
            # candidate_map[index] = {
            #     "step_index": index,
            #     "step_text": ...,
            #     "step_embedding": ...,
            #     "candidates": [...]
            # }
            #
            # The old planner treated the dictionary itself as
            # a candidate list.
            # -------------------------------------------------

            if isinstance(entry, dict):
                candidates = entry.get(
                    "candidates",
                    [],
                )
            else:
                # Backward compatibility with the older format.
                candidates = entry or []

            # Make a copy so sorting/filtering does not mutate
            # PromptSubGraphBuilder's candidate map.
            candidates = list(
                candidates
            )

            # -------------------------------------------------
            # Candidates must be executable domain operations.
            # -------------------------------------------------

            candidates = [
                candidate
                for candidate in candidates
                if _is_executable_candidate(
                    prompt_domain_subgraph,
                    candidate,
                )
            ]

            # -------------------------------------------------
            # Make sure candidates actually belong to THIS
            # semantic step.
            # -------------------------------------------------

            candidates = [
                candidate
                for candidate in candidates
                if self._candidate_belongs_to_step(
                    candidate,
                    step_index,
                )
            ]

            # -------------------------------------------------
            # Reject weak candidates rather than forcing a
            # semantically unrelated domain operation.
            # -------------------------------------------------

            candidates = [
                candidate
                for candidate in candidates
                if self._is_acceptable_candidate(
                    candidate
                )
            ]

            if not candidates:
                unsupported_steps.append(
                    {
                        "step_index": step_index,
                        "prompt_text": step.text,
                        "reason": (
                            "No sufficiently strong executable "
                            "domain operation matches the "
                            "requested semantic step."
                        ),
                    }
                )

                continue

            # -------------------------------------------------
            # Keep only the strongest candidates.
            # -------------------------------------------------

            candidates.sort(
                key=lambda candidate: (
                    candidate.get(
                        "lexical_score",
                        0.0,
                    ) >= 0.90,

                    candidate.get(
                        "source"
                    ) == "direct",

                    candidate.get(
                        "score",
                        0.0,
                    ),
                ),
                reverse=True,
            )

            candidates = candidates[
                : self.max_candidates_per_step
            ]

            new_beams = []

            # =================================================
            # Expand beam
            # =================================================

            for beam in beams:

                for candidate in candidates:

                    node_id = candidate.get(
                        "node_id"
                    )

                    if not node_id:
                        continue

                    # One domain node should not represent two
                    # different semantic operations.
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

                    relationship_score = (
                        _relationship_compatibility(
                            prompt_domain_subgraph,
                            beam[
                                "selected_ids"
                            ],
                            node_id,
                        )
                    )

                    connectivity_score = (
                        _connectivity_score(
                            prompt_domain_subgraph,
                            beam[
                                "selected_ids"
                            ],
                            node_id,
                        )
                    )

                    (
                        constraint_penalty,
                        violations,
                    ) = _rule_constraint_penalty(
                        prompt_domain_subgraph,
                        step.text,
                        node_id,
                    )

                    # -------------------------------------------------
                    # Explicit/direct semantic candidates receive a
                    # bonus.
                    # -------------------------------------------------

                    explicit_bonus = 0.0

                    if (
                        bool(step.explicit)
                        and candidate.get(
                            "source"
                        ) == "direct"
                    ):
                        explicit_bonus = (
                            EXPLICIT_DIRECT_BONUS
                        )

                    # -------------------------------------------------
                    # Neighborhood inference is weaker than a direct
                    # semantic match.
                    # -------------------------------------------------

                    inferred_penalty = 0.0

                    if (
                        candidate.get(
                            "source"
                        )
                        == "neighborhood"
                    ):
                        inferred_penalty = (
                            INFERRED_PENALTY
                        )

                    # -------------------------------------------------
                    # Penalize disconnected candidates only when
                    # there are already selected nodes.
                    # -------------------------------------------------

                    disconnected_penalty = 0.0

                    if (
                        beam["selected_ids"]
                        and connectivity_score <= 0.0
                    ):
                        disconnected_penalty = (
                            DISCONNECTED_PENALTY
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
                        "step_index": step_index,

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

                        "relationship_score": (
                            relationship_score
                        ),

                        "connectivity_score": (
                            connectivity_score
                        ),

                        "constraint_penalty": (
                            constraint_penalty
                        ),

                        "constraint_violations": (
                            violations
                        ),

                        "condition": getattr(
                            step,
                            "condition",
                            "",
                        ),

                        "branch": getattr(
                            step,
                            "branch",
                            "",
                        ),

                        "condition_negated": bool(
                            getattr(
                                step,
                                "condition_negated",
                                False,
                            )
                        ),

                        "composite": bool(
                            candidate.get(
                                "composite",
                                False,
                            )
                        ),
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

            # -------------------------------------------------
            # No beam could satisfy this step.
            # -------------------------------------------------

            if not new_beams:

                unsupported_steps.append(
                    {
                        "step_index": step_index,
                        "prompt_text": step.text,
                        "reason": (
                            "All acceptable domain operations "
                            "were already selected earlier "
                            "in the workflow."
                        ),
                    }
                )

                continue

            new_beams = (
                self._deduplicate_beams(
                    new_beams
                )
            )

            new_beams.sort(
                key=lambda beam: beam[
                    "score"
                ],
                reverse=True,
            )

            beams = new_beams[
                : self.beam_width
            ]

        # =====================================================
        # No usable workflow
        # =====================================================

        if not beams:

            return {
                "beam": [],
                "selection": [],
                "unsupported_steps": (
                    unsupported_steps
                ),
            }

        best_beam = beams[0]

        # =====================================================
        # Rule-based inferred operations
        # =====================================================

        prompt = " ".join(
            step.text
            for step in semantic_steps
            if getattr(
                step,
                "text",
                "",
            )
        )

        inferred_operation_ids = (
            _infer_operations_from_rules(
                prompt_domain_subgraph,
                best_beam[
                    "selected_ids"
                ],
                prompt,
            )
        )

        for operation_id in (
            inferred_operation_ids
        ):

            if (
                operation_id
                in best_beam[
                    "selected_ids"
                ]
            ):
                continue

            if not prompt_domain_subgraph.has_node(
                operation_id
            ):
                continue

            node = (
                prompt_domain_subgraph.nodes[
                    operation_id
                ]
            )

            if not _is_executable_node_type(
                node.get(
                    "node_type"
                )
            ):
                continue

            item = (
                self._build_inferred_selection_item(
                    operation_id,
                    node,
                    prompt,
                    source="rule_constraint",
                )
            )

            best_beam[
                "selection"
            ].append(item)

            best_beam[
                "selected_ids"
            ].append(operation_id)

        # =====================================================
        # Restore semantic execution order.
        #
        # Beam search scores candidates, but the prompt's
        # dependency structure determines execution order.
        # =====================================================

        best_beam[
            "selection"
        ] = self._restore_semantic_order(
            best_beam[
                "selection"
            ]
        )

        return {
            "beam": beams,

            "selection": best_beam[
                "selection"
            ],

            "unsupported_steps": (
                unsupported_steps
            ),
        }

    # =========================================================
    # Candidate validation
    # =========================================================

    def _candidate_belongs_to_step(
        self,
        candidate,
        step_index,
    ):
        """
        Prevent a candidate from another semantic step from
        being used for this step.

        PromptSubGraphBuilder now explicitly stores step_index
        on every candidate.
        """

        candidate_step_index = candidate.get(
            "step_index"
        )

        # New candidate format.
        if candidate_step_index is not None:
            return (
                candidate_step_index
                == step_index
            )

        # Older candidates may not contain step_index.
        #
        # Since search() is already iterating over the candidate
        # list belonging to this step, allow them for backward
        # compatibility.
        return True

    def _is_acceptable_candidate(
        self,
        candidate,
    ):
        """
        Do not force a weak semantic match.

        A candidate is accepted when there is reasonable direct
        evidence OR when it is a strong inferred candidate.

        This is intentionally not based on a single embedding
        threshold because the candidate score already combines
        semantic, lexical and domain-graph evidence.
        """

        score = float(
            candidate.get(
                "score",
                0.0,
            )
        )

        source = candidate.get(
            "source",
            "direct",
        )

        semantic_score = float(
            candidate.get(
                "semantic_score",
                0.0,
            )
        )

        lexical_score = float(
            candidate.get(
                "lexical_score",
                0.0,
            )
        )

        # -----------------------------------------------------
        # Direct candidates.
        #
        # We accept either:
        #
        #   strong lexical evidence
        #
        # OR
        #
        #   reasonable semantic + combined evidence.
        # -----------------------------------------------------

        if source == "direct":

            if lexical_score >= 0.45:
                return True

            if (
                semantic_score >= 0.45
                and score >= self.MIN_DIRECT_SCORE
            ):
                return True

            if score >= 0.35:
                return True

            return False

        # -----------------------------------------------------
        # Neighborhood candidates require stronger evidence.
        # -----------------------------------------------------

        if source == "neighborhood":

            return (
                score
                >= self.MIN_INFERRED_SCORE
                and semantic_score
                >= 0.40
            )

        # -----------------------------------------------------
        # Other explicit inferred sources.
        # -----------------------------------------------------

        return score >= self.MIN_INFERRED_SCORE

    # =========================================================
    # Beam deduplication
    # =========================================================

    def _deduplicate_beams(
        self,
        beams,
    ):
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

    # =========================================================
    # Semantic ordering
    # =========================================================

    def _restore_semantic_order(
        self,
        selection,
    ):
        """
        Keep the order dictated by the semantic parser.

        Stage 8 should not turn:

            check balance
            reject
            record
            notify

        into an arbitrary order based on domain-node IDs or
        beam scores.
        """

        return sorted(
            selection,
            key=lambda item: (
                item.get(
                    "step_index",
                    float("inf"),
                ),
                item.get(
                    "domain_node_id",
                    "",
                ),
            ),
        )

    # =========================================================
    # Rule-inferred selection
    # =========================================================

    def _build_inferred_selection_item(
        self,
        operation_id,
        node_data,
        prompt_text,
        source,
    ):
        return {
            "step_index": None,

            "prompt_text": prompt_text,

            "domain_node_id": operation_id,

            "domain_node_name": node_data.get(
                "name",
                operation_id,
            ),

            "domain_node_type": node_data.get(
                "node_type",
                "Operation",
            ),

            "explicit": False,

            "inferred": True,

            "source": source,

            "semantic_score": 0.0,

            "lexical_score": 0.0,

            "candidate_score": 0.0,

            "relationship_score": 0.0,

            "connectivity_score": 0.0,

            "constraint_penalty": 0.0,

            "constraint_violations": [],

            "condition": "",

            "branch": "",

            "condition_negated": False,

            "composite": False,
        }