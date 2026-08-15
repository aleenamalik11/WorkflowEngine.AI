"""
Stage 8 - Beam Search Planner.

The planner receives an explicit mapping:

    semantic step -> candidate domain nodes

and searches over those candidates.

It does NOT rediscover candidates by looking at graph node
prompt_text fields.

No Dijkstra.
No shortest-path routing.

Relationship weights are used as semantic/order bonuses.
"""

import networkx as nx


class BeamSearchPlanner:

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
    # Stage 8
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

        domain_graph = candidate_plan[
            "domain_graph"
        ]

        beams = [
            {
                "selection": [],
                "selected_ids": [],
                "score": 0.0,
            }
        ]

        for step_index, step in enumerate(
            semantic_steps
        ):

            candidates = candidate_map.get(
                step_index,
                [],
            )

            if not candidates:
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

                    # Prevent one operation from being
                    # selected repeatedly for multiple semantic
                    # concepts.
                    #
                    # Example of what this prevents:
                    #
                    # transfer funds
                    #      -> Process Transfer
                    # validate transfer
                    #      -> Process Transfer
                    # check balance
                    #      -> Process Transfer
                    #
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

                    relationship_bonus = (
                        self._relationship_bonus(
                            domain_graph,
                            beam["selected_ids"],
                            node_id,
                        )
                    )

                    explicit_bonus = (
                        0.15
                        if step.explicit
                        and candidate.get(
                            "source"
                        ) == "direct"
                        else 0.0
                    )

                    inferred_penalty = (
                        0.0
                        if candidate.get(
                            "source"
                        ) == "direct"
                        else -0.03
                    )

                    total_increment = (
                        candidate_score
                        + relationship_bonus
                        + explicit_bonus
                        + inferred_penalty
                    )

                    selection_item = {
                        "prompt_text": step.text,

                        "domain_node_id": node_id,

                        "domain_node_name":
                            candidate.get(
                                "name",
                                node_id,
                            ),

                        "explicit":
                            bool(step.explicit),

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

                        "relationship_bonus":
                            relationship_bonus,
                    }

                    new_beams.append(
                        {
                            "selection":
                                beam["selection"]
                                + [
                                    selection_item
                                ],

                            "selected_ids":
                                beam["selected_ids"]
                                + [node_id],

                            "score":
                                beam["score"]
                                + total_increment,
                        }
                    )

            # -------------------------------------------------
            # If every candidate was already selected, permit
            # the best candidate rather than silently dropping
            # the semantic step.
            # -------------------------------------------------

            if not new_beams:

                for beam in beams:

                    candidate = candidates[0]

                    node_id = candidate[
                        "node_id"
                    ]

                    selection_item = {
                        "prompt_text": step.text,
                        "domain_node_id": node_id,
                        "domain_node_name":
                            candidate.get(
                                "name",
                                node_id,
                            ),
                        "explicit":
                            bool(step.explicit),
                        "inferred": True,
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
                            candidate.get(
                                "score",
                                0.0,
                            ),
                        "relationship_bonus": 0.0,
                    }

                    new_beams.append(
                        {
                            "selection":
                                beam["selection"]
                                + [
                                    selection_item
                                ],
                            "selected_ids":
                                beam["selected_ids"]
                                + [node_id],
                            "score":
                                beam["score"]
                                + candidate.get(
                                    "score",
                                    0.0,
                                ),
                        }
                    )

            # -------------------------------------------------
            # Deduplicate identical beam states.
            # -------------------------------------------------

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

            "selection":
                beams[0]["selection"],
        }

    # =========================================================
    # Relationship-aware local bonus
    # =========================================================

    @staticmethod
    def _relationship_bonus(
        graph,
        selected_ids,
        candidate_id,
    ):
        """
        Reward candidates that have a strong ontology
        relationship with already selected concepts.

        This is local neighborhood reasoning.

        It is NOT shortest-path search.
        """

        if not selected_ids:
            return 0.0

        bonus = 0.0

        for selected_id in selected_ids:

            if not graph.has_node(
                selected_id
            ):
                continue

            if not graph.has_node(
                candidate_id
            ):
                continue

            # Direct edge in either direction.
            edges = []

            if graph.has_edge(
                selected_id,
                candidate_id,
            ):
                edges.append(
                    graph.edges[
                        selected_id,
                        candidate_id,
                    ]
                )

            if graph.has_edge(
                candidate_id,
                selected_id,
            ):
                edges.append(
                    graph.edges[
                        candidate_id,
                        selected_id,
                    ]
                )

            for edge in edges:

                classification = edge.get(
                    "classification"
                )

                if classification == "REQUIRED":
                    bonus = max(
                        bonus,
                        0.20,
                    )

                elif classification == "POSSIBLE":
                    bonus = max(
                        bonus,
                        0.10,
                    )

                elif classification == "CONTEXT":
                    bonus = max(
                        bonus,
                        0.03,
                    )

        return bonus

    # =========================================================
    # Beam deduplication
    # =========================================================

    @staticmethod
    def _deduplicate_beams(
        beams,
    ):
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

    # =========================================================
    # Stage 8 -> Stage 9
    # =========================================================

    def expand(
        self,
        search_result,
    ):
        """
        Convert the selected beam into the final workflow graph.

        This function is deliberately stateless.

        It does NOT depend on:
            self._last_search_result
            a previous call to search()
            hidden planner state

        The supplied search_result is the complete input.
        """

        if not search_result:
            raise RuntimeError(
                "BeamSearchPlanner.expand() received no search result."
            )

        selection = search_result.get(
            "selection",
            [],
        )

        if not selection:
            raise RuntimeError(
                "Beam search produced no selected workflow nodes."
            )

        graph = nx.DiGraph()

        selected_ids = []

        # -----------------------------------------------------
        # Create selected nodes.
        # -----------------------------------------------------

        for index, item in enumerate(
            selection
        ):

            node_id = item[
                "domain_node_id"
            ]

            if node_id not in graph:

                graph.add_node(
                    node_id,
                    name=item.get(
                        "domain_node_name",
                        node_id,
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
                    source=item.get(
                        "source",
                        "direct",
                    ),
                )

            selected_ids.append(
                node_id
            )

        # -----------------------------------------------------
        # Add ontology relationships between selected nodes.
        #
        # Only direct relationships are used.
        # No Dijkstra.
        # -----------------------------------------------------

        for source_index, source_id in enumerate(
            selected_ids
        ):

            for target_index, target_id in enumerate(
                selected_ids
            ):

                if (
                    source_index
                    == target_index
                ):
                    continue

                # A DiGraph cannot represent duplicate occurrences
                # of the same domain node. The beam normally avoids
                # duplicates, so this is mostly defensive.
                if source_id == target_id:
                    continue

                # Only use direct ontology edges.
                #
                # This is intentional:
                # neighborhood != shortest path.
                if graph.has_node(
                    source_id
                ) and graph.has_node(
                    target_id
                ):
                    pass

        # -----------------------------------------------------
        # The selected domain graph needs the ontology edges.
        #
        # We don't have the original candidate_plan here, so
        # reconstruct ordering from the selection itself.
        #
        # The semantic parser already gives us the intended
        # semantic order. Therefore consecutive selected concepts
        # receive a prompt-order edge.
        # -----------------------------------------------------

        for i in range(
            len(selected_ids) - 1
        ):

            source = selected_ids[i]
            target = selected_ids[i + 1]

            if source == target:
                continue

            if not graph.has_edge(
                source,
                target,
            ):

                graph.add_edge(
                    source,
                    target,
                    relation="PROMPT_PRECEDES",
                    origin="beam_selection",
                    inferred_context=True,
                    edge_type="mandatory",
                    weight=0.0,
                    classification="REQUIRED",
                    traverses_for_ordering=True,
                )

        return graph