"""
Stage 8 - Beam Search Planner.

Responsibilities:

    1. Select domain concepts for semantic steps.
    2. Use semantic + lexical candidate scores.
    3. Prefer locally related ontology concepts.
    4. Avoid repeatedly selecting the same domain concept.
    5. Preserve inferred neighborhood concepts.
    6. Expand the selected concepts into a workflow graph.

No Dijkstra.
No shortest-path search.

Relationship information is used locally only.

IMPORTANT -- selection order vs. execution order:

`search()` returns a `selection` list ordered by MATCHING/PROCESSING
order (the order semantic steps were evaluated in, e.g. the order the
LLM emitted `steps` in). That order is NOT an execution-order claim.
`expand()` must never treat two nodes being adjacent in `selection` as
"selection[i] happens before selection[i+1]" -- execution order comes
only from real directed relationships: domain-graph edges (Neo4j
ontology relationships) and PROMPT_DEPENDENCY edges (resolved from the
LLM's stated dependencies, added upstream in
contextual_subgraph_builder.py and present in `domain_graph` by the
time it reaches `expand()`). See the comment in `expand()` for the
concrete cycle this used to produce when that rule was violated.
"""

import networkx as nx


class BeamSearchPlanner:

    def __init__(
        self,
        beam_width=3,
        max_candidates_per_step=10,
    ):

        self.beam_width = (
            beam_width
        )

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

        semantic_steps = (
            candidate_plan[
                "semantic_steps"
            ]
        )

        candidate_map = (
            candidate_plan[
                "candidate_map"
            ]
        )

        domain_graph = (
            candidate_plan[
                "domain_graph"
            ]
        )

        beams = [
            {
                "selection": [],
                "selected_ids": [],
                "score": 0.0,
            }
        ]

        # -----------------------------------------------------
        # Process semantic steps in their semantic order.
        # -----------------------------------------------------

        for step_index, step in enumerate(
            semantic_steps
        ):

            candidates = (
                candidate_map.get(
                    step_index,
                    [],
                )
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

                    # -------------------------------------------------
                    # Do not map several semantic concepts to the
                    # same domain operation.
                    #
                    # This fixes the previous behavior:
                    #
                    # check balance
                    #       -> Process Transfer
                    # transfer funds
                    #       -> Process Transfer
                    #
                    # etc.
                    # -------------------------------------------------

                    if node_id in (
                        beam[
                            "selected_ids"
                        ]
                    ):
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
                            beam[
                                "selected_ids"
                            ],
                            node_id,
                        )
                    )

                    explicit_bonus = (
                        0.15
                        if (
                            step.explicit
                            and candidate.get(
                                "source"
                            )
                            == "direct"
                        )
                        else 0.0
                    )

                    inferred_penalty = (
                        -0.03
                        if candidate.get(
                            "source"
                        )
                        == "neighborhood"
                        else 0.0
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

                        "relationship_bonus":
                            relationship_bonus,
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
                        }
                    )

            # -----------------------------------------------------
            # If all candidates were already selected, do not
            # silently lose the semantic step.
            #
            # However, this fallback is deliberately weaker.
            # -----------------------------------------------------

            if not new_beams:

                for beam in beams:

                    candidate = (
                        candidates[0]
                    )

                    node_id = (
                        candidate[
                            "node_id"
                        ]
                    )

                    selection_item = {
                        "prompt_text":
                            step.text,

                        "domain_node_id":
                            node_id,

                        "domain_node_name":
                            candidate.get(
                                "name",
                                node_id,
                            ),

                        "explicit":
                            bool(
                                step.explicit
                            ),

                        "inferred":
                            True,

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

                        "relationship_bonus":
                            0.0,
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
                                + float(
                                    candidate.get(
                                        "score",
                                        0.0,
                                    )
                                ),
                        }
                    )

            # -----------------------------------------------------
            # Remove duplicate beam states.
            # -----------------------------------------------------

            new_beams = (
                self._deduplicate_beams(
                    new_beams
                )
            )

            new_beams.sort(
                key=lambda beam:
                    beam[
                        "score"
                    ],
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
                beams[0][
                    "selection"
                ],
        }

    # =========================================================
    # Local relationship bonus
    # =========================================================

    @staticmethod
    def _relationship_bonus(
        graph,
        selected_ids,
        candidate_id,
    ):

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

                classification = (
                    edge.get(
                        "classification"
                    )
                )

                weight = float(
                    edge.get(
                        "weight",
                        0.0,
                    )
                )

                if classification == (
                    "REQUIRED"
                ):

                    bonus = max(
                        bonus,
                        0.20
                        + max(
                            0.0,
                            0.05
                            * (
                                1.0
                                / max(
                                    weight,
                                    1.0,
                                )
                            ),
                        ),
                    )

                elif classification == (
                    "POSSIBLE"
                ):

                    bonus = max(
                        bonus,
                        0.10,
                    )

                elif classification == (
                    "CONTEXT"
                ):

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
    # Stage 8 -> Stage 9
    # =========================================================

    def expand(
        self,
        search_result,
        candidate_plan=None,
    ):
        """
        Convert the selected beam into the workflow graph.

        The contextual domain graph is copied into the resulting
        workflow graph only for relationships whose endpoints were
        actually selected.

        No shortest-path search is performed.
        """

        if not search_result:

            raise RuntimeError(
                "BeamSearchPlanner.expand() received "
                "no search result."
            )

        selection = (
            search_result.get(
                "selection",
                [],
            )
        )

        if not selection:

            raise RuntimeError(
                "Beam search produced no "
                "selected workflow nodes."
            )

        graph = nx.DiGraph()

        selected_ids = []

        # -----------------------------------------------------
        # Add selected nodes.
        # -----------------------------------------------------

        for item in selection:

            node_id = item[
                "domain_node_id"
            ]

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
            )

        # -----------------------------------------------------
        # Copy direct ontology relationships between selected
        # nodes.
        #
        # IMPORTANT:
        #
        # This is NOT path finding.
        #
        # We only copy relationships that already exist directly
        # in the contextual domain graph.
        # -----------------------------------------------------

        domain_graph = None

        if candidate_plan:

            domain_graph = (
                candidate_plan.get(
                    "domain_graph"
                )
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

                graph.add_edge(
                    source,
                    target,
                    **dict(data),
                )

        # -----------------------------------------------------
        # NOTE: there is intentionally no "prompt-order" /
        # list-adjacency fallback here anymore.
        #
        # `selection` (and therefore `selected_ids`) is ordered by
        # MATCHING/PROCESSING order -- the order semantic steps were
        # evaluated in -- not by execution order. Two nodes being
        # adjacent in that list carries no information about which
        # one must run first. The previous version of this method
        # treated adjacency as
        #     selected_ids[i] --PROMPT_PRECEDES--> selected_ids[i+1]
        # which was an architectural bug: it fabricated a directed
        # edge out of an accident of list position, and that edge
        # could directly contradict real domain relationships or
        # PROMPT_DEPENDENCY edges already copied in above (which DO
        # represent genuine ordering, resolved from Neo4j
        # relationships and the LLM's stated dependencies
        # respectively). Concretely, for "create a new account" this
        # produced:
        #     Create Account --PROMPT_PRECEDES--> Validate User
        #     Validate User  --PROMPT_DEPENDENCY--> Create User
        #     Create User    --PROMPT_DEPENDENCY--> Create Account
        # a cycle -- even though "Create Account" was simply the
        # first entry in the selection list because it was the
        # direct match for the whole prompt, not because it executes
        # first. A `nx.has_path()` guard could suppress that specific
        # symptom, but the correct fix is to never generate the
        # ordering claim in the first place.
        #
        # Execution order must come ONLY from edges that actually
        # assert an ordering claim: real domain-graph relationships
        # (added above) and PROMPT_DEPENDENCY edges (added upstream
        # in contextual_subgraph_builder.py, already present in
        # `domain_graph` and therefore already copied in above). A
        # selected node with no such relationship to any other
        # selected node is a genuine orphan: it stays in the graph as
        # an unordered node -- GraphPlanner.plan()'s topological sort
        # places nodes with no edges without needing one -- and no
        # edge is invented for it. If a caller needs orphans wired to
        # an explicit start/end, that must be its own explicit
        # graph-construction step, never inferred from selection-list
        # position.
        # -----------------------------------------------------

        return graph