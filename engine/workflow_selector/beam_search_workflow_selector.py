import networkx as nx

from helpers.beam_search_utils import _is_executable_candidate, EXPLICIT_DIRECT_BONUS, INFERRED_PENALTY, \
    DISCONNECTED_PENALTY, _relationship_compatibility, _connectivity_score, _rule_constraint_penalty, \
    _is_executable_node_type, _infer_operations_from_rules, _is_executable_selection_item


class BeamSearchWorkflowSelector:
    def __init__(
            self,
            beam_width=3,
            max_candidates_per_step=10,
    ):
        self.beam_width = beam_width
        self.max_candidates_per_step = max_candidates_per_step

    def search(
            self,
            candidate_plan,
    ):

        semantic_steps = candidate_plan["semantic_steps"]

        candidate_map = candidate_plan["candidate_map"]

        prompt_domain_subgraph = candidate_plan["prompt_domain_subgraph"]

        beams = [
            {
                "selection": [],
                "selected_ids": [],
                "score": 0.0,
                "constraint_violations": [],
            }
        ]
        unsupported_steps = []

        for step_index, step in enumerate(semantic_steps):

            candidates = candidate_map.get(step_index,[],)

            candidates = [
                candidate
                for candidate in candidates
                if _is_executable_candidate(
                    prompt_domain_subgraph,
                    candidate,
                )
            ]

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

                    if node_id in beam["selected_ids"]:
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
                            beam["selected_ids"],
                            node_id,
                        )
                    )

                    connectivity_score = (
                        _connectivity_score(
                            prompt_domain_subgraph,
                            beam["selected_ids"],
                            node_id,
                        )
                    )

                    constraint_penalty, violations = (
                        _rule_constraint_penalty(
                            prompt_domain_subgraph,
                            step.text,
                            node_id,
                        )
                    )

                    explicit_bonus = 0.0

                    if (
                            bool(step.explicit)
                            and candidate.get("source") == "direct"
                    ):
                        explicit_bonus = (
                            EXPLICIT_DIRECT_BONUS
                        )

                    inferred_penalty = 0.0

                    if candidate.get("source") == "neighborhood":
                        inferred_penalty = (
                            INFERRED_PENALTY
                        )

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
                        "condition": getattr(step, "condition", ""),
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
                unsupported_steps.append({
                    "step_index": step_index,
                    "prompt_text": step.text,
                    "reason": "All matching domain operations were already selected earlier in the plan.",
                })
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

        best_beam = beams[0]
        prompt = " ".join(step.text for step in semantic_steps if getattr(step, "text", ""))

        inferred_operation_ids = _infer_operations_from_rules(
            prompt_domain_subgraph,
            best_beam["selected_ids"],
            prompt,
        )

        for operation_id in inferred_operation_ids:
            if operation_id in best_beam["selected_ids"]:
                continue
            if not prompt_domain_subgraph.has_node(operation_id):
                continue
            node = prompt_domain_subgraph.nodes[operation_id]
            if not _is_executable_node_type(node.get("node_type")):
                continue
            item = self._build_inferred_selection_item(
                operation_id, node, prompt, source="rule_constraint"
            )
            best_beam["selection"].append(item)
            best_beam["selected_ids"].append(operation_id)

        return {
            "beam": beams,
            "selection": best_beam["selection"],
            "unsupported_steps": unsupported_steps,
        }

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

    def _build_inferred_selection_item(
            cls,
            operation_id,
            node_data,
            prompt_text,
            source,
    ):
        return {
            "prompt_text": prompt_text,
            "domain_node_id": operation_id,
            "domain_node_name": node_data.get("name", operation_id),
            "domain_node_type": node_data.get("node_type", "Operation"),
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
        }