class BeamSearchPlanner:

    def __init__(self, beam_width=3):
        self.beam_width = beam_width

    def search(self, candidate_plan):

        semantic_steps = candidate_plan[
            "semantic_steps"
        ]

        graph = candidate_plan[
            "domain_graph"
        ]

        beams = [
            {
                "selection": [],
                "score": 0.0,
            }
        ]

        for step in semantic_steps:

            candidates = self._candidates_for_step(
                graph,
                step,
            )

            if not candidates:
                continue

            new_beams = []

            for beam in beams:

                for node_id, node_data in candidates:

                    score = node_data.get(
                        "semantic_score",
                        0.0,
                    )

                    new_selection = (
                        beam["selection"]
                        + [
                            {
                                "prompt_text": step.text,
                                "domain_node_id": node_id,
                                "domain_node_name": node_data.get(
                                    "name",
                                    node_id,
                                ),
                                "explicit": step.explicit,
                            }
                        ]
                    )

                    new_beams.append(
                        {
                            "selection": new_selection,
                            "score": (
                                beam["score"]
                                + score
                            ),
                        }
                    )

            new_beams.sort(
                key=lambda b: b["score"],
                reverse=True,
            )

            beams = new_beams[
                :self.beam_width
            ]

        return {
            "beam": beams,
            "selection": (
                beams[0]["selection"]
                if beams
                else []
            ),
        }

    def _candidates_for_step(
        self,
        graph,
        step,
    ):

        candidates = []

        for node_id, data in graph.nodes(
            data=True
        ):

            prompt_text = data.get(
                "prompt_text"
            )

            if prompt_text:

                if self._similar(
                    step.text,
                    prompt_text,
                ):
                    candidates.append(
                        (
                            node_id,
                            data,
                        )
                    )

        # If no direct candidates exist,
        # use domain nodes attached through context.
        if not candidates:

            for node_id, data in graph.nodes(
                data=True
            ):

                candidates.append(
                    (
                        node_id,
                        data,
                    )
                )

        candidates.sort(
            key=lambda x: x[1].get(
                "semantic_score",
                0.0,
            ),
            reverse=True,
        )

        return candidates[:10]

    @staticmethod
    def _similar(a, b):

        a = set(
            a.lower().split()
        )

        b = set(
            b.lower().split()
        )

        if not a or not b:
            return False

        return bool(a & b)