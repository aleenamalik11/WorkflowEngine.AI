class BeamSearchWorkflowSelector:
    def __init__(
            self,
            beam_width=3,
            max_candidates_per_step=10,
    ):
        self.beam_width = beam_width
        self.max_candidates_per_step = max_candidates_per_step