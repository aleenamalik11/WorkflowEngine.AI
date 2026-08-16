import json
from typing import List, Optional

from models import (
    RegisteredFunction,
    FunctionMatch,
)

from utils import (
    EmbeddingService,
    functions_to_text,
    cosine_similarity,
)


class FunctionMatcher:

    def __init__(
        self,
        model_path,
        match_threshold: float = 0.70,
    ):
        """
        Parameters
        ----------
        model_path:
            Embedding model used for semantic function matching.

        match_threshold:
            Minimum cosine similarity required for a registered
            function to be considered a valid implementation.

        IMPORTANT:
            The matcher never guarantees a match.

            If the highest scoring function is below this
            threshold, the result is explicitly:

                matching function not found
        """

        self.embedding_service = EmbeddingService(
            model_path
        )

        self.functions: List[
            RegisteredFunction
        ] = []

        self.match_threshold = (
            float(match_threshold)
        )

    # ==========================================================
    # Load registered functions
    # ==========================================================

    def load(self, path):

        with open(
            path,
            "r",
            encoding="utf8",
        ) as f:

            raw = json.load(f)

        self.functions.clear()

        for item in raw:

            function = RegisteredFunction(

                name=item["name"],

                description=item.get(
                    "description",
                    "",
                ),

                inputs=item.get(
                    "inputs",
                    [],
                ),

                outputs=item.get(
                    "outputs",
                    [],
                ),
            )

            text = functions_to_text(
                function
            )

            function.embedding = (
                self.embedding_service.encode(
                    text
                )
            )

            self.functions.append(
                function
            )

    # ==========================================================
    # Match one semantic concept
    # ==========================================================

    def match(
        self,
        text: str,
    ) -> FunctionMatch:
        """
        Match a semantic/domain concept against registered
        functions.

        This method NEVER returns an arbitrary function merely
        because it is the closest candidate.

        If the best similarity is below the threshold, the
        semantic node remains unmatched.
        """

        text = (
            text or ""
        ).strip()

        if not text:

            return FunctionMatch(
                score=0.0,
                function=None,
                found=False,
                status="matching function not found",
                semantic_text=text,
            )

        if not self.functions:

            return FunctionMatch(
                score=0.0,
                function=None,
                found=False,
                status="matching function not found",
                semantic_text=text,
            )

        embedding = (
            self.embedding_service.encode(
                text
            )
        )

        best_score = -1.0

        best_function = None

        for function in self.functions:

            if function.embedding is None:
                continue

            score = cosine_similarity(
                embedding,
                function.embedding,
            )

            if score > best_score:

                best_score = score

                best_function = function

        # ------------------------------------------------------
        # No candidate at all
        # ------------------------------------------------------

        if best_function is None:

            return FunctionMatch(
                score=0.0,
                function=None,
                found=False,
                status="matching function not found",
                semantic_text=text,
            )

        # ------------------------------------------------------
        # Candidate exists but is not sufficiently similar
        # ------------------------------------------------------

        if best_score < self.match_threshold:

            return FunctionMatch(
                score=best_score,
                function=None,
                found=False,
                status="matching function not found",
                semantic_text=text,
            )

        # ------------------------------------------------------
        # Valid match
        # ------------------------------------------------------

        return FunctionMatch(
            score=best_score,
            function=best_function,
            found=True,
            status="matched",
            semantic_text=text,
        )

    # ==========================================================
    # Match many semantic concepts
    # ==========================================================

    def match_many(
        self,
        texts,
    ):

        return [
            self.match(text)
            for text in texts
        ]

    # ==========================================================
    # Top-k candidates
    # ==========================================================

    def top_matches(
        self,
        text,
        k=5,
    ) -> List[FunctionMatch]:
        """
        Return top-k candidates.

        Unlike `match()`, this method is useful for diagnostics
        and debugging.

        IMPORTANT:

        A caller must still respect `match_threshold`.

        A top candidate below the threshold is NOT a valid
        implementation.
        """

        embedding = (
            self.embedding_service.encode(
                text
            )
        )

        matches = []

        for function in self.functions:

            if function.embedding is None:
                continue

            score = cosine_similarity(
                embedding,
                function.embedding,
            )

            is_valid = (
                score >= self.match_threshold
            )

            matches.append(
                FunctionMatch(

                    score=score,

                    function=(
                        function
                        if is_valid
                        else None
                    ),

                    found=is_valid,

                    status=(
                        "matched"
                        if is_valid
                        else "matching function not found"
                    ),

                    semantic_text=text,
                )
            )

        matches.sort(
            key=lambda x: x.score,
            reverse=True,
        )

        return matches[:k]

    # ==========================================================
    # Configuration
    # ==========================================================

    def set_threshold(
        self,
        threshold: float,
    ):
        self.match_threshold = float(
            threshold
        )