import json

from models import (
    RegisteredFunction,
    FunctionMatch
)

from utils import (
    EmbeddingService,
    functions_to_text,
    cosine_similarity
)


class FunctionMatcher:

    def __init__(self, model_path):

        self.embedding_service = EmbeddingService(model_path)

        self.functions = []

    ##########################################################
    # Load registered functions
    ##########################################################

    def load(self, path):

        with open(path, "r", encoding="utf8") as f:
            raw = json.load(f)

        self.functions.clear()

        for item in raw:

            function = RegisteredFunction(

                name=item["name"],

                description=item.get("description", ""),

                inputs=item.get("inputs", []),

                outputs=item.get("outputs", [])

            )

            text = functions_to_text(function)

            function.embedding = self.embedding_service.encode(text)

            self.functions.append(function)

    ##########################################################
    # Match a single text/concept
    ##########################################################

    def match(self, text: str):

        embedding = self.embedding_service.encode(text)

        best_score = -1.0
        best_function = None

        for function in self.functions:

            score = cosine_similarity(
                embedding,
                function.embedding
            )

            if score > best_score:
                best_score = score
                best_function = function

        return FunctionMatch(
            score=best_score,
            function=best_function
        )

    ##########################################################
    # Match many concepts
    ##########################################################

    def match_many(self, texts):

        return [
            self.match(text)
            for text in texts
        ]

    ##########################################################
    # Return top-k matches
    ##########################################################

    def top_matches(self, text, k=5):

        embedding = self.embedding_service.encode(text)

        matches = []

        for function in self.functions:

            score = cosine_similarity(
                embedding,
                function.embedding
            )

            matches.append(

                FunctionMatch(

                    score=score,

                    function=function

                )

            )

        matches.sort(
            key=lambda x: x.score,
            reverse=True
        )

        return matches[:k]