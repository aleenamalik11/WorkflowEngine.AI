from dataclasses import dataclass, field
from typing import List

import spacy


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

@dataclass
class PromptAction:
    text: str
    lemma: str
    token_index: int


@dataclass
class PromptEntity:
    text: str
    root: str
    token_index: int


@dataclass
class PromptRelation:
    source: str
    target: str
    relation: str


@dataclass
class PromptAnalysis:
    actions: List[PromptAction] = field(default_factory=list)
    entities: List[PromptEntity] = field(default_factory=list)
    relations: List[PromptRelation] = field(default_factory=list)


# ---------------------------------------------------------
# Prompt Parser
# ---------------------------------------------------------

class WorkflowPromptParser:

    def __init__(self):

        print("Loading spaCy...")

        self.nlp = spacy.load("en_core_web_sm")

    # -----------------------------------------------------
    # Imperative repair
    # -----------------------------------------------------

    def _imperative_actions(self, doc):
        """Recover imperative verbs that spaCy tagged as nouns.

        A bare command such as ``Transfer funds`` has no subject, so the
        statistical tagger reads the leading token as a noun ("transfer" the
        thing) instead of a verb ("to transfer").  The sentence is therefore
        parsed again behind a polite cue, which restores the verb reading
        without touching the rest of the pipeline.
        """
        recovered = {}

        for sentence in doc.sents:

            first = sentence[0]

            if first.pos_ not in ("NOUN", "PROPN"):
                continue

            repaired = self.nlp("Please " + sentence.text)

            if len(repaired) < 2 or repaired[1].pos_ != "VERB":
                continue

            recovered[first.i] = PromptAction(
                text=first.text,
                lemma=repaired[1].lemma_.lower(),
                token_index=first.i,
            )

        return recovered

    # -----------------------------------------------------
    # Parse prompt
    # -----------------------------------------------------

    def parse(self, prompt: str) -> PromptAnalysis:

        doc = self.nlp(prompt)

        analysis = PromptAnalysis()

        ####################################################
        # Actions
        ####################################################

        imperative_actions = self._imperative_actions(doc)

        for token in doc:

            if token.i in imperative_actions:

                analysis.actions.append(imperative_actions[token.i])

                continue

            # spaCy can tag the verb in "after that notify teacher" as a
            # noun. Treat that narrow coordination pattern as an action.
            follows_after_that = (
                token.pos_ == "NOUN"
                and token.i >= 2
                and doc[token.i - 1].lower_ == "that"
                and doc[token.i - 2].lower_ in {"after", "before"}
            )

            if token.pos_ == "VERB" or follows_after_that:

                analysis.actions.append(

                    PromptAction(

                        text=token.text,
                        lemma=token.lemma_.lower(),
                        token_index=token.i

                    )

                )

        ####################################################
        # Entities
        ####################################################

        for chunk in doc.noun_chunks:

            analysis.entities.append(

                PromptEntity(

                    text=chunk.text,
                    root=chunk.root.lemma_.lower(),
                    token_index=chunk.root.i

                )

            )

        ####################################################
        # Dependency Relations
        ####################################################

        for token in doc:

            ################################################
            # before
            ################################################

            if token.text.lower() == "before":

                left = None
                right = None

                for child in token.head.subtree:

                    if child.pos_ == "VERB":

                        right = child.lemma_

                        break

                for ancestor in token.ancestors:

                    if ancestor.pos_ == "VERB":

                        left = ancestor.lemma_

                        break

                if left and right and left != right:

                    analysis.relations.append(

                        PromptRelation(

                            source=left,
                            target=right,
                            relation="before"

                        )

                    )

            ################################################
            # after
            ################################################

            elif token.text.lower() == "after":

                left = None
                right = None

                for child in token.head.subtree:

                    if child.pos_ == "VERB":

                        right = child.lemma_

                        break

                for ancestor in token.ancestors:

                    if ancestor.pos_ == "VERB":

                        left = ancestor.lemma_

                        break

                if left and right and left != right:

                    analysis.relations.append(

                        PromptRelation(

                            source=right,
                            target=left,
                            relation="before"

                        )

                    )

        ####################################################
        # Sequential verbs
        ####################################################

        verbs = [

            a.lemma

            for a in analysis.actions

        ]

        for i in range(len(verbs) - 1):

            relation = PromptRelation(

                source=verbs[i],

                target=verbs[i + 1],

                relation="sequence"

            )
            if relation not in analysis.relations:
                analysis.relations.append(relation)

        return analysis


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = WorkflowPromptParser()

    prompt = """
    Register a new student after validating documents
    and assign the student to Grade 8.
    """

    analysis = parser.parse(prompt)

    print("\nActions")
    print("----------------")

    for a in analysis.actions:
        print(a)

    print("\nEntities")
    print("----------------")

    for e in analysis.entities:
        print(e)

    print("\nRelations")
    print("----------------")

    for r in analysis.relations:
        print(r)
