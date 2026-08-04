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
    # Parse prompt
    # -----------------------------------------------------

    def parse(self, prompt: str) -> PromptAnalysis:

        doc = self.nlp(prompt)

        analysis = PromptAnalysis()

        ####################################################
        # Actions
        ####################################################

        for token in doc:

            if token.pos_ == "VERB":

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

                if left and right:

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

                if left and right:

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

            analysis.relations.append(

                PromptRelation(

                    source=verbs[i],

                    target=verbs[i + 1],

                    relation="sequence"

                )

            )

        return analysis


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = PromptParser()

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