"""Dependency-based fallback for conditional clauses embedded in an action."""


class ConditionalClauseSplitter:
    """Extract ``main action, if condition action`` without changing regex cases."""

    MARKERS = {"if", "unless", "once", "when", "whenever"}

    def __init__(self, nlp):
        self.nlp = nlp

    def split(self, sentence):
        """Return ``(main_text, conditional_action, condition)`` or ``None``."""
        if self.nlp is None:
            return None
        doc = self.nlp(sentence)
        for token in doc:
            marker = next((child for child in token.children
                           if child.dep_ == "mark" and child.lower_ in self.MARKERS), None)
            if marker is None or token.dep_ not in {"advcl", "pcomp"}:
                continue
            condition_tokens = set(token.subtree)
            # spaCy parses the common imperative shape as two adverbial
            # clauses under the final action root:
            #   check ... , if does not exist create ...
            #             ^condition       ^root/action
            root = next((item for item in doc if item.dep_ == "ROOT"), None)
            main_head = next(
                (child for child in (root.children if root else [])
                 if child.dep_ in {"advcl", "pcomp"} and child.i != token.i),
                None,
            )
            main_tokens = set(main_head.subtree) if main_head else set()
            action_tokens = [item for item in (root.subtree if root else [])
                             if item not in condition_tokens and item not in main_tokens
                             and item.dep_ != "punct"]
            main = " ".join(item.text for item in main_head.subtree).strip() if main_head else ""
            action = " ".join(item.text for item in action_tokens).strip()
            condition = " ".join(item.text for item in token.subtree
                                 if item.i != marker.i and item.dep_ != "punct").strip()
            if main and action and condition:
                return main, action, condition
        return None