import re

from helpers.parser_utils import (
    _load_nlp,
    _extract_explicit_steps,
    _nlp_entities,
)
from models import SemanticStep, SemanticInterpretation


class SimpleSemanticParser:
    """
    Deterministic semantic parser with NLP enrichment.

    Phases:
        1. Action / object / negation
        2. Clause segmentation / conditionals / dependencies
        3. Entities / parameters
        4. Simple coreference resolution
        5. Branching / else / otherwise / unless metadata

    This parser understands the user's language only.
    It does NOT resolve language to domain functions.
    """

    # ------------------------------------------------------------------
    # IMPORTANT:
    # Keep this list focused on actual action verbs.
    #
    # "record" was previously missing. This caused:
    #
    #   reject the transaction, record the failed transaction,
    #   and notify the customer
    #
    # to be parsed as:
    #
    #   reject the transaction, record the failed transaction
    #   notify the customer
    #
    # instead of three atomic actions.
    # ------------------------------------------------------------------
    ACTION_STARTERS = (
        "check",
        "verify",
        "validate",
        "confirm",
        "calculate",
        "create",
        "send",
        "notify",
        "mark",
        "retrieve",
        "get",
        "fetch",
        "process",
        "place",
        "open",
        "transfer",
        "debit",
        "credit",
        "persist",
        "generate",
        "update",
        "delete",
        "remove",
        "cancel",
        "close",
        "save",
        "load",
        "find",
        "search",
        "look",
        "ensure",
        "return",
        "reject",
        "approve",
        "deny",
        "add",

        # Logging / recording actions
        "record",
        "log",
        "store",

        # Common workflow actions
        "execute",
        "run",
        "start",
        "stop",
        "complete",
        "finish",
        "fail",
        "flag",
        "assign",
        "publish",
        "submit",
        "review",
        "authorize",
        "authenticate",
        "inspect",
        "evaluate",
        "compare",
    )

    CONDITION_MARKERS = (
        "not found",
        "does not exist",
        "doesn't exist",
        "do not exist",
        "don't exist",
        "is not available",
        "isn't available",
        "not available",
        "is missing",
        "are missing",
        "doesn't have",
        "does not have",
        "is invalid",
        "isn't valid",
        "not valid",
        "is empty",
        "isn't empty",
        "not empty",
        "is successful",
        "isn't successful",
        "fails",
        "failed",
        "if not",
        "if unavailable",
    )

    PRONOUNS = {
        "it",
        "this",
        "that",
        "one",
        "them",
        "they",
        "he",
        "she",
    }

    def __init__(self, use_nlp=True, nlp=None):
        self.use_nlp = use_nlp
        self._nlp = nlp

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def parse(self, prompt: str, domain_context=None):
        """Parse a workflow request into semantic steps."""

        cleaned_prompt = self._clean_prompt(prompt)

        if not cleaned_prompt:
            return SemanticInterpretation(
                intent="",
                steps=[],
                dependencies=[],
                explicit_steps=[],
                mentioned_entities=[],
                constraints=[],
            )

        # Phase 2:
        # Extract explicit actions / clauses.
        steps, dependencies = _extract_explicit_steps(
            self,
            cleaned_prompt,
        )

        # Phase 1 + Phase 3
        for step in steps:
            self._enrich_step(step)

        # Phase 4
        self._resolve_coreferences(steps)

        # Phase 5
        self._enrich_branches(
            steps,
            dependencies,
        )

        # Prompt-level entities
        entities = []

        if self.use_nlp:
            entities = _nlp_entities(
                self,
                cleaned_prompt,
            )

        return SemanticInterpretation(
            intent=(
                steps[0].text
                if steps
                else cleaned_prompt
            ),
            steps=steps,
            dependencies=dependencies,
            explicit_steps=steps,
            mentioned_entities=entities,
            constraints=[],
        )

    # ==========================================================
    # PHASE 1 + PHASE 3
    # ==========================================================

    def _enrich_step(self, step: SemanticStep):
        """Add linguistic information to a semantic step."""

        if not self.use_nlp:
            return

        nlp = self._load_nlp()

        if nlp is None:
            return

        self._extract_action_and_object(
            step,
            nlp,
        )

        self._extract_negation(
            step,
            nlp,
        )

        self._extract_step_entities(
            step,
            nlp,
        )

        self._extract_parameters(step)

    # ==========================================================
    # PHASE 1
    # ACTION / OBJECT
    # ==========================================================

    def _extract_action_and_object(self, step, nlp):
        doc = nlp(step.text)

        if len(doc) == 0:
            return

        root = next(
            (
                token
                for token in doc
                if token.dep_ == "ROOT"
            ),
            None,
        )

        if root is None:
            root = next(
                (
                    token
                    for token in doc
                    if token.pos_ in {"VERB", "AUX"}
                ),
                None,
            )

        if root is None:
            return

        if root.pos_ in {"VERB", "AUX"}:
            step.action = root.lemma_.lower()

        object_token = self._find_object_token(root)

        if object_token is not None:
            step.object = object_token.lemma_.lower()

    @staticmethod
    def _find_object_token(root):
        """Find the most likely semantic object."""

        # Direct object
        for child in root.children:
            if child.dep_ in {
                "dobj",
                "obj",
                "attr",
            }:
                return child

        # Prepositional object
        for child in root.children:
            if child.dep_ == "prep":
                for descendant in child.children:
                    if descendant.dep_ in {
                        "pobj",
                        "obj",
                    }:
                        return descendant

        # Oblique / nominal modifier
        for child in root.children:
            if child.dep_ in {
                "pobj",
                "obl",
                "nmod",
            }:
                if child.pos_ in {
                    "NOUN",
                    "PROPN",
                }:
                    return child

        return None

    # ==========================================================
    # PHASE 1
    # NEGATION
    # ==========================================================

    def _extract_negation(self, step, nlp):
        doc = nlp(step.text)

        if len(doc) == 0:
            return

        root = next(
            (
                token
                for token in doc
                if token.dep_ == "ROOT"
            ),
            None,
        )

        if root is None:
            return

        step.negated = any(
            token.dep_ == "neg"
            or token.lower_ in {
                "never",
                "no",
            }
            for token in root.subtree
        )

    # ==========================================================
    # PHASE 2
    # SENTENCE / CLAUSE PARSING
    # ==========================================================

    def _parse_sentence(self, sentence):
        sentence = self._clean_clause(sentence)

        if not sentence:
            return [], []

        # ELSE / OTHERWISE
        else_result = self._parse_else(sentence)

        if else_result is not None:
            return else_result

        # CONDITIONAL
        conditional = self._parse_conditional(sentence)

        if conditional is not None:
            return conditional

        # AFTER
        match = re.match(
            r"^after\s+([^,;]+),\s*"
            r"(?:then\s+)?(.+)$",
            sentence,
            re.IGNORECASE,
        )

        if match:
            prerequisite = self._clean_clause(
                match.group(1)
            )

            action_text = self._clean_clause(
                match.group(2)
            )

            return self._build_after_steps(
                prerequisite,
                action_text,
            )

        # X after Y
        match = re.match(
            r"^(.+?)\s+(?:only\s+)?after\s+(.+)$",
            sentence,
            re.IGNORECASE,
        )

        if match:
            final_text = self._clean_clause(
                match.group(1)
            )

            prerequisite_text = self._clean_clause(
                match.group(2)
            )

            prerequisite_steps, _ = self._make_steps(
                self._split_actions(
                    prerequisite_text
                )
            )

            final_steps, _ = self._make_steps(
                self._split_actions(
                    final_text
                )
            )

            steps = (
                prerequisite_steps
                + final_steps
            )

            return (
                steps,
                self._sequential_dependencies(steps),
            )

        # X before Y
        match = re.match(
            r"^(.+?)\s+before\s+(.+)$",
            sentence,
            re.IGNORECASE,
        )

        if match:
            return self._make_steps(
                [
                    self._clean_clause(match.group(1)),
                    self._clean_clause(match.group(2)),
                ]
            )

        # Normal sequential actions
        return self._make_steps(
            self._split_actions(sentence)
        )

    # ==========================================================
    # PHASE 2
    # CONDITIONAL PARSING
    # ==========================================================

    def _parse_conditional(self, sentence):
        """
        Handles:

            if account does not exist, create account

            check account, if it doesn't exist create account

            check account if it doesn't exist create account
        """

        # ------------------------------------------------------
        # 1. if condition, action
        # ------------------------------------------------------

        match = re.match(
            r"^if\s+(.+?),\s*"
            r"(?:then\s+)?(.+)$",
            sentence,
            re.IGNORECASE,
        )

        if match:
            condition = self._normalize_condition(
                match.group(1)
            )

            action_text = self._clean_clause(
                match.group(2)
            )

            return self._build_pure_conditional(
                condition,
                action_text,
            )

        # ------------------------------------------------------
        # 2. main action, if condition action
        # ------------------------------------------------------

        match = re.match(
            r"^(.+?),\s*if\s+(.+)$",
            sentence,
            re.IGNORECASE,
        )

        if match:
            main_text = self._clean_clause(
                match.group(1)
            )

            remainder = self._clean_clause(
                match.group(2)
            )

            result = self._split_condition_and_action(
                remainder
            )

            if result:
                condition, action_text = result

                return self._build_conditional_steps(
                    main_text,
                    condition,
                    action_text,
                )

        # ------------------------------------------------------
        # 3. main action if condition action
        # ------------------------------------------------------

        match = re.match(
            r"^(.+?)\s+if\s+(.+)$",
            sentence,
            re.IGNORECASE,
        )

        if match:
            main_text = self._clean_clause(
                match.group(1)
            )

            remainder = self._clean_clause(
                match.group(2)
            )

            result = self._split_condition_and_action(
                remainder
            )

            if result:
                condition, action_text = result

                return self._build_conditional_steps(
                    main_text,
                    condition,
                    action_text,
                )

        return None

    def _split_condition_and_action(self, text):
        """
        Example:

            it doesn't exist create a new one

        becomes:

            condition = it doesn't exist
            action = create a new one
        """

        text = self._clean_clause(text)

        if not text:
            return None

        condition_pattern = re.compile(
            r"(?:"
            + "|".join(
                re.escape(marker)
                for marker in sorted(
                    self.CONDITION_MARKERS,
                    key=len,
                    reverse=True,
                )
            )
            + r")",
            re.IGNORECASE,
        )

        condition_match = condition_pattern.search(text)

        if not condition_match:
            return None

        action_pattern = re.compile(
            r"\b(?:"
            + "|".join(
                re.escape(action)
                for action in sorted(
                    self.ACTION_STARTERS,
                    key=len,
                    reverse=True,
                )
            )
            + r")\b",
            re.IGNORECASE,
        )

        action_match = action_pattern.search(
            text,
            condition_match.end(),
        )

        if not action_match:
            return None

        condition = self._clean_clause(
            text[:action_match.start()]
        )

        action_text = self._clean_clause(
            text[action_match.start():]
        )

        if not condition or not action_text:
            return None

        return condition, action_text

    # ==========================================================
    # CONDITIONAL BUILDERS
    # ==========================================================

    def _build_pure_conditional(
        self,
        condition,
        action_text,
    ):
        condition = self._normalize_condition(
            condition
        )

        action_steps, _ = self._make_steps(
            self._split_actions(action_text),
            condition,
        )

        for step in action_steps:
            step.branch = "then"

        return action_steps, []

    def _build_conditional_steps(
        self,
        main_text,
        condition,
        action_text,
    ):
        main_steps, _ = self._make_steps(
            self._split_actions(main_text)
        )

        action_steps, _ = self._make_steps(
            self._split_actions(action_text),
            self._normalize_condition(condition),
        )

        for step in action_steps:
            step.branch = "then"

        if not main_steps:
            return action_steps, []

        if not action_steps:
            return main_steps, []

        dependencies = []

        for previous in main_steps:
            for current in action_steps:
                dependencies.append({
                    "before": previous.text,
                    "after": current.text,
                    "relation": "PROMPT_CONDITION",
                    "condition": current.condition,
                    "branch": "then",
                })

        return (
            main_steps + action_steps,
            dependencies,
        )

    def _build_after_steps(
        self,
        prerequisite,
        action_text,
    ):
        prerequisite_steps, _ = self._make_steps(
            self._split_actions(prerequisite)
        )

        action_steps, _ = self._make_steps(
            self._split_actions(action_text)
        )

        steps = (
            prerequisite_steps
            + action_steps
        )

        return (
            steps,
            self._sequential_dependencies(steps),
        )

    # ==========================================================
    # CONDITIONS
    # ==========================================================

    def _normalize_condition(self, condition):
        condition = self._clean_clause(condition)

        condition = re.sub(
            r"^if\s+",
            "",
            condition,
            flags=re.IGNORECASE,
        )

        condition = re.sub(
            r"^then\s+",
            "",
            condition,
            flags=re.IGNORECASE,
        )

        return condition.strip()

    # ==========================================================
    # PHASE 3
    # ENTITIES
    # ==========================================================

    def _extract_step_entities(self, step, nlp):
        doc = nlp(step.text)

        if len(doc) == 0:
            return

        step.entities = [
            {
                "text": entity.text,
                "label": entity.label_,
            }
            for entity in doc.ents
        ]

    # ==========================================================
    # PHASE 3
    # PARAMETERS
    # ==========================================================

    def _extract_parameters(self, step):
        parameters = []

        currency_matches = list(
            re.finditer(
                r"(?:[$€£¥]\s*\d+(?:[.,]\d+)?"
                r"|"
                r"\d+(?:[.,]\d+)?\s*"
                r"(?:USD|EUR|GBP|PKR))",
                step.text,
                re.IGNORECASE,
            )
        )

        currency_spans = [
            match.span()
            for match in currency_matches
        ]

        for match in currency_matches:
            parameters.append({
                "text": match.group(0),
                "type": "currency",
            })

        for match in re.finditer(
            r"\b\d+(?:[.,]\d+)?\b",
            step.text,
        ):
            if any(
                start <= match.start() < end
                for start, end in currency_spans
            ):
                continue

            parameters.append({
                "text": match.group(0),
                "type": "number",
            })

        step.parameters = parameters

    # ==========================================================
    # PHASE 4
    # COREFERENCE
    # ==========================================================

    def _resolve_coreferences(self, steps):
        """
        Resolve simple references:

            check existing account
            create a new one

        =>

            create a new account
        """

        previous_object = ""
        previous_entity = ""

        for step in steps:

            references = {}

            replacement = (
                previous_object
                or previous_entity
            )

            text = step.text

            if replacement:

                for pronoun in self.PRONOUNS:

                    pattern = (
                        r"\b"
                        + re.escape(pronoun)
                        + r"\b"
                    )

                    if not re.search(
                        pattern,
                        text,
                        re.IGNORECASE,
                    ):
                        continue

                    references[pronoun] = replacement

                    text = re.sub(
                        pattern,
                        replacement,
                        text,
                        flags=re.IGNORECASE,
                    )

            step.references = references
            step.resolved_text = text

            # Conditions
            if step.condition and replacement:

                condition = step.condition

                for pronoun in self.PRONOUNS:

                    pattern = (
                        r"\b"
                        + re.escape(pronoun)
                        + r"\b"
                    )

                    if re.search(
                        pattern,
                        condition,
                        re.IGNORECASE,
                    ):
                        references[pronoun] = replacement

                        condition = re.sub(
                            pattern,
                            replacement,
                            condition,
                            flags=re.IGNORECASE,
                        )

                step.condition = condition

            # Object itself may be a pronoun.
            current_object = getattr(
                step,
                "object",
                "",
            )

            if (
                current_object
                and current_object.lower()
                in self.PRONOUNS
                and replacement
            ):
                step.object = replacement
                current_object = replacement

            # Update referent after resolution.
            if (
                current_object
                and current_object.lower()
                not in self.PRONOUNS
            ):
                previous_object = current_object

            if getattr(step, "entities", None):
                previous_entity = (
                    step.entities[-1]["text"]
                )

    # ==========================================================
    # PHASE 5
    # ELSE / OTHERWISE
    # ==========================================================

    def _parse_else(self, sentence):

        match = re.match(
            r"^(.+?)\s+"
            r"(?:else|otherwise)\s+"
            r"(.+)$",
            sentence,
            re.IGNORECASE,
        )

        if not match:
            return None

        first_text = self._clean_clause(
            match.group(1)
        )

        else_text = self._clean_clause(
            match.group(2)
        )

        first_steps, first_dependencies = (
            self._parse_sentence(first_text)
        )

        else_steps, _ = self._make_steps(
            self._split_actions(else_text)
        )

        for step in else_steps:
            step.branch = "else"

        dependencies = list(first_dependencies)

        if first_steps and else_steps:

            condition = ""

            for step in reversed(first_steps):
                if step.condition:
                    condition = step.condition
                    break

            for previous in first_steps:
                for current in else_steps:
                    dependencies.append({
                        "before": previous.text,
                        "after": current.text,
                        "relation": "PROMPT_BRANCH",
                        "condition": condition,
                        "branch": "else",
                    })

        return (
            first_steps + else_steps,
            dependencies,
        )

    def _enrich_branches(self, steps, dependencies):

        for step in steps:

            if (
                getattr(step, "condition", "")
                and not getattr(step, "branch", "")
            ):
                step.branch = "then"

            if getattr(step, "condition", ""):
                step.condition_negated = (
                    self._condition_is_negated(
                        step.condition
                    )
                )

        for dependency in dependencies:

            if (
                dependency.get("relation")
                == "PROMPT_CONDITION"
            ):
                dependency.setdefault(
                    "branch",
                    "then",
                )

    @staticmethod
    def _condition_is_negated(condition):
        return bool(
            re.search(
                r"\b(?:"
                r"not|never|no|doesn't|don't|"
                r"isn't|aren't|cannot|can't|"
                r"failed|fails"
                r")\b",
                condition.lower(),
            )
        )

    # ==========================================================
    # ACTION SPLITTING
    # ==========================================================

    def _split_actions(self, text):
        """
        Split a natural-language action sequence into atomic actions.

        Examples:

            reject the transaction, record the failed transaction,
            and notify the customer

        becomes:

            reject the transaction
            record the failed transaction
            notify the customer

        Also handles:

            create account and notify customer

            validate payment, then process payment

            save result; notify user
        """

        text = self._clean_clause(text)

        if not text:
            return []

        # ------------------------------------------------------
        # Explicit separators
        # ------------------------------------------------------

        if ";" in text:
            parts = []

            for section in text.split(";"):
                parts.extend(
                    self._split_actions(section)
                )

            return [
                part
                for part in parts
                if part
            ]

        # ------------------------------------------------------
        # Normalize sequencing phrases
        # ------------------------------------------------------

        # "and then"
        text = re.sub(
            r"\s+and\s+then\s+",
            "; ",
            text,
            flags=re.IGNORECASE,
        )

        # "then"
        text = re.sub(
            r"\s+then\s+",
            "; ",
            text,
            flags=re.IGNORECASE,
        )

        # "after that"
        text = re.sub(
            r"\s+after\s+that\s+",
            "; ",
            text,
            flags=re.IGNORECASE,
        )

        # "followed by"
        text = re.sub(
            r"\s+followed\s+by\s+",
            "; ",
            text,
            flags=re.IGNORECASE,
        )

        # ------------------------------------------------------
        # Build action-start regex
        # ------------------------------------------------------

        action_regex = (
            r"(?:"
            + "|".join(
                re.escape(action)
                for action in sorted(
                    self.ACTION_STARTERS,
                    key=len,
                    reverse=True,
                )
            )
            + r")"
        )

        # ------------------------------------------------------
        # Split:
        #
        #   ", and create ..."
        #   ", and record ..."
        #   ", and notify ..."
        #
        # The action lookahead is important. We don't want to
        # blindly split every occurrence of "and".
        # ------------------------------------------------------

        text = re.sub(
            r",\s+and\s+(?="
            + action_regex
            + r"\b)",
            "; ",
            text,
            flags=re.IGNORECASE,
        )

        # ------------------------------------------------------
        # Also support:
        #
        #   "reject transaction, record transaction"
        #
        # without requiring "and".
        # ------------------------------------------------------

        text = re.sub(
            r",\s+(?="
            + action_regex
            + r"\b)",
            "; ",
            text,
            flags=re.IGNORECASE,
        )

        # ------------------------------------------------------
        # Support:
        #
        #   "create account and notify customer"
        #
        # Again, only split "and" when another known action
        # begins immediately after it.
        # ------------------------------------------------------

        text = re.sub(
            r"\s+and\s+(?="
            + action_regex
            + r"\b)",
            "; ",
            text,
            flags=re.IGNORECASE,
        )

        # ------------------------------------------------------
        # Convert pieces into clean clauses.
        # ------------------------------------------------------

        parts = [
            self._clean_clause(part)
            for part in text.split(";")
        ]

        cleaned_parts = []

        for part in parts:

            if not part:
                continue

            part = re.sub(
                r"^(?:and|then)\s+",
                "",
                part,
                flags=re.IGNORECASE,
            )

            part = self._clean_clause(part)

            if part:
                cleaned_parts.append(part)

        return cleaned_parts

    # ==========================================================
    # DEPENDENCIES
    # ==========================================================

    @staticmethod
    def _sequential_dependencies(steps):
        return [
            {
                "before": steps[index].text,
                "after": steps[index + 1].text,
            }
            for index in range(len(steps) - 1)
        ]

    # ==========================================================
    # STEP CREATION
    # ==========================================================

    @staticmethod
    def _make_steps(parts, condition=""):

        steps = [
            SemanticStep(
                text=part,
                explicit=True,
                reason="User-provided request.",
                condition=condition,
            )
            for part in parts
            if part
        ]

        return (
            steps,
            SimpleSemanticParser._sequential_dependencies(
                steps
            ),
        )

    # ==========================================================
    # UTILITIES
    # ==========================================================

    @staticmethod
    def _clean_prompt(prompt):
        lines = []

        for line in str(prompt or "").splitlines():

            line = re.sub(
                r"^\s*[-*]\s*",
                "",
                line,
            ).strip()

            if line:
                lines.append(line)

        return (
            "; ".join(lines)
            if len(lines) > 1
            else " ".join(lines).strip()
        )

    @staticmethod
    def _clean_clause(clause):
        return re.sub(
            r"^[\s,.:]+|[\s,.:]+$",
            "",
            clause,
        ).strip()

    def _load_nlp(self):
        return _load_nlp(self)