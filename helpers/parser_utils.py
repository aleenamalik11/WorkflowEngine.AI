import re


def _extract_explicit_steps(self, prompt):
    """
    Extract explicit semantic steps and prompt-level dependencies.

    This function is responsible only for understanding the user's
    language. It must NOT map steps to domain functions.

    Important:
        Conditional actions are kept as separate semantic steps.

    Example:

        Check the customer's account balance.
        If the balance is insufficient, reject the transaction,
        record the failed transaction, and notify the customer.

    becomes:

        1. Check the customer's account balance
        2. reject the transaction
        3. record the failed transaction
        4. notify the customer

    where steps 2-4 have the same condition.
    """

    text = self._clean_prompt(prompt)

    if not text:
        return [], []

    steps = []
    dependencies = []

    # ----------------------------------------------------------
    # Split sentences while preserving normal punctuation.
    # ----------------------------------------------------------

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    for sentence in sentences:

        sentence = self._clean_clause(sentence)

        if not sentence:
            continue

        sentence_steps, sentence_dependencies = (
            self._parse_sentence(sentence)
        )

        if not sentence_steps:
            continue

        # ------------------------------------------------------
        # Connect separate sentences sequentially.
        #
        # Example:
        #
        #   Check account.
        #   Send notification.
        #
        # becomes:
        #
        #   Check account -> Send notification
        # ------------------------------------------------------

        if steps:
            dependencies.append({
                "before": steps[-1].text,
                "after": sentence_steps[0].text,
                "relation": "PROMPT_DEPENDENCY",
            })

        steps.extend(sentence_steps)
        dependencies.extend(sentence_dependencies)

    # ----------------------------------------------------------
    # DO NOT create self-referential conditional dependencies.
    #
    # The old implementation generated:
    #
    #   step -> step
    #
    # for a conditional first step.
    #
    # That is not a meaningful workflow dependency and can later
    # produce cycles in the graph.
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # Ensure every conditional step has an explicit branch.
    # ----------------------------------------------------------

    for step in steps:

        condition = getattr(
            step,
            "condition",
            "",
        )

        if not condition:
            continue

        if not getattr(step, "branch", ""):
            step.branch = "then"

    # ----------------------------------------------------------
    # Build missing condition dependencies.
    #
    # A condition belongs to the action(s) governed by it.
    #
    # Example:
    #
    #   Check balance
    #   reject transaction       [condition]
    #   record failed transaction[condition]
    #   notify customer          [condition]
    #
    # Each conditional action is connected to the condition's
    # preceding step.
    #
    # We deliberately do NOT create:
    #
    #   reject -> record
    #   record -> notify
    #
    # here as PROMPT_CONDITION relations.
    #
    # Those are normal execution dependencies and should remain
    # separate from the condition relationship.
    # ----------------------------------------------------------

    for index, step in enumerate(steps):

        condition = getattr(
            step,
            "condition",
            "",
        )

        if not condition:
            continue

        already_has_condition_dependency = any(
            dependency.get("relation")
            == "PROMPT_CONDITION"
            and dependency.get("after")
            == step.text
            for dependency in dependencies
        )

        if already_has_condition_dependency:
            continue

        # ------------------------------------------------------
        # A conditional action normally depends on the step
        # immediately before the conditional block.
        #
        # For:
        #
        #   Check balance
        #   reject transaction
        #   record failed transaction
        #   notify customer
        #
        # the condition source is "Check balance" for all three
        # conditional actions.
        # ------------------------------------------------------

        condition_source = _find_condition_source(
            steps,
            index,
            condition,
        )

        if condition_source is None:
            continue

        dependencies.append({
            "before": condition_source.text,
            "after": step.text,
            "relation": "PROMPT_CONDITION",
            "condition": condition,
            "branch": getattr(
                step,
                "branch",
                "then",
            ) or "then",
        })

    return steps, _deduplicate_dependencies(dependencies)


def _find_condition_source(steps, conditional_index, condition):
    """
    Find the semantic step that establishes a condition.

    For a sequence such as:

        Check account balance
        reject transaction [balance insufficient]
        record transaction [balance insufficient]
        notify customer [balance insufficient]

    the source is the nearest preceding non-conditional step:

        Check account balance
    """

    normalized_condition = _normalize_text(condition)

    # ----------------------------------------------------------
    # First look backward through the current conditional block.
    #
    # If another conditional step has the same condition, reuse
    # its condition source when possible.
    # ----------------------------------------------------------

    for index in range(
        conditional_index - 1,
        -1,
        -1,
    ):
        previous = steps[index]

        previous_condition = getattr(
            previous,
            "condition",
            "",
        )

        if previous_condition:
            if (
                _normalize_text(previous_condition)
                == normalized_condition
            ):
                # Continue looking for the first step before
                # this conditional group.
                continue

            # Different condition means we crossed a condition
            # boundary.
            break

        # First non-conditional step encountered.
        return previous

    # ----------------------------------------------------------
    # Fallback: nearest preceding step.
    # ----------------------------------------------------------

    if conditional_index > 0:
        return steps[conditional_index - 1]

    # ----------------------------------------------------------
    # A condition with no preceding semantic action does not have
    # a condition source at this stage.
    #
    # The caller can still preserve the condition on the step.
    # Stage 10 can later create an explicit condition node.
    # ----------------------------------------------------------

    return None


def _deduplicate_dependencies(dependencies):
    """
    Remove duplicate dependencies while preserving order.

    Dependencies are compared using their complete semantic
    relationship rather than only before/after because the same
    pair of nodes may legitimately have different relation types.
    """

    result = []
    seen = set()

    for dependency in dependencies:

        key = (
            dependency.get("before", ""),
            dependency.get("after", ""),
            dependency.get("relation", ""),
            dependency.get("condition", ""),
            dependency.get("branch", ""),
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(dependency)

    return result


def _normalize_text(value):
    """Normalize text for safe semantic comparison."""

    return re.sub(
        r"\s+",
        " ",
        str(value or "").strip().lower(),
    )


def _nlp_entities(self, prompt):
    nlp = self._load_nlp()

    if nlp is None:
        return []

    doc = nlp(str(prompt or ""))

    return [
        {
            "text": chunk.text,
            "root": chunk.root.lemma_.lower(),
            "label": chunk.label_,
        }
        for chunk in doc.noun_chunks
    ]


def _load_nlp(self):
    if self._nlp is not None:
        return self._nlp or None

    try:
        import spacy

        self._nlp = spacy.load(
            "en_core_web_sm"
        )

    except (ImportError, OSError):
        self._nlp = False

    return self._nlp or None