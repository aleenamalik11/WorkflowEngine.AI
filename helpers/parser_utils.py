import re


def _extract_explicit_steps(self, prompt):
    text = self._clean_prompt(prompt)
    if not text:
        return [], []
    steps = []
    dependencies = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence = self._clean_clause(sentence)
        if not sentence:
            continue
        sentence_steps, sentence_dependencies = self._parse_sentence(sentence)
        if steps and sentence_steps:
            dependencies.append({"before": steps[-1].text,
                                 "after": sentence_steps[0].text})
        steps.extend(sentence_steps)
        dependencies.extend(sentence_dependencies)

    # Add a relation only when a parser supplied a condition without one.
    for i, step in enumerate(steps):
        if step.condition and not any(
                dependency.get("relation") == "PROMPT_CONDITION"
                and dependency.get("after") == step.text
                for dependency in dependencies
        ):
            # For conditional steps, create a PROMPT_CONDITION relation
            # that marks this step as conditional based on the previous step
            if i > 0:
                dependencies.append({
                    "before": steps[i - 1].text,
                    "after": step.text,
                    "relation": "PROMPT_CONDITION",
                    "condition": step.condition
                })
            else:
                # If first step has condition, it's a self-referential marker
                dependencies.append({
                    "before": step.text,
                    "after": step.text,
                    "relation": "PROMPT_CONDITION",
                    "condition": step.condition
                })

    return steps, dependencies

def _nlp_entities(self, prompt):
    nlp = self._load_nlp()
    if nlp is None:
        return []
    doc = nlp(str(prompt or ""))
    return [{"text": chunk.text, "root": chunk.root.lemma_.lower(),
             "label": chunk.label_} for chunk in doc.noun_chunks]

def _load_nlp(self):
    if self._nlp is not None:
        return self._nlp or None
    try:
        import spacy
        self._nlp = spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        self._nlp = False
    return self._nlp or None