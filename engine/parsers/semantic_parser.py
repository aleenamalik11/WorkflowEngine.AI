import re

from engine.parsers.conditional_clause_splitter import ConditionalClauseSplitter
from helpers.parser_utils import _load_nlp, _extract_explicit_steps, _nlp_entities
from models import SemanticStep, SemanticInterpretation


class SimpleSemanticParser:
    """Extract explicit actions without inventing workflow operations."""

    def __init__(self, use_nlp=True, nlp=None):
        self.use_nlp = use_nlp
        self._nlp = nlp

    def parse(self, prompt: str, domain_context=None):
        steps, dependencies = _extract_explicit_steps(prompt)
        entities = []
        if self.use_nlp:
            entities = _nlp_entities(prompt)
        return SemanticInterpretation(
            intent=steps[0].text if steps else str(prompt or "").strip(),
            steps=steps,
            dependencies=dependencies,
            explicit_steps=steps,
            mentioned_entities=entities,
        )



    def _parse_sentence(self, sentence):
        match = re.match(
            r"^if\s+(.+?),\s*(?:then\s+)?(.+)$",
            sentence, re.IGNORECASE,
        )
        if match:
            return self._make_steps(self._split_actions(match.group(2)), match.group(1))

        match = re.match(
            r"^after\s+([^,;]+),\s*(?:then\s+)?(.+)$",
            sentence, re.IGNORECASE,
        )
        if match:
            return self._make_steps(self._split_actions(match.group(2)), match.group(1))

        match = re.match(
            r"^(.+?)\s+(?:only\s+)?after\s+(.+)$",
            sentence, re.IGNORECASE,
        )
        if match:
            prerequisites = self._split_actions(match.group(2))
            actions, dependencies = self._make_steps(prerequisites)
            final_steps, _ = self._make_steps([match.group(1)])
            if actions:
                dependencies.append({"before": actions[-1].text,
                                     "after": final_steps[0].text})
            return actions + final_steps, dependencies

        match = re.match(r"^(.+?)\s+before\s+(.+)$", sentence, re.IGNORECASE)
        if match:
            return self._make_steps([match.group(1), match.group(2)])

        # Existing regex cases retain precedence; this is only for embedded
        # clauses such as "check the account, if it does not exist create".
        split = self._split_embedded_conditional(sentence)
        if split:
            main, action, condition = split
            main_steps, _ = self._make_steps(self._split_actions(main))
            action_steps, _ = self._make_steps(self._split_actions(action), condition)
            if main_steps and action_steps:
                return main_steps + action_steps, [{
                    "before": main_steps[-1].text,
                    "after": action_steps[0].text,
                    "relation": "PROMPT_CONDITION",
                    "condition": condition,
                }]

        return self._make_steps(self._split_actions(sentence))

    @staticmethod
    def _split_embedded_conditional(self, sentence):
        nlp = _load_nlp()
        if nlp is None:
            return None
        return ConditionalClauseSplitter(nlp).split(sentence)

    def _split_actions(self, text):
        if ";" in text:
            parts = []
            for section in text.split(";"):
                parts.extend(self._split_actions(section))
            return parts

        normalized_text = re.sub(
            r",\s+and\s+",
            "; ",
            text,
            flags=re.IGNORECASE,
        )
        if normalized_text != text:
            return self._split_actions(normalized_text)
        text = normalized_text

        text = re.sub(
            r",\s+and\s+(?=(?:[A-Za-z]+ing\b|check\b|verify\b|validate\b|"
            r"confirm\b|calculate\b|create\b|send\b|notify\b|mark\b|"
            r"retrieve\b|process\b|place\b|open\b|transfer\b|debit\b|"
            r"credit\b|persist\b|generate\b|update\b))",
            "; ",
            text,
            flags=re.IGNORECASE,
        )
        parts = re.split(
            r"\s*(?:;|,\s*(?:and\s+)?then\b)\s*|"
            r"\s+(?:and\s+then|then|after that|followed by)\s+|"
            r",\s*(?:and\s+)?(?=(?:[A-Za-z]+ing\b|check\b|verify\b|"
            r"validate\b|confirm\b|calculate\b|create\b|send\b|notify\b|mark\b|"
            r"retrieve\b|process\b|place\b|open\b|transfer\b|debit\b|credit\b|"
            r"persist\b|generate\b|update\b))|"
            r"\s+and\s+(?=(?:[A-Za-z]+ing\b|check\b|verify\b|validate\b|"
            r"confirm\b|calculate\b|create\b|send\b|notify\b|mark\b|retrieve\b|"
            r"process\b|place\b|open\b|transfer\b|debit\b|credit\b|persist\b|"
            r"generate\b|update\b))",
            text, re.IGNORECASE,
        )
        parts = [self._clean_clause(part) for part in parts]
        return [
            re.sub(r"^(?:and|then)\s+", "", part, flags=re.IGNORECASE)
            for part in parts
            if part
        ]

    @staticmethod
    def _make_steps(parts, condition=""):
        steps = [SemanticStep(
            text=part, explicit=True, reason="User-provided request.", condition=condition,
        ) for part in parts if part]
        return steps, [{"before": steps[index].text, "after": steps[index + 1].text}
                       for index in range(len(steps) - 1)]

    @staticmethod
    def _clean_prompt(prompt):
        lines = []
        for line in str(prompt or "").splitlines():
            line = re.sub(r"^\s*[-*]\s*", "", line).strip()
            if line:
                lines.append(line)
        return "; ".join(lines) if len(lines) > 1 else " ".join(lines).strip()

    @staticmethod
    def _clean_clause(clause):
        return re.sub(r"^[\s,.:]+|[\s,.:]+$", "", clause).strip()