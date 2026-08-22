"""Convert workflow prompts into explicit semantic steps."""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List


@dataclass
class SemanticStep:
    text: str
    explicit: bool = True
    reason: str = ""
    condition: str = ""
    domain_candidates: List[Any] = field(default_factory=list)


@dataclass
class SemanticInterpretation:
    intent: str
    steps: List[SemanticStep]
    dependencies: List[Dict[str, str]] = field(default_factory=list)
    explicit_steps: List[SemanticStep] = field(default_factory=list)
    inferred_steps: List[SemanticStep] = field(default_factory=list)
    mentioned_entities: List[Any] = field(default_factory=list)
    constraints: List[Any] = field(default_factory=list)

    def as_debug_dict(self):
        return {
            "intent": self.intent,
            "steps": [
                {"text": step.text, "explicit": step.explicit,
                 "reason": step.reason, "condition": step.condition}
                for step in self.steps
            ],
            "dependencies": self.dependencies,
            "mentioned_entities": self.mentioned_entities,
            "constraints": self.constraints,
        }


class SimpleSemanticParser:
    """Extract explicit actions without inventing workflow operations."""

    def __init__(self, use_nlp=True, nlp=None):
        self.use_nlp = use_nlp
        self._nlp = nlp

    def parse(self, prompt: str, domain_context=None):
        steps, dependencies = self._extract_explicit_steps(prompt)
        entities = []
        if self.use_nlp:
            entities = self._nlp_entities(prompt)
        return SemanticInterpretation(
            intent=steps[0].text if steps else str(prompt or "").strip(),
            steps=steps,
            dependencies=dependencies,
            explicit_steps=steps,
            mentioned_entities=entities,
        )

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
        return steps, dependencies

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

        return self._make_steps(self._split_actions(sentence))

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

    @staticmethod
    def _normalize(text):
        return " ".join(str(text).lower().strip().split())


class HybridSemanticParser(SimpleSemanticParser):
    def __init__(self, llm_service=None, enable_llm=True):
        super().__init__()
        self.llm_service = llm_service
        self.enable_llm = enable_llm

    def parse(self, prompt: str, domain_context=None):
        steps, dependencies = self._extract_explicit_steps(prompt)
        intent = self._infer_intent(prompt, steps)
        entities = self._nlp_entities(prompt)
        constraints = []
        if self.enable_llm and self.llm_service:
            result = self.llm_service.enrich_prompt(prompt)
            intent = result.get("intent", intent)
            llm_steps = [SemanticStep(
                text=item.get("text", "").strip(), explicit=True,
                reason="Explicit action extracted by the LLM.",
            ) for item in result.get("requested_actions", []) if item.get("text")]
            if llm_steps:
                steps = llm_steps
            dependencies = self._explicit_dependencies(result.get("relationships", [])) or dependencies
            entities = result.get("mentioned_entities", []) or entities
            constraints = result.get("constraints", [])
        return SemanticInterpretation(
            intent=intent, steps=self._merge_steps(steps, []),
            dependencies=dependencies, explicit_steps=steps,
            mentioned_entities=entities, constraints=constraints,
        )

    @staticmethod
    def _merge_steps(explicit_steps, inferred_steps):
        result, seen = [], set()
        for step in [*explicit_steps, *inferred_steps]:
            key = " ".join(step.text.lower().split())
            if key not in seen:
                seen.add(key)
                result.append(step)
        return result

    @staticmethod
    def _explicit_dependencies(relationships):
        return [{"before": item.get("before") or item.get("source"),
                 "after": item.get("after") or item.get("target")}
                for item in relationships if isinstance(item, dict)
                and (item.get("before") or item.get("source"))
                and (item.get("after") or item.get("target"))]

    @staticmethod
    def _infer_intent(prompt, steps):
        return steps[0].text if steps else str(prompt or "").strip()
