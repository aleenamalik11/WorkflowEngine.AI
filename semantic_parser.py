"""
Semantic parsing.

The LLM discovers semantic workflow concepts.

It does NOT choose implementation functions and does NOT create
the workflow graph.

The domain graph later grounds the concepts.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SemanticStep:

    text: str

    explicit: bool = True

    reason: str = ""

    domain_candidates: List[Any] = field(
        default_factory=list
    )


@dataclass
class SemanticInterpretation:

    intent: str

    steps: List[SemanticStep]

    dependencies: List[Dict[str, str]] = field(
        default_factory=list
    )

    explicit_steps: List[SemanticStep] = field(
        default_factory=list
    )

    inferred_steps: List[SemanticStep] = field(
        default_factory=list
    )

    def as_debug_dict(self):

        return {
            "intent": self.intent,

            "steps": [
                {
                    "text": step.text,
                    "explicit": step.explicit,
                    "reason": step.reason,
                }
                for step in self.steps
            ],

            "dependencies": self.dependencies,
        }


class HybridSemanticParser:

    def __init__(
        self,
        llm_service=None,
        enable_llm=True,
    ):

        self.llm_service = llm_service
        self.enable_llm = enable_llm

    def parse(
        self,
        prompt: str,
        domain_context=None,
    ):

        explicit_steps = (
            self._extract_explicit_steps(
                prompt
            )
        )

        intent = self._infer_intent(
            prompt,
            explicit_steps,
        )

        inferred_steps = []
        dependencies = []

        if (
            self.enable_llm
            and self.llm_service
        ):

            llm_result = (
                self.llm_service.enrich_prompt(
                    prompt,
                    domain_context=domain_context,
                )
            )

            intent = llm_result.get(
                "intent",
                intent,
            )

            inferred_steps = [
                SemanticStep(
                    text=item.get(
                        "text",
                        "",
                    ).strip(),
                    explicit=False,
                    reason=item.get(
                        "reason",
                        "",
                    ),
                )
                for item in llm_result.get(
                    "inferred_steps",
                    [],
                )
                if item.get("text")
            ]

            llm_explicit = [
                SemanticStep(
                    text=item.get(
                        "text",
                        "",
                    ).strip(),
                    explicit=True,
                    reason=item.get(
                        "reason",
                        "",
                    ),
                )
                for item in llm_result.get(
                    "explicit_steps",
                    [],
                )
                if item.get("text")
            ]

            if not explicit_steps and llm_explicit:
                explicit_steps = llm_explicit

            dependencies = (
                llm_result.get(
                    "dependencies",
                    [],
                )
            )

        steps = self._merge_steps(
            explicit_steps,
            inferred_steps,
        )

        return SemanticInterpretation(
            intent=intent,
            steps=steps,
            dependencies=dependencies,
            explicit_steps=explicit_steps,
            inferred_steps=inferred_steps,
        )

    def _merge_steps(
        self,
        explicit_steps,
        inferred_steps,
    ):

        result = []
        seen = set()

        for step in explicit_steps:

            key = self._normalize(
                step.text
            )

            if key not in seen:

                seen.add(key)
                result.append(step)

        for step in inferred_steps:

            key = self._normalize(
                step.text
            )

            if key not in seen:

                seen.add(key)
                result.append(step)

        return result

    @staticmethod
    def _normalize(text):

        return " ".join(
            text.lower()
            .strip()
            .split()
        )

    def _extract_explicit_steps(
        self,
        prompt,
    ):

        # Basic deterministic fallback.
        #
        # LLM enrichment can identify additional
        # semantic steps.

        separators = [
            ", then ",
            " then ",
            ";",
        ]

        text = prompt.strip()

        for separator in separators:

            if separator in text:

                parts = [
                    p.strip()
                    for p in text.split(
                        separator
                    )
                    if p.strip()
                ]

                return [
                    SemanticStep(
                        text=part,
                        explicit=True,
                        reason=(
                            "User-provided request."
                        ),
                    )
                    for part in parts
                ]

        return [
            SemanticStep(
                text=text,
                explicit=True,
                reason=(
                    "User-provided request."
                ),
            )
        ]

    def _infer_intent(
        self,
        prompt,
        explicit_steps,
    ):

        if explicit_steps:

            return explicit_steps[0].text

        return prompt.strip()