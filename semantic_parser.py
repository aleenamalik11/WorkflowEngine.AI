"""
Semantic parsing.

The parser identifies what the user is asking for.

LLM enrichment is used to discover implicit workflow concepts,
but the LLM does not create the workflow itself.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SemanticStep:

    text: str

    explicit: bool = True

    reason: str = ""

    # Filled later by domain matching.
    domain_candidates: List[Any] = field(default_factory=list)


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

    def parse(self, prompt: str) -> SemanticInterpretation:

        # ---------------------------------------------------------
        # Existing deterministic parsing
        # ---------------------------------------------------------

        explicit_steps = self._extract_explicit_steps(prompt)

        intent = self._infer_intent(prompt, explicit_steps)

        # ---------------------------------------------------------
        # LLM semantic enrichment
        # ---------------------------------------------------------

        inferred_steps = []
        dependencies = []

        if self.enable_llm and self.llm_service:

            llm_result = self.llm_service.enrich_prompt(
                prompt
            )

            intent = llm_result.get(
                "intent",
                intent,
            )

            inferred_steps = [
                SemanticStep(
                    text=item["text"],
                    explicit=False,
                    reason=item.get("reason", ""),
                )
                for item in llm_result.get(
                    "inferred_steps",
                    [],
                )
            ]

            # We also allow the LLM to identify explicit steps.
            #
            # However, deterministic parsing remains authoritative
            # when possible.
            llm_explicit = [
                SemanticStep(
                    text=item["text"],
                    explicit=True,
                    reason=item.get("reason", ""),
                )
                for item in llm_result.get(
                    "explicit_steps",
                    [],
                )
            ]

            if not explicit_steps and llm_explicit:
                explicit_steps = llm_explicit

            dependencies = llm_result.get(
                "dependencies",
                [],
            )

        # ---------------------------------------------------------
        # Remove duplicate concepts
        # ---------------------------------------------------------

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

        # Explicit steps always get priority.
        for step in explicit_steps:

            key = self._normalize(step.text)

            if key not in seen:
                seen.add(key)
                result.append(step)

        for step in inferred_steps:

            key = self._normalize(step.text)

            if key not in seen:
                seen.add(key)
                result.append(step)

        return result

    @staticmethod
    def _normalize(text):

        return " ".join(
            text.lower().strip().split()
        )

    # -------------------------------------------------------------
    # Keep your existing implementation here
    # -------------------------------------------------------------

    def _extract_explicit_steps(self, prompt):

        # IMPORTANT:
        # Replace this body with your existing Stage 1 parser.
        #
        # This fallback allows a simple prompt to become one step.
        #
        # Your existing parser should remain here instead.

        return [
            SemanticStep(
                text=prompt.strip(),
                explicit=True,
                reason="User-provided request.",
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