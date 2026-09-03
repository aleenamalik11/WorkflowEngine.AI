"""Optional LLM semantic enrichment layered over deterministic parsing."""

from engine.parsers.semantic_parser import SimpleSemanticParser
from models import SemanticInterpretation, SemanticStep


class HybridSemanticParser:
    """Keep explicit parsing deterministic and use the LLM only for semantics."""

    def __init__(self, llm_service=None, enable_llm=True):
        self.llm_service = llm_service
        self.enable_llm = bool(enable_llm)
        self._deterministic_parser = SimpleSemanticParser()

    def parse(self, prompt: str, domain_context=None):
        """Parse ``prompt`` and optionally enrich it with LLM semantic output."""
        interpretation = self._deterministic_parser.parse(
            prompt,
            domain_context=domain_context,
        )

        if not self.enable_llm or self.llm_service is None:
            return interpretation

        result = self.llm_service.enrich_prompt(
            prompt,
            domain_context=domain_context or [],
        )
        enriched_steps = [
            *self._steps_from_llm(
                result.get("explicit_steps", result.get("requested_actions", [])),
                explicit=True,
            ),
            *self._steps_from_llm(result.get("inferred_steps", []), explicit=False),
        ]

        steps = self._merge_steps(interpretation.steps, enriched_steps)
        dependencies = self._dependencies(
            result.get("dependencies", result.get("relationships", []))
        ) or interpretation.dependencies

        return SemanticInterpretation(
            intent=result.get("intent", interpretation.intent),
            steps=steps,
            dependencies=dependencies,
            explicit_steps=[step for step in steps if step.explicit],
            inferred_steps=[step for step in steps if not step.explicit],
            mentioned_entities=(
                result.get("mentioned_entities") or interpretation.mentioned_entities
            ),
            constraints=result.get("constraints", interpretation.constraints),
        )

    @staticmethod
    def _steps_from_llm(items, explicit):
        return [
            SemanticStep(
                text=item.get("text", "").strip(),
                explicit=explicit,
                reason=item.get("reason", "LLM semantic enrichment."),
            )
            for item in items
            if isinstance(item, dict) and item.get("text", "").strip()
        ]

    @staticmethod
    def _dependencies(relationships):
        return [
            {
                "before": item.get("before") or item.get("source"),
                "after": item.get("after") or item.get("target"),
            }
            for item in relationships
            if isinstance(item, dict)
            and (item.get("before") or item.get("source"))
            and (item.get("after") or item.get("target"))
        ]

    @staticmethod
    def _merge_steps(explicit_steps, enriched_steps):
        merged = []
        seen = set()
        for step in [*explicit_steps, *enriched_steps]:
            key = " ".join(step.text.lower().split())
            if key and key not in seen:
                seen.add(key)
                merged.append(step)
        return merged
