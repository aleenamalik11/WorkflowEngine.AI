from helpers.parser_utils import _extract_explicit_steps, _nlp_entities
from models import SemanticInterpretation, SemanticStep


class HybridSemanticParser():
    def __init__(self, llm_service=None, enable_llm=True):
        super().__init__()
        self.llm_service = llm_service
        self.enable_llm = enable_llm

    def parse(self, prompt: str, domain_context=None):
        steps, dependencies = _extract_explicit_steps(prompt)
        intent = self._infer_intent(prompt, steps)
        entities = _nlp_entities(prompt)
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