"""
Stage 1 output contract.

Every prompt -- simple or complex, spaCy-parsed or LLM-parsed -- is
normalized into this same structure before anything touches the domain
graph. This is what lets Stage 2 swap parsers without Stage 3+ caring.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SemanticOperation:
    text: str            # "transfer the funds"
    intent: str           # normalized lemma/phrase: "transfer funds"
    negated: bool = False
    condition: Optional[str] = None   # e.g. "if balance sufficient"


@dataclass
class SemanticEntity:
    text: str
    type_hint: str = "DomainEntity"   # DomainEntity | Actor | Component


@dataclass
class SemanticActor:
    text: str


@dataclass
class SemanticEvent:
    text: str


@dataclass
class SemanticRule:
    text: str


@dataclass
class PromptConstraint:
    """
    A relationship *inferred from the user's language*, not from the
    domain graph. Never written back to Neo4j -- consumed only by the
    planner as a soft/hard ordering constraint.
    """
    source: str            # intent string
    target: str            # intent string
    relation: str           # "PROMPT_PRECEDES" | "PROMPT_AND" | "PROMPT_CONDITION" | ...
    hard: bool = True       # hard = must hold in final plan; soft = preference


@dataclass
class SemanticInterpretation:
    operations: List[SemanticOperation] = field(default_factory=list)
    entities: List[SemanticEntity] = field(default_factory=list)
    actors: List[SemanticActor] = field(default_factory=list)
    events: List[SemanticEvent] = field(default_factory=list)
    rules: List[SemanticRule] = field(default_factory=list)
    constraints: List[PromptConstraint] = field(default_factory=list)
    source: str = "rule_based"   # "rule_based" | "llm"
    raw_prompt: str = ""

    def as_debug_dict(self):
        return {
            "source": self.source,
            "operations": [o.__dict__ for o in self.operations],
            "entities": [e.__dict__ for e in self.entities],
            "actors": [a.__dict__ for a in self.actors],
            "events": [e.__dict__ for e in self.events],
            "rules": [r.__dict__ for r in self.rules],
            "constraints": [c.__dict__ for c in self.constraints],
        }