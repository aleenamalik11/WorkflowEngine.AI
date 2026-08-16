from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


# ==========================================================
# DATASET MODELS
# ==========================================================

@dataclass
class TrainingExample:
    """
    One row from the training dataset.
    """

    prompt: str

    workflow: dict


# ==========================================================
# KNOWLEDGE GRAPH
# ==========================================================

@dataclass
class GraphNode:
    """
    Represents a concept inside the knowledge graph.

    Examples:

        Student
        Course
        Transfer Funds
        Check Balance
        GPA
    """

    id: str

    name: str

    node_type: str

    embedding: Optional[List[float]] = None

    metadata: Dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    """
    Relationship between two graph nodes.
    """

    source: str

    target: str

    relation: str

    weight: float = 1.0


# ==========================================================
# REGISTERED FUNCTIONS
# ==========================================================

@dataclass
class RegisteredFunction:
    """
    Represents one function exported by the workflow engine.
    """

    name: str

    description: str

    inputs: List[str]

    outputs: List[str]

    embedding: Optional[List[float]] = None


# ==========================================================
# FUNCTION MATCHING
# ==========================================================

@dataclass
class FunctionMatch:
    """
    Result of matching a semantic/domain concept against
    the registered functions.

    A match is considered usable only when `found` is True.

    The matcher must never replace a domain concept with a
    function simply because that function happens to be the
    closest available candidate.
    """

    score: float = 0.0

    function: Optional[RegisteredFunction] = None

    found: bool = False

    status: str = "matching function not found"

    semantic_text: str = ""

    def to_dict(self) -> dict:
        """
        Convert the function match into JSON-safe data.
        """

        if self.function is None:
            return {
                "Found": False,
                "FunctionName": None,
                "Description": "",
                "Inputs": [],
                "Outputs": [],
                "Score": self.score,
                "Status": self.status,
                "SemanticText": self.semantic_text,
            }

        return {
            "Found": self.found,
            "FunctionName": self.function.name,
            "Description": self.function.description,
            "Inputs": list(self.function.inputs or []),
            "Outputs": list(self.function.outputs or []),
            "Score": self.score,
            "Status": self.status,
            "SemanticText": self.semantic_text,
        }


# ==========================================================
# WORKFLOW FUNCTION DETAILS
# ==========================================================

@dataclass
class WorkflowFunctionDetails:
    """
    Implementation details attached to a semantic workflow node.

    The workflow node itself represents the domain meaning.

    This object represents the implementation mapping.

    Example:

        Workflow node:
            Name = "Debit Account"

        FunctionDetails:
            FunctionName = "DebitAccountAsync"

    If no registered function can safely implement the semantic
    node:

        Found = False
        FunctionName = None
        Status = "matching function not found"

    The semantic node remains in the workflow.
    """

    found: bool = False

    function_name: Optional[str] = None

    description: str = ""

    inputs: List[str] = field(default_factory=list)

    outputs: List[str] = field(default_factory=list)

    score: float = 0.0

    status: str = "matching function not found"

    semantic_text: str = ""

    @classmethod
    def from_match(
        cls,
        match: Optional[FunctionMatch],
        semantic_text: str,
    ):
        """
        Build implementation details from a FunctionMatch.
        """

        if match is None:
            return cls(
                found=False,
                function_name=None,
                description="",
                inputs=[],
                outputs=[],
                score=0.0,
                status="matching function not found",
                semantic_text=semantic_text,
            )

        if not match.found or match.function is None:
            return cls(
                found=False,
                function_name=None,
                description="",
                inputs=[],
                outputs=[],
                score=match.score,
                status="matching function not found",
                semantic_text=semantic_text,
            )

        return cls(
            found=True,
            function_name=match.function.name,
            description=match.function.description,
            inputs=list(match.function.inputs or []),
            outputs=list(match.function.outputs or []),
            score=match.score,
            status="matched",
            semantic_text=semantic_text,
        )

    def to_dict(self) -> dict:
        """
        Convert to the JSON representation expected by the
        workflow engine.
        """

        return {
            "Found": self.found,
            "FunctionName": self.function_name,
            "Description": self.description,
            "Inputs": list(self.inputs),
            "Outputs": list(self.outputs),
            "Score": self.score,
            "Status": self.status,
            "SemanticText": self.semantic_text,
        }


# ==========================================================
# PLANNER
# ==========================================================

@dataclass
class PlannerNode:
    """
    Intermediate planner representation.

    `function_name` is retained for backward compatibility with
    older planner code, but Stage 10/11 should derive the actual
    function mapping from the selected semantic node.
    """

    id: str

    function_name: str

    node_type: str = "custom"

    metadata: Dict = field(default_factory=dict)


@dataclass
class PlannerEdge:

    source: str

    target: str

    transition: str = "success"


@dataclass
class WorkflowPlan:
    """
    AI planner output before converting to C# WorkflowModel.
    """

    name: str

    nodes: List[PlannerNode]

    edges: List[PlannerEdge]

    inputs: List[str]


# ==========================================================
# SEARCH RESULTS
# ==========================================================

@dataclass
class SearchResult:

    score: float

    workflow: WorkflowPlan


# ==========================================================
# HELPER
# ==========================================================

def function_details_from_match(
    match: Optional[FunctionMatch],
    semantic_text: str,
) -> WorkflowFunctionDetails:
    """
    Convenience helper used by Stage 10/11.
    """

    return WorkflowFunctionDetails.from_match(
        match=match,
        semantic_text=semantic_text,
    )