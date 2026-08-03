from dataclasses import dataclass, field
from typing import List, Dict, Optional


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

    Examples

    Student
    Course
    TransferFunds
    CheckBalance
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
# PLANNER
# ==========================================================

@dataclass
class PlannerNode:

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
# MATCHED FUNCTION
# ==========================================================

@dataclass
class FunctionMatch:

    score: float

    function: RegisteredFunction