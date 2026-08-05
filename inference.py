import json

from prompt_builder_graph import PromptGraphBuilder
from graph_matcher import GraphMatcher
from candidate_retriever import CandidateRetriever
from graph_planner import GraphPlanner
from workflow_generator import WorkflowGenerator

from function_matcher import FunctionMatcher
from utils import (
    load_graph,
    EmbeddingService
)
from workflow_prompt_parser import WorkflowPromptParser

###############################################################
# CONFIG
###############################################################

EMBEDDING_MODEL =  "sentence-transformers/all-MiniLM-L6-v2"

DOMAIN_GRAPH = "models/domain_graph.pkl"

FUNCTIONS = "functions.json"

###############################################################
# LOAD COMPONENTS
###############################################################

print("=" * 70)
print("Loading AI Engine")
print("=" * 70)

###############################################################
# Parser
###############################################################

parser = WorkflowPromptParser()

###############################################################
# Embeddings
###############################################################

embedding_service = EmbeddingService(
    EMBEDDING_MODEL
)

###############################################################
# Prompt Graph Builder
###############################################################

prompt_graph_builder = PromptGraphBuilder(embedding_service)

###############################################################
# Domain Graph
###############################################################

domain_graph = load_graph(
    DOMAIN_GRAPH
)

###############################################################
# Matcher
###############################################################

graph_matcher = GraphMatcher(
    embedding_service,
    domain_graph
)

###############################################################
# Candidate Retriever
###############################################################

candidate_retriever = CandidateRetriever(
    domain_graph
)

###############################################################
# Planner
###############################################################

planner = GraphPlanner()

###############################################################
# Function Matcher
###############################################################

function_matcher = FunctionMatcher(
    EMBEDDING_MODEL
)

function_matcher.load(
    FUNCTIONS
)

###############################################################
# Workflow Generator
###############################################################

generator = WorkflowGenerator(
    function_matcher
)

###############################################################
# PROMPT
###############################################################

prompt = input("\nEnter workflow prompt:\n> ")

###############################################################
# STEP 1
###############################################################

print("\n[1] Parsing Prompt...")

analysis = parser.parse(
    prompt
)

###############################################################
# STEP 2
###############################################################

print("[2] Building Prompt Graph...")

prompt_graph = prompt_graph_builder.build(
    analysis
)

###############################################################
# STEP 3
###############################################################

print("[3] Matching Concepts...")

matched_graph = graph_matcher.match(
    prompt_graph
)

###############################################################
# STEP 4
###############################################################

print("[4] Retrieving Candidates...")

candidate_graph = candidate_retriever.retrieve(
    matched_graph
)

###############################################################
# STEP 5
###############################################################

print("[5] Ranking Missing Concepts...")

ranked_candidates = candidate_retriever.rank_missing_nodes(
    matched_graph,
    candidate_graph
)

###############################################################
# STEP 6
###############################################################

print("[6] Planning Workflow...")

plan = planner.plan(
    matched_graph,
    candidate_graph,
    ranked_candidates
)

###############################################################
# STEP 7
###############################################################

print("[7] Generating Workflow JSON...")

workflow = generator.generate(
    plan,
    workflow_name="Generated Workflow"
)

###############################################################
# OUTPUT
###############################################################

print()

print("=" * 70)
print("Generated Workflow")
print("=" * 70)

print()

print(json.dumps(
    workflow,
    indent=4
))
