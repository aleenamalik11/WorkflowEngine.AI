import json

from prompt_builder_graph import PromptGraphBuilder
from graph_matcher import GraphMatcher
from beam_search_planner import BeamSearchPlanner
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

TOP_K = 5

BEAM_WIDTH = 3

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
# Beam Search Planner
###############################################################

beam_planner = BeamSearchPlanner(
    beam_width=BEAM_WIDTH
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

print("[3] Matching Concepts (Top K candidates per action)...")

candidate_plan = graph_matcher.candidates(
    prompt_graph,
    k=TOP_K
)

graph_matcher.print_candidates(
    candidate_plan
)

###############################################################
# STEP 4
###############################################################

print("\n[4] Running Beam Search Planner...")

search_result = beam_planner.search(
    candidate_plan
)

beam_planner.print_search(
    search_result
)

###############################################################
# STEP 5
###############################################################

print("\n[5] Expanding shortest domain paths...")

workflow_graph = beam_planner.expand(
    search_result
)

plan = planner.plan(
    workflow_graph
)

planner.print_plan(
    plan
)

###############################################################
# STEP 6
###############################################################

print("\n[6] Generating Workflow JSON...")

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
