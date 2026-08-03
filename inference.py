import json

from graph_search import GraphSearch
from function_matcher import FunctionMatcher
from planner import WorkflowPlanner

###############################################################
# CONFIGURATION
###############################################################

MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"

GRAPH_PATH = "models/workflow_graph.gpickle"

FAISS_INDEX = "models/workflow.index"

METADATA = "models/workflow_metadata.pkl"

REGISTERED_FUNCTIONS = "functions.json"

OUTPUT = "generated_workflow.json"

###############################################################
# LOAD COMPONENTS
###############################################################

print("=" * 60)
print("Loading AI Components")
print("=" * 60)

graph_search = GraphSearch(
    MODEL_PATH,
    GRAPH_PATH,
    FAISS_INDEX,
    METADATA
)

matcher = FunctionMatcher(MODEL_PATH)

matcher.load(REGISTERED_FUNCTIONS)

planner = WorkflowPlanner(
    graph_search,
    matcher
)

###############################################################
# INFERENCE LOOP
###############################################################

while True:

    print()

    prompt = input("Admin Prompt (exit to quit): ")

    if prompt.lower() == "exit":
        break

    print()

    print("Generating workflow...")

    workflow = planner.plan(prompt)

    print()

    print("=" * 60)
    print("Workflow")
    print("=" * 60)

    print()

    print("Name:", workflow.name)

    print()

    print("Nodes")

    for node in workflow.nodes:

        print(
            node.id,
            node.function_name
        )

    print()

    print("Connections")

    for edge in workflow.edges:

        print(
            edge.source,
            "--",
            edge.transition,
            "-->",
            edge.target
        )

    ###########################################################
    # Save
    ###########################################################

    output = {

        "name": workflow.name,

        "inputs": workflow.inputs,

        "nodes": [

            {

                "id": n.id,

                "function": n.function_name,

                "type": n.node_type

            }

            for n in workflow.nodes

        ],

        "connections": [

            {

                "source": e.source,

                "target": e.target,

                "transition": e.transition

            }

            for e in workflow.edges

        ]

    }

    with open(
            OUTPUT,
            "w",
            encoding="utf8") as f:

        json.dump(
            output,
            f,
            indent=4
        )

    print()

    print("Saved to generated_workflow.json")

    print()