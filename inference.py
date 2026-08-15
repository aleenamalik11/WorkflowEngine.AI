"""
Command-line entry point for WorkflowEngine.AI.

Usage:

    python inference.py "transfer funds"

or:

    python inference.py "check balance, then transfer funds"
"""

import sys
import json
import os

from pipeline import WorkflowPipeline

from llm_service import LLMService

from domain_graph_client import (
    InMemoryDomainGraph,
    DomainNode,
    DomainRelationship,
)

from utils import EmbeddingService

from function_matcher import FunctionMatcher


def build_domain_graph():
    """
    Replace this with your actual Neo4jDomainGraph configuration
    when running against the real domain graph.

    Keeping this function here makes inference easy to run locally.
    """

    nodes = {

        "validate_transfer":
            DomainNode(
                id="validate_transfer",
                name="Validate Transfer Request",
                node_type="Operation",
            ),

        "check_balance":
            DomainNode(
                id="check_balance",
                name="Check Balance",
                node_type="Operation",
            ),

        "process_transfer":
            DomainNode(
                id="process_transfer",
                name="Process Transfer",
                node_type="Operation",
            ),

        "generate_transfer_response":
            DomainNode(
                id="generate_transfer_response",
                name="Generate Transfer Response",
                node_type="Operation",
            ),

        "transfer_funds":
            DomainNode(
                id="transfer_funds",
                name="Transfer Funds",
                node_type="Operation",
            ),
    }

    relationships = [

        DomainRelationship(
            "validate_transfer",
            "check_balance",
            "OPERATION_PRECEDES",
        ),

        DomainRelationship(
            "check_balance",
            "process_transfer",
            "OPERATION_PRECEDES",
        ),

        DomainRelationship(
            "process_transfer",
            "generate_transfer_response",
            "OPERATION_PRECEDES",
        ),

        DomainRelationship(
            "validate_transfer",
            "transfer_funds",
            "OPERATION_PRECEDES",
        ),

        DomainRelationship(
            "transfer_funds",
            "generate_transfer_response",
            "OPERATION_PRECEDES",
        ),
    ]

    return InMemoryDomainGraph(
        nodes,
        relationships,
    )


def main():

    # ---------------------------------------------------------
    # Get prompt
    # ---------------------------------------------------------

    if len(sys.argv) > 1:

        prompt = " ".join(
            sys.argv[1:]
        )

    else:

        prompt = input(
            "Enter workflow prompt: "
        ).strip()

    if not prompt:

        print("No prompt supplied.")

        return

    print(
        f"\nUser prompt:\n{prompt}\n"
    )

    # ---------------------------------------------------------
    # LLM
    # ---------------------------------------------------------

    llm_service = LLMService(
        model=os.getenv(
            "HF_MODEL",
            "Qwen/Qwen2.5-7B-Instruct",
        )
    )

    # ---------------------------------------------------------
    # Embeddings
    # ---------------------------------------------------------

    embedding_service = EmbeddingService(
        os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
    )

    # ---------------------------------------------------------
    # Domain graph
    # ---------------------------------------------------------

    domain_client = build_domain_graph()

    # ---------------------------------------------------------
    # Function matcher
    # ---------------------------------------------------------

    function_matcher = FunctionMatcher(
        os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
    )

    function_matcher.load(
        os.getenv(
            "FUNCTIONS_FILE",
            "functions.json",
        )
    )

    # ---------------------------------------------------------
    # Pipeline
    # ---------------------------------------------------------

    pipeline = WorkflowPipeline(
        domain_client=domain_client,
        embedding_service=embedding_service,
        function_matcher=function_matcher,
        llm_service=llm_service,
        beam_width=3,
        top_k=5,
        neighborhood_depth=1,
        verbose=True,
    )

    # ---------------------------------------------------------
    # Execute
    # ---------------------------------------------------------

    workflow, debug = pipeline.run(
        prompt,
        workflow_name="Generated Workflow",
    )

    # ---------------------------------------------------------
    # Result
    # ---------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print("FINAL WORKFLOW")

    print(
        "=" * 70
    )

    print(
        json.dumps(
            workflow,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()