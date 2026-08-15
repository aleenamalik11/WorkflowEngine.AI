"""
Command-line entry point.

Examples:

    python inference.py "transfer funds"

    python inference.py "check balance, then transfer funds"

The command-line layer is intentionally thin.

All workflow reasoning happens inside WorkflowPipeline.
"""

import json
import os
import sys

from pipeline import WorkflowPipeline

from llm_service import LLMService

from domain_graph_client import (
    InMemoryDomainGraph,
    DomainNode,
    DomainRelationship,
)

from utils import EmbeddingService

from function_matcher import FunctionMatcher


def build_domain_graph(
    embedding_service,
):
    """
    Local/demo domain ontology.

    Replace this with Neo4jDomainGraph when connecting
    to the real ontology.
    """

    nodes = {

        "validate_transfer":
            DomainNode(
                id="validate_transfer",
                name="Validate Transfer Request",
                node_type="Operation",
                description=(
                    "Validate the transfer request "
                    "before processing."
                ),
                aliases=[
                    "validate transfer",
                    "validate request",
                    "verify transfer request",
                ],
            ),

        "check_balance":
            DomainNode(
                id="check_balance",
                name="Check Balance",
                node_type="Operation",
                description=(
                    "Check the available account balance "
                    "before transferring funds."
                ),
                aliases=[
                    "check account balance",
                    "verify balance",
                    "check available funds",
                ],
            ),

        "process_transfer":
            DomainNode(
                id="process_transfer",
                name="Process Transfer",
                node_type="Operation",
                description=(
                    "Process and execute a funds transfer."
                ),
                aliases=[
                    "execute transfer",
                    "process funds transfer",
                    "execute funds transfer",
                ],
            ),

        "generate_transfer_response":
            DomainNode(
                id="generate_transfer_response",
                name="Generate Transfer Response",
                node_type="Operation",
                description=(
                    "Generate the result of the transfer "
                    "operation."
                ),
                aliases=[
                    "transfer response",
                    "return transfer result",
                    "generate response",
                ],
            ),

        "transfer_funds":
            DomainNode(
                id="transfer_funds",
                name="Transfer Funds",
                node_type="Operation",
                description=(
                    "Transfer funds from one account "
                    "to another account."
                ),
                aliases=[
                    "fund transfer",
                    "move money",
                    "send funds",
                ],
            ),
    }

    relationships = [

        DomainRelationship(
            "validate_transfer",
            "check_balance",
            "OPERATION_PRECEDES",
        ),

        DomainRelationship(
            "validate_transfer",
            "check_balance",
            "OPERATION_REQUIRES",
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
            "OPERATION_REQUIRES",
        ),

        DomainRelationship(
            "transfer_funds",
            "process_transfer",
            "OPERATION_PRECEDES",
        ),

        DomainRelationship(
            "transfer_funds",
            "generate_transfer_response",
            "OPERATION_PRODUCES",
        ),
    ]

    return InMemoryDomainGraph(
        nodes,
        relationships,
        embedding_service=embedding_service,
    )


def main():

    # ---------------------------------------------------------
    # Prompt
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

        print(
            "No prompt supplied."
        )

        return

    print(
        f"\nUser prompt:\n{prompt}\n"
    )

    # ---------------------------------------------------------
    # Embedding model
    # ---------------------------------------------------------

    embedding_model = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    embedding_service = (
        EmbeddingService(
            embedding_model
        )
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
    # Domain graph
    # ---------------------------------------------------------

    domain_client = build_domain_graph(
        embedding_service
    )

    # ---------------------------------------------------------
    # Function matcher
    # ---------------------------------------------------------

    function_matcher = (
        FunctionMatcher(
            embedding_model
        )
    )

    functions_file = os.getenv(
        "FUNCTIONS_FILE",
        "functions.json",
    )

    function_matcher.load(
        functions_file
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
    # Final result
    # ---------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL WORKFLOW"
    )

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