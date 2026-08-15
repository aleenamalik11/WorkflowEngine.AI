"""
Command-line entry point.

Examples:

    python inference.py

Then enter prompts interactively:

    transfer funds

    check balance, then transfer funds

    I want to send $500 to another account. Make sure
    the request is valid and that I have enough funds first.

The command-line layer is intentionally thin.

All workflow reasoning happens inside WorkflowPipeline.

The domain ontology is loaded from Neo4j. The in-memory graph
is NOT used here because Neo4j is the source of truth.
"""

import json
import os

from neo4j import GraphDatabase

from pipeline import WorkflowPipeline
from llm_service import LLMService
from domain_graph_client import Neo4jDomainGraph
from utils import EmbeddingService
from function_matcher import FunctionMatcher


def create_neo4j_driver():
    """
    Create the Neo4j driver.

    Local development configuration.
    """

    uri = os.getenv(
        "NEO4J_URI",
        "bolt://localhost:7687",
    )

    username = os.getenv(
        "NEO4J_USER",
        "neo4j",
    )

    # Temporary hardcoded password for local development.
    # Move this to an environment variable before committing.
    password = "helloworld"

    print(
        f"Neo4j URI: {uri}"
    )

    print(
        f"Neo4j user: {username}"
    )

    return GraphDatabase.driver(
        uri,
        auth=(
            username,
            password,
        ),
    )


def main():

    # =========================================================
    # Configuration
    # =========================================================

    embedding_model = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    hf_model = os.getenv(
        "HF_MODEL",
        "Qwen/Qwen2.5-7B-Instruct",
    )

    neo4j_database = os.getenv(
        "NEO4J_DATABASE",
        "neo4j",
    )

    fulltext_index = os.getenv(
        "NEO4J_FULLTEXT_INDEX",
        "domainNodeSearch",
    )

    functions_file = os.getenv(
        "FUNCTIONS_FILE",
        "functions.json",
    )

    # =========================================================
    # Embedding service
    # =========================================================

    print(
        "Loading embedding model..."
    )

    embedding_service = EmbeddingService(
        embedding_model
    )

    # =========================================================
    # LLM
    # =========================================================

    print(
        "Initializing LLM service..."
    )

    llm_service = LLMService(
        model=hf_model
    )

    # =========================================================
    # Neo4j
    # =========================================================

    print(
        "Connecting to Neo4j..."
    )

    driver = create_neo4j_driver()

    try:

        # -----------------------------------------------------
        # Verify connection
        # -----------------------------------------------------

        driver.verify_connectivity()

        print(
            "Neo4j connection established."
        )

        # =====================================================
        # Domain graph
        # =====================================================

        domain_client = Neo4jDomainGraph(
            driver=driver,
            database=neo4j_database,
            embedding_service=embedding_service,
            fulltext_index=fulltext_index,
        )

        # =====================================================
        # Function matcher
        # =====================================================

        print(
            "Loading registered functions..."
        )

        function_matcher = FunctionMatcher(
            embedding_model
        )

        function_matcher.load(
            functions_file
        )

        print(
            f"Loaded "
            f"{len(function_matcher.functions)} "
            f"registered functions."
        )

        # =====================================================
        # Pipeline
        # =====================================================

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

        # =====================================================
        # Interactive prompt loop
        # =====================================================

        print(
            "\n"
            + "=" * 70
        )

        print(
            "WORKFLOW ENGINE"
        )

        print(
            "=" * 70
        )

        print(
            "\nEnter a workflow prompt."
            "\nType 'exit', 'quit', or 'q' to stop."
            "\nType 'debug' to toggle pipeline debug output."
        )

        verbose = True

        while True:

            print(
                "\n"
                + "-" * 70
            )

            try:

                prompt = input(
                    "Enter workflow prompt: "
                ).strip()

            except (
                KeyboardInterrupt,
                EOFError,
            ):

                print(
                    "\nExiting Workflow Engine."
                )

                break

            # -------------------------------------------------
            # Empty prompt
            # -------------------------------------------------

            if not prompt:

                print(
                    "Please enter a workflow prompt."
                )

                continue

            # -------------------------------------------------
            # Exit
            # -------------------------------------------------

            if prompt.lower() in {
                "exit",
                "quit",
                "q",
            }:

                print(
                    "Exiting Workflow Engine."
                )

                break

            # -------------------------------------------------
            # Toggle debug
            # -------------------------------------------------

            if prompt.lower() == "debug":

                verbose = not verbose

                pipeline.verbose = verbose

                print(
                    "Debug logging: "
                    + (
                        "ON"
                        if verbose
                        else "OFF"
                    )
                )

                continue

            # -------------------------------------------------
            # Execute workflow
            # -------------------------------------------------

            print(
                f"\nUser prompt:\n{prompt}\n"
            )

            try:

                workflow, debug = pipeline.run(
                    prompt,
                    workflow_name="Generated Workflow",
                )

                # ---------------------------------------------
                # Final workflow
                # ---------------------------------------------

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

            except Exception as exc:

                # One bad prompt must NOT terminate the
                # interactive application.

                print(
                    "\n"
                    + "=" * 70
                )

                print(
                    "WORKFLOW GENERATION FAILED"
                )

                print(
                    "=" * 70
                )

                print(
                    f"{type(exc).__name__}: {exc}"
                )

                continue

    finally:

        # =====================================================
        # Close Neo4j
        # =====================================================

        driver.close()

        print(
            "\nNeo4j connection closed."
        )


if __name__ == "__main__":
    main()