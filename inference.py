"""
Command-line entry point.

Neo4j is the source of truth for the domain graph.

The application continuously accepts workflow prompts.

Examples:

    transfer funds

    check account balance, then transfer funds

    I want to send $500 to my savings account.
    Make sure there is enough money first.

Commands:

    exit
    quit
    q

    debug
"""

import json
import os

from neo4j import GraphDatabase

from pipeline import WorkflowPipeline

from llm_service import LLMService

from domain_graph_client import (
    Neo4jDomainGraph,
)

from utils import EmbeddingService

from function_matcher import FunctionMatcher


# =============================================================
# Neo4j
# =============================================================

def create_neo4j_driver():

    uri = os.getenv(
        "NEO4J_URI",
        "bolt://localhost:7687",
    )

    username = os.getenv(
        "NEO4J_USERNAME",
        "neo4j",
    )

    password = "helloworld"

    database = os.getenv(
        "NEO4J_DATABASE",
        "neo4j",
    )

    print(
        f"Neo4j URI: {uri}"
    )

    print(
        f"Neo4j user: {username}"
    )

    driver = GraphDatabase.driver(
        uri,
        auth=(
            username,
            password,
        ),
    )

    driver.verify_connectivity()

    print(
        "Neo4j connection established."
    )

    return driver, database


# =============================================================
# Main
# =============================================================

def main():

    # ---------------------------------------------------------
    # Embedding model
    # ---------------------------------------------------------

    embedding_model = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    print(
        "Loading embedding model..."
    )

    embedding_service = (
        EmbeddingService(
            embedding_model
        )
    )

    # ---------------------------------------------------------
    # LLM
    # ---------------------------------------------------------

    llm_model = os.getenv(
        "HF_MODEL",
        "Qwen/Qwen2.5-7B-Instruct",
    )

    print(
        "Initializing LLM service..."
    )

    llm_service = (
        LLMService(
            model=llm_model
        )
    )

    # ---------------------------------------------------------
    # Neo4j
    # ---------------------------------------------------------

    print(
        "Connecting to Neo4j..."
    )

    driver, database = (
        create_neo4j_driver()
    )

    try:

        # -----------------------------------------------------
        # Domain graph
        # -----------------------------------------------------

        domain_client = (
            Neo4jDomainGraph(
                driver=driver,
                database=database,
                embedding_service=(
                    embedding_service
                ),
                fulltext_index=os.getenv(
                    "NEO4J_FULLTEXT_INDEX",
                    "domainNodeSearch",
                ),
            )
        )

        # -----------------------------------------------------
        # Function matcher
        # -----------------------------------------------------

        print(
            "Loading registered functions..."
        )

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

        print(
            f"Loaded "
            f"{len(function_matcher.functions)} "
            f"registered functions."
        )

        # -----------------------------------------------------
        # Pipeline
        # -----------------------------------------------------

        pipeline = (
            WorkflowPipeline(

                domain_client=(
                    domain_client
                ),

                embedding_service=(
                    embedding_service
                ),

                function_matcher=(
                    function_matcher
                ),

                llm_service=(
                    llm_service
                ),

                beam_width=3,

                top_k=5,

                neighborhood_depth=1,

                verbose=True,
            )
        )

        # -----------------------------------------------------
        # Interactive loop
        # -----------------------------------------------------

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
        )

        print(
            "Type 'exit', 'quit', or 'q' to stop."
        )

        print(
            "Type 'debug' to toggle debug logging."
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
            # Empty input
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

                pipeline.verbose = (
                    verbose
                )

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
            # Run workflow
            # -------------------------------------------------

            print(
                f"\nUser prompt:\n{prompt}\n"
            )

            try:

                workflow, debug = (
                    pipeline.run(
                        prompt,
                        workflow_name=(
                            "Generated Workflow"
                        ),
                    )
                )

                # -------------------------------------------------
                # Final workflow
                # -------------------------------------------------

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

            except Exception as error:

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
                    f"{type(error).__name__}: "
                    f"{error}"
                )

                # Keep the application alive.
                continue

    finally:

        driver.close()

        print(
            "\nNeo4j connection closed."
        )


if __name__ == "__main__":
    main()