"""
End-to-end regression test runner for WorkflowEngine.AI.

This script runs the REAL inference.py entry point against multiple prompts.
It does not mock the embedding model, LLM, Neo4j, function matcher, or pipeline.

The prompts are sent to one inference.py process so the expensive services are
initialized once. The complete console output and the extracted final workflow
for every prompt are written to one JSON file.

Usage:

    python test_pipeline.py

Optional:

    python test_pipeline.py --inference inference.py --output pipeline_results.json

    python test_pipeline.py --timeout 900

The script expects to be run from the WorkflowEngine.AI repository root.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PROMPTS = [
    "transfer funds",
    "check account balance",
    "create a new account",
    "issue a card for an account",
    "retrieve an account",
    "update an account",
    "debit an account",
    "credit an account",
    "create a transaction",
    "transfer funds after checking the account balance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WorkflowEngine.AI end-to-end against multiple prompts."
    )
    parser.add_argument(
        "--inference",
        default="inference.py",
        help="Path to the real inference.py entry point.",
    )
    parser.add_argument(
        "--output",
        default="pipeline_test_results.json",
        help="JSON file containing all test results.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Maximum runtime in seconds for the complete pipeline process.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Run a custom prompt. Can be supplied multiple times.",
    )
    return parser.parse_args()


def extract_json_after_marker(text: str, marker: str) -> Any | None:
    """
    Find the first JSON value after marker.

    JSONDecoder.raw_decode is used instead of a regex so nested workflow
    dictionaries/lists are handled correctly.
    """
    marker_index = text.find(marker)
    if marker_index == -1:
        return None

    payload = text[marker_index + len(marker):].lstrip()

    decoder = json.JSONDecoder()

    try:
        value, _ = decoder.raw_decode(payload)
        return value
    except json.JSONDecodeError:
        return None


def extract_final_workflow(output: str) -> dict[str, Any] | None:
    value = extract_json_after_marker(output, "FINAL WORKFLOW")
    return value if isinstance(value, dict) else None


def extract_error(output: str) -> str | None:
    marker = "WORKFLOW GENERATION FAILED"
    index = output.find(marker)

    if index == -1:
        return None

    tail = output[index + len(marker):]

    # The inference entry point prints:
    #   TypeError: ...
    #   ValueError: ...
    # etc.
    match = re.search(
        r"\n([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)): (.*?)(?:\n|$)",
        tail,
    )

    if match:
        return f"{match.group(1)}: {match.group(2).strip()}"

    return tail.strip().splitlines()[0] if tail.strip() else "Unknown error"


def validate_workflow(workflow: dict[str, Any]) -> list[str]:
    """
    Structural regression checks for the final Stage 10/11 workflow.

    These checks intentionally do not require an exact workflow because the
    semantic/LLM stages can legitimately produce different valid structures.
    """
    errors: list[str] = []

    nodes = workflow.get("Nodes")

    if not isinstance(nodes, list):
        errors.append("Workflow.Nodes is not a list.")
        return errors

    if not nodes:
        errors.append("Workflow contains no nodes.")
        return errors

    node_ids = []
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"Node {index} is not an object.")
            continue

        node_id = node.get("Id")
        if not node_id:
            errors.append(f"Node {index} has no Id.")
        else:
            node_ids.append(node_id)

        if not node.get("Name"):
            errors.append(f"Node {index} has no semantic Name.")

        # New Stage 10/11 contract:
        # function details are separate from the semantic node name.
        if "Function" not in node:
            errors.append(
                f"Node '{node.get('Name', node_id)}' has no Function field."
            )
        else:
            function = node["Function"]

            # None is VALID: it means no matching registered function exists.
            if function is not None and not isinstance(function, dict):
                errors.append(
                    f"Node '{node.get('Name', node_id)}' has invalid Function data."
                )

            if isinstance(function, dict):
                if not function.get("Name"):
                    errors.append(
                        f"Node '{node.get('Name', node_id)}' has Function data "
                        "without Function.Name."
                    )

                # A semantic node's Name must not simply be replaced by the
                # registered function name.
                if function.get("Name") == node.get("Name"):
                    errors.append(
                        f"Node '{node.get('Name')}' appears to use the function "
                        "name as its semantic node Name."
                    )

    if len(node_ids) != len(set(node_ids)):
        errors.append("Workflow contains duplicate node IDs.")

    start_node_id = workflow.get("StartNodeId")

    if not start_node_id:
        errors.append("Workflow has no StartNodeId.")
    elif start_node_id not in set(node_ids):
        errors.append(
            f"StartNodeId '{start_node_id}' does not reference a workflow node."
        )

    connections = workflow.get("Connections")

    if not isinstance(connections, dict):
        errors.append("Workflow.Connections is not an object.")
        return errors

    known_ids = set(node_ids)

    for source_id, transitions in connections.items():
        if source_id not in known_ids:
            errors.append(
                f"Connection source '{source_id}' does not reference a workflow node."
            )

        if not isinstance(transitions, dict):
            errors.append(
                f"Connections for '{source_id}' are not an object."
            )
            continue

        for relation, target in transitions.items():
            if target != "Done" and target not in known_ids:
                errors.append(
                    f"Connection '{source_id}' --{relation}--> '{target}' "
                    "references a nonexistent node."
                )

    return errors


def run_pipeline(
    inference_path: Path,
    prompts: list[str],
    timeout: int,
) -> tuple[int, str, str]:
    """
    Run the real inference.py ONCE.

    All prompts are supplied through stdin. The final 'q' terminates the
    interactive loop after every prompt has been processed.
    """
    input_text = "\n".join([*prompts, "q"]) + "\n"

    completed = subprocess.run(
        [sys.executable, str(inference_path)],
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        cwd=str(inference_path.parent),
    )

    return completed.returncode, completed.stdout, completed.stderr


def split_prompt_outputs(console_output: str, prompts: list[str]) -> list[str]:
    """
    Split the single inference.py console log into one section per prompt.

    inference.py prints 'User prompt:' before each pipeline invocation.
    """
    marker = "User prompt:\n"
    positions = []

    start = 0
    while True:
        index = console_output.find(marker, start)
        if index == -1:
            break
        positions.append(index)
        start = index + len(marker)

    sections: list[str] = []

    for i, position in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(console_output)
        sections.append(console_output[position:end])

    # If a prompt failed before the expected marker was printed, preserve the
    # fact that it could not be associated with a workflow.
    while len(sections) < len(prompts):
        sections.append("")

    return sections[: len(prompts)]


def build_results(
    prompts: list[str],
    sections: list[str],
    return_code: int,
    stderr: str,
) -> dict[str, Any]:
    tests = []

    for index, prompt in enumerate(prompts):
        section = sections[index] if index < len(sections) else ""

        workflow = extract_final_workflow(section)
        pipeline_error = extract_error(section)

        validation_errors = (
            validate_workflow(workflow)
            if workflow is not None
            else ["No final workflow JSON was produced."]
        )

        if pipeline_error:
            validation_errors.insert(0, pipeline_error)

        passed = (
            return_code == 0
            and workflow is not None
            and not validation_errors
        )

        tests.append(
            {
                "test_number": index + 1,
                "prompt": prompt,
                "status": "PASS" if passed else "FAIL",
                "validation_errors": validation_errors,
                "workflow": workflow,
                "raw_output": section,
            }
        )

    passed_count = sum(test["status"] == "PASS" for test in tests)

    return {
        "test_run": {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "return_code": return_code,
            "prompt_count": len(prompts),
            "passed": passed_count,
            "failed": len(tests) - passed_count,
        },
        "tests": tests,
        "stderr": stderr,
    }


def main() -> int:
    args = parse_args()

    inference_path = Path(args.inference).resolve()
    output_path = Path(args.output).resolve()

    prompts = args.prompts or DEFAULT_PROMPTS

    if not prompts:
        print("No prompts supplied.")
        return 2

    if not inference_path.exists():
        print(f"ERROR: inference.py was not found: {inference_path}")
        return 2

    print("=" * 70)
    print("WORKFLOW ENGINE - END-TO-END PIPELINE TEST")
    print("=" * 70)
    print(f"Pipeline: {inference_path}")
    print(f"Prompts:  {len(prompts)}")
    print(f"Output:   {output_path}")
    print()
    print("Running the real inference.py once for all prompts...")
    print()

    try:
        return_code, stdout, stderr = run_pipeline(
            inference_path=inference_path,
            prompts=prompts,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired as exc:
        print(f"ERROR: pipeline timed out after {args.timeout} seconds.")

        timeout_results = {
            "test_run": {
                "return_code": None,
                "status": "TIMEOUT",
                "prompt_count": len(prompts),
            },
            "tests": [
                {
                    "test_number": i + 1,
                    "prompt": prompt,
                    "status": "FAIL",
                    "validation_errors": [
                        f"Pipeline timed out after {args.timeout} seconds."
                    ],
                    "workflow": None,
                }
                for i, prompt in enumerate(prompts)
            ],
            "stdout": exc.stdout,
            "stderr": exc.stderr,
        }

        output_path.write_text(
            json.dumps(timeout_results, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"Partial results saved to: {output_path}")
        return 1

    sections = split_prompt_outputs(stdout, prompts)

    results = build_results(
        prompts=prompts,
        sections=sections,
        return_code=return_code,
        stderr=stderr,
    )

    # Keep the complete console log as well. This is extremely useful when a
    # regression occurs in Stage 3/4/5/6/8/9 before Stage 10/11.
    results["complete_console_output"] = stdout

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    for test in results["tests"]:
        status = test["status"]
        print(f"[{status}] {test['test_number']}: {test['prompt']}")

        for error in test["validation_errors"]:
            print(f"       - {error}")

        if test["workflow"]:
            nodes = test["workflow"].get("Nodes", [])
            print(f"       nodes: {len(nodes)}")

    print()
    print(
        f"Passed: {results['test_run']['passed']} / "
        f"{results['test_run']['prompt_count']}"
    )
    print(
        f"Failed: {results['test_run']['failed']} / "
        f"{results['test_run']['prompt_count']}"
    )
    print()
    print(f"Complete results saved to:")
    print(output_path)

    return 0 if results["test_run"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())