"""Graph-first, end-to-end workflow generation pipeline.

The domain graph is the semantic source of truth. LLM use is limited to
enriching the semantic interpretation; it never selects functions, creates
workflow nodes, or emits workflow JSON.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Tuple

# ``python orchestrator/workflow_pipeline.py`` puts only the orchestrator
# directory on sys.path. Include the project root before importing packages.
if __package__ in {None, ""}:
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from engine.parsers.hybrid_semantic_parser import HybridSemanticParser
from engine.parsers.semantic_parser import SimpleSemanticParser
from engine.subgraph_builder.prompt_subgrapgh_builder import PromptSubGraphBuilder
from engine.workflow_generator.workflow_generator import WorkflowGenerator
from engine.workflow_graph_builder.workflow_graph_builder import WorkflowGraphBuilder
from engine.workflow_selector.beam_search_workflow_selector import (
    BeamSearchWorkflowSelector,
)


class WorkflowPipeline:
    """Orchestrate semantic parsing through workflow JSON generation."""

    def __init__(
        self,
        domain_graph_service,
        embedding_service,
        function_matcher,
        parser=None,
        llm_service=None,
        hybrid_mode: bool = False,
        beam_width=3,
        top_k=5,
        neighborhood_depth=1,
        verbose=True,
        subgraph_builder=None,
        beam_selector=None,
        graph_builder=None,
        generator=None,
    ):
        self.domain_graph_service = domain_graph_service
        self.embedding_service = embedding_service
        self.function_matcher = function_matcher
        self.llm_service = llm_service
        self.hybrid_mode = bool(hybrid_mode)
        self.top_k = int(top_k)
        self.neighborhood_depth = int(neighborhood_depth)
        self.verbose = bool(verbose)

        if parser is not None:
            self.parser = parser
        elif self.hybrid_mode:
            if llm_service is None:
                raise ValueError(
                    "hybrid_mode=True requires llm_service when no custom parser is supplied."
                )
            self.parser = HybridSemanticParser(
                llm_service=llm_service,
                enable_llm=True,
            )
        else:
            self.parser = SimpleSemanticParser()

        self.subgraph_builder = subgraph_builder or PromptSubGraphBuilder(
            embedding_service,
            domain_graph_service,
        )
        self.beam_selector = beam_selector or BeamSearchWorkflowSelector(
            beam_width=beam_width
        )
        self.graph_builder = graph_builder or WorkflowGraphBuilder(domain_graph_service)
        self.generator = generator or WorkflowGenerator(function_matcher)

    def run(
        self,
        prompt: str | None = None,
        workflow_name: str = "Generated Workflow",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate a workflow for one prompt."""
        if prompt is None:
            prompt = input("Workflow request> ").strip()

        if not prompt:
            raise ValueError("A workflow request is required.")

        domain_context = self._build_domain_context()
        interpretation = self._parse_prompt(prompt, domain_context)

        debug: Dict[str, Any] = {
            "semantic_interpretation": self._interpretation_debug(
                interpretation
            ),
        }

        self._log(
            "STAGE 1/2 - Semantic interpretation",
            debug["semantic_interpretation"],
        )

        candidate_plan = self.subgraph_builder.build(
            interpretation,
            k=self.top_k,
            neighborhood_depth=self.neighborhood_depth,
        )

        self._normalise_candidate_map(candidate_plan)

        debug.update(self._subgraph_debug(candidate_plan))

        self._log(
            "STAGES 3-7 - Contextual domain subgraph",
            debug["domain_graph_edges"],
        )

        search_result = self.beam_selector.search(candidate_plan)

        debug["beam_search"] = search_result.get("beam", [])
        debug["unsupported_steps"] = search_result.get(
            "unsupported_steps",
            [],
        )

        self._log(
            "STAGE 8 - Beam candidates",
            debug["beam_search"],
        )

        if not search_result.get("selection"):
            raise RuntimeError(
                "Beam search produced no selected workflow nodes. "
                "The domain graph contained no usable executable candidate."
            )

        plan = self.graph_builder.build(
            search_result,
            candidate_plan=candidate_plan,
        )

        debug["execution_order"] = [
            plan["graph"].nodes[node_id].get("name", node_id)
            for node_id in plan.get("execution_order", [])
        ]

        self._log(
            "STAGE 9 - Execution order",
            debug["execution_order"],
        )

        workflow = self.generator.generate(
            plan,
            workflow_name=workflow_name,
        )

        workflow["Inputs"] = self._resolve_inputs(workflow)

        self._log(
            "STAGE 10/11 - Workflow JSON",
            workflow,
        )

        return workflow, debug

    def run_forever(
        self,
        workflow_name: str = "Generated Workflow",
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        """Continuously accept workflow prompts."""
        output_fn(
            "Workflow pipeline ready. Enter a request, or 'exit' to stop."
        )

        while True:
            try:
                prompt = input_fn("\nWorkflow request> ").strip()
            except EOFError:
                output_fn("Workflow pipeline stopped.")
                return
            except KeyboardInterrupt:
                output_fn("\nWorkflow pipeline stopped.")
                return

            if not prompt:
                continue

            if prompt.lower() in {"exit", "quit", "q"}:
                output_fn("Workflow pipeline stopped.")
                return

            try:
                workflow, _ = self.run(
                    prompt,
                    workflow_name=workflow_name,
                )
            except Exception:
                import traceback
                traceback.print_exc()
                output_fn("Workflow generation failed. See traceback above.")
                continue

            output_fn(
                json.dumps(
                    workflow,
                    indent=2,
                    default=str,
                )
            )

    def _parse_prompt(
        self,
        prompt: str,
        domain_context: List[Dict[str, Any]],
    ):
        try:
            return self.parser.parse(
                prompt,
                domain_context=domain_context,
            )
        except TypeError:
            return self.parser.parse(prompt)

    def _build_domain_context(self) -> List[Dict[str, Any]]:
        try:
            nodes = self.domain_graph_service.all_nodes()
        except Exception:
            return []

        return [
            {
                "id": node.id,
                "name": node.name,
                "type": node.node_type,
                "types": list(node.types or []),
                "description": node.description or "",
                "aliases": list(node.aliases or []),
            }
            for node in nodes
        ]

    @staticmethod
    def _normalise_candidate_map(
        candidate_plan: Dict[str, Any],
    ) -> None:
        candidate_map = candidate_plan.get("candidate_map", {})

        for index, value in list(candidate_map.items()):
            if isinstance(value, dict) and "candidates" in value:
                candidate_map[index] = value["candidates"]

    @staticmethod
    def _interpretation_debug(interpretation) -> Dict[str, Any]:
        return {
            "intent": getattr(interpretation, "intent", ""),
            "steps": [
                {
                    "text": step.text,
                    "explicit": bool(
                        getattr(step, "explicit", True)
                    ),
                    "condition": getattr(
                        step,
                        "condition",
                        "",
                    ),
                }
                for step in getattr(interpretation, "steps", [])
            ],
            "dependencies": list(
                getattr(
                    interpretation,
                    "dependencies",
                    [],
                )
            ),
            "mentioned_entities": list(
                getattr(
                    interpretation,
                    "mentioned_entities",
                    [],
                )
            ),
            "constraints": list(
                getattr(
                    interpretation,
                    "constraints",
                    [],
                )
            ),
        }

    @staticmethod
    def _subgraph_debug(
        candidate_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        graph = candidate_plan.get("prompt_domain_subgraph")
        edges = []

        if graph is not None:
            edges = [
                {
                    "source": source,
                    "target": target,
                    **data,
                }
                for source, target, data in graph.edges(
                    data=True
                )
            ]

        return {
            "candidate_map": candidate_plan.get(
                "candidate_map",
                {},
            ),
            "domain_graph_edges": edges,
        }

    @staticmethod
    def _resolve_inputs(
        workflow: Dict[str, Any],
    ) -> List[str]:
        inputs: List[str] = []

        for node in workflow.get("Nodes", []):
            details = node.get("FunctionDetails", {})

            if not details.get("Found", False):
                continue

            for input_name in details.get("Inputs", []):
                if input_name not in inputs:
                    inputs.append(input_name)

        return inputs

    def _log(self, title: str, payload: Any) -> None:
        if not self.verbose:
            return

        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
        print(json.dumps(payload, indent=2, default=str))


def create_neo4j_driver():
    """Create and verify the Neo4j connection."""
    from neo4j import GraphDatabase

    uri = os.getenv(
        "NEO4J_URI",
        "bolt://localhost:7687",
    )
    username = os.getenv(
        "NEO4J_USERNAME",
        os.getenv("NEO4J_USER", "neo4j"),
    )
    password = os.getenv(
        "NEO4J_PASSWORD",
        "helloworld",
    )
    database = os.getenv(
        "NEO4J_DATABASE",
        "neo4j",
    )

    print(f"Neo4j URI: {uri}")
    print(f"Neo4j user: {username}")

    driver = GraphDatabase.driver(
        uri,
        auth=(username, password),
    )

    driver.verify_connectivity()

    print("Neo4j connection established.")

    return driver, database


def _load_domain_accessor(
    driver,
    database,
    embedding_service,
):
    """Load the legacy hyphenated accessor module."""
    accessor_path = (
        Path(__file__).resolve().parents[1]
        / "data_access"
        / "domain-graph-accessor.py"
    )

    spec = importlib.util.spec_from_file_location(
        "domain_graph_accessor",
        accessor_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to load domain graph accessor: {accessor_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.DomainGraphService(
        driver=driver,
        database=database,
        embedding_service=embedding_service,
        fulltext_index=os.getenv(
            "NEO4J_FULLTEXT_INDEX",
            "domainNodeSearch",
        ),
    )


def main():
    """Start the continuously interactive workflow engine."""
    from services.domain_graph_service import DomainGraphService
    from services.embedding_service import EmbeddingService
    from services.llm_service import LLMService
    from engine.workflow_generator.function_matcher import FunctionMatcher

    embedding_model = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    print("Loading embedding model...")
    embedding_service = EmbeddingService(embedding_model)

    hybrid_mode = os.getenv(
        "WORKFLOW_HYBRID_MODE",
        "false",
    ).lower() in {"1", "true", "yes"}

    llm_service = LLMService(
        model=os.getenv(
            "HF_MODEL",
            "Qwen/Qwen2.5-7B-Instruct",
        )
    )

    print("Connecting to Neo4j...")
    driver, database = create_neo4j_driver()

    try:
        accessor = _load_domain_accessor(
            driver,
            database,
            embedding_service,
        )

        domain_graph_service = DomainGraphService(
            embedding_service,
            accessor,
        )

        print("Loading registered functions...")

        function_matcher = FunctionMatcher(embedding_model)

        from pathlib import Path

        PROJECT_ROOT = Path(__file__).resolve().parent.parent

        functions_file = os.getenv(
            "FUNCTIONS_FILE",
            str(PROJECT_ROOT / "functions.json"),
        )

        function_matcher.load(functions_file)

        print(
            f"Loaded {len(function_matcher.functions)} "
            "registered functions."
        )

        pipeline = WorkflowPipeline(
            domain_graph_service=domain_graph_service,
            embedding_service=embedding_service,
            function_matcher=function_matcher,
            llm_service=llm_service,
            hybrid_mode=hybrid_mode,
            beam_width=3,
            top_k=5,
            neighborhood_depth=1,
            verbose=True,
        )

        pipeline.run_forever(
            workflow_name="Generated Workflow",
        )

    finally:
        driver.close()
        print("\nNeo4j connection closed.")


if __name__ == "__main__":
    main()