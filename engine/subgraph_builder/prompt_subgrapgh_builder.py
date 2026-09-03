import math
import networkx as nx

from helpers.similarity_utils import _cosine_similarity, _lexical_similarity, _combined_score, _relationship_relevance, \
    _node_text
from helpers.utils import _add_domain_node
from models import SemanticInterpretation

class PromptSubGraphBuilder:
    def __init__(self, embedding_service, domain_graph_service):
        self._embedding_service = embedding_service
        self._domain_graph_service = domain_graph_service

    def build (self,
            interpretation: SemanticInterpretation,
            k=5,
            neighborhood_depth=1,
    ):
        prompt_domain_subgraph = nx.DiGraph()

        candidate_map = {}

        candidates_debug = []
        context_attachments = []
        constraint_edges = []
        conditional_dependencies = []

        for step_index, step in enumerate(interpretation.steps):
            step_embedding = self._embedding_service.encode(step.text)

            candidates = self._domain_graph_service.candidate_nodes(step.text, step_embedding, k=max(k, 25))

            step_candidates = []

            for candidate in candidates:
                node = candidate.node
                candidate_embedding = candidate.embedding

                if candidate_embedding is None:
                    candidate_embedding = self._embedding_service.encode(
                        _node_text(node)
                    )

                semantic_similarity = _cosine_similarity(
                    candidate_embedding,
                    step_embedding,
                )

                lexical_similarity = _lexical_similarity(
                    step.text,
                    node
                )

                score = _combined_score(lexical_similarity, semantic_similarity, candidate.score)

                item = {
                    "node_id": node.id,
                    "name": node.name,
                    "node_type": node.node_type,
                    "score": score,
                    "lexical_score": lexical_similarity,
                    "semantic_score": semantic_similarity,
                    "explicit": bool(step.explicit),
                    "inferred": not bool(step.explicit),
                    "source": "direct",
                    "prompt_text": step.text,
                }

                existing_candidate = next(
                    (
                        candidate
                        for candidate in step_candidates
                        if candidate["node_id"] == item["node_id"]
                    ),
                    None,
                )

                if existing_candidate is None:
                    step_candidates.append(item)
                elif item["score"] > existing_candidate["score"]:
                    item["explicit"] = (
                            item["explicit"]
                            or existing_candidate["explicit"]
                    )
                    item["inferred"] = not item["explicit"]

                    index = step_candidates.index(existing_candidate)
                    step_candidates[index] = item


                _add_domain_node(
                    prompt_domain_subgraph,
                    node,
                    item,
                )

            prompt_domain_subgraph, inferred_candidates = (
                self.expand_neighborhood(
                    prompt_domain_subgraph,
                    step,
                    step_embedding,
                    step_candidates,
                    neighborhood_depth,
                )
            )

            step_candidates.extend(inferred_candidates)

            step_candidates.sort(
                key=lambda x: x["score"],
                reverse=True,
            )

            step_candidates = step_candidates[
                : max(k * 2, 10)
            ]

            candidate_map[step_index] = {
                "step_embedding": step_embedding,
                "candidates": step_candidates,
            }

        prompt_domain_subgraph = self.set_execution_order(
            interpretation,
            prompt_domain_subgraph,
            candidate_map)

        candidate_plan = {
            "prompt_domain_subgraph": prompt_domain_subgraph,
            "candidate_map": candidate_map,
            "semantic_steps": interpretation.steps,
            "intent": interpretation.intent,
        }

        return candidate_plan

    def expand_neighborhood(
            self,
            prompt_subgraph,
            step,
            step_embedding,
            step_candidates,
            neighborhood_depth,
    ):
        neighborhood_seed_ids = [
            item["node_id"]
            for item in step_candidates
        ]

        inferred_candidates = []
        for seed_id in neighborhood_seed_ids:

            step_candidate_neighbors = self._domain_graph_service.neighborhood(seed_id, neighborhood_depth)

            for step_candidate_neighbor in step_candidate_neighbors:
                source = self._domain_graph_service.get_node(step_candidate_neighbor.source_id)
                target = self._domain_graph_service.get_node(step_candidate_neighbor.target_id)

                if source is None or target is None:
                    continue

                _add_domain_node(
                    prompt_subgraph,
                    source,
                    None,
                )

                _add_domain_node(
                    prompt_subgraph,
                    target,
                    None,
                )

                # Preserve the ontology edge.
                prompt_subgraph.add_edge(
                    source.id,
                    target.id,
                    relation=step_candidate_neighbor.relation,
                    inferred_context=True,
                    origin="domain",
                )

                inferred_candidates.extend(
                    self.infer_candidates(
                        step,
                        step_embedding,
                        step_candidates,
                        source,
                        target,
                        step_candidate_neighbor,
                    )
                )

        return prompt_subgraph, inferred_candidates

    def infer_candidates(
            self,
            step,
            step_embedding,
            step_candidates,
            source,
            target,
            relationship,
    ):
        inferred_candidates = []

        for neighbor in (
                source,
                target,
        ):

            if neighbor.id in {
                item["node_id"]
                for item in step_candidates
            }:
                continue

            neighbor_embedding = (
                neighbor.embedding
            )

            # Some graph implementations may not store
            # embeddings on nodes. Generate one from the
            # semantic text if necessary.
            if neighbor_embedding is None:
                neighbor_embedding = (
                    self._embedding_service.encode(
                        _node_text(neighbor)
                    )
                )

            lexical_score = _lexical_similarity(
                step.text,
                neighbor
            )

            semantic_score = _cosine_similarity(
                step_embedding,
                neighbor_embedding,
            )

            relation_score = _relationship_relevance(
                relationship.relation
            )

            contextual_score = (
                    0.50 * semantic_score
                    + 0.25 * lexical_score
                    + 0.25 * relation_score
            )

            # Don't pull every arbitrary neighbor into
            # the workflow.
            #
            # The threshold is deliberately permissive
            # because this is contextual inference rather
            # than final function matching.
            if contextual_score < 0.25:
                continue

            inferred_item = {
                "node_id": neighbor.id,
                "name": neighbor.name,
                "node_type": neighbor.node_type,
                "score": contextual_score,
                "lexical_score": lexical_score,
                "semantic_score": semantic_score,
                "explicit": False,
                "inferred": True,
                "source": "neighborhood",
                "prompt_text": step.text,
                "relation_score": relation_score,
            }

            inferred_candidates.append(inferred_item)

        return inferred_candidates

    def set_execution_order(
            self,
            interpretation: SemanticInterpretation,
            prompt_subgraph,
            candidate_map):

        for dependency in interpretation.dependencies:

            before = dependency.get("before")
            after = dependency.get("after")

            relation = dependency.get(
                "relation",
                "PROMPT_DEPENDENCY"
            )

            if not before or not after:
                continue

            before_candidate_embedding = self._embedding_service.encode(before)
            after_candidate_embedding = self._embedding_service.encode(after)

            before_node = self.best_candidate_for_embedding(
                candidate_map,
                before_candidate_embedding,
            )

            after_node = self.best_candidate_for_embedding(
                candidate_map,
                after_candidate_embedding,
            )

            if before_node is None or after_node is None:
                continue

            if before_node == after_node:
                continue

            # For PROMPT_CONDITION, we store the condition as metadata instead of creating an edge
            if relation == "PROMPT_CONDITION":
                condition = dependency.get("condition", "")
                # Store condition on the target node for downstream processing
                if prompt_subgraph.has_node(after_node):
                    prompt_subgraph.nodes[after_node]["condition"] = condition
                    prompt_subgraph.nodes[after_node]["is_conditional"] = True
                    # Also mark the source node as the condition check
                    if prompt_subgraph.has_node(before_node):
                        prompt_subgraph.nodes[before_node]["is_condition_check"] = True
                        prompt_subgraph.nodes[before_node]["condition_for"] = after_node
                continue

            prompt_subgraph.add_edge(
                before_node,
                after_node,
                relation="PROMPT_DEPENDENCY",
                inferred_context=True,
                origin="prompt",
            )

        return prompt_subgraph

    def best_candidate_for_embedding(
            self,
            candidate_map,
            dependency_embedding,
    ):
        best_entry = None
        best_similarity = float("-inf")

        for entry in candidate_map.values():
            similarity = _cosine_similarity(
                dependency_embedding,
                entry["step_embedding"],
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry

        if best_entry is None:
            return None

        candidates = best_entry["candidates"]

        if not candidates:
            return None

        best_candidate = max(
            candidates,
            key=lambda candidate: candidate.get("score", 0.0),
        )

        if best_similarity < 0.5:  # tune experimentally
            return None

        return best_candidate.get("node_id")