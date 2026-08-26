import math
import networkx as nx

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

        for step_index, step in interpretation.steps:
            step_embedding = self._embedding_service.get_embedding(step.text)

            candidates = self._domain_graph_service.candidate_nodes(step.text, step_embedding, k=max(k, 25))

            step_candidates = []

            for candidate in candidates:
                node = candidate.node
                candidate_embedding = get_or_create_embedding(candidate)

                semantic_similarity = _cosine_similarity(candidate_embedding, step_embedding)
                lexical_similarity = _lexical_similarity(candidate.text, step.text)

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

                existing_candidate = step_candidates.get(item["node_id"])

                if existing_candidate is None:
                    step_candidates.append(item)
                elif item["score"] > existing_candidate["score"]:
                    item["explicit"] = (
                            item["explicit"]
                            or existing["explicit"]
                    )

                    item["inferred"] = (
                        not item["explicit"]
                    )


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

            candidate_map[step_index] = (
                step_candidates
            )

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
                neighbor.name,
                neighbor.aliases,
                neighbor.description,
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