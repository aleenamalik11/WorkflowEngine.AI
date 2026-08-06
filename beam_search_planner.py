import networkx as nx


class BeamSearchPlanner:
    """Select the best candidate workflow with beam search.

    The matcher no longer commits to a single domain node per prompt action.
    It proposes the top K candidates, and this planner picks the sequence of
    candidates that scores best as a whole.  Dijkstra is demoted to a
    subroutine: it only evaluates (and later expands) the transition between
    two consecutive candidates.
    """

    ###############################################################
    # Scoring configuration
    ###############################################################

    DEFAULT_BEAM_WIDTH = 3

    SEMANTIC_WEIGHT = 0.6
    GRAPH_WEIGHT = 0.4

    PATH_BONUS = 10.0
    WEIGHT_PENALTY = 1.0
    HOP_PENALTY = 0.2
    MISSING_PATH_PENALTY = 5.0

    RELATION_BONUS = {
        "mandatory": 1.0,
        "alternative": 0.25,
        "optional": 0.0,
        "deprecated": -2.0,
    }

    def __init__(self,
                 beam_width=DEFAULT_BEAM_WIDTH,
                 semantic_weight=SEMANTIC_WEIGHT,
                 graph_weight=GRAPH_WEIGHT,
                 path_bonus=PATH_BONUS,
                 weight_penalty=WEIGHT_PENALTY,
                 hop_penalty=HOP_PENALTY,
                 missing_path_penalty=MISSING_PATH_PENALTY):

        self.beam_width = max(1, int(beam_width))
        self.semantic_weight = semantic_weight
        self.graph_weight = graph_weight
        self.path_bonus = path_bonus
        self.weight_penalty = weight_penalty
        self.hop_penalty = hop_penalty
        self.missing_path_penalty = missing_path_penalty

    ###############################################################
    # Transition evaluation (Dijkstra as a subroutine)
    ###############################################################

    def _shortest_path(self, domain_graph, source, target, cache):
        """Cached weighted shortest path between two candidate nodes."""
        key = (source, target)
        if key in cache:
            return cache[key]

        if source == target:
            path = [source]
        else:
            try:
                path = nx.dijkstra_path(
                    domain_graph,
                    source=source,
                    target=target,
                    weight="weight",
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                path = None

        cache[key] = path
        return path

    def _transition_score(self, domain_graph, source, target, cache):
        """Graph part of the score: reward valid, short, cheap domain paths."""
        path = self._shortest_path(domain_graph, source, target, cache)

        if path is None:
            return -self.missing_path_penalty, None

        hops = len(path) - 1
        score = self.path_bonus - self.hop_penalty * hops

        for path_source, path_target in zip(path, path[1:]):
            edge = domain_graph.edges[path_source, path_target]
            score -= self.weight_penalty * float(edge.get("weight", 1))
            score += self.RELATION_BONUS.get(
                str(edge.get("edge_type", "mandatory")).lower(), 0.0
            )
            score += float(edge.get("confidence", 1.0))

        return score, path

    ###############################################################
    # Beam Search
    ###############################################################

    def search(self, candidate_plan):
        """Return the best candidate sequence for the ordered prompt actions.

        ``candidate_plan`` is the structure produced by
        ``GraphMatcher.candidates``.
        """
        domain_graph = candidate_plan.get("domain_graph") or nx.DiGraph()
        actions = [
            action for action in candidate_plan.get("actions", [])
            if action.get("candidates")
        ]

        path_cache = {}

        # Each beam is (score, [selection, ...]).
        beams = [(0.0, [])]

        for action in actions:

            expanded = []

            for score, selection in beams:

                for candidate in action["candidates"]:

                    step_score = self.semantic_weight * float(
                        candidate.get("similarity", 0.0)
                    )

                    transition = None

                    if selection:
                        graph_score, path = self._transition_score(
                            domain_graph,
                            selection[-1]["domain_node_id"],
                            candidate["domain_node_id"],
                            path_cache,
                        )
                        step_score += self.graph_weight * graph_score
                        transition = {
                            "source": selection[-1]["domain_node_id"],
                            "target": candidate["domain_node_id"],
                            "path": path,
                            "graph_score": graph_score,
                        }

                    step = {
                        "prompt_node_id": action["prompt_node_id"],
                        "prompt_text": action["prompt_text"],
                        "domain_node_id": candidate["domain_node_id"],
                        "domain_node_name": candidate["domain_node_name"],
                        "similarity": candidate.get("similarity", 0.0),
                        "step_score": step_score,
                        "transition": transition,
                    }

                    expanded.append((score + step_score, selection + [step]))

            if not expanded:
                break

            # Keep only the best N partial workflows.
            expanded.sort(key=lambda item: item[0], reverse=True)
            beams = expanded[:self.beam_width]

        best_score, best_selection = beams[0]

        return {
            "selection": best_selection,
            "score": best_score,
            "beam": [
                {
                    "score": score,
                    "nodes": [step["domain_node_name"] for step in selection],
                }
                for score, selection in beams
            ],
            "skipped_actions": [
                action for action in candidate_plan.get("actions", [])
                if not action.get("candidates")
            ],
            "domain_graph": domain_graph,
        }

    ###############################################################
    # Expand the winning sequence into a workflow graph
    ###############################################################

    def expand(self, search_result):
        """Turn the winning candidate sequence into a workflow graph."""
        domain_graph = search_result.get("domain_graph") or nx.DiGraph()
        selection = search_result.get("selection", [])

        workflow_graph = nx.DiGraph()
        selected_ids = {step["domain_node_id"] for step in selection}
        unreachable_pairs = []

        for step in selection:
            node_id = step["domain_node_id"]
            node_data = dict(
                domain_graph.nodes[node_id]
                if node_id in domain_graph.nodes else {}
            )
            node_data.setdefault("name", step["domain_node_name"] or str(node_id))
            node_data["score"] = step["similarity"]
            node_data["prompt_text"] = step["prompt_text"]
            node_data["inferred"] = False
            workflow_graph.add_node(node_id, **node_data)

        for previous, current in zip(selection, selection[1:]):
            transition = current.get("transition") or {}
            path = transition.get("path")

            if not path:
                unreachable_pairs.append({
                    "source": previous["domain_node_id"],
                    "target": current["domain_node_id"],
                    "prompt_relation": "sequence",
                })
                continue

            for node_id in path:
                if node_id in workflow_graph:
                    continue
                node_data = dict(domain_graph.nodes[node_id])
                node_data["inferred"] = node_id not in selected_ids
                node_data.setdefault(
                    "prompt_text", node_data.get("name", str(node_id))
                )
                workflow_graph.add_node(node_id, **node_data)

            for path_source, path_target in zip(path, path[1:]):
                workflow_graph.add_edge(
                    path_source,
                    path_target,
                    **dict(domain_graph.edges[path_source, path_target]),
                )

        workflow_graph.graph["beam_score"] = search_result.get("score", 0.0)
        workflow_graph.graph["beam"] = search_result.get("beam", [])
        workflow_graph.graph["beam_selection"] = selection
        workflow_graph.graph["unreachable_prompt_pairs"] = unreachable_pairs
        workflow_graph.graph["unmatched_prompt_actions"] = search_result.get(
            "skipped_actions", []
        )

        return workflow_graph

    ###############################################################
    # Convenience entry point
    ###############################################################

    def plan(self, candidate_plan):
        """Run beam search and expand the winner in a single call."""
        return self.expand(self.search(candidate_plan))

    ###############################################################
    # Pretty Print
    ###############################################################

    @staticmethod
    def print_search(search_result):

        print()
        print("=" * 60)
        print("Beam Search")
        print("=" * 60)
        print()

        print("Beam")
        for entry in search_result.get("beam", []):
            print(f"  {entry['score']:.3f}  " + " -> ".join(entry["nodes"]))

        print()
        print("Best candidate workflow")
        for step in search_result.get("selection", []):
            print(
                f"  {step['prompt_text']} -> {step['domain_node_name']} "
                f"(similarity={step['similarity']:.3f}, "
                f"step={step['step_score']:.3f})"
            )

        skipped = search_result.get("skipped_actions", [])
        if skipped:
            print()
            print("Actions without candidates")
            for action in skipped:
                print("  " + str(action.get("prompt_text")))
