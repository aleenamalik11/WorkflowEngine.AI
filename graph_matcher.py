import networkx as nx
from utils import cosine_similarity


class GraphMatcher:
    EDGE_TYPE_WEIGHTS = {
        "mandatory": 1,
        "alternative": 3,
        "optional": 5,
        "deprecated": 100,
    }

    def __init__(self,
                 embedding_service,
                 domain_graph,
                 rebel_model_name=None,
                 device=None):

        self.embedding_service = embedding_service
        self.domain_graph = domain_graph

        self.rebel_model_name = rebel_model_name
        self.device = device
        self.tokenizer = None
        self.rebel_model = None

    ###############################################################
    # Helper to iterate nodes regardless of graph storage format
    ###############################################################

    def _get_domain_nodes(self):
        """
        Retrieves domain nodes as (node_id, node_data) pairs,
        supporting both NetworkX DiGraph and standard dict representations.
        """
        if hasattr(self.domain_graph, "nodes"):
            return self.domain_graph.nodes(data=True)
        elif isinstance(self.domain_graph, dict) and "nodes" in self.domain_graph:
            nodes_data = self.domain_graph["nodes"]
            if isinstance(nodes_data, dict):
                return nodes_data.items()
            elif isinstance(nodes_data, list):
                return [(node.get("id", i), node) for i, node in enumerate(nodes_data)]

        raise TypeError("domain_graph must be a NetworkX Graph or a dict containing a 'nodes' key.")

    def _as_domain_digraph(self):
        """Return the persisted domain graph in a form suitable for path queries."""
        if isinstance(self.domain_graph, nx.Graph):
            graph = nx.DiGraph()
            graph.add_nodes_from(self.domain_graph.nodes(data=True))
            graph.add_edges_from(
                (source, target, self._edge_attributes(data))
                for source, target, data in self.domain_graph.edges(data=True)
            )
            return graph

        graph = nx.DiGraph()

        for node_id, node_data in self._get_domain_nodes():
            graph.add_node(node_id, **node_data)

        if not isinstance(self.domain_graph, dict):
            return graph

        # Training persists edges as dictionaries.  The adjacency fallback
        # also supports graphs produced by older training runs.
        edges = self.domain_graph.get("edges", [])
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            source = edge.get("source")
            target = edge.get("target")
            if source is not None and target is not None:
                graph.add_edge(source, target, **self._edge_attributes(edge))

        if graph.number_of_edges() == 0:
            for source, neighbours in self.domain_graph.get("adjacency", {}).items():
                for neighbour in neighbours:
                    if isinstance(neighbour, dict):
                        target = neighbour.get("target")
                        edge_data = neighbour
                    else:
                        target = neighbour
                        edge_data = {}
                    if target is not None:
                        graph.add_edge(
                            source, target, **self._edge_attributes(edge_data)
                        )

        return graph

    @classmethod
    def _edge_attributes(cls, edge_data):
        """Normalize persisted domain-edge metadata for workflow consumers."""
        edge_data = dict(edge_data)
        edge_data["relation"] = edge_data.get(
            "relation", edge_data.get("transition", "success")
        )
        edge_type = edge_data.get(
            "edge_type", edge_data.get("edge_category", "mandatory")
        )
        edge_type = str(edge_type).strip().lower()
        if edge_type not in cls.EDGE_TYPE_WEIGHTS:
            edge_type = "mandatory"
        edge_data["edge_type"] = edge_type
        edge_data.setdefault("weight", cls.EDGE_TYPE_WEIGHTS[edge_type])
        edge_data.setdefault("confidence", 1.0)
        return edge_data

    ###############################################################
    # REBEL Triplet Extraction Helper (Fixed Parser)
    ###############################################################

    def _extract_rebel_triplets(self, text):
        """
        Runs REBEL to parse raw text into (subject, relation, object) triplets safely.
        """
        if not text or not text.strip():
            return []

        self._load_rebel()
        if self.tokenizer is None or self.rebel_model is None:
            return []

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=256,
            truncation=True
        ).to(self.device)

        gen_kwargs = {
            "max_length": 128,
            "length_penalty": 0,
            "num_beams": 2,
            "num_return_sequences": 1,
        }

        import torch

        with torch.no_grad():
            generated_tokens = self.rebel_model.generate(
                **inputs,
                **gen_kwargs
            )

        decoded_text = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=False
        )[0]

        triplets = []
        current = 'text'
        subject, relation, object_ = '', '', ''

        tokens = decoded_text.replace("<s>", "").replace("</s>", "").replace("<pad>", "").split()
        for token in tokens:
            if token == "<triplet>":
                if subject.strip() and relation.strip() and object_.strip():
                    triplets.append((subject.strip(), relation.strip(), object_.strip()))
                subject, relation, object_ = '', '', ''
                current = 't'
            elif token == "<subj>":
                current = 's'
            elif token == "<obj>":
                current = 'o'
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token

        if subject.strip() and relation.strip() and object_.strip():
            triplets.append((subject.strip(), relation.strip(), object_.strip()))

        return triplets

    def _load_rebel(self):
        """Load REBEL only when relation extraction was explicitly enabled."""
        if self.tokenizer is not None or not self.rebel_model_name:
            return

        import os
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.rebel_model_name)
        self.rebel_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.rebel_model_name
        ).to(self.device)
        self.rebel_model.eval()

    ###############################################################
    # Extract concepts from Prompt Graph via REBEL
    ###############################################################

    def _extract_concepts(self, prompt_graph):
        """
        Extracts actionable concepts from prompt graph nodes using REBEL
        triplets and graph edge relationships.
        """
        concepts = []

        for node_id, node in prompt_graph.nodes(data=True):
            # Workflow steps come from action nodes. Entity nodes provide
            # context through ``acts_on`` edges and must not become steps.
            if node.get("type") and node["type"] != "Action":
                continue

            node_name = node.get("name", str(node_id))
            node_description = node.get("description", node_name)

            # 1. Gather connected target entities from prompt graph edges
            graph_entities = []
            for _, target, edge in prompt_graph.out_edges(node_id, data=True):
                if edge.get("relation") != "acts_on":
                    continue
                entity = prompt_graph.nodes[target]
                graph_entities.append(entity.get("name", str(target)))

            # 2. Extract relationships via REBEL on node text
            rebel_triplets = (
                self._extract_rebel_triplets(node_description)
                if self.rebel_model_name else []
            )

            rebel_entities = []
            for subj, rel, obj in rebel_triplets:
                if obj.lower() not in [e.lower() for e in rebel_entities]:
                    rebel_entities.append(obj)
                if subj.lower() != node_name.lower() and subj.lower() not in [e.lower() for e in rebel_entities]:
                    rebel_entities.append(subj)

            # Combine graph targets + REBEL extracted entities cleanly
            all_entities = list(dict.fromkeys(graph_entities + rebel_entities))

            if all_entities:
                concept_text = f"{node_name} {' '.join(all_entities)}"
            else:
                concept_text = node_name

            concepts.append({
                "node_id": node_id,
                "text": concept_text
            })

        return concepts

    ###############################################################
    # Top K candidates per prompt action (input for beam search)
    ###############################################################

    @staticmethod
    def _node_similarity(query_embedding, node):
        """Best similarity between the concept and a domain node.

        A domain node is embedded twice: once from its full ontology text
        (name + types + BRS statements) and once from its bare name.  Short
        prompts match the name, long prompts match the description, so the
        stronger of the two signals is used.
        """
        embedding = node.get("embedding")
        if embedding is None:
            return None

        score = cosine_similarity(query_embedding, embedding)

        name_embedding = node.get("name_embedding")
        if name_embedding is not None:
            score = max(
                score, cosine_similarity(query_embedding, name_embedding)
            )

        return score

    def candidates(self, prompt_graph, k=5, threshold=0.35):
        """Propose the top ``k`` domain nodes for every prompt action.

        Candidate selection is deliberately *not* resolved here: the beam
        search planner decides which combination of candidates forms the best
        workflow, because the best local match is not always the best step in
        a sequence.
        """
        concepts = self._extract_concepts(prompt_graph)
        domain_nodes = list(self._get_domain_nodes())
        domain_digraph = self._as_domain_digraph()

        actions = []

        for concept in concepts:
            query_embedding = self.embedding_service.encode(concept["text"])

            scored = []
            for node_id, node in domain_nodes:
                score = self._node_similarity(query_embedding, node)
                if score is None:
                    continue

                scored.append((score, node_id, node))

            scored.sort(key=lambda item: item[0], reverse=True)

            top = scored[:max(1, int(k))]

            candidates = [
                {
                    "domain_node_id": node_id,
                    "domain_node_name": node.get("name", str(node_id)),
                    "domain_node_types": node.get("types", []),
                    "executable": node.get("executable"),
                    "similarity": score,
                }
                for score, node_id, node in top
                if score >= threshold
            ]

            actions.append({
                "prompt_node_id": concept["node_id"],
                "prompt_text": concept["text"],
                "candidates": candidates,
                "top_candidates": [
                    {
                        "domain_node_id": node_id,
                        "domain_node_name": node.get("name", str(node_id)),
                        "domain_node_types": node.get("types", []),
                        "executable": node.get("executable"),
                        "similarity": score,
                    }
                    for score, node_id, node in top
                ],
                "best_similarity": scored[0][0] if scored else None,
                "threshold": threshold,
            })

        return {
            "actions": actions,
            "domain_graph": domain_digraph,
            "concept_count": len(concepts),
            "domain_node_count": len(domain_nodes),
            "embedded_domain_node_count": sum(
                1 for _, node in domain_nodes if node.get("embedding") is not None
            ),
        }

    @staticmethod
    def print_candidates(candidate_plan):
        """Print the candidate shortlist produced for each prompt action."""
        print("\nTop candidates per action")

        actions = candidate_plan.get("actions", [])

        print(
            f"  concepts={candidate_plan.get('concept_count', len(actions))}  "
            f"domain nodes={candidate_plan.get('domain_node_count', 'n/a')}  "
            f"embedded={candidate_plan.get('embedded_domain_node_count', 'n/a')}"
        )

        if not actions:
            print(
                "  no concepts extracted from the prompt graph: "
                "the prompt produced no Action nodes"
            )
            return

        for action in actions:
            print(f"- {action['prompt_text']}")
            if not action["candidates"]:
                best = action.get("best_similarity")
                best_str = f"{best:.3f}" if best is not None else "n/a"
                print(
                    f"    no candidate above threshold "
                    f"{action['threshold']:.3f} (best={best_str})"
                )
                for candidate in action.get("top_candidates", []):
                    print(
                        f"      rejected {candidate['domain_node_name']} "
                        f"{candidate.get('domain_node_types', [])} "
                        f"({candidate['similarity']:.3f})"
                    )
                continue
            for candidate in action["candidates"]:
                print(
                    f"    {candidate['domain_node_name']} "
                    f"{candidate.get('domain_node_types', [])} "
                    f"({candidate['similarity']:.3f})"
                )

    ###############################################################
    # Match concepts against Domain Graph via Cosine Similarity
    ###############################################################

    def match(self, prompt_graph, threshold=0.35):
        concepts = self._extract_concepts(prompt_graph)
        matched_graph = nx.DiGraph()

        prompt_to_domain_map = {}
        match_diagnostics = []
        domain_nodes = list(self._get_domain_nodes())

        for concept in concepts:
            # 1. Embed user concept
            query_embedding = self.embedding_service.encode(concept["text"])

            best_score = -1.0
            best_node = None
            best_node_id = None

            # 2. Compare against domain embeddings via Cosine Similarity
            for node_id, node in domain_nodes:
                embedding = node.get("embedding")
                if embedding is None:
                    continue

                score = cosine_similarity(query_embedding, embedding)

                if score > best_score:
                    best_score = score
                    best_node = node
                    best_node_id = node_id

            accepted = best_node is not None and best_score >= threshold
            match_diagnostics.append({
                "prompt_node_id": concept["node_id"],
                "prompt_text": concept["text"],
                "best_domain_node_id": best_node_id,
                "best_domain_node_name": (
                    best_node.get("name", str(best_node_id))
                    if best_node is not None else None
                ),
                "score": best_score,
                "threshold": threshold,
                "accepted": accepted,
            })

            # 3. Filter by similarity threshold
            if not accepted:
                continue

            # Persisted graph edges are keyed by the domain node key, rather
            # than by the generated ``id`` stored in node metadata.
            domain_id = best_node_id
            prompt_to_domain_map[concept["node_id"]] = domain_id

            if domain_id in matched_graph:
                # Multiple prompt actions can resolve to one domain concept.
                # Keep the strongest match and preserve every source action as
                # diagnostics instead of silently overwriting prompt_text.
                existing = matched_graph.nodes[domain_id]
                existing["matched_prompt_texts"].append(concept["text"])
                existing["matched_prompt_node_ids"].append(concept["node_id"])
                if best_score > existing["score"]:
                    existing["score"] = best_score
                    existing["prompt_text"] = concept["text"]
                continue

            matched_graph.add_node(
                domain_id,
                **best_node,
                score=best_score,
                prompt_text=concept["text"],
                matched_prompt_texts=[concept["text"]],
                matched_prompt_node_ids=[concept["node_id"]],
                inferred=False,
            )

        ###########################################################
        # 4. Build structure from domain-validated paths
        ###########################################################

        # Prompt edges describe the requested ordering.  They are deliberately
        # not copied into the result; each mapped pair is connected only when
        # the domain graph contains a directed path between them.
        domain_digraph = self._as_domain_digraph()
        explicitly_matched = set(prompt_to_domain_map.values())
        unreachable_pairs = []

        for source, target, prompt_edge in prompt_graph.edges(data=True):
            source_match = prompt_to_domain_map.get(source)
            target_match = prompt_to_domain_map.get(target)

            if not source_match or not target_match or source_match == target_match:
                continue

            try:
                path = nx.dijkstra_path(
                    domain_digraph,
                    source=source_match,
                    target=target_match,
                    weight="weight",
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                unreachable_pairs.append({
                    "source": source_match,
                    "target": target_match,
                    "prompt_relation": prompt_edge.get("relation"),
                })
                continue

            for node_id in path:
                if node_id in matched_graph:
                    continue
                node_data = dict(domain_digraph.nodes[node_id])
                node_data["inferred"] = node_id not in explicitly_matched
                if node_data["inferred"]:
                    node_data.setdefault("prompt_text", node_data.get("name", str(node_id)))
                matched_graph.add_node(node_id, **node_data)

            for path_source, path_target in zip(path, path[1:]):
                edge_data = self._edge_attributes(
                    domain_digraph.edges[path_source, path_target]
                )
                matched_graph.add_edge(path_source, path_target, **edge_data)

        # Kept as graph metadata so callers can surface a useful diagnostic
        # without manufacturing a non-domain edge as a fallback.
        matched_graph.graph["unreachable_prompt_pairs"] = unreachable_pairs
        matched_graph.graph["match_diagnostics"] = match_diagnostics
        matched_graph.graph["unmatched_prompt_actions"] = [
            item for item in match_diagnostics if not item["accepted"]
        ]
        matched_graph.graph["duplicate_domain_matches"] = [
            {
                "domain_node_id": node_id,
                "domain_node_name": node.get("name", str(node_id)),
                "prompt_texts": node["matched_prompt_texts"],
            }
            for node_id, node in matched_graph.nodes(data=True)
            if len(node.get("matched_prompt_node_ids", [])) > 1
        ]

        return matched_graph

    ###############################################################
    # Output Debug Helper
    ###############################################################

    @staticmethod
    def print_graph(graph):
        print("\n" + "=" * 60)
        print("Matched Graph")
        print("=" * 60 + "\n")

        print("Nodes")
        for _, node in graph.nodes(data=True):
            score_str = f"({node['score']:.3f})" if "score" in node else ""
            print(f"- {node.get('name', 'Unknown')} {score_str}")

        print("\nEdges")
        for source, target, edge in graph.edges(data=True):
            s_name = graph.nodes[source].get("name", source)
            t_name = graph.nodes[target].get("name", target)
            print(f"{s_name} -- {edge.get('relation', 'connected_to')} --> {t_name}")

    @staticmethod
    def print_match_diagnostics(graph):
        """Print why each prompt action was accepted or rejected."""
        print("\nMatch diagnostics")
        for item in graph.graph.get("match_diagnostics", []):
            outcome = "accepted" if item["accepted"] else "rejected"
            score = item["score"]
            print(
                f"- {item['prompt_text']} -> {item['best_domain_node_name']} "
                f"(score={score:.3f}, threshold={item['threshold']:.3f}, {outcome})"
            )
