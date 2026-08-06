import networkx as nx
from utils import cosine_similarity


class GraphMatcher:

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
            if self.domain_graph.is_directed():
                return self.domain_graph
            return nx.DiGraph(self.domain_graph)

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
                graph.add_edge(source, target, **edge)

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
                        graph.add_edge(source, target, **edge_data)

        return graph

    @staticmethod
    def _edge_attributes(edge_data):
        """Normalize persisted domain-edge metadata for workflow consumers."""
        edge_data = dict(edge_data)
        edge_data["relation"] = edge_data.get(
            "relation", edge_data.get("transition", "success")
        )
        edge_data.setdefault("weight", 1)
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
    # Match concepts against Domain Graph via Cosine Similarity
    ###############################################################

    def match(self, prompt_graph, threshold=0.35):
        concepts = self._extract_concepts(prompt_graph)
        matched_graph = nx.DiGraph()

        prompt_to_domain_map = {}
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

            # 3. Filter by similarity threshold
            if best_node is None or best_score < threshold:
                continue

            # Persisted graph edges are keyed by the domain node key, rather
            # than by the generated ``id`` stored in node metadata.
            domain_id = best_node_id
            prompt_to_domain_map[concept["node_id"]] = domain_id

            matched_graph.add_node(
                domain_id,
                **best_node,
                score=best_score,
                prompt_text=concept["text"],
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
                path = nx.shortest_path(
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
