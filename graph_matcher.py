import networkx as nx

from utils import cosine_similarity


class GraphMatcher:

    def __init__(self,
                 embedding_service,
                 domain_graph):

        self.embedding_service = embedding_service
        self.domain_graph = domain_graph

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

    ###############################################################
    # Extract concepts from Prompt Graph
    ###############################################################

    def _extract_concepts(self, prompt_graph):
        concepts = []

        for node_id, node in prompt_graph.nodes(data=True):
            node_type = node.get("type", "")
            if node_type and node_type != "Action":
                continue

            # Extract target entities connected via "acts_on"
            entities = []
            for _, target, edge in prompt_graph.out_edges(node_id, data=True):
                if edge.get("relation") == "acts_on":
                    entity = prompt_graph.nodes[target]
                    entities.append(entity.get("name", str(target)))

            node_name = node.get("name", str(node_id))
            concept_text = f"{node_name} {' '.join(entities)}" if entities else node_name

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

        # Direct mapping: prompt_node_id -> matched domain_node_id
        prompt_to_domain_map = {}

        for concept in concepts:
            # 1. Embed user concept
            query_embedding = self.embedding_service.encode(concept["text"])

            best_score = -1.0
            best_node = None

            # 2. Compare against domain embeddings via Cosine Similarity
            for node_id, node in self._get_domain_nodes():
                embedding = node.get("embedding")
                if embedding is None:
                    continue

                score = cosine_similarity(query_embedding, embedding)

                if score > best_score:
                    best_score = score
                    best_node = node

            # 3. Filter by similarity threshold
            if best_node is None or best_score < threshold:
                continue

            domain_id = best_node.get("id", best_node.get("name"))
            prompt_to_domain_map[concept["node_id"]] = domain_id

            matched_graph.add_node(
                domain_id,
                **best_node,
                score=best_score,
                prompt_text=concept["text"]
            )

        ###########################################################
        # 4. Map Prompt Graph Edges to Domain Nodes
        ###########################################################

        for source, target, edge in prompt_graph.edges(data=True):
            source_match = prompt_to_domain_map.get(source)
            target_match = prompt_to_domain_map.get(target)

            if source_match and target_match and source_match != target_match:
                matched_graph.add_edge(
                    source_match,
                    target_match,
                    relation=edge.get("relation", "connected_to")
                )

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