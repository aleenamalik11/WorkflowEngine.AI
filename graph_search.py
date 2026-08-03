import faiss
import numpy as np

from typing import List

from utils import (
    EmbeddingService,
    load_graph,
    load_pickle,
    cosine_similarity
)

##############################################################
# GraphSearch
##############################################################

class GraphSearch:

    def __init__(
            self,
            model_path,
            graph_path,
            faiss_index,
            metadata_file):

        self.embedding_service = EmbeddingService(model_path)

        self.graph = load_graph(graph_path)

        self.index = faiss.read_index(faiss_index)

        self.metadata = load_pickle(metadata_file)

    ##########################################################
    # Search Workflows
    ##########################################################

    def search_workflows(
            self,
            prompt,
            top_k=5):

        embedding = self.embedding_service.encode(prompt)

        scores, indices = self.index.search(

            np.array([embedding]),

            top_k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            results.append({

                "score": float(score),

                "workflow": self.metadata[idx]

            })

        return results

    ##########################################################
    # Search Nodes
    ##########################################################

    def search_nodes(
            self,
            text,
            top_k=10):

        query = self.embedding_service.encode(text)

        results = []

        for node_id, data in self.graph.nodes(data=True):

            if data.get("embedding") is None:
                continue

            score = cosine_similarity(

                query,

                data["embedding"]

            )

            results.append({

                "id": node_id,

                "name": data["name"],

                "score": float(score)

            })

        results.sort(

            key=lambda x: x["score"],

            reverse=True

        )

        return results[:top_k]

    ##########################################################
    # Search Edges
    ##########################################################

    def search_edges(
            self,
            relation):

        matches = []

        for source, target, data in self.graph.edges(data=True):

            if relation.lower() in data["relation"].lower():

                matches.append({

                    "source": source,

                    "target": target,

                    "relation": data["relation"]

                })

        return matches

    ##########################################################
    # Retrieve Workflow Fragment
    ##########################################################

    def workflow_fragment(
            self,
            start_node,
            depth=4):

        visited = set()

        fragment = []

        self.__dfs(

            start_node,

            depth,

            visited,

            fragment

        )

        return fragment

    ##########################################################

    def __dfs(
            self,
            node,
            depth,
            visited,
            fragment):

        if depth == 0:
            return

        if node in visited:
            return

        visited.add(node)

        fragment.append(node)

        for neighbor in self.graph.successors(node):

            self.__dfs(

                neighbor,

                depth - 1,

                visited,

                fragment

            )