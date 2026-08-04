import uuid
import networkx as nx

from sentence_transformers import SentenceTransformer

from workflow_prompt_parser import (
    PromptAnalysis,
    PromptAction,
    PromptEntity
)


class PromptGraphBuilder:

    def __init__(self,
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2"):

        self.embedding_model = SentenceTransformer(
            embedding_model
        )

    ###############################################################
    # Build Prompt Graph
    ###############################################################

    def build(self,
              analysis: PromptAnalysis):

        graph = nx.DiGraph()

        ###########################################################
        # Create Action Nodes
        ###########################################################

        action_lookup = {}

        for action in analysis.actions:

            node_id = str(uuid.uuid4())

            embedding = self.embedding_model.encode(
                action.lemma,
                normalize_embeddings=True
            )

            graph.add_node(

                node_id,

                id=node_id,

                name=action.lemma,

                original_text=action.text,

                type="Action",

                embedding=embedding

            )

            action_lookup[action.lemma] = node_id

        ###########################################################
        # Create Entity Nodes
        ###########################################################

        entity_lookup = {}

        for entity in analysis.entities:

            node_id = str(uuid.uuid4())

            embedding = self.embedding_model.encode(
                entity.root,
                normalize_embeddings=True
            )

            graph.add_node(

                node_id,

                id=node_id,

                name=entity.root,

                original_text=entity.text,

                type="Entity",

                embedding=embedding

            )

            entity_lookup[entity.root] = node_id

        ###########################################################
        # Attach Entity to Closest Action
        ###########################################################

        for entity in analysis.entities:

            nearest_action = None

            distance = 999999

            for action in analysis.actions:

                d = abs(

                    entity.token_index -
                    action.token_index

                )

                if d < distance:

                    distance = d
                    nearest_action = action

            if nearest_action is None:
                continue

            if nearest_action.lemma not in action_lookup:
                continue

            if entity.root not in entity_lookup:
                continue

            graph.add_edge(

                action_lookup[nearest_action.lemma],

                entity_lookup[entity.root],

                relation="acts_on"

            )

        ###########################################################
        # Action Relationships
        ###########################################################

        for relation in analysis.relations:

            if relation.source not in action_lookup:
                continue

            if relation.target not in action_lookup:
                continue

            graph.add_edge(

                action_lookup[relation.source],

                action_lookup[relation.target],

                relation=relation.relation

            )

        return graph

    ###############################################################
    # Print Graph
    ###############################################################

    @staticmethod
    def print_graph(graph):

        print()
        print("=" * 60)
        print("Prompt Graph")
        print("=" * 60)

        print()

        print("Nodes")

        for node_id, node in graph.nodes(data=True):

            print(node)

        print()

        print("Edges")

        for source, target, edge in graph.edges(data=True):

            print(

                graph.nodes[source]["name"],
                "--",
                edge["relation"],
                "-->",
                graph.nodes[target]["name"]

            )


###############################################################
# Test
###############################################################

if __name__ == "__main__":

    from workflow_prompt_parser import WorkflowPromptParser

    parser = WorkflowPromptParser()

    builder = PromptGraphBuilder()

    prompt = """
    Register a new student after validating documents
    and assign the student to Grade 8.
    """

    analysis = parser.parse(prompt)

    graph = builder.build(analysis)

    builder.print_graph(graph)