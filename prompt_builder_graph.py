import uuid
import networkx as nx

from workflow_prompt_parser import (
    PromptAnalysis,
    PromptAction,
    PromptEntity
)
from utils import EmbeddingService


class PromptGraphBuilder:

    def __init__(self, embedding_service=None):
        self.embedding_service = embedding_service or EmbeddingService()

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

            embedding = self.embedding_service.encode(action.lemma)

            graph.add_node(

                node_id,

                id=node_id,

                name=action.lemma,

                original_text=action.text,

                description=action.text,

                type="Action",

                embedding=embedding

            )

            action_lookup[action.token_index] = node_id

        ###########################################################
        # Create Entity Nodes
        ###########################################################

        entity_lookup = {}

        for entity in analysis.entities:

            node_id = str(uuid.uuid4())

            embedding = self.embedding_service.encode(entity.root)

            graph.add_node(

                node_id,

                id=node_id,

                name=entity.root,

                original_text=entity.text,

                description=entity.text,

                type="Entity",

                embedding=embedding

            )

            entity_lookup[entity.token_index] = node_id

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

            if nearest_action.token_index not in action_lookup:
                continue

            if entity.token_index not in entity_lookup:
                continue

            graph.add_edge(

                action_lookup[nearest_action.token_index],

                entity_lookup[entity.token_index],

                relation="acts_on"

            )

        ###########################################################
        # Action Relationships
        ###########################################################

        for relation in analysis.relations:
            source_id = next(
                (action_lookup[action.token_index] for action in analysis.actions
                 if action.lemma == relation.source),
                None,
            )
            target_id = next(
                (action_lookup[action.token_index] for action in analysis.actions
                 if action.lemma == relation.target),
                None,
            )
            if source_id and target_id and source_id != target_id:
                graph.add_edge(source_id, target_id, relation=relation.relation)

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
