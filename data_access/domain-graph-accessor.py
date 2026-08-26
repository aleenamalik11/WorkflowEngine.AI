from typing import Optional

from helpers.utils import _to_domain_node
from models import DomainNode, DomainRelationship


class DomainGraphService():

    def __init__(
        self,
        driver,
        database="neo4j",
        embedding_service=None,
        fulltext_index="domainNodeSearch",
    ):
        self.driver = driver

        self.database = database

        self.embedding_service = (
            embedding_service
        )

        # Kept for compatibility/configuration.

        self.fulltext_index = (
            fulltext_index
        )

        self._embedding_cache = {}

        # Cache complete domain nodes as well.
        self._node_cache = None

    # =========================================================
    # Retrieve all nodes
    # =========================================================

    def all_nodes(self) -> list[DomainNode]:

        if self._node_cache is not None:
            return list(
                self._node_cache.values()
            )

        # The ontology importer creates :GraphNode.
        #
        # Use the label explicitly rather than MATCH (n),
        # so unrelated Neo4j application data is not accidentally
        # treated as domain ontology.

        cypher = """
        MATCH (n:GraphNode)
        RETURN n
        """

        with self.driver.session(database=self.database) as session:

            rows = session.run(cypher)

            nodes = {}

            for row in rows:

                node = _to_domain_node(
                    row["n"]
                )

                if node.id:
                    nodes[node.id] = node


        return list(nodes.values())

    # =========================================================
    # Neighborhood
    # =========================================================

    def neighborhood(
        self,
        node_id,
        depth=1,
    ) -> list[DomainRelationship]:

        depth = max(
            1,
            int(depth),
        )

        cypher = f"""
        MATCH (n:GraphNode {{id: $id}})
              -[r*1..{depth}]-
              (m:GraphNode)

        UNWIND r AS rel

        RETURN DISTINCT
            startNode(rel).id AS source_id,
            type(rel) AS relation,
            endNode(rel).id AS target_id
        """

        with self.driver.session(database=self.database) as session:

            rows = session.run(cypher,id=node_id)

            relationships = []

            seen = set()

            for row in rows:

                source_id = row[
                    "source_id"
                ]

                target_id = row[
                    "target_id"
                ]

                relation = row[
                    "relation"
                ]

                key = (
                    source_id,
                    relation,
                    target_id,
                )

                if key in seen:
                    continue

                seen.add(key)

                relationships.append(
                    DomainRelationship(
                        source_id=source_id,
                        target_id=target_id,
                        relation=relation,
                    )
                )

            return relationships

    # =========================================================
    # Get one node
    # =========================================================

    def get_node(
        self,
        node_id,
    ) -> Optional[DomainNode]:

        # First check cache.

        if self._node_cache is not None:

            cached = self._node_cache.get(
                node_id
            )

            if cached is not None:
                return cached

        cypher = """
        MATCH (n:GraphNode {id: $id})
        RETURN n
        LIMIT 1
        """

        with self.driver.session(
            database=self.database
        ) as session:

            row = session.run(cypher, id=node_id).single()

            if not row:
                return None

            node = _to_domain_node(
                row["n"]
            )

            return node

