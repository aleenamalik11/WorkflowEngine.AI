import json
import uuid

from models import (
    WorkflowPlan,
    PlannerNode,
    PlannerEdge
)


class WorkflowPlanner:

    def __init__(self,
                 graph_search,
                 function_matcher):

        self.graph_search = graph_search
        self.function_matcher = function_matcher

    ##########################################################
    # Main entry
    ##########################################################

    def plan(self, prompt):

        ######################################################
        # 1 Retrieve similar workflows
        ######################################################

        workflows = self.graph_search.search_workflows(
            prompt,
            top_k=3
        )

        if not workflows:
            raise Exception("No similar workflow found.")

        ######################################################
        # 2 Choose best workflow
        ######################################################

        best = workflows[0]["workflow"]

        workflow = best["workflow_json"]

        # Excel stores JSON as string
        if isinstance(workflow, str):
            workflow = json.loads(workflow)

        ######################################################
        # 3 Match every workflow node to a registered function
        ######################################################

        planner_nodes = []
        id_map = {}

        for workflow_node in workflow["Nodes"]:

            matches = self.function_matcher.top_matches(
                workflow_node["Name"],
                k=1
            )

            if not matches:
                print(f"Warning: No function found for {workflow_node['Name']}")
                continue

            match = matches[0]

            planner_node = PlannerNode(
                id=str(uuid.uuid4()),
                function_name=match.function.name,
                node_type=workflow_node.get("Type", "custom")
            )

            planner_nodes.append(planner_node)

            # map ORIGINAL workflow node id -> planner node id
            id_map[workflow_node["Id"]] = planner_node.id

        ######################################################
        # 4 Build planner edges
        ######################################################

        planner_edges = []

        for source_id, transitions in workflow["Connections"].items():

            if source_id not in id_map:
                continue

            for transition, target_id in transitions.items():

                # Skip terminal states
                if target_id in ("Done", "End", "Stop"):
                    continue

                if target_id not in id_map:
                    continue

                planner_edges.append(

                    PlannerEdge(
                        source=id_map[source_id],
                        target=id_map[target_id],
                        transition=transition
                    )

                )

        ######################################################
        # 5 Determine start node
        ######################################################

        start_node = workflow.get("StartNodeId")

        if start_node in id_map:
            start_node = id_map[start_node]
        else:
            start_node = planner_nodes[0].id if planner_nodes else None

        ######################################################
        # 6 Return workflow plan
        ######################################################

        return WorkflowPlan(
            name=workflow["Name"],
            nodes=planner_nodes,
            edges=planner_edges,
            inputs=workflow.get("Inputs", []),
        )