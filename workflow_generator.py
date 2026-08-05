import uuid


class WorkflowGenerator:

    def __init__(self, function_matcher):
        self.function_matcher = function_matcher

    ###############################################################
    # Helper to resolve match output safely
    ###############################################################

    def _get_function_from_match(self, match_item):
        """Extracts function metadata regardless of return type."""
        if hasattr(match_item, "function"):
            return match_item.function
        elif isinstance(match_item, tuple) and len(match_item) > 0:
            return match_item[0]
        elif isinstance(match_item, dict):
            return match_item.get("function", match_item)
        return match_item

    ###############################################################
    # Generate Workflow JSON
    ###############################################################

    def generate(self, plan, workflow_name="Generated Workflow"):

        # 1. Safely extract graph and execution order from plan
        if isinstance(plan, dict):
            graph = plan.get("graph")
            execution_order = plan.get("execution_order") or plan.get("order") or plan.get("path") or []
        else:
            graph = plan
            execution_order = []

        # Fallback to graph nodes if execution_order wasn't provided or empty
        if not execution_order and graph is not None:
            if hasattr(graph, "nodes"):
                execution_order = list(graph.nodes())
            elif isinstance(graph, dict) and "nodes" in graph:
                execution_order = list(graph["nodes"].keys()) if isinstance(graph["nodes"], dict) else [
                    n.get("id", i) for i, n in enumerate(graph["nodes"])
                ]

        workflow = {
            "Id": str(uuid.uuid4()),
            "Name": workflow_name,
            "Version": "1.0",
            "StartNodeId": None,
            "Inputs": [],
            "Nodes": [],
            "Connections": {}
        }

        workflow_node_lookup = {}

        if graph is None:
            return workflow

        ###########################################################
        # Create Workflow Nodes
        ###########################################################

        for node_id in execution_order:

            # Retrieve node metadata safely
            if hasattr(graph, "nodes"):
                node_data = graph.nodes[node_id] if node_id in graph.nodes else {}
            elif isinstance(graph, dict) and "nodes" in graph:
                nodes = graph["nodes"]
                node_data = nodes.get(node_id, {}) if isinstance(nodes, dict) else {}
            else:
                node_data = {}

            node_name = node_data.get("name", str(node_id))
            match_text = node_data.get("prompt_text", node_name)

            #######################################################
            # Match domain concept to registered function
            #######################################################

            # Match the function from the user's original concept when it is
            # available; the domain-node label can be a related workflow step.
            matches = self.function_matcher.top_matches(match_text, k=1)

            function_obj = None
            if matches:
                function_obj = self._get_function_from_match(matches[0])

            workflow_node_id = str(uuid.uuid4())
            workflow_node_lookup[node_id] = workflow_node_id

            # Extract fields safely whether function_obj is a class, dict, or object
            func_name = getattr(function_obj, "name", None) or (function_obj.get("name") if isinstance(function_obj, dict) else node_name)
            func_desc = getattr(function_obj, "description", None) or (function_obj.get("description") if isinstance(function_obj, dict) else "")
            func_inputs = getattr(function_obj, "inputs", None) or (function_obj.get("inputs") if isinstance(function_obj, dict) else [])
            func_outputs = getattr(function_obj, "outputs", None) or (function_obj.get("outputs") if isinstance(function_obj, dict) else [])

            workflow["Nodes"].append({
                "Id": workflow_node_id,
                "Name": func_name,
                "Type": "CustomNode",
                "Description": func_desc,
                "Inputs": func_inputs,
                "Outputs": func_outputs,
                "Inferred": node_data.get("inferred", False)
            })

        ###########################################################
        # Start Node
        ###########################################################

        if execution_order:
            first = execution_order[0]
            if first in workflow_node_lookup:
                workflow["StartNodeId"] = workflow_node_lookup[first]
            elif workflow["Nodes"]:
                workflow["StartNodeId"] = workflow["Nodes"][0]["Id"]
        elif workflow["Nodes"]:
            workflow["StartNodeId"] = workflow["Nodes"][0]["Id"]

        ###########################################################
        # Connections
        ###########################################################

        edges = []
        if hasattr(graph, "edges"):
            edges = graph.edges(data=True)
        elif isinstance(graph, dict) and "edges" in graph:
            raw_edges = graph["edges"]
            for e in raw_edges:
                if isinstance(e, (tuple, list)) and len(e) >= 2:
                    data = e[2] if len(e) > 2 and isinstance(e[2], dict) else {}
                    edges.append((e[0], e[1], data))
                elif isinstance(e, dict):
                    edges.append((e.get("source"), e.get("target"), e))

        for source, target, edge_data in edges:
            if source not in workflow_node_lookup or target not in workflow_node_lookup:
                continue

            src_id = workflow_node_lookup[source]
            tgt_id = workflow_node_lookup[target]

            workflow["Connections"].setdefault(src_id, {})
            relation = edge_data.get("relation", "success") if isinstance(edge_data, dict) else "success"
            workflow["Connections"][src_id][relation] = tgt_id

        ###########################################################
        # Sequential Fallback if graph edges were disconnected
        ###########################################################

        if not workflow["Connections"] and len(workflow["Nodes"]) > 1:
            for i in range(len(workflow["Nodes"]) - 1):
                src_id = workflow["Nodes"][i]["Id"]
                tgt_id = workflow["Nodes"][i + 1]["Id"]
                workflow["Connections"][src_id] = {"success": tgt_id}

        ###########################################################
        # Default terminal state
        ###########################################################

        for node in workflow["Nodes"]:
            node_id = node["Id"]
            if node_id not in workflow["Connections"]:
                workflow["Connections"][node_id] = {
                    "success": "Done"
                }

        return workflow

    ###############################################################
    # Pretty Print
    ###############################################################

    @staticmethod
    def print(workflow):
        print()
        print("=" * 70)
        print(workflow.get("Name", "Generated Workflow"))
        print("=" * 70)
        print()

        print("Start Node")
        print(workflow.get("StartNodeId"))
        print()

        print("Nodes")
        for node in workflow.get("Nodes", []):
            status = "(Inferred)" if node.get("Inferred") else "(Matched)"
            print(node.get("Name"), status)

        print()
        print("Connections")
        for source, transitions in workflow.get("Connections", {}).items():
            print(source)
            for key, target in transitions.items():
                print("   ", key, "->", target)
