from graph_search import GraphSearch

search = GraphSearch(
    "sentence-transformers/all-MiniLM-L6-v2",
    "models/workflow_graph.gpickle",
    "models/workflow.index",
    "models/workflow_metadata.pkl"
)

results = search.search_workflows(
    "Enroll Ahmed in Data Structures"
)

print(results)