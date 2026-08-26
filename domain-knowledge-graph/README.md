# Domain Knowledge Graph — ontology mapping

This project maps an LLM-extracted BRS graph into a controlled, Neo4j-ready
ontology graph. It deliberately keeps three concerns separate:

```text
raw BRS extraction -> ontology mapping + validation -> canonical graph -> normalization
```

The mapper is deterministic: it applies the vocabulary and constraints in an
ontology file; it does not ask an LLM to reinterpret the graph. This makes the
mapping repeatable and makes every rejected relationship reviewable. The
original node categories and relationship wording are kept as provenance. The
mapper does not silently create missing nodes or merge potentially distinct
concepts.

## End-to-end data flow

```text
LLM output (data/raw/*.json)
    │  category lists + relationships with source, target, and wording
    ▼
run_mapping.py
    │
    ├── ontology/core_ontology.json: allowed node categories and relationships
    ├── data/mapped/*_mapped.json: accepted canonical graph
    └── data/reports/*_mapping_report.json: items needing review
    ▼
import_neo4j.py
    ▼
Neo4j nodes, labels, typed relationships, and flat provenance properties
```

## 1. Raw LLM extraction

The LLM is responsible for extracting a graph from a BRS or other source
document. Its output belongs in `data/raw/` and is treated as an immutable
source artifact. It has category arrays such as `actors`, `components`,
`operations`, `entities`, `events`, and `rules`, followed by a `relationships`
array.

For example, the raw banking extraction contains these separate facts:

```json
{
  "components": [
    {"id": "fund_transfer_service", "name": "Fund Transfer Service"}
  ],
  "operations": [
    {"id": "transfer_funds", "name": "Transfer Funds"}
  ],
  "relationships": [
    {
      "source": "fund_transfer_service",
      "target": "transfer_funds",
      "type": "executes"
    }
  ]
}
```

Each relationship refers to nodes by ID. Newer extractions use the `type`
field for the LLM's wording; older ones may use `raw_type`. The mapper accepts
either form. The raw file may contain explanatory prose after its JSON object:
the reader intentionally parses only the first JSON value.

Keep the extraction vocabulary unconstrained. For example, an LLM can emit
`runs`, `handles`, `owns`, or `produces`; the next stage decides whether that
wording has a safe, valid canonical meaning.

## 2. Ontology mapper

Run the mapper with:

```bash
python3 run_mapping.py data/raw/banking_brs_raw.json
```

`run_mapping.py` reads the raw object, loads `ontology/core_ontology.json`,
and writes two artifacts:

- `data/mapped/banking_brs_raw_mapped.json` — the accepted canonical graph.
- `data/reports/banking_brs_raw_mapping_report.json` — the audit/review
  result, including every relationship that was not accepted.

### Node mapping

The ontology's `node_types` object maps raw category names to canonical types:

```json
{
  "actors": "Actor",
  "components": "Component",
  "operations": "Operation",
  "entities": "DomainEntity",
  "events": "Event",
  "rules": "Rule"
}
```

For every record in one of those lists, the mapper creates a canonical node
with `id`, `name`, `types`, and `provenance`. It groups entries with the same
raw ID rather than creating multiple graph nodes. Importantly, this is not a
semantic merge: all original records are retained in `provenance.raw_records`.

For example, `customer` appears in the raw `actors` and `entities` lists, so
the mapped representation is one multi-type node:

```json
{
  "id": "customer",
  "name": "Customer",
  "types": ["Actor", "DomainEntity"],
  "provenance": {
    "raw_categories": ["actors", "entities"]
  }
}
```

The mapping report flags such repeated IDs as `duplicate_raw_ids` for a human
to review during a later normalization step. A raw category absent from
`node_types` is not mapped.

### Relationship mapping and validation

Every relationship definition in the ontology specifies three things:

1. A canonical relationship type, such as `COMPONENT_EXECUTES`.
2. The raw words that may mean it, in `raw_terms`.
3. Valid source and target node types.

The definition used by the selected example is:

```json
{
  "type": "COMPONENT_EXECUTES",
  "raw_terms": ["executes", "execute", "runs", "run"],
  "source_types": ["Component"],
  "target_types": ["Operation"]
}
```

For each raw relationship the mapper performs these checks in order:

1. Resolve `source` and `target` IDs in the mapped-node index. If either node
   is missing, the relationship is placed in `unresolved_references`; no node
   is invented.
2. Normalize the raw wording by lowercasing it and treating spaces, `_`, and
   `-` equivalently. Find ontology definitions whose `raw_terms` contain that
   normalized wording. No match goes to `unmapped_relationships`.
3. Check the types of both resolved nodes against each candidate definition.
   A relationship is valid when the source shares at least one permitted
   `source_type` and the target shares at least one permitted `target_type`.
   Candidates with an incompatible direction or endpoint type go to
   `invalid_relationships`.
4. Use the first ontology definition that passes those tests, preserving the
   original wording and relationship position as provenance.

Therefore the raw edge `fund_transfer_service --executes--> transfer_funds`
becomes:

```json
{
  "id": "edge:6",
  "source": "fund_transfer_service",
  "target": "transfer_funds",
  "type": "COMPONENT_EXECUTES",
  "condition": null,
  "provenance": {
    "raw_type": "executes",
    "relationship_index": 6
  }
}
```

Some words deliberately have more than one possible canonical meaning. For
example, `produces` can map to `OPERATION_PRODUCES` when the target is a
`DomainEntity`, or to `OPERATION_PRODUCES_EVENT` when it is an `Event`. The
endpoint-type validation removes this ambiguity.

### Mapping report and ontology changes

Always inspect the report after a mapping run. It contains counts plus:

- `duplicate_raw_ids`: identical raw IDs in more than one recognized category.
- `unresolved_references`: edges whose source or target ID was not mapped.
- `unmapped_relationships`: raw wording not present in any `raw_terms` list.
- `invalid_relationships`: recognized wording whose endpoint types/direction
  violate the ontology.

To support a system-specific vocabulary, copy `ontology/core_ontology.json`,
extend its node categories or relationship definitions, and pass it explicitly:

```bash
python3 run_mapping.py data/raw/my_system.json \
  --ontology ontology/my_system_ontology.json
```

Do not edit the raw extraction to force it through the ontology. Update the
versioned ontology when a new term or valid relationship is genuinely needed.

## Run

```bash
python3 run_mapping.py data/raw/banking_brs_raw.json
python3 run_mapping.py /path/to/system_brs.json --ontology ontology/my_system_ontology.json
```

The input may contain explanatory text after its JSON object (as does the
attached BRS export); only the first JSON value is read. The command writes:

- `data/mapped/<input-name>_mapped.json` — mapped nodes, edges, and provenance
- `data/reports/<input-name>_mapping_report.json` — unresolved references and
  unmapped/invalid relationships for review

## Reusable ontology

The controlled vocabulary is in `ontology/core_ontology.json`. It maps common
extraction categories to `Actor`, `Operation`, `DomainEntity`, `Event`, and
`Rule`, then maps relationships only when their source and target types are
allowed. It contains no banking concepts.

For a specific system, copy the core ontology, extend its allowed types and
relationships, and pass that file through `--ontology`. Keep the raw extractor
generic; ontology mapping is a separate, versioned pipeline stage.

## Raw inputs

Place each LLM extraction in `data/raw/` and never edit it during mapping or
normalization. The included `banking_brs_raw.json` is the supplied raw graph;
its mapped and normalized variants must remain separate artifacts.

## Import into Neo4j

The Neo4j importer reads only the mapped artifact, not the raw LLM output.
Install the driver, then load it:

```bash
python3 -m pip install neo4j
python3 import_neo4j.py data/mapped/banking_brs_raw_mapped.json --password "$NEO4J_PASSWORD"
```

### What the importer creates

The importer first creates this constraint, if it does not already exist:

```cypher
CREATE CONSTRAINT graph_node_id IF NOT EXISTS
FOR (node:GraphNode) REQUIRE node.id IS UNIQUE
```

It then groups mapped nodes by their complete set of types and imports them in
batches. Each node is matched by `id` with `MERGE`, so rerunning the same
import updates the existing node rather than creating a duplicate. Every node
has the base `GraphNode` label and every mapped ontology type as a label:

```cypher
(:GraphNode:Component {id: 'fund_transfer_service', ...})
(:GraphNode:Operation {id: 'transfer_funds', ...})
(:GraphNode:Actor:DomainEntity {id: 'customer', ...})
```

The importer groups edges by their mapped `type`. It matches the two endpoint
nodes by `GraphNode.id` and `MERGE`s a relationship identified by its stable
mapped-edge ID. The selected example becomes this Neo4j pattern:

```cypher
(:GraphNode:Component {id: 'fund_transfer_service'})
  -[:COMPONENT_EXECUTES {id: 'edge:6', raw_type: 'executes'}]->
(:GraphNode:Operation {id: 'transfer_funds'})
```

This means the canonical ontology relationship (`COMPONENT_EXECUTES`) is the
Neo4j relationship type, while the original LLM wording (`executes`) remains a
relationship property for traceability. Node properties include `id`, `name`,
`types`, and the original categories. Neo4j properties cannot store nested
maps, so `provenance.raw_records` is serialized into the string property
`raw_records_json`.

### Inspecting the graph

Open Neo4j Browser at `http://localhost:7474` and use Bolt clients, including
this importer, through `bolt://localhost:7687`. Then run:

```cypher
MATCH (source)-[relationship]->(target)
RETURN source, relationship, target
LIMIT 100;
```

To inspect the example edge only:

```cypher
MATCH (source:Component {id: 'fund_transfer_service'})
      -[relationship:COMPONENT_EXECUTES]->
      (target:Operation {id: 'transfer_funds'})
RETURN source, relationship, target;
```
