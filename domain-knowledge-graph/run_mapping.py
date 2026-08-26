#!/usr/bin/env python3
"""Run ontology mapping and write reviewable artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from argparse import ArgumentParser

from mapping.ontology_mapper import OntologyMapper, read_first_json


ROOT = Path(__file__).resolve().parent
DEFAULT_ONTOLOGY = ROOT / "ontology" / "core_ontology.json"


def main() -> None:
    parser = ArgumentParser(description="Map a raw BRS graph to a controlled ontology.")
    parser.add_argument("input", type=Path, help="Raw BRS JSON file")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY,
                        help="Ontology JSON file (defaults to the generic core ontology)")
    args = parser.parse_args()

    input_name = args.input.stem.replace(" ", "_").lower()
    mapped_output = ROOT / "data" / "mapped" / f"{input_name}_mapped.json"
    report_output = ROOT / "data" / "reports" / f"{input_name}_mapping_report.json"
    raw = read_first_json(args.input)
    mapped, report = OntologyMapper(args.ontology).map_graph(raw)
    for output, value in ((mapped_output, mapped), (report_output, report)):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    summary = report["summary"]
    print(
        f"Mapped {summary['mapped_node_count']} nodes and "
        f"{summary['mapped_relationship_count']}/{summary['raw_relationship_count']} relationships."
    )
    print(f"Review report: {report_output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
