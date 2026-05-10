from __future__ import annotations

from typing import Iterable, List, Tuple

from ..core.harness import Harness, HarnessExecutionError
from ..core.modules import ModuleSpec


def validate_schema_compatibility(module: ModuleSpec, anchor: str, primitive: str, harness: Harness) -> None:
    if anchor not in harness.modules:
        raise ValueError(f"Unknown insertion vertex: {anchor}")
    if primitive not in {"pre-insert", "post-insert", "guard-insert", "branch-insert"}:
        raise ValueError(f"Unknown insertion primitive: {primitive}")
    if primitive in {"pre-insert", "post-insert"} and not module.input_schema:
        raise ValueError(f"Module {module.name} must declare an input schema for {primitive}")
    if primitive == "branch-insert" and "respond" not in harness.modules:
        raise ValueError("Branch inserts require a respond module in the harness")


def compute_edges_for_insertion(
    harness: Harness,
    module_name: str,
    vertex: str,
    primitive: str,
) -> List[Tuple[str, str]]:
    outgoing = [dst for src, dst in harness.edges if src == vertex]
    incoming = [src for src, dst in harness.edges if dst == vertex]
    if primitive == "pre-insert":
        harness.edges = [(src, dst) for src, dst in harness.edges if dst != vertex]
        return [(src, module_name) for src in incoming] + [(module_name, vertex)]
    if primitive == "post-insert":
        harness.edges = [(src, dst) for src, dst in harness.edges if src != vertex]
        return [(vertex, module_name)] + [(module_name, dst) for dst in outgoing]
    if primitive == "guard-insert":
        if incoming:
            return [(incoming[0], module_name), (module_name, vertex)]
        return [(module_name, vertex)]
    return [(vertex, module_name), (module_name, "respond")]


def apply_graph_surgery(
    harness: Harness,
    module: ModuleSpec,
    insertion_vertex: str,
    primitive: str,
) -> tuple[Harness, List[Tuple[str, str]]]:
    validate_schema_compatibility(module, insertion_vertex, primitive, harness)
    candidate = harness.copy()
    edges = compute_edges_for_insertion(candidate, module.name, insertion_vertex, primitive)
    candidate.add_module(module, edges)
    return candidate, edges


def assert_acyclic_after_update(harness: Harness) -> None:
    try:
        harness.topological_order()
    except HarnessExecutionError as exc:
        raise ValueError("Graph surgery produced an invalid DAG") from exc


def edge_set_snapshot(edges: Iterable[Tuple[str, str]]) -> List[str]:
    return [f"{src}->{dst}" for src, dst in sorted(edges)]
