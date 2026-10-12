from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .graph import ComputeGraph, Op, Region, verify_compute_graph
from .lowering import TensorProgram


def form_regions(graph: ComputeGraph) -> ComputeGraph:
    verify_compute_graph(graph)
    regions = tuple(
        Region(
            id=f"region:{operation.id.removeprefix('op:')}",
            source_op_ids=(operation.id,),
            external_inputs=_external_inputs((operation,), graph),
            live_outputs=_live_outputs((operation,), graph),
            semantic_facts=_semantic_facts(operation, graph),
            tensor_programs=(operation.tensor_program,)
            if operation.tensor_program is not None
            else (),
        )
        for operation in graph.ops.values()
    )
    result = replace(graph, regions=regions)
    verify_compute_graph(result)
    _verify_regions(result)
    return result


def dump_region_graph(graph: ComputeGraph, path: str | Path) -> Path:
    operation_regions = {
        operation_id: region
        for region in graph.regions
        for operation_id in region.source_op_ids
    }
    lines = [
        "# RegionGraph",
        f"values={len(graph.values)} operations={len(graph.ops)} "
        f"regions={len(graph.regions)} inputs={len(graph.inputs)} "
        f"outputs={len(graph.outputs)}",
        "",
        "## Regions",
    ]
    for region in graph.regions:
        lines.append(
            f"{region.id} source_ops={list(region.source_op_ids)} "
            f"external_inputs={list(region.external_inputs)} "
            f"live_outputs={list(region.live_outputs)} "
            f"semantic_facts={region.semantic_facts!r}"
        )

    lines.extend(["", "## Dependencies"])
    for consumer in graph.regions:
        for input_id in consumer.external_inputs:
            producer_id = graph.values[input_id].producer
            if producer_id is None:
                continue
            producer = operation_regions[producer_id]
            lines.append(f"{producer.id} -> {consumer.id} value={input_id}")

    lines.extend(["", "## Graph Outputs"])
    for output_id in graph.outputs:
        producer_id = graph.values[output_id].producer
        region_id = operation_regions[producer_id].id if producer_id else "boundary"
        lines.append(f"{region_id} -> output value={output_id}")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _external_inputs(operations: tuple[Op, ...], graph: ComputeGraph) -> tuple[str, ...]:
    produced = {result_id for operation in operations for result_id in operation.results}
    inputs: list[str] = []
    for operation in operations:
        for operand_id in operation.operands:
            if operand_id not in produced and operand_id not in inputs:
                inputs.append(operand_id)
    return tuple(inputs)


def _live_outputs(operations: tuple[Op, ...], graph: ComputeGraph) -> tuple[str, ...]:
    operation_ids = {operation.id for operation in operations}
    produced = {result_id for operation in operations for result_id in operation.results}
    outputs: list[str] = []
    for result_id in graph.outputs:
        if result_id in produced and result_id not in outputs:
            outputs.append(result_id)
    for operation in operations:
        for result_id in operation.results:
            if result_id in outputs:
                continue
            users = graph.values[result_id].users
            if any(user not in operation_ids for user in users):
                outputs.append(result_id)
    return tuple(outputs)


def _semantic_facts(operation: Op, graph: ComputeGraph) -> dict[str, object]:
    program = operation.tensor_program
    if not isinstance(program, TensorProgram) or program.family != "contraction":
        return {}

    for load in program.loads:
        view_operation = _producing_view(load.value_id, graph)
        if view_operation is None:
            continue
        view_program = view_operation.tensor_program
        assert isinstance(view_program, TensorProgram)
        view_load = view_program.loads[0]
        if load.strides != view_load.strides:
            continue
        contraction_map = _index_map(load.axes, load.strides)
        view_map = _index_map(view_load.axes, view_load.strides)
        return {
            "family": "view_plus_gemm",
            "contraction": operation.id,
            "view": view_operation.id,
            "view_result": load.value_id,
            "view_source": view_load.value_id,
            "contraction_index_map": contraction_map,
            "view_index_map": view_map,
            "composed_index_map": _compose_index_maps(
                load.axes, view_load.axes, view_load.strides
            ),
            "materialization": "none",
            "alias": True,
        }
    return {}


def _producing_view(value_id: str, graph: ComputeGraph) -> Op | None:
    producer_id = graph.values[value_id].producer
    if producer_id is None:
        return None
    operation = graph.ops[producer_id]
    program = operation.tensor_program
    if (
        not isinstance(program, TensorProgram)
        or program.family != "view"
        or program.store.owns_storage
        or len(program.loads) != 1
        or value_id not in operation.results
    ):
        return None
    return operation


def _index_map(
    axes: tuple[str | int, ...], strides: tuple[int, ...]
) -> dict[str, object]:
    return {
        "terms": tuple(
            {"axis": axis, "stride": stride}
            for axis, stride in zip(axes, strides)
            if not isinstance(axis, int) or axis != 0
        ),
        "offset": sum(
            axis * stride
            for axis, stride in zip(axes, strides)
            if isinstance(axis, int)
        ),
    }


def _compose_index_maps(
    contraction_axes: tuple[str | int, ...],
    view_axes: tuple[str | int, ...],
    view_strides: tuple[int, ...],
) -> dict[str, object]:
    terms = [
        {
            "axis": contraction_axis,
            "view_axis": view_axis,
            "stride": view_stride,
        }
        for contraction_axis, view_axis, view_stride in zip(
            contraction_axes, view_axes, view_strides
        )
        if not isinstance(contraction_axis, int)
    ]
    offset = sum(
        contraction_axis * view_stride
        for contraction_axis, view_stride in zip(contraction_axes, view_strides)
        if isinstance(contraction_axis, int)
    )
    return {"terms": tuple(terms), "offset": offset}


def _verify_regions(graph: ComputeGraph) -> None:
    source_ids = [operation.id for operation in graph.ops.values()]
    region_ids = [region.id for region in graph.regions]
    covered_ids = [
        operation_id for region in graph.regions for operation_id in region.source_op_ids
    ]
    if len(set(region_ids)) != len(region_ids) or covered_ids != source_ids:
        raise ValueError("Regions do not cover every source operation exactly once")

    operation_regions = {
        operation_id: region
        for region in graph.regions
        for operation_id in region.source_op_ids
    }
    for region in graph.regions:
        operations = tuple(graph.ops[operation_id] for operation_id in region.source_op_ids)
        if region.external_inputs != _external_inputs(operations, graph):
            raise ValueError(f"Region {region.id} has incorrect external inputs")
        if region.live_outputs != _live_outputs(operations, graph):
            raise ValueError(f"Region {region.id} has incorrect live outputs")
        for result_id in region.live_outputs:
            producer = graph.values[result_id].producer
            if producer not in region.source_op_ids:
                raise ValueError(f"Region {region.id} outputs a foreign value")
        for operand_id in region.external_inputs:
            producer = graph.values[operand_id].producer
            if producer is not None and operation_regions[producer] is region:
                raise ValueError(f"Region {region.id} marks an internal value external")
