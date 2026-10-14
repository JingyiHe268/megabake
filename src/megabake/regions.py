from __future__ import annotations

from dataclasses import replace
from collections.abc import Mapping
from pathlib import Path
from math import prod

from .graph import ComputeGraph, Op, Region, RegionCandidate, verify_compute_graph
from .lowering import (
    Binary,
    Constant,
    IndexedLoad,
    Accumulator,
    AxisIndex,
    Cast,
    Load,
    ScalarExpression,
    TensorProgram,
    Unary,
)


_DTYPE_BYTES = {
    "torch.bool": 1,
    "torch.uint8": 1,
    "torch.int8": 1,
    "torch.int16": 2,
    "torch.int32": 4,
    "torch.int64": 8,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.float32": 4,
    "torch.float64": 8,
}


def form_regions(graph: ComputeGraph) -> ComputeGraph:
    verify_compute_graph(graph)
    candidates_by_tail = _fusion_candidates(graph)
    regions = tuple(
        replace(
            Region(
                id=f"region:{operation.id.removeprefix('op:')}",
                source_op_ids=(operation.id,),
                external_inputs=_external_inputs((operation,), graph),
                live_outputs=_live_outputs((operation,), graph),
                semantic_facts=_semantic_facts(operation, graph),
                tensor_programs=(operation.tensor_program,)
                if operation.tensor_program is not None
                else (),
            ),
            fusion_candidates=candidates_by_tail.get(operation.id, ()),
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
        for candidate in region.fusion_candidates:
            lines.append(
                f"  candidate={candidate.id} source_ops={list(candidate.source_op_ids)} "
                f"executable={candidate.executable} rejections={list(candidate.rejections)}"
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


def _fusion_candidates(graph: ComputeGraph) -> dict[str, tuple[RegionCandidate, ...]]:
    candidates: dict[str, list[RegionCandidate]] = {}
    for operation in graph.ops.values():
        candidate = _scalar_fusion_candidate(operation, graph)
        if candidate is not None:
            candidates.setdefault(operation.id, []).append(candidate)

    scalar_candidates = [
        candidate
        for values in candidates.values()
        for candidate in values
        if candidate.semantic_facts["family"] == "scalar_chain"
    ]
    for operation_id, values in list(candidates.items()):
        candidates[operation_id] = [
            candidate
            for candidate in values
            if candidate.semantic_facts["family"] != "scalar_chain"
            or not any(
                set(candidate.source_op_ids) < set(other.source_op_ids)
                for other in scalar_candidates
            )
        ]

    gated_candidate = _gated_mlp_candidate(graph)
    if gated_candidate is not None:
        tail_operation = gated_candidate.source_op_ids[-1]
        candidates.setdefault(tail_operation, []).append(gated_candidate)
    return {tail: tuple(values) for tail, values in candidates.items()}


def _scalar_fusion_candidate(tail: Op, graph: ComputeGraph) -> RegionCandidate | None:
    if not isinstance(tail.tensor_program, TensorProgram):
        return None
    if tail.tensor_program.family not in {"elementwise", "copy"}:
        return None

    operations = _scalar_fusion_operations(tail, graph)
    if len(operations) < 2:
        return None

    operation_ids = {operation.id for operation in operations}
    rejections: list[str] = []
    if any(_has_illegal_effects(operation) for operation in operations):
        rejections.append("source effects do not fit a vector-chain body")
    if any(
        operation.results[0] in graph.outputs
        or any(
            user not in operation_ids
            for user in graph.values[operation.results[0]].users
        )
        for operation in operations[:-1]
    ):
        rejections.append("an intermediate value has an external use")
    if not all(
        _same_chain_layout(operation.results[0], tail.results[0], graph)
        for operation in operations
    ):
        rejections.append("source ownership or broadcast contract changes")

    fused_program = _compose_scalar_chain(operations, graph)
    external_inputs = tuple(
        load.value_id for load in fused_program.loads
    ) if fused_program is not None else _external_inputs(operations, graph)
    return RegionCandidate(
        id=f"region:{tail.id.removeprefix('op:')}:fused",
        source_op_ids=tuple(operation.id for operation in operations),
        external_inputs=external_inputs,
        live_outputs=(tail.results[0],),
        tensor_program=fused_program,
        semantic_facts={
            "family": "scalar_chain",
            "fusion": "vector_chain",
            "source_casts": "preserved",
            "packing_cost_bytes": 0,
            "conversion_cost_bytes": 0,
        },
        cost_bytes=sum(
            _value_bytes(operation.results[0], graph) * 2
            for operation in operations[:-1]
        ),
        rejections=tuple(rejections),
    )


def _scalar_fusion_operations(tail: Op, graph: ComputeGraph) -> tuple[Op, ...]:
    operations: list[Op] = []
    visited: set[str] = set()

    def visit(operation: Op) -> None:
        if operation.id in visited:
            return
        visited.add(operation.id)
        program = operation.tensor_program
        if not isinstance(program, TensorProgram):
            return
        for load in program.loads:
            producer_id = graph.values[load.value_id].producer
            if producer_id is None:
                continue
            producer = graph.ops[producer_id]
            if _is_scalar_operation(producer):
                visit(producer)
        operations.append(operation)

    visit(tail)
    return tuple(operations)


def _is_scalar_operation(operation: Op) -> bool:
    program = operation.tensor_program
    return isinstance(program, TensorProgram) and program.family in {
        "elementwise",
        "copy",
    }


def _compose_scalar_chain(
    chain: tuple[Op, ...], graph: ComputeGraph
) -> TensorProgram | None:
    tail = chain[-1]
    tail_program = tail.tensor_program
    if not isinstance(tail_program, TensorProgram):
        return None

    leaf_loads: list[IndexedLoad] = []
    leaf_ids: dict[str, str] = {}
    expressions: dict[str, ScalarExpression] = {}
    for operation in chain:
        program = operation.tensor_program
        if not isinstance(program, TensorProgram):
            return None
        replacements: dict[str, ScalarExpression] = {}
        for load in program.loads:
            if load.value_id in expressions:
                replacements[load.id] = expressions[load.value_id]
                continue
            leaf_id = leaf_ids.get(load.value_id)
            if leaf_id is None:
                leaf_id = f"load:{len(leaf_loads)}"
                leaf_loads.append(
                    IndexedLoad(
                        id=leaf_id,
                        value_id=load.value_id,
                        axes=load.axes,
                        strides=load.strides,
                        dtype=load.dtype,
                        source_shape=load.source_shape,
                    )
                )
                leaf_ids[load.value_id] = leaf_id
            replacements[load.id] = Load(leaf_id, load.dtype)
        expressions[operation.results[0]] = _substitute_loads(
            program.combine, replacements
        )

    return replace(
        tail_program,
        loads=tuple(leaf_loads),
        combine=expressions[tail.results[0]],
    )


def _substitute_loads(
    expression: ScalarExpression, replacements: Mapping[str, ScalarExpression]
) -> ScalarExpression:
    if isinstance(expression, Load):
        return replacements[expression.load_id]
    if isinstance(expression, Constant):
        return expression
    if isinstance(expression, Unary):
        return replace(expression, operand=_substitute_loads(expression.operand, replacements))
    if isinstance(expression, Binary):
        return replace(
            expression,
            lhs=_substitute_loads(expression.lhs, replacements),
            rhs=_substitute_loads(expression.rhs, replacements),
        )
    if isinstance(expression, (AxisIndex, Accumulator)):
        return expression
    if isinstance(expression, Cast):
        return replace(expression, operand=_substitute_loads(expression.operand, replacements))
    if isinstance(expression, Where):
        return replace(
            expression,
            condition=_substitute_loads(expression.condition, replacements),
            true_value=_substitute_loads(expression.true_value, replacements),
            false_value=_substitute_loads(expression.false_value, replacements),
        )
    raise ValueError(f"unsupported scalar expression {expression!r}")


def _gated_mlp_candidate(graph: ComputeGraph) -> RegionCandidate | None:
    for multiply in graph.ops.values():
        program = multiply.tensor_program
        if not isinstance(program, TensorProgram) or program.family != "elementwise":
            continue
        if not isinstance(program.combine, Binary) or program.combine.operator != "mul":
            continue
        value_by_load = {load.id: load.value_id for load in program.loads}
        operands = [value_by_load.get(_load_id(program.combine.lhs)), value_by_load.get(_load_id(program.combine.rhs))]
        if any(operand is None for operand in operands):
            continue
        activation_id = None
        up_id = None
        activation_operations: tuple[Op, ...] = ()
        for operand_id in operands:
            producer_id = graph.values[operand_id].producer
            if producer_id is None:
                continue
            producer_program = graph.ops[producer_id].tensor_program
            decomposed_silu = _decomposed_silu_operations(graph.ops[producer_id], graph)
            if decomposed_silu is not None:
                activation_id = operand_id
                activation_operations = decomposed_silu
            elif (
                isinstance(producer_program, TensorProgram)
                and isinstance(producer_program.combine, Unary)
                and producer_program.combine.operator == "silu"
            ):
                activation_id = operand_id
                activation_operations = (graph.ops[producer_id],)
            else:
                up_id = operand_id
        if activation_id is None or up_id is None:
            continue
        up_producer_id = graph.values[up_id].producer
        if up_producer_id is None:
            continue
        activation = activation_operations[-1]
        up = graph.ops[up_producer_id]
        gate_id = activation_operations[0].operands[0] if activation_operations else None
        if gate_id is None:
            continue
        gate_producer_id = graph.values[gate_id].producer
        if gate_producer_id is None:
            continue
        gate = graph.ops[gate_producer_id]
        if (
            not activation_operations
            or not _is_contraction(gate)
            or not _is_contraction(up)
        ):
            continue
        gate_program = gate.tensor_program
        up_program = up.tensor_program
        assert isinstance(gate_program, TensorProgram)
        assert isinstance(up_program, TensorProgram)
        if (
            gate_program.iteration_domains != up_program.iteration_domains
            or gate_program.reduction_axes != up_program.reduction_axes
            or gate_program.loads[0].value_id != up_program.loads[0].value_id
        ):
            continue

        source_op_ids = (
            gate.id,
            up.id,
            *(operation.id for operation in activation_operations),
            multiply.id,
        )
        operations = tuple(graph.ops[operation_id] for operation_id in source_op_ids)
        external_outputs = _external_outputs(operations, graph)
        rejections = []
        if any(_has_illegal_effects(operation) for operation in operations):
            rejections.append("source effects do not fit a paired GEMM epilogue")
        if not (
            _same_tensor_layout(gate.results[0], up.results[0], graph)
            and _same_tensor_layout(activation.results[0], multiply.results[0], graph)
        ):
            rejections.append("numerical or ownership contract fails")
        if external_outputs:
            rejections.append("secondary outputs require multi-store epilogue support")
        rejections.append("paired GEMM provider is not implemented")

        output_value = graph.values[multiply.results[0]]
        output_elements = prod(output_value.shape or ())
        element_bytes = _DTYPE_BYTES.get(output_value.dtype or "", 0)
        separate_output_bytes = 2 * output_elements * element_bytes
        paired_output_bytes = 0 if not external_outputs else separate_output_bytes
        packing_cost_bytes = output_elements * element_bytes
        conversion_cost_bytes = (
            0
            if _same_strides(gate.results[0], up.results[0], graph)
            else separate_output_bytes
        )
        scalar_saved_bytes = sum(
            _value_bytes(operation.results[0], graph) * 2
            for operation in activation_operations
            if operation.results[0] not in external_outputs
        )
        return RegionCandidate(
            id=f"region:{multiply.id.removeprefix('op:')}:gated_mlp",
            source_op_ids=source_op_ids,
            external_inputs=_external_inputs(operations, graph),
            live_outputs=tuple(
                dict.fromkeys((multiply.results[0], *external_outputs))
            ),
            tensor_program=None,
            semantic_facts={
                "family": "gated_mlp",
                "fusion": "paired_gemm_epilogue",
                "gate_contraction": gate.id,
                "up_contraction": up.id,
                "activation": activation.id,
                "combine": multiply.id,
                "secondary_outputs": external_outputs,
                "effects": tuple(
                    (operation.id, operation.effects) for operation in operations
                ),
                "separate_work_bytes": separate_output_bytes + scalar_saved_bytes,
                "paired_work_bytes": paired_output_bytes + packing_cost_bytes + conversion_cost_bytes,
                "packing_cost_bytes": packing_cost_bytes,
                "conversion_cost_bytes": conversion_cost_bytes,
                "scalar_fusion_saved_bytes": scalar_saved_bytes,
                "source_casts": "preserved",
            },
            cost_bytes=paired_output_bytes + packing_cost_bytes + conversion_cost_bytes,
            rejections=tuple(rejections),
        )
    return None


def _load_id(expression: ScalarExpression) -> str | None:
    return expression.load_id if isinstance(expression, Load) else None


def _decomposed_silu_operations(tail: Op, graph: ComputeGraph) -> tuple[Op, ...] | None:
    expected = (
        ("convert_element_type", "elementwise"),
        ("neg", "elementwise"),
        ("exp", "elementwise"),
        ("add", "elementwise"),
        ("div", "elementwise"),
        ("convert_element_type", "elementwise"),
    )
    chain = [tail]
    current = tail
    if not _matches_operation(current, *expected[-1]):
        return None
    for operation_name, family in reversed(expected[:-1]):
        if not current.operands:
            return None
        operand_index = 1 if current.id.removeprefix("op:").startswith("div") else 0
        producer_id = graph.values[current.operands[operand_index]].producer
        if producer_id is None:
            return None
        current = graph.ops[producer_id]
        if not _matches_operation(current, operation_name, family):
            return None
        chain.insert(0, current)
    cast_to_float, negation, exponential, increment, division, cast_to_source = chain
    if (
        division.operands != (cast_to_float.results[0], increment.results[0])
        or negation.operands != (cast_to_float.results[0],)
        or exponential.operands != (negation.results[0],)
        or increment.operands != (exponential.results[0],)
        or cast_to_source.operands != (division.results[0],)
    ):
        return None
    return tuple(chain)


def _matches_operation(operation: Op, name_prefix: str, family: str) -> bool:
    return (
        operation.id.removeprefix("op:").startswith(name_prefix)
        and isinstance(operation.tensor_program, TensorProgram)
        and operation.tensor_program.family == family
    )


def _has_illegal_effects(operation: Op) -> bool:
    return bool(
        operation.effects.get("unknown")
        or operation.effects.get("mutates_inputs")
    )


def _is_contraction(operation: Op) -> bool:
    return isinstance(operation.tensor_program, TensorProgram) and operation.tensor_program.family == "contraction"


def _external_outputs(operations: tuple[Op, ...], graph: ComputeGraph) -> tuple[str, ...]:
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
            if any(user not in operation_ids for user in graph.values[result_id].users):
                outputs.append(result_id)
    return tuple(outputs)


def _same_tensor_layout(left_id: str, right_id: str, graph: ComputeGraph) -> bool:
    left = graph.values[left_id]
    right = graph.values[right_id]
    return left.shape == right.shape and left.dtype == right.dtype and left.strides == right.strides


def _same_chain_layout(left_id: str, right_id: str, graph: ComputeGraph) -> bool:
    left = graph.values[left_id]
    right = graph.values[right_id]
    return left.shape == right.shape and left.strides == right.strides


def _same_strides(left_id: str, right_id: str, graph: ComputeGraph) -> bool:
    return graph.values[left_id].strides == graph.values[right_id].strides


def _value_bytes(value_id: str, graph: ComputeGraph) -> int:
    value = graph.values[value_id]
    return prod(value.shape or ()) * _DTYPE_BYTES.get(value.dtype or "", 0)


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
    candidate_ids = [
        candidate.id
        for region in graph.regions
        for candidate in region.fusion_candidates
    ]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Fusion candidate IDs are not unique")
    if set(candidate_ids).intersection(region_ids):
        raise ValueError("Fusion candidate IDs collide with region IDs")
    for region in graph.regions:
        for candidate in region.fusion_candidates:
            if not set(candidate.source_op_ids).issubset(graph.ops):
                raise ValueError(f"Candidate {candidate.id} references an unknown operation")
            if any(
                operation_id not in region.source_op_ids
                for operation_id in candidate.source_op_ids
            ):
                if candidate.source_op_ids[-1] not in region.source_op_ids:
                    raise ValueError(f"Candidate {candidate.id} is not attached to its tail region")
            for value_id in (*candidate.external_inputs, *candidate.live_outputs):
                if value_id not in graph.values:
                    raise ValueError(f"Candidate {candidate.id} references an unknown value")
            if any(
                graph.values[value_id].producer not in candidate.source_op_ids
                for value_id in candidate.live_outputs
            ):
                raise ValueError(f"Candidate {candidate.id} outputs a foreign value")
            if any(
                graph.values[value_id].producer in candidate.source_op_ids
                for value_id in candidate.external_inputs
            ):
                raise ValueError(f"Candidate {candidate.id} marks an internal value external")
            if candidate.tensor_program is not None and candidate.tensor_program.store.value_id not in candidate.live_outputs:
                raise ValueError(f"Candidate {candidate.id} does not output its program result")
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
