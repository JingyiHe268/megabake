from __future__ import annotations

from dataclasses import replace
from collections.abc import Callable, Mapping
from typing import Any
from pathlib import Path
from math import prod
import operator

from .graph import ComputeGraph, Op, Region, RegionCandidate, verify_compute_graph
from .lowering import (
    attention_access_axes,
    attention_tensor_program,
    AxisAffine,
    AxisPredicate,
    AxisTerm,
    Binary,
    Constant,
    IndexedLoad,
    IterationDomain,
    Accumulator,
    AxisIndex,
    Cast,
    Load,
    ScalarExpression,
    TensorProgram,
    Unary,
    TensorStore,
    Where,
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
        stateful = _stateful_attention_candidate(operation, graph)
        if stateful is not None:
            candidates.setdefault(operation.id, []).append(stateful)
        declared = _declared_attention_candidate(operation, graph)
        if declared is not None:
            candidates.setdefault(operation.id, []).append(declared)
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
    for candidate in _rms_norm_candidates(graph):
        tail_operation = candidate.source_op_ids[-1]
        candidates.setdefault(tail_operation, []).append(candidate)
    for candidate in _rope_candidates(graph):
        tail_operation = candidate.source_op_ids[-1]
        candidates.setdefault(tail_operation, []).append(candidate)
    for candidate in _attention_candidates(graph):
        tail_operation = candidate.source_op_ids[-1]
        candidates.setdefault(tail_operation, []).append(candidate)
    return {tail: tuple(values) for tail, values in candidates.items()}


def _declared_attention_candidate(
    operation: Op, graph: ComputeGraph
) -> RegionCandidate | None:
    """``megabake::attention``: the model declares its attention, so the
    region needs no pattern matching and no layout inference."""
    if not _matches_target(operation, "megabake.attention.default"):
        return None
    program = operation.tensor_program
    if not isinstance(program, TensorProgram) or program.family != "attention":
        return None
    loads = {load.id: load for load in program.loads}
    query, key = loads["attention:q"], loads["attention:k"]
    batch, _, query_heads, head_dim = query.source_shape
    kv_heads = key.source_shape[2]
    output_id = operation.results[0]
    return RegionCandidate(
        id=f"region:{operation.id.removeprefix('op:')}:attention",
        source_op_ids=(operation.id,),
        external_inputs=operation.operands,
        live_outputs=(output_id,),
        tensor_program=program,
        semantic_facts={
            "family": "attention",
            "algorithm": "stable_softmax",
            # Hugging Face eager numerics, which the reference implements.
            "numerical_policy": {
                "score_dtype": "torch.bfloat16",
                "statistics_dtype": "torch.float32",
                "probability_dtype": "torch.bfloat16",
                "accumulator_dtype": "torch.float32",
            },
            "q_access": query,
            "k_access": key,
            "v_access": loads["attention:v"],
            "head_grouping": {
                "batch": batch,
                "query_heads": query_heads,
                "kv_heads": kv_heads,
                "group_size": query_heads // kv_heads,
                "head_dim": head_dim,
            },
            "scale": _scalar_operand(operation, 4),
            "mask": (
                {"kind": "additive_tensor", "block_skip_proof": None}
                if "attention:mask" in loads
                else {"kind": "none"}
            ),
            "score_expression": 0.0,
            "requested_outputs": (output_id,),
            "boundary_output": output_id if output_id in graph.outputs else None,
            "cache_effects": {
                "reads": (),
                "appends": (),
                "updates": (),
                "policy": "read_only",
            },
            "hook_restrictions": {
                "score": "scaled dot product",
                "mask": "additive tensor",
            },
        },
        cost_bytes=0,
    )


def _stateful_attention_candidate(
    operation: Op, graph: ComputeGraph
) -> RegionCandidate | None:
    if not _matches_target(operation, "megabake.kv_attention.default"):
        return None
    program = operation.tensor_program
    if (
        not isinstance(program, TensorProgram)
        or program.family != "attention"
        or not program.state_accesses
    ):
        return None
    state = program.state_accesses[0]
    loads = {load.id: load for load in program.loads}
    output_id = operation.results[0]
    return RegionCandidate(
        id=f"region:{operation.id.removeprefix('op:')}:kv_attention",
        source_op_ids=(operation.id,),
        external_inputs=operation.operands,
        live_outputs=(output_id,),
        tensor_program=program,
        semantic_facts={
            "family": "attention",
            "algorithm": "stable_softmax",
            "numerical_policy": {
                "score_dtype": "torch.bfloat16",
                "statistics_dtype": "torch.float32",
                "probability_dtype": "torch.bfloat16",
                "accumulator_dtype": "torch.float32",
            },
            "q_access": loads["attention:q"],
            "k_access": loads["attention:k"],
            "v_access": loads["attention:v"],
            "head_grouping": {
                "batch": loads["attention:q"].source_shape[0],
                "query_heads": loads["attention:q"].source_shape[2],
                "kv_heads": loads["attention:k"].source_shape[2],
                "group_size": (
                    loads["attention:q"].source_shape[2]
                    // loads["attention:k"].source_shape[2]
                ),
                "head_dim": loads["attention:q"].source_shape[3],
            },
            "scale": loads["attention:q"].source_shape[3] ** -0.5,
            "mask": {
                "kind": "stateful_causal",
                "causal_offset": 0,
                "block_skip_proof": "runtime valid-length bound",
            },
            "score_expression": 0.0,
            "requested_outputs": (output_id,),
            "boundary_output": output_id if output_id in graph.outputs else None,
            "cache_effects": {
                "reads": (state.key_cache_value_id, state.value_cache_value_id),
                "appends": (
                    loads["attention:new_k"].value_id,
                    loads["attention:new_v"].value_id,
                ),
                "updates": (
                    state.key_cache_value_id,
                    state.value_cache_value_id,
                    state.valid_length_value_id,
                ),
                "policy": "stateful",
            },
            "kv_state": state,
            "hook_restrictions": {
                "score": "row-local dot product with source-order K/V selection",
                "mask": "causal selection by runtime valid length",
            },
        },
        cost_bytes=(
            2
            * loads["attention:q"].source_shape[0]
            * state.query_length
            * loads["attention:k"].source_shape[2]
            * loads["attention:q"].source_shape[3]
            * 2
        ),
    )


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


def _rms_norm_candidates(graph: ComputeGraph) -> tuple[RegionCandidate, ...]:
    candidates = []
    for operation in graph.ops.values():
        program = operation.tensor_program
        if (
            not isinstance(program, TensorProgram)
            or program.family != "reduction"
            or program.source_target != "aten.mean.dim"
        ):
            continue
        candidate = _rms_norm_candidate(operation, graph)
        if candidate is not None:
            candidates.append(candidate)
    return tuple(candidates)


def _rms_norm_candidate(mean: Op, graph: ComputeGraph) -> RegionCandidate | None:
    chain = _rms_norm_chain(mean, graph)
    if chain is None:
        return None
    operations, x_id, weight_id, epsilon = chain
    tail = operations[-1]
    tail_value = graph.values[tail.results[0]]
    if tail_value.shape is None or tail_value.dtype is None or tail_value.strides is None:
        return None
    x_value = graph.values[x_id]
    if x_value.shape is None or x_value.dtype is None or x_value.strides is None:
        return None
    if len(x_value.shape) < 2 or x_value.shape[-1] <= 0:
        return None

    output_domains = tuple(
        IterationDomain(f"o{index}", size)
        for index, size in enumerate(tail_value.shape)
    )
    reduction_domain = IterationDomain("r0", x_value.shape[-1])
    output_axes = tuple(domain.axis for domain in output_domains)
    leaf_loads: list[IndexedLoad] = []

    def leaf_load(value_id: str, axes: tuple[object, ...]) -> Load:
        value = graph.values[value_id]
        load = IndexedLoad(
            id=f"load:{len(leaf_loads)}",
            value_id=value_id,
            axes=axes,
            strides=value.strides or (),
            dtype=value.dtype or "",
            source_shape=value.shape,
        )
        leaf_loads.append(load)
        return Load(load.id, load.dtype)

    x_reduction_load = leaf_load(
        x_id, (*output_axes[:-1], reduction_domain.axis)
    )
    x_post_load = leaf_load(x_id, output_axes)
    weight_reduction_load = (
        leaf_load(weight_id, (reduction_domain.axis,)) if weight_id is not None else None
    )
    weight_post_load = (
        leaf_load(weight_id, output_axes[-1:]) if weight_id is not None else None
    )
    residual_id = None
    if (
        tail.tensor_program is not None
        and isinstance(tail.tensor_program.combine, Binary)
        and tail.tensor_program.combine.operator == "add"
    ):
        for operand_id in tail.operands:
            if operand_id != operations[-2].results[0]:
                residual_id = operand_id
    residual_load = (
        leaf_load(residual_id, output_axes)
        if residual_id is not None
        else None
    )

    expressions: dict[str, ScalarExpression] = {}
    post_expressions: dict[str, ScalarExpression] = {}
    x_expression_ids = {x_id}
    if (
        operations[0].attributes.get("target") == "prims.convert_element_type.default"
        and operations[0].operands[0] == x_id
    ):
        x_expression_ids.add(operations[0].results[0])
    after_mean = False
    for operation in operations:
        program = operation.tensor_program
        if not isinstance(program, TensorProgram):
            return None
        if operation is mean:
            input_expression = expressions.get(program.loads[0].value_id)
            if input_expression is None:
                return None
            combine = _substitute_loads(
                program.combine,
                {program.loads[0].id: input_expression},
            )
            expressions[operation.results[0]] = program.post
            post_expressions[operation.results[0]] = program.post
            after_mean = True
            continue
        target_expressions = post_expressions if after_mean else expressions
        replacements: dict[str, ScalarExpression] = {}
        for load in program.loads:
            expression = target_expressions.get(load.value_id)
            if expression is None:
                if load.value_id in x_expression_ids:
                    expression = x_post_load if after_mean else x_reduction_load
                elif weight_reduction_load is not None and load.value_id == weight_id:
                    expression = (
                        weight_post_load if after_mean else weight_reduction_load
                    )
                elif residual_load is not None and load.value_id == residual_id:
                    expression = residual_load
                else:
                    return None
            replacements[load.id] = expression
        target_expressions[operation.results[0]] = _substitute_loads(
            program.combine, replacements
        )

    program = TensorProgram(
        family="reduction",
        source_target="rms_norm",
        decomposition="row sum of squares with generated pre- and post-reduction expressions",
        iteration_domains=(*output_domains, reduction_domain),
        reduction_axes=(reduction_domain.axis,),
        loads=tuple(leaf_loads),
        init=mean.tensor_program.init if isinstance(mean.tensor_program, TensorProgram) else None,
        combine=combine,
        post=post_expressions[tail.results[0]],
        accumulator_dtype=mean.tensor_program.accumulator_dtype
        if isinstance(mean.tensor_program, TensorProgram)
        else None,
        store=TensorStore(
            tail_value.id,
            tail_value.shape,
            tail_value.dtype,
            tail_value.strides,
            True,
        ),
        predicates=tuple(
            AxisPredicate(domain.axis, domain.size)
            for domain in (*output_domains, reduction_domain)
        ),
    )
    operation_ids = {operation.id for operation in operations}
    rejections: list[str] = []
    if any(_has_illegal_effects(operation) for operation in operations):
        rejections.append("source effects do not fit a fused norm body")
    if any(
        any(user not in operation_ids for user in graph.values[operation.results[0]].users)
        for operation in operations[:-1]
    ):
        rejections.append("a norm intermediate value has an external use")

    source_casts = tuple(
        f"{graph.values[operation.operands[0]].dtype}->{graph.values[operation.results[0]].dtype}"
        for operation in operations
        if operation.attributes.get("target") == "prims.convert_element_type.default"
    )
    return RegionCandidate(
        id=f"region:{tail.id.removeprefix('op:')}:rms_norm",
        source_op_ids=tuple(operation.id for operation in operations),
        external_inputs=_external_inputs(tuple(operations), graph),
        live_outputs=_live_outputs(tuple(operations), graph),
        tensor_program=program,
        semantic_facts={
            "family": "rms_norm",
            "reduction_axis": len(x_value.shape) - 1,
            "width": x_value.shape[-1],
            "epsilon": epsilon,
            "affine": weight_id,
            "residual": residual_id,
            "source_casts": source_casts,
        },
        cost_bytes=sum(
            _value_bytes(operation.results[0], graph)
            for operation in operations[:-1]
        ),
        rejections=tuple(rejections),
    )


def _rms_norm_chain(
    mean: Op, graph: ComputeGraph
) -> tuple[tuple[Op, ...], str, str | None, float] | None:
    mean_program = mean.tensor_program
    if not isinstance(mean_program, TensorProgram):
        return None
    if len(mean_program.reduction_axes) != 1:
        return None
    pow_operation = _producer(mean.operands[0], graph)
    if pow_operation is None or not _matches_target(
        pow_operation, "aten.pow.Tensor_Scalar"
    ):
        return None
    cast_operation = _producer(pow_operation.operands[0], graph)
    if cast_operation is not None and _matches_target(
        cast_operation, "prims.convert_element_type.default"
    ):
        x_id = cast_operation.operands[0]
    else:
        x_id = pow_operation.operands[0]
        cast_operation = None

    add = _sole_user(mean, graph)
    if add is None or not (
        _matches_target(add, "aten.add.Scalar")
        or _matches_target(add, "aten.add.Tensor")
    ):
        return None
    rsqrt = _sole_user(add, graph)
    if rsqrt is None or not _matches_target(rsqrt, "aten.rsqrt.default"):
        return None
    scale = _sole_user(rsqrt, graph)
    if scale is None or not _matches_target(scale, "aten.mul.Tensor"):
        return None
    scale_x_id = cast_operation.results[0] if cast_operation is not None else x_id
    if scale_x_id not in scale.operands or rsqrt.results[0] not in scale.operands:
        return None

    def output_cast_of(operation: Op) -> Op | None:
        user = _sole_user(operation, graph)
        if (
            user is not None
            and _matches_target(user, "prims.convert_element_type.default")
            and graph.values[user.operands[0]].dtype
            != graph.values[user.results[0]].dtype
        ):
            return user
        return None

    def affine_of(operation: Op) -> tuple[Op, str] | None:
        user = _sole_user(operation, graph)
        if (
            user is None
            or not _matches_target(user, "aten.mul.Tensor")
            or operation.results[0] not in user.operands
        ):
            return None
        weight = next(
            (operand_id for operand_id in user.operands if operand_id != operation.results[0]),
            None,
        )
        return (user, weight) if weight in graph.inputs else None

    # The weight multiply may come before the output cast or, as in Hugging
    # Face's LlamaRMSNorm (`weight * hidden.to(input_dtype)`), after it.
    tail = scale
    affine: Op | None = None
    weight_id: str | None = None
    output_cast = output_cast_of(tail)
    if output_cast is not None:
        tail = output_cast
        matched = affine_of(tail)
        if matched is not None:
            affine, weight_id = matched
            tail = affine
    else:
        matched = affine_of(tail)
        if matched is not None:
            affine, weight_id = matched
            tail = affine
            output_cast = output_cast_of(tail)
            if output_cast is not None:
                tail = output_cast

    residual = _sole_user(tail, graph)
    if not (
        residual is not None
        and _matches_target(residual, "aten.add.Tensor")
        and tail.results[0] in residual.operands
    ):
        residual = None
    else:
        residual_id = next(
            operand_id
            for operand_id in residual.operands
            if operand_id != tail.results[0]
        )
        if (
            residual_id not in graph.inputs
            or graph.values[residual_id].shape != graph.values[tail.results[0]].shape
        ):
            residual = None
        else:
            tail = residual

    operations = [
        operation
        for operation in graph.ops.values()
        if operation.id
        in {
            operation.id
            for operation in (
                cast_operation,
                pow_operation,
                mean,
                add,
                rsqrt,
                scale,
                affine,
                output_cast,
                residual,
            )
            if operation is not None
        }
    ]
    add_program = add.tensor_program
    if (
        not isinstance(add_program, TensorProgram)
        or not isinstance(add_program.combine, Binary)
        or not isinstance(add_program.combine.rhs, Constant)
        or not isinstance(add_program.combine.rhs.value, float)
    ):
        return None
    return tuple(operations), x_id, weight_id, add_program.combine.rhs.value


def _rope_candidates(graph: ComputeGraph) -> tuple[RegionCandidate, ...]:
    candidates = []
    for operation in graph.ops.values():
        program = operation.tensor_program
        if program is None or program.family != "concat":
            continue
        candidate = _rope_candidate(operation, graph) or _rotate_half_rope_candidate(
            operation, graph
        )
        if candidate is not None:
            candidates.append(candidate)
    return tuple(candidates)


def _rotate_half_rope_candidate(concat: Op, graph: ComputeGraph) -> RegionCandidate | None:
    """Match ``x * cos + cat(-x[..., h:], x[..., :h]) * sin``.

    This is Hugging Face's ``apply_rotary_pos_emb`` with ``rotate_half``. The
    multiplies and the add keep their own captured programs; only the concat
    becomes a select on the last-axis index between offset loads of ``x``.
    """
    concat_program = concat.tensor_program
    if not isinstance(concat_program, TensorProgram) or len(concat.operands) != 2:
        return None
    output = graph.values[concat.results[0]]
    if output.shape is None or output.dtype is None or output.shape[-1] % 2:
        return None
    rank = len(output.shape)
    half = output.shape[-1] // 2
    negate = _producer(concat.operands[0], graph)
    if negate is None or not _matches_target(negate, "aten.neg.default"):
        return None
    first = _last_axis_slice(negate.operands[0], graph)
    second = _last_axis_slice(concat.operands[1], graph)
    if first is None or second is None or first[0] != second[0]:
        return None
    x_id = first[0]
    if first[1] != half or second[1] != 0:
        return None
    if any(
        graph.values[value_id].shape is None or graph.values[value_id].shape[-1] != half
        for value_id in (negate.results[0], concat.operands[1])
    ):
        return None
    x = graph.values[x_id]
    if x.shape != output.shape or x.strides is None or x.dtype != output.dtype:
        return None

    rotate_multiply = _sole_user(concat, graph)
    if rotate_multiply is None or not _binary_operator(rotate_multiply, "mul"):
        return None
    add = _sole_user(rotate_multiply, graph)
    if add is None or not _binary_operator(add, "add"):
        return None
    x_multiply = _producer(
        next(
            (operand for operand in add.operands if operand != rotate_multiply.results[0]),
            rotate_multiply.results[0],
        ),
        graph,
    )
    if (
        x_multiply is None
        or x_multiply.id == rotate_multiply.id
        or not _binary_operator(x_multiply, "mul")
        or x_id not in x_multiply.operands
    ):
        return None
    operations = (negate, concat, x_multiply, rotate_multiply, add)
    programs = [operation.tensor_program for operation in operations]
    if any(not isinstance(program, TensorProgram) for program in programs):
        return None
    if any(
        len(program.iteration_domains) != rank
        for program in (x_multiply.tensor_program, rotate_multiply.tensor_program, add.tensor_program)
    ):
        return None

    output_axes = tuple(domain.axis for domain in add.tensor_program.iteration_domains)
    final_axis = output_axes[-1]
    # The captured load of x gives its axis map (unit axes are literal 0).
    x_axes = next(
        (tuple(load.axes) for load in x_multiply.tensor_program.loads if load.value_id == x_id),
        None,
    )
    if x_axes is None or x_axes[-1] != final_axis:
        return None
    leaf_loads: list[IndexedLoad] = []
    leaf_ids: dict[tuple[str, tuple[Any, ...]], str] = {}

    def leaf(value_id: str, axes: tuple[Any, ...], strides: tuple[int, ...], dtype: str, shape) -> Load:
        key = (value_id, axes)
        if key not in leaf_ids:
            leaf_ids[key] = f"load:{len(leaf_loads)}"
            leaf_loads.append(IndexedLoad(leaf_ids[key], value_id, axes, strides, dtype, shape))
        return Load(leaf_ids[key], dtype)

    def x_at(offset: int) -> Load:
        axis = final_axis if offset == 0 else AxisAffine((AxisTerm(final_axis),), offset=offset)
        return leaf(x_id, (*x_axes[:-1], axis), x.strides, x.dtype, x.shape)

    def compose(operation: Op, known: Mapping[str, ScalarExpression]) -> ScalarExpression | None:
        program = operation.tensor_program
        replacements: dict[str, ScalarExpression] = {}
        for load in program.loads:
            if load.value_id in known:
                replacements[load.id] = known[load.value_id]
            elif load.value_id == x_id and tuple(load.axes) == x_axes:
                replacements[load.id] = x_at(0)
            elif load.value_id != x_id and all(
                axis in output_axes or isinstance(axis, int) for axis in load.axes
            ):
                replacements[load.id] = leaf(
                    load.value_id, tuple(load.axes), tuple(load.strides), load.dtype, load.source_shape
                )
            else:
                return None
        return _substitute_loads(program.combine, replacements)

    negated = compose(negate, {negate.operands[0]: x_at(half)})
    if negated is None:
        return None
    rotated = Where(
        Binary(
            "lt",
            AxisIndex(final_axis, "torch.int64"),
            Constant(half, "torch.int64"),
            "torch.bool",
        ),
        negated,
        x_at(-half),
        output.dtype,
    )
    rotated_product = compose(rotate_multiply, {concat.results[0]: rotated})
    x_product = compose(x_multiply, {})
    if rotated_product is None or x_product is None:
        return None
    combined = compose(
        add,
        {
            rotate_multiply.results[0]: rotated_product,
            x_multiply.results[0]: x_product,
        },
    )
    if combined is None:
        return None
    fused_program = replace(
        add.tensor_program,
        source_target="rope",
        decomposition="rotate_half rotation: offset loads of x selected by the last-axis index",
        loads=tuple(leaf_loads),
        combine=combined,
    )
    operation_ids = {operation.id for operation in operations}
    rejections: list[str] = []
    if any(_has_illegal_effects(operation) for operation in operations):
        rejections.append("source effects do not fit a fused RoPE body")
    if any(
        any(user not in operation_ids for user in graph.values[operation.results[0]].users)
        for operation in operations[:-1]
    ):
        rejections.append("a RoPE intermediate value has an external use")
    ordered = tuple(operation for operation in graph.ops.values() if operation.id in operation_ids)
    return RegionCandidate(
        id=f"region:{add.id.removeprefix('op:')}:rope",
        source_op_ids=tuple(operation.id for operation in ordered),
        external_inputs=_external_inputs(ordered, graph),
        live_outputs=_live_outputs(ordered, graph),
        tensor_program=fused_program,
        semantic_facts={
            "family": "rope",
            "axis": rank - 1,
            "width": output.shape[-1],
            "rotary_width": output.shape[-1],
            "rotation": "rotate_half: i <-> i + width / 2 with the first half negated",
            "pass_through": 0,
            "source_casts": (),
        },
        cost_bytes=sum(
            _value_bytes(operation.results[0], graph)
            for operation in ordered[:-1]
            if isinstance(operation.tensor_program, TensorProgram)
            and operation.tensor_program.store.owns_storage
        ),
        rejections=tuple(rejections),
    )


def _rope_candidate(concat: Op, graph: ComputeGraph) -> RegionCandidate | None:
    program = concat.tensor_program
    if program is None or program.family != "concat" or len(concat.operands) != 2:
        return None
    if not isinstance(program, TensorProgram):
        return None
    concat_axis = next(
        (
            index
            for index, axis in enumerate(program.loads[0].axes)
            if isinstance(axis, AxisAffine)
        ),
        None,
    )
    if concat_axis != len(graph.values[concat.results[0]].shape or ()) - 1:
        return None
    sub = _producer(concat.operands[0], graph)
    add = _producer(concat.operands[1], graph)
    if sub is None or add is None:
        return None
    if not (
        _binary_operator(sub, "sub")
        and _binary_operator(add, "add")
        and _binary_operator(_producer(sub.operands[0], graph), "mul")
        and _binary_operator(_producer(sub.operands[1], graph), "mul")
        and _binary_operator(_producer(add.operands[0], graph), "mul")
        and _binary_operator(_producer(add.operands[1], graph), "mul")
    ):
        return None

    roots: dict[str, int] = {}
    multiplies = (
        _producer(sub.operands[0], graph),
        _producer(sub.operands[1], graph),
        _producer(add.operands[0], graph),
        _producer(add.operands[1], graph),
    )
    for multiply in multiplies:
        if multiply is None or len(multiply.operands) != 2:
            return None
        for operand_id in multiply.operands:
            root = _view_root(operand_id, graph)
            if root is None:
                return None
            roots[root] = roots.get(root, 0) + 1
    if len(roots) != 3 or sorted(roots.values()) != [2, 2, 4]:
        return None
    x_id = next(value_id for value_id, count in roots.items() if count == 4)
    trig_ids = tuple(value_id for value_id, count in roots.items() if count == 2)
    cos_id, sin_id = trig_ids
    if any(
        (
            producer := _producer(root_id, graph)
        ) is not None
        and _matches_target(producer, "prims.convert_element_type.default")
        for root_id in roots
    ):
        return None
    operations = _rope_operations(concat, (x_id, cos_id, sin_id), graph)
    if any("convert_element_type" in str(op.attributes.get("target")) for op in operations):
        return None

    output = graph.values[concat.results[0]]
    x = graph.values[x_id]
    cos = graph.values[cos_id]
    sin = graph.values[sin_id]
    if (
        output.shape is None
        or output.strides is None
        or x.shape != output.shape
        or cos.shape is None
        or sin.shape is None
        or len(cos.shape) != len(sin.shape)
        or cos.shape != sin.shape
        or cos.shape[-1] % 2
        or output.shape[-1] < cos.shape[-1]
        or output.dtype not in {"torch.float32", "torch.bfloat16"}
        or not {x.dtype, cos.dtype, sin.dtype, output.dtype} == {output.dtype}
        or any(stride < 0 for stride in (*x.strides, *cos.strides, *sin.strides, *output.strides))
    ):
        return None
    rank = len(output.shape)
    rotary_width = cos.shape[-1]
    half_width = rotary_width // 2
    if any(
        cos.shape[index] != output.shape[index]
        for index in range(len(cos.shape) - 1)
    ):
        return None

    output_axes = tuple(f"o{index}" for index in range(rank))
    final_axis = output_axes[-1]
    leading_axes = output_axes[:-1]
    # Each half reads x and the trig tables at its own source slice; derive
    # those last-axis starts instead of assuming one indexing convention.
    first_width = graph.values[concat.operands[0]].shape
    if first_width is None or first_width[-1] != half_width:
        return None
    load_ids: dict[tuple[str, int], str] = {}
    loads_list: list[IndexedLoad] = []

    def operand_load(operand_id: str, branch_start: int) -> Load | None:
        located = _last_axis_slice(operand_id, graph)
        if located is None:
            return None
        root_id, start = located
        root = graph.values[root_id]
        if root.shape is None or root.strides is None or len(root.shape) != rank:
            return None
        offset = start - branch_start
        key = (root_id, offset)
        if key not in load_ids:
            load_ids[key] = f"load:{len(loads_list)}"
            axis = (
                final_axis
                if offset == 0
                else AxisAffine((AxisTerm(final_axis),), offset=offset)
            )
            loads_list.append(
                IndexedLoad(
                    load_ids[key],
                    root_id,
                    (*leading_axes, axis),
                    root.strides,
                    root.dtype or "",
                    root.shape,
                )
            )
        return Load(load_ids[key], root.dtype or "")

    branch_loads = []
    for multiply, branch_start in zip(multiplies, (0, 0, half_width, half_width)):
        operands = tuple(
            operand_load(operand_id, branch_start) for operand_id in multiply.operands
        )
        if any(operand is None for operand in operands):
            return None
        branch_loads.append(operands)
    passthrough = operand_load(x_id, 0)
    if passthrough is None:
        return None
    loads = tuple(loads_list)
    dtype = output.dtype or ""
    compute_dtype = "torch.float32" if dtype in {"torch.bfloat16", "torch.float16"} else dtype

    def source_binary(
        operator: str, lhs: ScalarExpression, rhs: ScalarExpression
    ) -> ScalarExpression:
        if compute_dtype == dtype:
            return Binary(operator, lhs, rhs, dtype)
        lhs = Cast(lhs, compute_dtype, "round_nearest_even")
        rhs = Cast(rhs, compute_dtype, "round_nearest_even")
        return Cast(
            Binary(operator, lhs, rhs, compute_dtype),
            dtype,
            "round_nearest_even",
        )

    axis_index = AxisIndex(final_axis, "torch.int64")
    products = [source_binary("mul", *operands) for operands in branch_loads]
    first = source_binary("sub", products[0], products[1])
    second = source_binary("add", products[2], products[3])
    expression = Where(
        Binary(
            "lt",
            axis_index,
            Constant(half_width, "torch.int64"),
            "torch.bool",
        ),
        first,
        Where(
            Binary(
                "lt",
                axis_index,
                Constant(rotary_width, "torch.int64"),
                "torch.bool",
            ),
            second,
            passthrough,
            dtype,
        ),
        dtype,
    )
    fused_program = TensorProgram(
        family="elementwise",
        source_target="rope",
        decomposition="paired half-width rotation with an optional pass-through tail",
        iteration_domains=tuple(
            IterationDomain(axis, size) for axis, size in zip(output_axes, output.shape)
        ),
        reduction_axes=(),
        loads=loads,
        init=None,
        combine=expression,
        post=None,
        accumulator_dtype=None,
        store=TensorStore(output.id, output.shape, output.dtype or "", output.strides, True),
        predicates=tuple(
            AxisPredicate(axis, size) for axis, size in zip(output_axes, output.shape)
        ),
    )
    operation_ids = {operation.id for operation in operations}
    rejections: list[str] = []
    if any(_has_illegal_effects(operation) for operation in operations):
        rejections.append("source effects do not fit a fused RoPE body")
    if any(
        any(user not in operation_ids for user in graph.values[operation.results[0]].users)
        for operation in operations[:-1]
    ):
        rejections.append("a RoPE intermediate value has an external use")
    return RegionCandidate(
        id=f"region:{concat.id.removeprefix('op:')}:rope",
        source_op_ids=tuple(operation.id for operation in operations),
        external_inputs=_external_inputs(tuple(operations), graph),
        live_outputs=_live_outputs(tuple(operations), graph),
        tensor_program=fused_program,
        semantic_facts={
            "family": "rope",
            "axis": rank - 1,
            "width": output.shape[-1],
            "rotary_width": rotary_width,
            "rotation": "i <-> i + rotary_width / 2",
            "pass_through": output.shape[-1] - rotary_width,
            "source_casts": (),
        },
        cost_bytes=sum(
            _value_bytes(operation.results[0], graph)
            for operation in operations[:-1]
            if isinstance(operation.tensor_program, TensorProgram)
            and operation.tensor_program.store.owns_storage
        ),
        rejections=tuple(rejections),
    )


def _producer(value_id: str, graph: ComputeGraph) -> Op | None:
    operation_id = graph.values[value_id].producer
    return graph.ops.get(operation_id) if operation_id is not None else None


def _attention_head_axes(value: Any) -> tuple[int, ...]:
    """Axes that can hold the heads: their stride steps over one head dim.

    A size-1 axis has no meaningful stride, so with one KV head (MQA) or one
    token both axes can qualify; the caller resolves that with the score shape.
    """
    if value.strides is None or len(value.strides) != 4:
        return ()
    head_stride = value.shape[3] * value.strides[3]
    return tuple(
        axis
        for axis in (2, 1)
        if value.strides[axis] == head_stride or value.shape[axis] == 1
    )


def _attention_axes(
    q: Any, k: Any, v: Any, scores: tuple[int, ...] | None
) -> tuple[int, int, int] | None:
    """Head axes of q, k and v that agree with each other and with the
    ``[batch * query_heads, query_length, kv_length]`` score matrix."""
    for q_axis in _attention_head_axes(q):
        query_heads, query_length = q.shape[q_axis], q.shape[3 - q_axis]
        for k_axis in _attention_head_axes(k):
            kv_heads, kv_length = k.shape[k_axis], k.shape[3 - k_axis]
            for v_axis in _attention_head_axes(v):
                if (
                    query_heads % kv_heads
                    or v.shape[v_axis] != kv_heads
                    or v.shape[3 - v_axis] != kv_length
                ):
                    continue
                if scores is not None and tuple(scores) != (
                    q.shape[0] * query_heads,
                    query_length,
                    kv_length,
                ):
                    continue
                return q_axis, k_axis, v_axis
    return None


def _attention_candidates(graph: ComputeGraph) -> tuple[RegionCandidate, ...]:
    candidates = []
    for operation in graph.ops.values():
        if _matches_target(operation, "aten.bmm.default"):
            candidate = _attention_candidate(operation, graph)
            if candidate is not None:
                candidates.append(candidate)
    return tuple(candidates)


def _attention_candidate(value_bmm: Op, graph: ComputeGraph) -> RegionCandidate | None:
    value_program = value_bmm.tensor_program
    if not _is_contraction(value_bmm) or not isinstance(value_program, TensorProgram):
        return None
    if len(value_program.loads) != 2:
        return None

    probability_root = _producer_until(
        value_program.loads[0].value_id,
        lambda operation: _matches_target(operation, "aten.div.Tensor"),
        graph,
    )
    if probability_root is None:
        return None
    division = probability_root
    softmax = _softmax_operations(division, graph)
    if softmax is None:
        return None
    algorithm, score_value, softmax_operations = softmax
    score_chain = _score_chain(score_value, graph)
    if score_chain is None:
        return None
    score_add, scale_operation, mask_value, score_bmm, score_path = score_chain
    score_program = score_bmm.tensor_program
    if (
        not _is_contraction(score_bmm)
        or not isinstance(score_program, TensorProgram)
        or len(score_program.loads) != 2
    ):
        return None
    softmax_input = (
        score_add.results[0] if score_add is not None else scale_operation.results[0]
    )
    softmax_input_cast = _producer(score_value, graph)
    if (
        softmax_input_cast is not None
        and _matches_target(softmax_input_cast, "prims.convert_element_type.default")
        and softmax_input_cast.operands == (softmax_input,)
    ):
        softmax_input_operations = (softmax_input_cast,)
    else:
        softmax_input_operations = ()

    q_root, q_path = _value_root_path(score_program.loads[0].value_id, graph)
    k_root, k_path = _value_root_path(score_program.loads[1].value_id, graph)
    v_root, v_path = _value_root_path(value_program.loads[1].value_id, graph)
    q = graph.values[q_root]
    k = graph.values[k_root]
    v = graph.values[v_root]
    if any(value.shape is None or len(value.shape) != 4 for value in (q, k, v)):
        return None
    scores = graph.values[score_bmm.results[0]].shape
    axes = _attention_axes(q, k, v, scores)
    if axes is None:
        return None
    q_head_axis, k_head_axis, v_head_axis = axes
    query_heads = q.shape[q_head_axis]
    query_length = q.shape[3 - q_head_axis]
    kv_heads = k.shape[k_head_axis]
    kv_length = k.shape[3 - k_head_axis]
    head_dim = q.shape[-1]
    if (
        query_heads % kv_heads
        or head_dim != k.shape[-1]
        or head_dim != v.shape[-1]
        or v.shape[v_head_axis] != kv_heads
    ):
        return None

    group_size = query_heads // kv_heads
    batch_head = AxisAffine((AxisTerm("bh", divisor=query_heads),))
    query_head = AxisAffine((AxisTerm("bh", modulus=query_heads),))
    kv_head = AxisAffine(
        (AxisTerm("bh", divisor=query_heads, modulus=group_size),)
    )
    q_load = IndexedLoad(
        "attention:q",
        q_root,
        attention_access_axes(q_head_axis, batch_head, query_head, "s"),
        q.strides or (),
        q.dtype or "",
        q.shape,
    )
    k_load = IndexedLoad(
        "attention:k",
        k_root,
        attention_access_axes(k_head_axis, batch_head, kv_head, "k"),
        k.strides or (),
        k.dtype or "",
        k.shape,
    )
    v_load = IndexedLoad(
        "attention:v",
        v_root,
        attention_access_axes(v_head_axis, batch_head, kv_head, "k"),
        v.strides or (),
        v.dtype or "",
        v.shape,
    )
    score_dtype = graph.values[scale_operation.results[0]].dtype or ""
    score_expression = _score_expression(
        q_load, k_load, scale_operation, mask_value, score_dtype, graph
    )
    if score_expression is None:
        return None
    mask_semantics = _mask_semantics(mask_value, score_dtype, graph)
    condition_loads = mask_semantics.get("condition_loads")
    mask_condition = (
        condition_loads[0].value_id
        if isinstance(condition_loads, tuple) and condition_loads
        else None
    )
    attention_mask = (
        mask_value
        if mask_semantics.get("kind") == "additive_tensor"
        else mask_condition
        if mask_semantics.get("kind") == "typed_where"
        else None
    )

    probability_value = graph.values[value_program.loads[0].value_id]
    probability_path = _producer_path(value_program.loads[0].value_id, division, graph)
    if probability_path is None:
        return None
    boundary_output, output_path = _boundary_output_path(
        value_bmm.results[0], graph
    )
    operations = _ordered_operations(
        (
            *q_path,
            *k_path,
            *v_path,
            score_bmm,
            *score_path,
            scale_operation,
            *(() if score_add is None else (score_add,)),
            *softmax_input_operations,
            *softmax_operations,
            *probability_path,
            value_bmm,
            *output_path,
        ),
        graph,
    )
    attention_live_outputs = _live_outputs(operations, graph)
    if value_bmm.results[0] not in attention_live_outputs:
        attention_live_outputs = (
            value_bmm.results[0],
            *attention_live_outputs,
        )
    rejections: list[str] = []
    if any(_has_illegal_effects(operation) for operation in operations):
        rejections.append("source effects do not fit a composable attention body")
    external_inputs = tuple(
        dict.fromkeys(
            (
                q_root,
                k_root,
                v_root,
                *((attention_mask,) if attention_mask is not None else ()),
            )
        )
    )
    return RegionCandidate(
        id=f"region:{value_bmm.id.removeprefix('op:')}:attention",
        source_op_ids=tuple(operation.id for operation in operations),
        external_inputs=external_inputs,
        live_outputs=attention_live_outputs,
        tensor_program=attention_tensor_program(
            value_bmm.results[0],
            q_root,
            k_root,
            v_root,
            q,
            k,
            v,
            axes,
            attention_mask,
            graph,
        ),
        semantic_facts={
            "family": "attention",
            "algorithm": algorithm,
            "numerical_policy": {
                "score_dtype": score_dtype,
                "statistics_dtype": graph.values[score_value].dtype,
                "probability_dtype": probability_value.dtype,
                "accumulator_dtype": score_program.accumulator_dtype,
            },
            "q_access": q_load,
            "k_access": k_load,
            "v_access": v_load,
            "head_grouping": {
                "batch": q.shape[0],
                "query_heads": query_heads,
                "kv_heads": kv_heads,
                "group_size": group_size,
                "head_dim": head_dim,
            },
            "scale": _scalar_operand(scale_operation, 1),
            "mask": mask_semantics,
            "score_expression": score_expression,
            "requested_outputs": attention_live_outputs,
            "boundary_output": boundary_output,
            "cache_effects": _cache_effects(k_root, v_root, graph),
            "hook_restrictions": {
                "score": "row-local additive bias over query and key coordinates only",
                "mask": "typed pre-softmax selection; block skipping requires a mask proof",
            },
        },
        cost_bytes=sum(
            _value_bytes(operation.results[0], graph)
            for operation in operations[:-1]
            if isinstance(operation.tensor_program, TensorProgram)
            and operation.tensor_program.store.owns_storage
        ),
        rejections=tuple(rejections),
    )


def _producer_until(
    value_id: str, predicate: Callable[[Op], bool], graph: ComputeGraph
) -> Op | None:
    visited: set[str] = set()
    while value_id not in visited:
        visited.add(value_id)
        operation = _producer(value_id, graph)
        if operation is None:
            return None
        if predicate(operation):
            return operation
        program = operation.tensor_program
        if (
            not isinstance(program, TensorProgram)
            or program.family not in {"view", "copy", "elementwise"}
            or len(operation.operands) != 1
        ):
            return None
        value_id = operation.operands[0]
    return None


def _cache_effects(key_root: str, value_root: str, graph: ComputeGraph) -> dict[str, object]:
    key_state, key_append = _cache_access(key_root, graph)
    value_state, value_append = _cache_access(value_root, graph)
    updates = tuple(
        value_id
        for value_id in (key_root, value_root)
        if value_id in graph.outputs
    )
    if updates:
        policy = "functional_append"
    elif _in_place_cache_update((key_state, value_state, key_root, value_root), graph):
        policy = "in_place"
    else:
        policy = "read_only"
    return {
        "reads": (key_state, value_state),
        "appends": tuple(
            value_id
            for value_id in (key_append, value_append)
            if value_id is not None
        ),
        "updates": updates,
        "policy": policy,
    }


def _cache_access(value_id: str, graph: ComputeGraph) -> tuple[str, str | None]:
    operation = _producer(value_id, graph)
    if (
        operation is not None
        and _matches_target(operation, "aten.cat.default")
        and len(operation.operands) == 2
        and _is_sequence_append(operation, graph)
    ):
        return operation.operands[0], operation.operands[1]
    return value_id, None


def _is_sequence_append(operation: Op, graph: ComputeGraph) -> bool:
    # A cache append extends a boundary cache along a non-feature axis; a
    # feature-axis cat (for example rotate-half RoPE) is ordinary computation.
    cache = graph.values.get(operation.operands[0])
    if cache is None or cache.shape is None:
        return False
    arguments = operation.attributes.get("fx_args", ())
    dimension = arguments[1] if len(arguments) > 1 else 0
    if not isinstance(dimension, int):
        return False
    if dimension % len(cache.shape) == len(cache.shape) - 1:
        return False
    seen: set[str] = set()
    pending = [cache.id]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        if current in graph.inputs:
            return True
        value = graph.values.get(current)
        if value is None:
            continue
        pending.extend(value.storage_relation.aliases)
        producer = _producer(current, graph)
        if producer is not None and len(producer.operands) == 1:
            program = producer.tensor_program
            if isinstance(program, TensorProgram) and program.family in {"view", "copy"}:
                pending.append(producer.operands[0])
    return False


def _in_place_cache_update(cache_values: tuple[str, ...], graph: ComputeGraph) -> bool:
    return any(
        operation.effects.get("mutates_inputs")
        and operation.operands
        and operation.operands[0] in cache_values
        for operation in graph.ops.values()
    )


def _softmax_operations(
    division: Op, graph: ComputeGraph
) -> tuple[str, str, tuple[Op, ...]] | None:
    if not _matches_target(division, "aten.div.Tensor") or len(division.operands) != 2:
        return None
    exponential = _producer(division.operands[0], graph)
    subtraction = (
        _producer(exponential.operands[0], graph)
        if exponential is not None
        else None
    )
    if (
        exponential is None
        or subtraction is None
        or not _matches_target(exponential, "aten.exp.default")
        or not _matches_target(subtraction, "aten.sub.Tensor")
        or len(subtraction.operands) != 2
    ):
        return None

    maximum = _producer(subtraction.operands[1], graph)
    if (
        maximum is not None
        and _matches_target(maximum, "aten.amax.default")
        and exponential.operands == (subtraction.results[0],)
        and _producer(division.operands[1], graph) is not None
        and _matches_target(_producer(division.operands[1], graph), "aten.sum.dim_IntList")
        and _producer(_producer(division.operands[1], graph).operands[0], graph)
        is exponential
    ):
        return "stable_softmax", subtraction.operands[0], (
            maximum,
            subtraction,
            exponential,
            _producer(division.operands[1], graph),
            division,
        )

    maximum_projection = _producer(subtraction.operands[1], graph)
    sum_projection = _producer(division.operands[1], graph)
    if (
        maximum_projection is None
        or sum_projection is None
        or operation_target(maximum_projection) != str(operator.getitem)
        or operation_target(sum_projection) != str(operator.getitem)
    ):
        return None
    statistics = _producer(maximum_projection.operands[0], graph)
    if (
        statistics is None
        or not _matches_target(statistics, "prims.prepare_softmax_online.default")
        or maximum_projection.operands[0] != statistics.results[0]
        or maximum_projection.attributes.get("fx_args", (None, None))[1] != 0
        or sum_projection.attributes.get("fx_args", (None, None))[1] != 1
        or statistics.results[1] not in sum_projection.operands
        or statistics.operands != (subtraction.operands[0],)
        or exponential.operands != (subtraction.results[0],)
    ):
        return None
    return "online_softmax", subtraction.operands[0], (
        statistics,
        maximum_projection,
        sum_projection,
        subtraction,
        exponential,
        division,
    )


def _score_chain(
    score_value: str, graph: ComputeGraph
) -> tuple[Op | None, Op, str | None, Op, tuple[Op, ...]] | None:
    operation = _producer(score_value, graph)
    if (
        operation is not None
        and _matches_target(operation, "prims.convert_element_type.default")
        and len(operation.operands) == 1
    ):
        score_value = operation.operands[0]
        operation = _producer(score_value, graph)
    mask_value: str | None = None
    if operation is not None and _matches_target(operation, "aten.add.Tensor"):
        if len(operation.operands) != 2:
            return None
        mask_value = operation.operands[1]
        scale_operation = _producer(operation.operands[0], graph)
    else:
        scale_operation = operation
    if scale_operation is None or not _matches_target(scale_operation, "aten.mul.Tensor"):
        return None
    score_path = _producer_path_until(
        scale_operation.operands[0],
        lambda candidate: _matches_target(candidate, "aten.bmm.default"),
        graph,
    )
    score_bmm = score_path[-1] if score_path is not None else None
    if score_bmm is None or not isinstance(_scalar_operand(scale_operation, 1), (int, float)):
        return None
    return (
        operation if mask_value is not None else None,
        scale_operation,
        mask_value,
        score_bmm,
        score_path[:-1],
    )


def _producer_path_until(
    value_id: str, predicate: Callable[[Op], bool], graph: ComputeGraph
) -> tuple[Op, ...] | None:
    operations: list[Op] = []
    visited: set[str] = set()
    while value_id not in visited:
        visited.add(value_id)
        operation = _producer(value_id, graph)
        if operation is None:
            return None
        if predicate(operation):
            return (*operations, operation)
        program = operation.tensor_program
        if (
            not isinstance(program, TensorProgram)
            or program.family not in {"view", "copy"}
            or len(operation.operands) != 1
        ):
            return None
        operations.append(operation)
        value_id = operation.operands[0]
    return None


def _boundary_output_path(
    value_id: str, graph: ComputeGraph
) -> tuple[str, tuple[Op, ...]]:
    operations: list[Op] = []
    current = value_id
    while current not in graph.outputs:
        users = graph.values[current].users
        if len(users) != 1:
            break
        operation = graph.ops.get(users[0])
        program = operation.tensor_program if operation is not None else None
        if (
            operation is None
            or not isinstance(program, TensorProgram)
            or program.family != "view"
            or len(operation.operands) != 1
        ):
            break
        operations.append(operation)
        current = operation.results[0]
    return current, tuple(operations)


def _producer_path(value_id: str, stop: Op, graph: ComputeGraph) -> tuple[Op, ...] | None:
    operations: list[Op] = []
    visited: set[str] = set()
    while value_id not in visited:
        visited.add(value_id)
        operation = _producer(value_id, graph)
        if operation is None:
            return None
        if operation.id == stop.id:
            return tuple(operations)
        program = operation.tensor_program
        if (
            not isinstance(program, TensorProgram)
            or program.family not in {"view", "copy", "elementwise"}
            or len(operation.operands) != 1
        ):
            return None
        operations.append(operation)
        value_id = operation.operands[0]
    return None


def _value_root(value_id: str, graph: ComputeGraph) -> str:
    while True:
        operation = _producer(value_id, graph)
        program = operation.tensor_program if operation is not None else None
        if (
            operation is None
            or not isinstance(program, TensorProgram)
            or program.family not in {"view", "copy"}
            or len(operation.operands) != 1
        ):
            return value_id
        value_id = operation.operands[0]


def _value_root_path(
    value_id: str, graph: ComputeGraph
) -> tuple[str, tuple[Op, ...]]:
    operations: list[Op] = []
    current = value_id
    while (
        graph.values[current].shape is None
        or len(graph.values[current].shape) != 4
    ):
        operation = _producer(current, graph)
        program = operation.tensor_program if operation is not None else None
        if (
            operation is None
            or not isinstance(program, TensorProgram)
            or program.family not in {"view", "copy"}
            or len(operation.operands) != 1
        ):
            return current, tuple(operations)
        operations.append(operation)
        current = operation.operands[0]
    while True:
        operation = _producer(current, graph)
        program = operation.tensor_program if operation is not None else None
        operand_shape = (
            graph.values[operation.operands[0]].shape
            if operation is not None and operation.operands
            else None
        )
        if (
            operation is None
            or not isinstance(program, TensorProgram)
            or program.family not in {"view", "copy"}
            or len(operation.operands) != 1
            or operand_shape is None
            or len(operand_shape) < 4
        ):
            return current, tuple(operations)
        operations.append(operation)
        current = operation.operands[0]


def _score_expression(
    q_load: IndexedLoad,
    k_load: IndexedLoad,
    scale_operation: Op,
    mask_value: str | None,
    dtype: str,
    graph: ComputeGraph,
) -> ScalarExpression | None:
    rounding = _cast_rounding_for(dtype)
    product = Binary(
        "mul",
        Cast(Load(q_load.id, q_load.dtype), dtype, rounding),
        Cast(Load(k_load.id, k_load.dtype), dtype, rounding),
        dtype,
    )
    scaled = Binary(
        "mul", product, Constant(_scalar_operand(scale_operation, 1), dtype), dtype
    )
    if mask_value is None:
        return scaled
    mask = _mask_semantics(mask_value, dtype, graph)
    mask_expression = mask.get("expression")
    if not isinstance(mask_expression, ScalarExpression):
        return None
    return Binary("add", scaled, Cast(mask_expression, dtype, rounding), dtype)


def _mask_semantics(
    value_id: str, dtype: str, graph: ComputeGraph
) -> dict[str, object]:
    operation = _producer(value_id, graph)
    if operation is not None and _matches_target(operation, "aten.where.self"):
        program = operation.tensor_program
        if isinstance(program, TensorProgram):
            loads = tuple(_semantic_mask_load(load) for load in program.loads)
            condition = _producer(_condition_value(program, graph), graph)
            condition_program = condition.tensor_program if condition is not None else None
            return {
                "kind": "typed_where",
                "expression": program.combine,
                "loads": loads,
                "condition_expression": (
                    condition_program.combine
                    if isinstance(condition_program, TensorProgram)
                    else None
                ),
                "condition_loads": (
                    tuple(_semantic_mask_load(load) for load in condition_program.loads)
                    if isinstance(condition_program, TensorProgram)
                    else ()
                ),
                "causal_offset": _causal_offset(condition, graph),
                "window_offsets": _window_offsets(condition, graph),
                "true_value": _scalar_mask_value(program.loads[1], graph),
                "false_value": _scalar_mask_value(program.loads[2], graph),
                "block_skip_proof": "none",
            }
    value = graph.values[value_id]
    load = IndexedLoad(
        "attention:mask",
        value_id,
        tuple(
            semantic_axis if size != 1 else 0
            for size, semantic_axis in zip(value.shape or (), ("b", "h", "s", "k"))
        ),
        value.strides or (),
        value.dtype or "",
        value.shape,
    )
    return {
        "kind": "additive_tensor",
        "expression": Load(load.id, load.dtype),
        "loads": (load,),
        "block_skip_proof": "none",
    }


def _semantic_mask_load(load: IndexedLoad) -> IndexedLoad:
    axes = {
        "o0": "b",
        "o1": "h",
        "o2": "s",
        "o3": "k",
    }
    return replace(
        load,
        axes=tuple(axes.get(axis, axis) if isinstance(axis, str) else axis for axis in load.axes),
    )


def _scalar_mask_value(load: IndexedLoad, graph: ComputeGraph) -> float | None:
    operation = _producer(load.value_id, graph)
    if (
        operation is not None
        and isinstance(operation.tensor_program, TensorProgram)
        and operation.tensor_program.family == "copy"
    ):
        source = _producer(operation.tensor_program.loads[0].value_id, graph)
        if (
            source is not None
            and isinstance(source.tensor_program, TensorProgram)
            and isinstance(source.tensor_program.combine, Constant)
        ):
            value = source.tensor_program.combine.value
            return value if isinstance(value, (int, float)) else None
    if (
        operation is not None
        and _matches_target(operation, "aten.full.default")
        and len(operation.attributes.get("fx_args", ())) >= 2
    ):
        value = operation.attributes["fx_args"][1]
        return value if isinstance(value, (int, float)) else None
    if (
        operation is None
        or not _matches_target(operation, "aten.scalar_tensor.default")
        or not operation.attributes.get("fx_args")
    ):
        return None
    value = operation.attributes["fx_args"][0]
    return value if isinstance(value, (int, float)) else None


def _condition_value(program: TensorProgram, graph: ComputeGraph) -> str:
    condition = program.combine
    if isinstance(condition, Where) and isinstance(condition.condition, Load):
        return next(
            load.value_id for load in program.loads if load.id == condition.condition.load_id
        )
    return program.loads[0].value_id


def _causal_offset(condition: Op | None, graph: ComputeGraph) -> int | None:
    if condition is None or not _matches_target(condition, "aten.le.Tensor"):
        return None
    return _comparison_offset(condition, graph)


def _window_offsets(condition: Op | None, graph: ComputeGraph) -> tuple[int, int] | None:
    if condition is None or not _matches_target(condition, "aten.bitwise_and.Tensor"):
        return None
    if len(condition.operands) != 2:
        return None
    upper = _producer(condition.operands[0], graph)
    lower = _producer(condition.operands[1], graph)
    if (
        upper is None
        or lower is None
        or not _matches_target(upper, "aten.le.Tensor")
        or not _matches_target(lower, "aten.ge.Tensor")
    ):
        return None
    lower_offset = _comparison_offset(lower, graph)
    upper_offset = _comparison_offset(upper, graph)
    if lower_offset is None or upper_offset is None:
        return None
    return lower_offset, upper_offset


def _comparison_offset(condition: Op, graph: ComputeGraph) -> int | None:
    program = condition.tensor_program
    if not isinstance(program, TensorProgram) or len(program.loads) != 2:
        return None
    offset = _producer(program.loads[1].value_id, graph)
    if offset is None or not (
        _matches_target(offset, "aten.add.Tensor")
        or _matches_target(offset, "aten.sub.Tensor")
    ):
        return None
    value = _scalar_operand(offset, 1)
    if not isinstance(value, int):
        return None
    return -value if _matches_target(offset, "aten.sub.Tensor") else value


def _scalar_operand(operation: Op, index: int) -> object:
    arguments = operation.attributes.get("fx_args", ())
    return arguments[index] if index < len(arguments) else None


def operation_target(operation: Op) -> object:
    return operation.attributes.get("target")


def _cast_rounding_for(dtype: str) -> str:
    return "round_nearest_even" if dtype in {"torch.bfloat16", "torch.float16"} else "exact"


def _ordered_operations(operations: tuple[Op, ...], graph: ComputeGraph) -> tuple[Op, ...]:
    by_id = {operation.id: operation for operation in operations}
    return tuple(
        operation
        for operation_id, operation in graph.ops.items()
        if operation_id in by_id
    )


def _sole_user(operation: Op, graph: ComputeGraph) -> Op | None:
    users = graph.values[operation.results[0]].users
    if len(users) != 1:
        return None
    return graph.ops.get(users[0])


def _matches_target(operation: Op, target: str) -> bool:
    return operation.attributes.get("target") == target


def _binary_operator(operation: Op | None, operator: str) -> bool:
    if operation is None or not isinstance(operation.tensor_program, TensorProgram):
        return False
    return (
        isinstance(operation.tensor_program.combine, Binary)
        and operation.tensor_program.combine.operator == operator
    )


def _transitive_operations(operation: Op, graph: ComputeGraph) -> tuple[Op, ...]:
    operation_ids = {operation.id}
    frontier = list(operation.operands)
    while frontier:
        value_id = frontier.pop()
        producer = _producer(value_id, graph)
        if producer is None or producer.id in operation_ids:
            continue
        operation_ids.add(producer.id)
        frontier.extend(producer.operands)
    return tuple(
        candidate for candidate in graph.ops.values() if candidate.id in operation_ids
    )


def _rope_operations(
    concat: Op, root_ids: tuple[str, str, str], graph: ComputeGraph
) -> tuple[Op, ...]:
    operation_ids = {concat.id}
    frontier = list(concat.operands)
    roots = set(root_ids)
    while frontier:
        value_id = frontier.pop()
        if value_id in roots:
            continue
        producer = _producer(value_id, graph)
        if producer is None or producer.id in operation_ids:
            continue
        operation_ids.add(producer.id)
        frontier.extend(producer.operands)
    return tuple(
        candidate for candidate in graph.ops.values() if candidate.id in operation_ids
    )


def _last_axis_slice(value_id: str, graph: ComputeGraph) -> tuple[str, int] | None:
    """Return the view root and start offset of a unit-step last-axis slice."""
    start = 0
    while True:
        producer = _producer(value_id, graph)
        if producer is None:
            return value_id, start
        if _matches_target(producer, "aten.unsqueeze.default"):
            source = graph.values[producer.operands[0]]
            arguments = producer.attributes.get("fx_args", ())
            dimension = arguments[1] if len(arguments) > 1 else None
            if (
                source.shape is None
                or not isinstance(dimension, int)
                or dimension % (len(source.shape) + 1) == len(source.shape)
            ):
                return None
        elif _matches_target(producer, "aten.slice.Tensor"):
            source = graph.values[producer.operands[0]]
            arguments = tuple(producer.attributes.get("fx_args", ()))
            dimension, slice_start, _, step = (
                *arguments[1:5],
                *(0, 0, None, 1)[len(arguments[1:5]):],
            )
            slice_start = 0 if slice_start is None else slice_start
            if (
                source.shape is None
                or not isinstance(dimension, int)
                or not isinstance(slice_start, int)
                or step not in (None, 1)
            ):
                return None
            if dimension % len(source.shape) == len(source.shape) - 1:
                if slice_start < 0:
                    slice_start += source.shape[-1]
                start += max(0, min(slice_start, source.shape[-1]))
        else:
            return value_id, start
        value_id = producer.operands[0]


def _view_root(value_id: str, graph: ComputeGraph) -> str | None:
    while True:
        producer = _producer(value_id, graph)
        if producer is None:
            return value_id
        if not _matches_target(producer, "aten.slice.Tensor") and not _matches_target(
            producer, "aten.unsqueeze.default"
        ):
            return value_id
        value_id = producer.operands[0]


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
