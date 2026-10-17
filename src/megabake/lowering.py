from __future__ import annotations

import operator
import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from .graph import ComputeGraph, Op, Value


@dataclass(frozen=True)
class IterationDomain:
    axis: str
    size: int


@dataclass(frozen=True)
class AxisPredicate:
    axis: str
    upper_bound: int


@dataclass(frozen=True)
class AxisTerm:
    axis: str
    scale: int = 1
    divisor: int = 1
    modulus: int = 0


@dataclass(frozen=True)
class AxisAffine:
    terms: tuple[AxisTerm, ...]
    offset: int = 0


@dataclass(frozen=True)
class DataIndex:
    load_id: str


@dataclass(frozen=True)
class IndexedLoad:
    id: str
    value_id: str
    axes: tuple[str | int | AxisAffine | DataIndex, ...]
    strides: tuple[int, ...]
    dtype: str
    source_shape: tuple[int, ...] | None = None


@dataclass(frozen=True)
class TensorStore:
    value_id: str
    shape: tuple[int, ...]
    dtype: str
    strides: tuple[int, ...]
    owns_storage: bool


@dataclass(frozen=True)
class Constant:
    value: int | float | bool
    dtype: str
    weak_type: bool = False


@dataclass(frozen=True)
class Load:
    load_id: str
    dtype: str


@dataclass(frozen=True)
class Accumulator:
    dtype: str


@dataclass(frozen=True)
class Unary:
    operator: str
    operand: ScalarExpression
    dtype: str


@dataclass(frozen=True)
class AxisIndex:
    axis: str
    dtype: str
    start: int = 0
    step: int = 1


@dataclass(frozen=True)
class Binary:
    operator: str
    lhs: ScalarExpression
    rhs: ScalarExpression
    dtype: str


@dataclass(frozen=True)
class Where:
    condition: ScalarExpression
    true_value: ScalarExpression
    false_value: ScalarExpression
    dtype: str


@dataclass(frozen=True)
class Cast:
    operand: ScalarExpression
    dtype: str
    rounding: str


ScalarExpression = (
    Constant
    | Load
    | Accumulator
    | Unary
    | AxisIndex
    | Binary
    | Where
    | Cast
)

AxisExpression = str | int | AxisAffine | DataIndex


@dataclass(frozen=True)
class TensorProgram:
    family: str
    source_target: str
    decomposition: str
    iteration_domains: tuple[IterationDomain, ...]
    reduction_axes: tuple[str, ...]
    loads: tuple[IndexedLoad, ...]
    init: Constant | None
    combine: ScalarExpression
    post: ScalarExpression | None
    accumulator_dtype: str | None
    store: TensorStore
    predicates: tuple[AxisPredicate, ...]
    state_accesses: tuple[KVCacheAccess, ...] = ()


@dataclass(frozen=True)
class KVCacheLayout:
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class KVCacheAccess:
    key_cache_value_id: str
    value_cache_value_id: str
    valid_length_value_id: str
    key_cache_load_id: str
    value_cache_load_id: str
    new_key_load_id: str
    new_value_load_id: str
    valid_length_load_id: str
    capacity: int
    query_length: int
    key_layout: KVCacheLayout
    value_layout: KVCacheLayout
    storage_identity: tuple[str, str]
    read_range: tuple[int | str, int | str] = (0, "valid_length")
    write_range: tuple[int | str, int | str] = (
        "valid_length",
        "valid_length + query_length",
    )
    source_order: tuple[str, ...] = (
        "read_valid_length",
        "read_cache_prefix",
        "read_new_key_value",
        "write_key_cache",
        "write_value_cache",
        "write_valid_length",
    )
    capacity_guard: str = "0 <= valid_length and valid_length + query_length <= capacity"
    storage_class: str = "bounded_contiguous"
    valid_length_specialized: bool = False


@dataclass(frozen=True)
class CanonicalProgram:
    family: str
    source_target: str
    decomposition: str
    result_roles: tuple[str, ...]
    semantics: Mapping[str, Any]


class UnsupportedLowering(ValueError):
    pass


LoweringRule = Callable[
    [torch.fx.Node, Op, ComputeGraph], TensorProgram | CanonicalProgram
]


@torch.library.custom_op(
    "megabake::kv_attention",
    mutates_args=("key_cache", "value_cache", "valid_length"),
)
def kv_attention(
    source: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    valid_length: torch.Tensor,
) -> torch.Tensor:
    if source.shape[1] != 1:
        raise RuntimeError("megabake kv_attention supports one-token decode only")
    start = int(valid_length.item())
    end = start + source.shape[1]
    if start < 0 or end > key_cache.shape[1]:
        raise RuntimeError(
            f"KV cache length {start} plus {source.shape[1]} exceeds capacity "
            f"{key_cache.shape[1]}"
        )
    key_cache[:, start:end].copy_(key)
    value_cache[:, start:end].copy_(value)
    valid_length.fill_(end)
    query = source.permute(0, 2, 1, 3)
    keys = key_cache[:, :end].permute(0, 2, 1, 3)
    values = value_cache[:, :end].permute(0, 2, 1, 3)
    if query.shape[1] != keys.shape[1]:
        group_size = query.shape[1] // keys.shape[1]
        keys = keys.repeat_interleave(group_size, dim=1)
        values = values.repeat_interleave(group_size, dim=1)
    scores = (query @ keys.transpose(-2, -1)).float() * source.shape[-1] ** -0.5
    probabilities = torch.softmax(scores, dim=-1).to(source.dtype)
    output = probabilities @ values
    return output.permute(0, 2, 1, 3).reshape(
        source.shape[0] * source.shape[2], source.shape[1], source.shape[3]
    )


@kv_attention.register_fake
def _kv_attention_fake(
    source: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    valid_length: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (
            source.shape[0] * source.shape[2],
            source.shape[1],
            source.shape[3],
        ),
        dtype=source.dtype,
        device=source.device,
    )


KV_ATTENTION_OP = torch.ops.megabake.kv_attention.default


@torch.library.custom_op("megabake::attention", mutates_args=())
def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """Softmax attention with Hugging Face eager numerics.

    ``query`` is ``[batch, query, heads, dim]``; ``key`` and ``value`` are
    ``[batch, key, kv_heads, dim]`` with ``heads`` a multiple of ``kv_heads``;
    ``mask`` is additive and broadcasts to ``[batch, heads, query, key]``.
    Returns ``[batch * heads, query, dim]``, the layout of ``kv_attention``.
    """
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    if query.shape[1] != key.shape[1]:
        group_size = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(group_size, dim=1)
        value = value.repeat_interleave(group_size, dim=1)
    scores = torch.matmul(query, key.transpose(2, 3)) * scale
    if mask is not None:
        scores = scores + mask
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.matmul(probabilities, value)
    return output.reshape(-1, output.shape[2], output.shape[3])


@attention.register_fake
def _attention_fake(query, key, value, mask, scale):
    return torch.empty(
        (query.shape[0] * query.shape[2], query.shape[1], query.shape[3]),
        dtype=query.dtype,
        device=query.device,
    )


ATTENTION_OP = torch.ops.megabake.attention.default


def _node_value(node: torch.fx.Node, graph: ComputeGraph) -> Value:
    value_id = f"value:{node.name}"
    value = graph.values.get(value_id)
    if value is None or value.source_fx_id != f"fx:{node.name}":
        raise UnsupportedLowering(f"FX node {node.name} does not have one value")
    return value


def _result_value(op: Op, graph: ComputeGraph) -> Value:
    if len(op.results) != 1:
        raise UnsupportedLowering("only single-result operations are supported")
    value = graph.values[op.results[0]]
    if value.shape is None or value.dtype is None or value.strides is None:
        raise UnsupportedLowering(f"result {value.id} lacks static tensor metadata")
    return value


def _domains(prefix: str, shape: Sequence[int]) -> tuple[IterationDomain, ...]:
    return tuple(IterationDomain(f"{prefix}{index}", size) for index, size in enumerate(shape))


def _predicates(domains: Sequence[IterationDomain]) -> tuple[AxisPredicate, ...]:
    return tuple(AxisPredicate(domain.axis, domain.size) for domain in domains)


def _load(value: Value, axes: Sequence[AxisExpression], index: int) -> IndexedLoad:
    if value.strides is None:
        raise UnsupportedLowering(f"value {value.id} lacks strides")
    return IndexedLoad(
        id=f"load:{index}",
        value_id=value.id,
        axes=tuple(axes),
        strides=value.strides,
        dtype=value.dtype or "",
        source_shape=value.shape,
    )


def _affine(*terms: AxisTerm, offset: int = 0) -> AxisAffine:
    return AxisAffine(terms, offset)


def axis_references(axis: AxisExpression) -> tuple[str, ...]:
    if isinstance(axis, str):
        return (axis,)
    if isinstance(axis, int):
        return ()
    if isinstance(axis, DataIndex):
        return ()
    return tuple(dict.fromkeys(term.axis for term in axis.terms))


def axis_coordinate(axis: AxisExpression, coordinates: Mapping[str, int]) -> int:
    if isinstance(axis, str):
        return coordinates[axis]
    if isinstance(axis, int):
        return axis
    value = axis.offset
    for term in axis.terms:
        coordinate = coordinates[term.axis] // term.divisor
        if term.modulus:
            coordinate %= term.modulus
        value += coordinate * term.scale
    return value


def _broadcast_axes(value: Value, output_shape: Sequence[int]) -> tuple[str | int, ...]:
    if value.shape is None:
        raise UnsupportedLowering(f"value {value.id} lacks a shape")
    rank_difference = len(output_shape) - len(value.shape)
    if rank_difference < 0:
        raise UnsupportedLowering(f"value {value.id} has too many dimensions to broadcast")
    axes: list[str | int] = []
    for dimension, size in enumerate(value.shape):
        output_dimension = dimension + rank_difference
        if size == 1:
            axes.append(0)
        elif size == output_shape[output_dimension]:
            axes.append(f"o{output_dimension}")
        else:
            raise UnsupportedLowering(
                f"value {value.id} cannot broadcast {size} to {output_shape[output_dimension]}"
            )
    return tuple(axes)


def _constant(value: int | float | bool) -> Constant:
    if isinstance(value, bool):
        dtype = "torch.bool"
    elif isinstance(value, int):
        dtype = "torch.int64"
    elif isinstance(value, float):
        dtype = "torch.float32"
    else:
        raise UnsupportedLowering(f"unsupported scalar constant {value!r}")
    return Constant(value=value, dtype=dtype, weak_type=True)


def _zero(dtype: str) -> Constant:
    return Constant(value=0, dtype=dtype, weak_type=False)


def _expression_argument(
    argument: Any, load: Load | None
) -> ScalarExpression:
    if isinstance(argument, torch.fx.Node):
        if load is None:
            raise UnsupportedLowering("missing tensor load for FX node argument")
        return load
    if isinstance(argument, (int, float, bool)):
        return _constant(argument)
    raise UnsupportedLowering(f"unsupported scalar argument {argument!r}")


def _elementwise_program(
    node: torch.fx.Node,
    op: Op,
    graph: ComputeGraph,
    family: str,
    build_expression: Callable[[list[Load]], ScalarExpression],
    decomposition: str = "none",
) -> TensorProgram:
    result = _result_value(op, graph)
    if result.shape is None:
        raise UnsupportedLowering("result shape is unknown")
    domains = _domains("o", result.shape)
    loads: list[IndexedLoad] = []
    load_expressions: list[Load] = []
    arguments = [*node.args, *node.kwargs.values()]
    for index, argument in enumerate(arguments):
        if not isinstance(argument, torch.fx.Node):
            continue
        value = _node_value(argument, graph)
        load = _load(value, _broadcast_axes(value, result.shape), index)
        loads.append(load)
        load_expressions.append(Load(load.id, load.dtype))
    expression = build_expression(load_expressions)
    return TensorProgram(
        family=family,
        source_target=str(node.target),
        decomposition=decomposition,
        iteration_domains=domains,
        reduction_axes=(),
        loads=tuple(loads),
        init=None,
        combine=expression,
        post=None,
        accumulator_dtype=None,
        store=TensorStore(
            result.id, result.shape, result.dtype or "", result.strides, True
        ),
        predicates=_predicates(domains),
    )


def _binary_program(
    node: torch.fx.Node,
    op: Op,
    graph: ComputeGraph,
    operator: str,
) -> TensorProgram:
    result = _result_value(op, graph)
    alpha = node.kwargs.get("alpha", 1)

    def build(loads: list[Load]) -> ScalarExpression:
        if len(node.args) != 2:
            raise UnsupportedLowering("binary operation must have two positional arguments")
        left_argument, right_argument = node.args
        left = _expression_argument(left_argument, loads[0] if loads else None)
        right = _expression_argument(right_argument, loads[-1] if loads else None)
        if operator in {"add", "sub"} and alpha != 1:
            right = Binary(
                "mul",
                right,
                _constant(alpha),
                result.dtype or "",
            )
        return Binary(operator, left, right, result.dtype or "")

    return _elementwise_program(node, op, graph, "elementwise", build)


def _unary_program(
    node: torch.fx.Node, op: Op, graph: ComputeGraph, operator: str
) -> TensorProgram:
    def build(loads: list[Load]) -> ScalarExpression:
        if not loads:
            raise UnsupportedLowering("unary operation has no tensor operand")
        return Unary(operator, loads[0], _result_value(op, graph).dtype or "")

    return _elementwise_program(node, op, graph, "elementwise", build)


def _logical_program(
    node: torch.fx.Node, op: Op, graph: ComputeGraph, operator: str
) -> TensorProgram:
    operands = [argument for argument in node.args if isinstance(argument, torch.fx.Node)]
    if len(operands) != 2 or any(
        _node_value(operand, graph).dtype != "torch.bool" for operand in operands
    ):
        raise UnsupportedLowering(f"{operator} is lowered for boolean tensors only")
    return _binary_program(node, op, graph, operator)


def _indexed_program(
    node: torch.fx.Node,
    op: Op,
    graph: ComputeGraph,
    loads: Sequence[IndexedLoad],
    expression: ScalarExpression,
    decomposition: str,
) -> TensorProgram:
    """An elementwise program over the result with explicitly indexed loads."""
    result = _result_value(op, graph)
    domains = _domains("o", result.shape)
    return TensorProgram(
        family="elementwise",
        source_target=str(node.target),
        decomposition=decomposition,
        iteration_domains=domains,
        reduction_axes=(),
        loads=tuple(loads),
        init=None,
        combine=expression,
        post=None,
        accumulator_dtype=None,
        store=TensorStore(
            result.id, result.shape, result.dtype or "", result.strides, True
        ),
        predicates=_predicates(domains),
    )


def _within(axis: str, start: int, end: int) -> ScalarExpression:
    index = AxisIndex(axis, "torch.int64")
    return Binary(
        "bitwise_and",
        Binary("ge", index, Constant(start, "torch.int64"), "torch.bool"),
        Binary("lt", index, Constant(end, "torch.int64"), "torch.bool"),
        "torch.bool",
    )


def _value_load(load: IndexedLoad) -> Load:
    return Load(load.id, load.dtype)


def _slice_scatter_program(node: torch.fx.Node, op: Op, graph: ComputeGraph) -> TensorProgram:
    """``out = base`` with ``src`` written into ``base[..., start:end, ...]``."""
    result = _result_value(op, graph)
    base, source = (_node_value(argument, graph) for argument in node.args[:2])
    rank = len(result.shape)
    dim = (node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", 0)) % rank
    size = result.shape[dim]
    start = node.args[3] if len(node.args) > 3 else node.kwargs.get("start")
    end = node.args[4] if len(node.args) > 4 else node.kwargs.get("end")
    step = node.args[5] if len(node.args) > 5 else node.kwargs.get("step", 1)
    start = 0 if start is None else (start + size if start < 0 else start)
    end = size if end is None else min(end + size if end < 0 else end, size)
    if not isinstance(step, int) or step < 1:
        raise UnsupportedLowering("slice_scatter needs a positive integer step")
    axes = [f"o{index}" for index in range(rank)]
    base_load = _load(base, axes, 0)
    source_axes = list(axes)
    # Inside the slice o = start + step * j, so j = o // step - start // step.
    source_axes[dim] = _affine(AxisTerm(axes[dim], divisor=step), offset=-(start // step))
    source_load = _load(source, source_axes, 1)
    inside = _within(axes[dim], start, end)
    if step != 1:
        offset = Binary(
            "sub", AxisIndex(axes[dim], "torch.int64"), Constant(start, "torch.int64"), "torch.int64"
        )
        on_step = Binary(
            "eq",
            Binary("remainder", offset, Constant(step, "torch.int64"), "torch.int64"),
            Constant(0, "torch.int64"),
            "torch.bool",
        )
        inside = Binary("bitwise_and", inside, on_step, "torch.bool")
    expression = Where(
        inside,
        _value_load(source_load),
        _value_load(base_load),
        result.dtype or "",
    )
    return _indexed_program(
        node, op, graph, (base_load, source_load), expression, "select source inside the slice"
    )


def _select_scatter_program(node: torch.fx.Node, op: Op, graph: ComputeGraph) -> TensorProgram:
    """``out = base`` with ``src`` written into ``base.select(dim, index)``."""
    result = _result_value(op, graph)
    base, source = (_node_value(argument, graph) for argument in node.args[:2])
    rank = len(result.shape)
    dim, index = node.args[2] % rank, node.args[3]
    index = index + result.shape[dim] if index < 0 else index
    axes = [f"o{position}" for position in range(rank)]
    base_load = _load(base, axes, 0)
    source_load = _load(source, axes[:dim] + axes[dim + 1 :], 1)
    expression = Where(
        _within(axes[dim], index, index + 1),
        _value_load(source_load),
        _value_load(base_load),
        result.dtype or "",
    )
    return _indexed_program(
        node, op, graph, (base_load, source_load), expression, "select source at the index"
    )


def _constant_pad_program(node: torch.fx.Node, op: Op, graph: ComputeGraph) -> TensorProgram:
    """``constant_pad_nd``: a shifted load; out-of-range reads take the fill value."""
    result = _result_value(op, graph)
    source = _node_value(node.args[0], graph)
    pad = node.args[1]
    fill = node.args[2] if len(node.args) > 2 else node.kwargs.get("value", 0)
    rank = len(result.shape)
    axes: list[Any] = [f"o{index}" for index in range(rank)]
    inside: ScalarExpression | None = None
    for pair in range(len(pad) // 2):
        dim = rank - 1 - pair
        low = pad[2 * pair]
        axes[dim] = _affine(AxisTerm(f"o{dim}"), offset=-low)
        within = _within(f"o{dim}", low, low + source.shape[dim])
        inside = within if inside is None else Binary("bitwise_and", inside, within, "torch.bool")
    load = _load(source, axes, 0)
    expression: ScalarExpression = _value_load(load)
    if fill != 0 and inside is not None:
        # Loads outside the source read zero; other fill values need a select.
        expression = Where(inside, expression, _typed_constant(fill, result.dtype or ""), result.dtype or "")
    return _indexed_program(node, op, graph, (load,), expression, "shifted load with fill")


def _functional_copy_program(node: torch.fx.Node, op: Op, graph: ComputeGraph) -> TensorProgram:
    """``aten.copy(self, src)``: ``src`` broadcast to ``self`` and cast to its dtype."""
    result = _result_value(op, graph)
    source = _node_value(node.args[1], graph)
    load = _load(source, _broadcast_axes(source, result.shape), 0)
    expression = Cast(
        _value_load(load),
        result.dtype or "",
        _cast_rounding(source.dtype or "", result.dtype or ""),
    )
    return _indexed_program(node, op, graph, (load,), expression, "broadcast copy")


def _state_write_program(node: torch.fx.Node, op: Op, graph: ComputeGraph) -> TensorProgram:
    """``aten.copy_(state, src)`` on a graph input: write ``src`` into the
    input's storage. AOT keeps input mutations as these trailing copies; the
    storage-order dependencies run the write after every read of the old state.
    """
    destination = _node_value(node.args[0], graph)
    if destination.id not in graph.inputs:
        raise UnsupportedLowering("in-place copies are lowered into graph inputs only")
    program = _functional_copy_program(node, op, graph)
    result = _result_value(op, graph)
    if result.storage_relation.storage_id != destination.storage_relation.storage_id:
        raise UnsupportedLowering("in-place copy result does not alias its destination")
    return replace(program, store=replace(program.store, owns_storage=False))


def _where_program(node: torch.fx.Node, op: Op, graph: ComputeGraph) -> TensorProgram:
    result = _result_value(op, graph)

    def build(loads: list[Load]) -> ScalarExpression:
        if len(node.args) != 3 or len(loads) != 3:
            raise UnsupportedLowering("where must have three tensor arguments")
        condition, true_value, false_value = loads
        return Where(condition, true_value, false_value, result.dtype or "")

    return _elementwise_program(node, op, graph, "elementwise", build)


def _cast_rounding(source_dtype: str, target_dtype: str) -> str:
    source = getattr(torch, source_dtype.removeprefix("torch."), None)
    target = getattr(torch, target_dtype.removeprefix("torch."), None)
    if not isinstance(source, torch.dtype) or not isinstance(target, torch.dtype):
        return "pytorch_default"
    if source == target:
        return "exact"
    if source.is_floating_point and not target.is_floating_point:
        return "truncate_toward_zero"
    if source.is_floating_point and target.is_floating_point:
        return "round_nearest_even"
    return "pytorch_default"


def _cast_program(node: torch.fx.Node, op: Op, graph: ComputeGraph) -> TensorProgram:
    result = _result_value(op, graph)
    input_value = _node_value(node.args[0], graph)

    def build(loads: list[Load]) -> ScalarExpression:
        if not loads:
            raise UnsupportedLowering("cast has no tensor operand")
        return Cast(
            loads[0],
            result.dtype or "",
            _cast_rounding(input_value.dtype or "", result.dtype or ""),
        )

    return _elementwise_program(node, op, graph, "elementwise", build, "explicit cast")


def _view_program(node: torch.fx.Node, op: Op, graph: ComputeGraph) -> TensorProgram:
    result = _result_value(op, graph)
    source = _node_value(node.args[0], graph)
    if result.shape is None or result.strides is None:
        raise UnsupportedLowering("view result lacks shape or strides")
    if source.shape is None or source.dtype != result.dtype:
        raise UnsupportedLowering("only same-dtype views are supported")
    if (
        result.storage_relation.unknown
        or result.storage_relation.storage_id != source.storage_relation.storage_id
    ):
        raise UnsupportedLowering("view result does not alias its source")
    domains = _domains("o", result.shape)
    load = IndexedLoad(
        id="load:0",
        value_id=source.id,
        axes=tuple(domain.axis for domain in domains),
        strides=result.strides,
        dtype=source.dtype or "",
    )
    expression = Load(load.id, load.dtype)
    aliases_source = (
        not result.storage_relation.unknown
        and result.storage_relation.storage_id == source.storage_relation.storage_id
    )
    return TensorProgram(
        family="view",
        source_target=str(node.target),
        decomposition="coordinate map through result strides over source storage",
        iteration_domains=domains,
        reduction_axes=(),
        loads=(load,),
        init=None,
        combine=expression,
        post=None,
        accumulator_dtype=None,
        store=TensorStore(
            result.id,
            result.shape,
            result.dtype or "",
            result.strides,
            not aliases_source,
        ),
        predicates=_predicates(domains),
    )


def _copy_program(node: torch.fx.Node, op: Op, graph: ComputeGraph) -> TensorProgram:
    def build(loads: list[Load]) -> ScalarExpression:
        if not loads:
            raise UnsupportedLowering("copy has no tensor operand")
        return loads[0]

    return _elementwise_program(
        node, op, graph, "copy", build, "materializing copy"
    )


def _typed_constant(value: Any, dtype: str) -> Constant:
    if not isinstance(value, (bool, int, float)):
        raise UnsupportedLowering(f"unsupported scalar constant {value!r}")
    return Constant(value=value, dtype=dtype)


def _constant_program(node: torch.fx.Node, op: Op, graph: ComputeGraph) -> TensorProgram:
    result = _result_value(op, graph)
    if result.shape is None:
        raise UnsupportedLowering("constant result lacks a shape")
    value = node.args[1] if node.target is torch.ops.aten.full.default else node.args[0]
    domains = _domains("o", result.shape)
    return TensorProgram(
        family="elementwise",
        source_target=str(node.target),
        decomposition="constant generator",
        iteration_domains=domains,
        reduction_axes=(),
        loads=(),
        init=None,
        combine=_typed_constant(value, result.dtype or ""),
        post=None,
        accumulator_dtype=None,
        store=TensorStore(
            result.id, result.shape, result.dtype or "", result.strides, True
        ),
        predicates=_predicates(domains),
    )


def _attribute_constant_program(
    node: torch.fx.Node, op: Op, graph: ComputeGraph
) -> TensorProgram:
    result = _result_value(op, graph)
    value = node.meta.get("val")
    scalar = value.item() if isinstance(value, torch.Tensor) else None
    if not isinstance(scalar, (bool, int, float)):
        source_graph = getattr(graph.runtime_context, "graph", None)
        value = getattr(source_graph, node.target, None)
        scalar = value.item() if isinstance(value, torch.Tensor) else None
    if not isinstance(value, torch.Tensor):
        raise UnsupportedLowering("tensor attribute value is unavailable")
    if not isinstance(scalar, (bool, int, float)):
        raise UnsupportedLowering("tensor attribute is not a concrete scalar")
    if result.shape is None or result.shape != ():
        raise UnsupportedLowering("only scalar tensor attributes are supported")
    domains = _domains("o", result.shape)
    return TensorProgram(
        family="elementwise",
        source_target=str(node.target),
        decomposition="scalar tensor attribute",
        iteration_domains=domains,
        reduction_axes=(),
        loads=(),
        init=None,
        combine=_typed_constant(scalar, result.dtype or ""),
        post=None,
        accumulator_dtype=None,
        store=TensorStore(
            result.id, result.shape, result.dtype or "", result.strides, True
        ),
        predicates=_predicates(domains),
    )


def _iota_program(node: torch.fx.Node, op: Op, graph: ComputeGraph) -> TensorProgram:
    result = _result_value(op, graph)
    length = node.args[0]
    if result.shape != (length,):
        raise UnsupportedLowering("iota result shape does not match its length")
    domains = _domains("o", result.shape)
    return TensorProgram(
        family="elementwise",
        source_target=str(node.target),
        decomposition="affine axis generator",
        iteration_domains=domains,
        reduction_axes=(),
        loads=(),
        init=None,
        combine=AxisIndex(
            axis="o0",
            dtype=result.dtype or "",
            start=node.kwargs.get("start", 0),
            step=node.kwargs.get("step", 1),
        ),
        post=None,
        accumulator_dtype=None,
        store=TensorStore(
            result.id, result.shape, result.dtype or "", result.strides, True
        ),
        predicates=_predicates(domains),
    )


def _tuple_projection_program(
    node: torch.fx.Node, op: Op, graph: ComputeGraph
) -> TensorProgram:
    result = _result_value(op, graph)
    result_index = node.args[1]
    if not isinstance(result_index, int) or not 0 <= result_index < len(op.operands):
        raise UnsupportedLowering("tuple projection has an invalid result index")
    source = graph.values[op.operands[result_index]]
    if source.shape is None or source.dtype != result.dtype:
        raise UnsupportedLowering("tuple projection source and result differ")
    domains = _domains("o", result.shape or ())
    load = IndexedLoad(
        id="load:0",
        value_id=source.id,
        axes=tuple(domain.axis for domain in domains),
        strides=result.strides or (),
        dtype=source.dtype or "",
    )
    return TensorProgram(
        family="view",
        source_target=str(node.target),
        decomposition="pure tuple result projection",
        iteration_domains=domains,
        reduction_axes=(),
        loads=(load,),
        init=None,
        combine=Load(load.id, load.dtype),
        post=None,
        accumulator_dtype=None,
        store=TensorStore(
            result.id,
            result.shape or (),
            result.dtype or "",
            result.strides or (),
            False,
        ),
        predicates=_predicates(domains),
    )
def _dtype_value(dtype_name: str) -> torch.dtype:
    dtype = getattr(torch, dtype_name.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise UnsupportedLowering(f"unknown dtype {dtype_name}")
    return dtype


def _accumulator_dtype(result_dtype: str, source_dtypes: Sequence[str]) -> str:
    result = _dtype_value(result_dtype)
    sources = [_dtype_value(dtype) for dtype in source_dtypes if dtype]
    if result in {torch.float16, torch.bfloat16} and any(
        dtype in {torch.float16, torch.bfloat16} for dtype in sources
    ):
        return str(torch.float32)
    return result_dtype


def _cast_load(load: Load, dtype: str) -> Cast:
    return Cast(load, dtype, _cast_rounding(load.dtype, dtype))


def _contraction_post(
    accumulator_dtype: str,
    result_dtype: str,
    bias_load: Load | None,
    beta: int | float = 1,
) -> ScalarExpression:
    combined = Accumulator(accumulator_dtype)
    if bias_load is None or beta == 0:
        combined_value = combined
    else:
        bias = _cast_load(Load(bias_load.id, bias_load.dtype), accumulator_dtype)
        if beta != 1:
            bias = Binary("mul", bias, _constant(beta), accumulator_dtype)
        combined_value = Binary("add", combined, bias, accumulator_dtype)
    return Cast(
        combined_value,
        result_dtype,
        _cast_rounding(accumulator_dtype, result_dtype),
    )


def _contraction_program(
    node: torch.fx.Node,
    op: Op,
    graph: ComputeGraph,
    left: Value,
    right: Value,
    logical_right_shape: tuple[int, int],
    right_axes: tuple[str | int, ...],
    bias: Value | None,
    alpha: int | float,
    beta: int | float,
) -> TensorProgram:
    result = _result_value(op, graph)
    if result.shape is None or len(result.shape) != 2:
        raise UnsupportedLowering("contraction result must be a static 2-D tensor")
    if left.shape is None or right.shape is None or len(left.shape) != 2 or len(right.shape) != 2:
        raise UnsupportedLowering("contraction operands must be static 2-D tensors")
    if left.shape[1] != logical_right_shape[0]:
        raise UnsupportedLowering("contraction reduction dimensions do not match")
    if result.shape != (left.shape[0], logical_right_shape[1]):
        raise UnsupportedLowering("contraction result shape does not match operands")

    output_domains = _domains("o", result.shape)
    reduction_domains = (IterationDomain("r0", left.shape[1]),)
    left_load = _load(left, ("o0", "r0"), 0)
    right_load = _load(right, right_axes, 1)
    loads = [left_load, right_load]
    bias_load: Load | None = None
    if bias is not None:
        bias_load = _load(bias, _broadcast_axes(bias, result.shape), 2)
        loads.append(bias_load)

    accumulator_dtype = _accumulator_dtype(
        result.dtype or "", [left.dtype or "", right.dtype or ""]
    )
    product = Binary(
        "mul",
        _cast_load(Load(left_load.id, left_load.dtype), accumulator_dtype),
        _cast_load(Load(right_load.id, right_load.dtype), accumulator_dtype),
        accumulator_dtype,
    )
    if alpha != 1:
        product = Binary("mul", product, _constant(alpha), accumulator_dtype)
    combine = Binary(
        "add",
        Accumulator(accumulator_dtype),
        product,
        accumulator_dtype,
    )
    post = _contraction_post(
        accumulator_dtype,
        result.dtype or "",
        bias_load,
        beta,
    )
    return TensorProgram(
        family="contraction",
        source_target=str(node.target),
        decomposition="indexed multiply-accumulate with explicit result cast",
        iteration_domains=(*output_domains, *reduction_domains),
        reduction_axes=("r0",),
        loads=tuple(loads),
        init=_zero(accumulator_dtype),
        combine=combine,
        post=post,
        accumulator_dtype=accumulator_dtype,
        store=TensorStore(
            result.id, result.shape, result.dtype or "", result.strides, True
        ),
        predicates=_predicates((*output_domains, *reduction_domains)),
    )


def _linear_program(node: torch.fx.Node, op: Op, graph: ComputeGraph) -> TensorProgram:
    if len(node.args) not in {2, 3}:
        raise UnsupportedLowering("linear must have two or three arguments")
    input_value = _node_value(node.args[0], graph)
    weight_value = _node_value(node.args[1], graph)
    if weight_value.shape is None or len(weight_value.shape) != 2:
        raise UnsupportedLowering("linear weight must be 2-D")
    if input_value.shape is None or len(input_value.shape) != 2:
        raise UnsupportedLowering("only 2-D linear inputs are supported")
    result_shape = _result_value(op, graph).shape
    if (
        result_shape is None
        or input_value.shape[1] != weight_value.shape[1]
        or result_shape != (input_value.shape[0], weight_value.shape[0])
    ):
        raise UnsupportedLowering("linear input, weight, and output shapes do not match")
    bias_value = _node_value(node.args[2], graph) if len(node.args) == 3 else None
    return _contraction_program(
        node,
        op,
        graph,
        input_value,
        weight_value,
        (weight_value.shape[1], weight_value.shape[0]),
        ("o1", "r0"),
        bias_value,
        1,
        1,
    )


def _mm_program(node: torch.fx.Node, op: Op, graph: ComputeGraph) -> TensorProgram:
    if len(node.args) != 2:
        raise UnsupportedLowering("mm must have two arguments")
    return _contraction_program(
        node,
        op,
        graph,
        _node_value(node.args[0], graph),
        _node_value(node.args[1], graph),
        _node_value(node.args[1], graph).shape,
        ("r0", "o1"),
        None,
        1,
        1,
    )


def _addmm_program(node: torch.fx.Node, op: Op, graph: ComputeGraph) -> TensorProgram:
    if len(node.args) != 3:
        raise UnsupportedLowering("addmm must have three arguments")
    return _contraction_program(
        node,
        op,
        graph,
        _node_value(node.args[1], graph),
        _node_value(node.args[2], graph),
        _node_value(node.args[2], graph).shape,
        ("r0", "o1"),
        _node_value(node.args[0], graph),
        node.kwargs.get("alpha", 1),
        node.kwargs.get("beta", 1),
    )


def _bmm_program(node: torch.fx.Node, op: Op, graph: ComputeGraph) -> TensorProgram:
    if len(node.args) != 2:
        raise UnsupportedLowering("bmm must have two arguments")
    left = _node_value(node.args[0], graph)
    right = _node_value(node.args[1], graph)
    result = _result_value(op, graph)
    if (
        left.shape is None
        or right.shape is None
        or result.shape is None
        or len(left.shape) != 3
        or len(right.shape) != 3
        or len(result.shape) != 3
    ):
        raise UnsupportedLowering("bmm operands and result must be static 3-D tensors")
    batch, rows, reduction = left.shape
    right_batch, right_reduction, columns = right.shape
    if (
        batch != right_batch
        or reduction != right_reduction
        or result.shape != (batch, rows, columns)
    ):
        raise UnsupportedLowering("bmm operand and result shapes do not match")
    output_domains = (
        IterationDomain("o0", batch),
        IterationDomain("o1", rows),
        IterationDomain("o2", columns),
    )
    reduction_domain = IterationDomain("r0", reduction)
    left_load = _load(left, ("o0", "o1", "r0"), 0)
    right_load = _load(right, ("o0", "r0", "o2"), 1)
    accumulator_dtype = _accumulator_dtype(
        result.dtype or "", [left.dtype or "", right.dtype or ""]
    )
    product = Binary(
        "mul",
        _cast_load(Load(left_load.id, left_load.dtype), accumulator_dtype),
        _cast_load(Load(right_load.id, right_load.dtype), accumulator_dtype),
        accumulator_dtype,
    )
    return TensorProgram(
        family="contraction",
        source_target=str(node.target),
        decomposition="batched indexed multiply-accumulate with result cast",
        iteration_domains=(*output_domains, reduction_domain),
        reduction_axes=("r0",),
        loads=(left_load, right_load),
        init=_zero(accumulator_dtype),
        combine=Binary(
            "add", Accumulator(accumulator_dtype), product, accumulator_dtype
        ),
        post=_contraction_post(accumulator_dtype, result.dtype or "", None),
        accumulator_dtype=accumulator_dtype,
        store=TensorStore(
            result.id, result.shape, result.dtype or "", result.strides, True
        ),
        predicates=_predicates((*output_domains, reduction_domain)),
    )


def _normalized_dimension(dimension: int, rank: int) -> int:
    normalized = dimension + rank if dimension < 0 else dimension
    if not 0 <= normalized < rank:
        raise UnsupportedLowering(f"dimension {dimension} is invalid for rank {rank}")
    return normalized


def _concat_program(
    node: torch.fx.Node, op: Op, graph: ComputeGraph
) -> TensorProgram:
    result = _result_value(op, graph)
    if result.shape is None:
        raise UnsupportedLowering("concat result lacks a shape")
    operand_nodes = node.args[0]
    if not isinstance(operand_nodes, Sequence) or not operand_nodes:
        raise UnsupportedLowering("concat requires a nonempty operand list")
    axis = _normalized_dimension(node.args[1], len(result.shape))
    operands = tuple(_node_value(operand_node, graph) for operand_node in operand_nodes)
    if any(value.shape is None or value.strides is None for value in operands):
        raise UnsupportedLowering("concat operands need static shapes and strides")
    if any(len(value.shape) != len(result.shape) for value in operands):
        raise UnsupportedLowering("concat operands must share the result rank")
    for value in operands:
        for dimension, size in enumerate(value.shape or ()):
            if dimension != axis and size != result.shape[dimension]:
                raise UnsupportedLowering("concat operand shape does not match the result")
    if sum(value.shape[axis] for value in operands) != result.shape[axis]:
        raise UnsupportedLowering("concat operand extents do not cover the result")

    output_axes = tuple(f"o{index}" for index in range(len(result.shape)))
    offsets: list[int] = []
    offset = 0
    for value in operands:
        offsets.append(offset)
        offset += value.shape[axis]
    loads = tuple(
        _load(
            value,
            tuple(
                AxisAffine(
                    (AxisTerm(output_axes[dimension]),),
                    offset=-offsets[index],
                )
                if dimension == axis
                else output_axes[dimension]
                for dimension in range(len(output_axes))
            ),
            index,
        )
        for index, value in enumerate(operands)
    )
    expression: ScalarExpression = Load(loads[-1].id, loads[-1].dtype)
    for (value, load), offset in reversed(list(zip(zip(operands, loads), offsets))):
        coordinate = AxisIndex(axis=output_axes[axis], dtype="torch.int64")
        condition = Binary(
            "bitwise_and",
            Binary("ge", coordinate, _constant(offset), "torch.bool"),
            Binary(
                "lt",
                coordinate,
                _constant(offset + value.shape[axis]),
                "torch.bool",
            ),
            "torch.bool",
        )
        expression = Where(
            condition,
            Load(load.id, load.dtype),
            expression,
            result.dtype or "",
        )

    domains = _domains("o", result.shape)
    return TensorProgram(
        family="concat",
        source_target=str(node.target),
        decomposition="ordered multi-source materializing copy",
        iteration_domains=domains,
        reduction_axes=(),
        loads=loads,
        init=None,
        combine=expression,
        post=None,
        accumulator_dtype=None,
        store=TensorStore(
            result.id, result.shape, result.dtype or "", result.strides, True
        ),
        predicates=_predicates(domains),
    )


def _broadcast_shape(shapes: Sequence[tuple[int, ...]]) -> tuple[int, ...]:
    rank = len(shapes[0])
    if any(len(shape) != rank for shape in shapes):
        raise UnsupportedLowering("gather index tensors must have the same rank")
    result: list[int] = []
    for dimension in range(rank):
        sizes = {shape[dimension] for shape in shapes}
        if len(sizes) > 2 or (len(sizes) == 2 and 1 not in sizes):
            raise UnsupportedLowering("gather index tensors are not broadcastable")
        result.append(max(sizes))
    return tuple(result)


def _indexed_gather_program(
    node: torch.fx.Node,
    op: Op,
    graph: ComputeGraph,
    data: Value,
    index_values: tuple[Value, ...],
) -> TensorProgram:
    result = _result_value(op, graph)
    if data.shape is None or result.shape is None:
        raise UnsupportedLowering("gather operands need static shapes")
    if any(value.shape is None for value in index_values):
        raise UnsupportedLowering("gather indices need static shapes")
    if not index_values or len(index_values) > len(data.shape):
        raise UnsupportedLowering("gather index count does not match the source rank")

    index_shape = _broadcast_shape(tuple(value.shape for value in index_values))
    trailing_shape = data.shape[len(index_values) :]
    if result.shape != (*index_shape, *trailing_shape):
        raise UnsupportedLowering("gather result shape does not match its operands")

    output_axes = tuple(f"o{index}" for index in range(len(result.shape)))
    index_axes = output_axes[: len(index_shape)]
    index_loads = tuple(
        IndexedLoad(
            id=f"gather:index:{index}",
            value_id=index_value.id,
            axes=_broadcast_axes(index_value, index_shape),
            strides=index_value.strides or (),
            dtype=index_value.dtype or "",
            source_shape=index_value.shape,
        )
        for index, index_value in enumerate(index_values)
    )
    data_axes: tuple[str | int | AxisAffine | DataIndex, ...] = tuple(
        DataIndex(load.id) for load in index_loads
    ) + output_axes[len(index_shape) :]
    data_load = IndexedLoad(
        id="gather:data",
        value_id=data.id,
        axes=data_axes,
        strides=data.strides or (),
        dtype=data.dtype or "",
        source_shape=data.shape,
    )
    domains = tuple(
        IterationDomain(axis, size) for axis, size in zip(output_axes, result.shape)
    )
    return TensorProgram(
        family="gather",
        source_target=str(node.target),
        decomposition="source elements selected by broadcast integer indices",
        iteration_domains=domains,
        reduction_axes=(),
        loads=(*index_loads, data_load),
        init=None,
        combine=Load(data_load.id, data_load.dtype),
        post=None,
        accumulator_dtype=None,
        store=TensorStore(
            result.id, result.shape, result.dtype or "", result.strides, True
        ),
        predicates=_predicates(domains),
    )


def _embedding_program(
    node: torch.fx.Node, op: Op, graph: ComputeGraph
) -> TensorProgram:
    weight = _node_value(node.args[0], graph)
    indices = _node_value(node.args[1], graph)
    return _indexed_gather_program(node, op, graph, weight, (indices,))


def _index_program(node: torch.fx.Node, op: Op, graph: ComputeGraph) -> TensorProgram:
    source = _node_value(node.args[0], graph)
    index_nodes = node.args[1]
    if not isinstance(index_nodes, Sequence) or not index_nodes:
        raise UnsupportedLowering("index requires a nonempty index list")
    return _indexed_gather_program(
        node,
        op,
        graph,
        source,
        tuple(_node_value(index_node, graph) for index_node in index_nodes),
    )


def _softmax_statistics_program(
    node: torch.fx.Node, op: Op, graph: ComputeGraph
) -> CanonicalProgram:
    source = _node_value(node.args[0], graph)
    if source.shape is None or len(op.results) != 2:
        raise UnsupportedLowering("online softmax requires a tensor and two results")
    axis = _normalized_dimension(node.args[1], len(source.shape))
    for result_id in op.results:
        result = graph.values[result_id]
        expected_shape = tuple(
            1 if dimension == axis else size
            for dimension, size in enumerate(source.shape)
        )
        if result.shape != expected_shape:
            raise UnsupportedLowering("online softmax result shape does not match its axis")
    return CanonicalProgram(
        family="softmax_statistics",
        source_target=str(node.target),
        decomposition="online maximum and sum-exp statistics",
        result_roles=("max", "sum_exp"),
        semantics={
            "input": source.id,
            "axis": axis,
            "input_shape": source.shape,
            "numerical_policy": "online_softmax",
        },
    )


def _normalized_dims(dims: Any, rank: int) -> tuple[int, ...]:
    if dims is None:
        return tuple(range(rank))
    if isinstance(dims, int):
        values = [dims]
    elif isinstance(dims, Sequence):
        values = list(dims)
    else:
        raise UnsupportedLowering(f"unsupported reduction dimensions {dims!r}")
    normalized = [dimension + rank if dimension < 0 else dimension for dimension in values]
    if len(set(normalized)) != len(normalized) or any(
        dimension < 0 or dimension >= rank for dimension in normalized
    ):
        raise UnsupportedLowering(f"invalid reduction dimensions {dims!r}")
    return tuple(normalized)


def _reduction_program(
    node: torch.fx.Node,
    op: Op,
    graph: ComputeGraph,
    is_mean: bool,
    operator: str = "add",
    accumulator_dtype: str | None = None,
    post: ScalarExpression | None = None,
) -> TensorProgram:
    if not node.args:
        raise UnsupportedLowering("reduction has no arguments")
    source = _node_value(node.args[0], graph)
    result = _result_value(op, graph)
    if source.shape is None or result.shape is None:
        raise UnsupportedLowering("reduction operands need static shapes")
    dims = _normalized_dims(node.args[1] if len(node.args) > 1 else None, len(source.shape))
    keepdim = bool(node.args[2]) if len(node.args) > 2 else False
    output_domains = _domains("o", result.shape)
    reduction_domains = tuple(
        IterationDomain(f"r{index}", source.shape[dimension])
        for index, dimension in enumerate(dims)
    )
    if not dims:
        raise UnsupportedLowering("empty reduction is not supported")
    if operator == "maximum" and any(source.shape[dimension] == 0 for dimension in dims):
        raise UnsupportedLowering("amax requires a nonempty reduction extent")

    axes: list[str | int] = []
    output_dimension = 0
    for dimension in range(len(source.shape)):
        if dimension in dims:
            axes.append(f"r{dims.index(dimension)}")
        else:
            if keepdim:
                axes.append(f"o{dimension}")
            else:
                axes.append(f"o{output_dimension}")
                output_dimension += 1
    if keepdim:
        expected_output_shape = tuple(
            1 if dimension in dims else size
            for dimension, size in enumerate(source.shape)
        )
    else:
        expected_output_shape = tuple(
            size for dimension, size in enumerate(source.shape) if dimension not in dims
        )
    if result.shape != expected_output_shape:
        raise UnsupportedLowering("reduction output shape does not match its dimensions")

    load = _load(source, axes, 0)
    accumulator_dtype = accumulator_dtype or result.dtype or ""
    accumulator_load = _cast_load(Load(load.id, load.dtype), accumulator_dtype)
    combine = Binary(
        operator,
        Accumulator(accumulator_dtype),
        accumulator_load,
        accumulator_dtype,
    )
    reduction_size = 1
    for domain in reduction_domains:
        reduction_size *= domain.size
    decomposition = "ordered add reduction"
    if post is None and is_mean:
        post: ScalarExpression = Binary(
            "div",
            Cast(
                Accumulator(accumulator_dtype),
                accumulator_dtype,
                _cast_rounding(accumulator_dtype, accumulator_dtype),
            ),
            _constant(reduction_size),
            accumulator_dtype,
        )
        decomposition = "sum divided by the reduction extent"
    elif post is None:
        post = Cast(
            Accumulator(accumulator_dtype),
            accumulator_dtype,
            _cast_rounding(accumulator_dtype, accumulator_dtype),
        )
        decomposition = (
            "ordered maximum reduction" if operator == "maximum" else "ordered add reduction"
        )
    return TensorProgram(
        family="reduction",
        source_target=str(node.target),
        decomposition=decomposition,
        iteration_domains=(*output_domains, *reduction_domains),
        reduction_axes=tuple(domain.axis for domain in reduction_domains),
        loads=(load,),
        init=(
            _reduction_identity(accumulator_dtype, operator)
            if operator == "maximum"
            else _zero(accumulator_dtype)
        ),
        combine=combine,
        post=post,
        accumulator_dtype=accumulator_dtype,
        store=TensorStore(
            result.id, result.shape, result.dtype or "", result.strides, True
        ),
        predicates=_predicates((*output_domains, *reduction_domains)),
    )


def _amax_program(node: torch.fx.Node, op: Op, graph: ComputeGraph) -> TensorProgram:
    return _reduction_program(node, op, graph, is_mean=False, operator="maximum")


def _any_program(node: torch.fx.Node, op: Op, graph: ComputeGraph) -> TensorProgram:
    source = _node_value(node.args[0], graph)
    if source.dtype != "torch.bool":
        raise UnsupportedLowering("any.dim currently supports boolean tensors")
    return _reduction_program(
        node,
        op,
        graph,
        is_mean=False,
        accumulator_dtype="torch.int32",
        post=Binary(
            "gt",
            Accumulator("torch.int32"),
            _zero("torch.int32"),
            "torch.bool",
        ),
    )


def _reduction_identity(dtype: str, operator: str) -> Constant:
    if operator != "maximum":
        return _zero(dtype)
    if dtype in {
        "torch.float16",
        "torch.bfloat16",
        "torch.float32",
        "torch.float64",
    }:
        return Constant(float("-inf"), dtype)
    if dtype == "torch.bool":
        return Constant(False, dtype)
    torch_dtype = getattr(torch, dtype.removeprefix("torch."))
    return Constant(torch.iinfo(torch_dtype).min, dtype)


def _int_tuple(value: Any, name: str, length: int, minimum: int = 1) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or len(value) != length:
        raise UnsupportedLowering(f"{name} must have {length} values")
    values = tuple(value)
    if any(not isinstance(item, int) or item < minimum for item in values):
        raise UnsupportedLowering(f"{name} values must be integers of at least {minimum}")
    return values


def _convolution_program(
    node: torch.fx.Node, op: Op, graph: ComputeGraph
) -> TensorProgram:
    if len(node.args) != 9:
        raise UnsupportedLowering("convolution must have nine arguments")
    input_value = _node_value(node.args[0], graph)
    weight_value = _node_value(node.args[1], graph)
    bias_node = node.args[2]
    if input_value.shape is None or weight_value.shape is None:
        raise UnsupportedLowering("convolution operands lack static shapes")
    spatial = len(input_value.shape) - 2
    if spatial not in (1, 2) or len(weight_value.shape) != spatial + 2:
        raise UnsupportedLowering("only static 1-D and 2-D convolution is supported")
    stride = _int_tuple(node.args[3], "convolution stride", spatial)
    padding = _int_tuple(node.args[4], "convolution padding", spatial, minimum=0)
    dilation = _int_tuple(node.args[5], "convolution dilation", spatial)
    if node.args[6] or any(value for value in node.args[7]):
        raise UnsupportedLowering("transposed convolution is not supported")
    groups = node.args[8]
    if not isinstance(groups, int) or groups <= 0:
        raise UnsupportedLowering("convolution groups must be a positive integer")

    result = _result_value(op, graph)
    # A 1-D convolution is the 2-D one with a single row.
    batch, input_channels, *input_extent = input_value.shape
    output_channels, channels_per_group, *kernel = weight_value.shape
    if input_channels != channels_per_group * groups or output_channels % groups:
        raise UnsupportedLowering("convolution groups do not match its operands")
    output_shape = (
        batch,
        output_channels,
        *(
            (extent + 2 * pad - dilate * (size - 1) - 1) // step + 1
            for extent, pad, dilate, size, step in zip(
                input_extent, padding, dilation, kernel, stride, strict=True
            )
        ),
    )
    if result.shape != output_shape:
        raise UnsupportedLowering("convolution result shape does not match its operands")

    kernel_width = kernel[-1]
    kernel_extent = math.prod(kernel)
    channels_per_output_group = output_channels // groups
    width = _affine(
        AxisTerm(f"o{spatial + 1}", scale=stride[-1]),
        AxisTerm("r0", scale=dilation[-1], modulus=kernel_width),
        offset=-padding[-1],
    )
    if spatial == 2:
        kernel_height = kernel[0]
        input_spatial = (
            _affine(
                AxisTerm("o2", scale=stride[0]),
                AxisTerm("r0", scale=dilation[0], divisor=kernel_width, modulus=kernel_height),
                offset=-padding[0],
            ),
            width,
        )
        weight_spatial = (
            _affine(AxisTerm("r0", divisor=kernel_width, modulus=kernel_height)),
            _affine(AxisTerm("r0", modulus=kernel_width)),
        )
    else:
        input_spatial = (width,)
        weight_spatial = (_affine(AxisTerm("r0", modulus=kernel_width)),)
    input_load = _load(
        input_value,
        (
            AxisAffine((AxisTerm("o0"),)),
            _affine(
                AxisTerm("o1", scale=channels_per_group, divisor=channels_per_output_group),
                AxisTerm("r0", divisor=kernel_extent),
            ),
            *input_spatial,
        ),
        0,
    )
    weight_load = _load(
        weight_value,
        ("o1", _affine(AxisTerm("r0", divisor=kernel_extent)), *weight_spatial),
        1,
    )
    loads = [input_load, weight_load]
    bias_load: IndexedLoad | None = None
    if bias_node is not None:
        bias_value = _node_value(bias_node, graph)
        if bias_value.shape != (output_channels,):
            raise UnsupportedLowering("convolution bias shape does not match its output channels")
        bias_load = _load(bias_value, ("o1",), 2)
        loads.append(bias_load)

    accumulator_dtype = _accumulator_dtype(
        result.dtype or "",
        [input_value.dtype or "", weight_value.dtype or ""],
    )
    product = Binary(
        "mul",
        _cast_load(Load(input_load.id, input_load.dtype), accumulator_dtype),
        _cast_load(Load(weight_load.id, weight_load.dtype), accumulator_dtype),
        accumulator_dtype,
    )
    combine = Binary(
        "add", Accumulator(accumulator_dtype), product, accumulator_dtype
    )
    post = _contraction_post(
        accumulator_dtype,
        result.dtype or "",
        bias_load,
    )
    output_domains = _domains("o", result.shape)
    reduction_domains = (
        IterationDomain("r0", channels_per_group * kernel_extent),
    )
    return TensorProgram(
        family="contraction",
        source_target=str(node.target),
        decomposition=(
            "indexed multiply-accumulate with padding, stride, dilation, and groups"
        ),
        iteration_domains=(*output_domains, *reduction_domains),
        reduction_axes=("r0",),
        loads=tuple(loads),
        init=_zero(accumulator_dtype),
        combine=combine,
        post=post,
        accumulator_dtype=accumulator_dtype,
        store=TensorStore(
            result.id, result.shape, result.dtype or "", result.strides, True
        ),
        predicates=_predicates((*output_domains, *reduction_domains)),
    )


def _conv2d_program(node: torch.fx.Node, op: Op, graph: ComputeGraph) -> TensorProgram:
    if not 3 <= len(node.args) <= 8:
        raise UnsupportedLowering("conv2d must have three to eight arguments")
    args = list(node.args)
    args.extend([None] * (8 - len(args)))
    input_node, weight_node, bias_node, stride, padding, dilation, _, groups = args
    convolution_node = copy.copy(node)
    convolution_node.target = torch.ops.aten.convolution.default
    convolution_node.args = (
        input_node,
        weight_node,
        bias_node,
        stride or (1, 1),
        padding or (0, 0),
        dilation or (1, 1),
        False,
        (0, 0),
        groups or 1,
    )
    return _convolution_program(convolution_node, op, graph)


def _avg_pool2d_program(
    node: torch.fx.Node, op: Op, graph: ComputeGraph
) -> TensorProgram:
    if not 2 <= len(node.args) <= 7:
        raise UnsupportedLowering("avg_pool2d must have two to seven arguments")
    args = list(node.args)
    args.extend([None] * (7 - len(args)))
    _, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override = args[:7]
    ceil_mode = bool(ceil_mode)
    count_include_pad = True if count_include_pad is None else bool(count_include_pad)
    if ceil_mode or not count_include_pad or divisor_override is not None:
        raise UnsupportedLowering(
            "only count_include_pad avg_pool2d without ceil mode is supported"
        )
    kernel = _int_tuple(kernel_size, "avg_pool2d kernel", 2)
    pool_stride = _int_tuple(stride or kernel_size, "avg_pool2d stride", 2)
    pool_padding = _int_tuple(padding or (0, 0), "avg_pool2d padding", 2)
    source = _node_value(node.args[0], graph)
    result = _result_value(op, graph)
    if source.shape is None or len(source.shape) != 4:
        raise UnsupportedLowering("avg_pool2d requires a static 4-D input")
    batch, channels, input_height, input_width = source.shape
    output_shape = (
        batch,
        channels,
        (input_height + 2 * pool_padding[0] - kernel[0]) // pool_stride[0] + 1,
        (input_width + 2 * pool_padding[1] - kernel[1]) // pool_stride[1] + 1,
    )
    if result.shape != output_shape:
        raise UnsupportedLowering("avg_pool2d result shape does not match its input")

    kernel_extent = kernel[0] * kernel[1]
    load = _load(
        source,
        (
            "o0",
            "o1",
            _affine(
                AxisTerm("o2", scale=pool_stride[0]),
                AxisTerm("r0", divisor=kernel[1], modulus=kernel[0]),
                offset=-pool_padding[0],
            ),
            _affine(
                AxisTerm("o3", scale=pool_stride[1]),
                AxisTerm("r0", modulus=kernel[1]),
                offset=-pool_padding[1],
            ),
        ),
        0,
    )
    accumulator_dtype = _accumulator_dtype(result.dtype or "", [source.dtype or ""])
    combine = Binary(
        "add",
        Accumulator(accumulator_dtype),
        _cast_load(Load(load.id, load.dtype), accumulator_dtype),
        accumulator_dtype,
    )
    post = Cast(
        Binary(
            "div",
            Accumulator(accumulator_dtype),
            _constant(kernel_extent),
            accumulator_dtype,
        ),
        result.dtype or "",
        _cast_rounding(accumulator_dtype, result.dtype or ""),
    )
    output_domains = _domains("o", result.shape)
    reduction_domains = (IterationDomain("r0", kernel_extent),)
    return TensorProgram(
        family="reduction",
        source_target=str(node.target),
        decomposition="indexed window sum divided by the window extent",
        iteration_domains=(*output_domains, *reduction_domains),
        reduction_axes=("r0",),
        loads=(load,),
        init=_zero(accumulator_dtype),
        combine=combine,
        post=post,
        accumulator_dtype=accumulator_dtype,
        store=TensorStore(
            result.id, result.shape, result.dtype or "", result.strides, True
        ),
        predicates=_predicates((*output_domains, *reduction_domains)),
    )


def _gather_program(node: torch.fx.Node, op: Op, graph: ComputeGraph) -> TensorProgram:
    if len(node.args) != 3:
        raise UnsupportedLowering("gather must have three arguments")
    source = _node_value(node.args[0], graph)
    dimension = _normalized_dimension(node.args[1], len(source.shape or ()))
    index = _node_value(node.args[2], graph)
    result = _result_value(op, graph)
    if source.shape is None or index.shape is None or result.shape is None:
        raise UnsupportedLowering("gather operands need static shapes")
    if index.shape != result.shape or source.shape != result.shape:
        raise UnsupportedLowering("gather operands and result must have the same shape")
    if index.id not in graph.inputs:
        raise UnsupportedLowering("data-dependent gather requires a runtime index input")
    if index.dtype not in {
        "torch.int32",
        "torch.int64",
    }:
        raise UnsupportedLowering("gather indices must be int32 or int64")

    output_axes = tuple(f"o{dimension_index}" for dimension_index in range(len(result.shape)))
    index_load = _load(index, output_axes, 0)
    source_axes = tuple(
        DataIndex(index_load.id) if dimension_index == dimension else axis
        for dimension_index, axis in enumerate(output_axes)
    )
    source_load = _load(source, source_axes, 1)
    return TensorProgram(
        family="elementwise",
        source_target=str(node.target),
        decomposition="data-indexed gather with a validated runtime index domain",
        iteration_domains=_domains("o", result.shape),
        reduction_axes=(),
        loads=(index_load, source_load),
        init=None,
        combine=Load(source_load.id, source_load.dtype),
        post=None,
        accumulator_dtype=None,
        store=TensorStore(
            result.id, result.shape, result.dtype or "", result.strides, True
        ),
        predicates=_predicates(_domains("o", result.shape)),
    )


def _kv_attention_program(node: torch.fx.Node, op: Op, graph: ComputeGraph) -> TensorProgram:
    del node
    if len(op.operands) != 6:
        raise UnsupportedLowering("KV attention expects source, K, V, two caches, and length")
    source, new_key, new_value, key_cache, value_cache, valid_length = (
        graph.values[value_id] for value_id in op.operands
    )
    output = _result_value(op, graph)
    if any(
        value.shape is None or value.strides is None or value.dtype is None
        for value in (source, new_key, new_value, key_cache, value_cache, valid_length)
    ):
        raise UnsupportedLowering("KV attention operands lack static tensor metadata")

    batch, query_length, query_heads, head_dim = source.shape
    _, new_length, kv_heads, _ = new_key.shape
    _, capacity, cache_kv_heads, _ = key_cache.shape
    group_size = query_heads // kv_heads
    contiguous_cache = (
        capacity * kv_heads * head_dim,
        kv_heads * head_dim,
        head_dim,
        1,
    )
    if (
        query_length != 1
        or new_length != query_length
        or new_key.shape != new_value.shape
        or key_cache.shape != value_cache.shape
        or cache_kv_heads != kv_heads
        or query_heads % kv_heads
        or group_size <= 0
        or key_cache.strides != contiguous_cache
        or value_cache.strides != contiguous_cache
        or valid_length.shape != (1,)
        or valid_length.dtype != "torch.int32"
        or output.shape != (batch * query_heads, query_length, head_dim)
        or source.dtype != "torch.bfloat16"
        or new_key.dtype != "torch.bfloat16"
        or key_cache.dtype != "torch.bfloat16"
    ):
        raise UnsupportedLowering(
            "KV attention supports one-token decode over two contiguous BF16 caches"
        )

    batch_head = AxisAffine((AxisTerm("bh", divisor=query_heads),))
    query_head = AxisAffine((AxisTerm("bh", modulus=query_heads),))
    kv_head = AxisAffine(
        (AxisTerm("bh", divisor=query_heads, modulus=group_size),)
    )
    loads = (
        IndexedLoad(
            "attention:q", source.id, (batch_head, "s", query_head, "d"),
            source.strides, source.dtype, source.shape,
        ),
        IndexedLoad(
            "attention:new_k", new_key.id, (batch_head, 0, kv_head, "d"),
            new_key.strides, new_key.dtype, new_key.shape,
        ),
        IndexedLoad(
            "attention:new_v", new_value.id, (batch_head, 0, kv_head, "d"),
            new_value.strides, new_value.dtype, new_value.shape,
        ),
        IndexedLoad(
            "attention:k", key_cache.id, (batch_head, "k", kv_head, "d"),
            key_cache.strides, key_cache.dtype, key_cache.shape,
        ),
        IndexedLoad(
            "attention:v", value_cache.id, (batch_head, "k", kv_head, "d"),
            value_cache.strides, value_cache.dtype, value_cache.shape,
        ),
        IndexedLoad(
            "attention:valid_length", valid_length.id, (0,),
            valid_length.strides, valid_length.dtype, valid_length.shape,
        ),
    )
    zero = Constant(0.0, "torch.float32")
    state_access = KVCacheAccess(
        key_cache_value_id=key_cache.id,
        value_cache_value_id=value_cache.id,
        valid_length_value_id=valid_length.id,
        key_cache_load_id="attention:k",
        value_cache_load_id="attention:v",
        new_key_load_id="attention:new_k",
        new_value_load_id="attention:new_v",
        valid_length_load_id="attention:valid_length",
        capacity=capacity,
        query_length=query_length,
        key_layout=KVCacheLayout(key_cache.shape, key_cache.strides, key_cache.dtype),
        value_layout=KVCacheLayout(
            value_cache.shape, value_cache.strides, value_cache.dtype
        ),
        storage_identity=(key_cache.id, value_cache.id),
    )
    return TensorProgram(
        family="attention",
        source_target="megabake.kv_attention",
        decomposition="stateful decode attention with bounded contiguous KV cache",
        iteration_domains=(
            IterationDomain("bh", batch * query_heads),
            IterationDomain("s", query_length),
            IterationDomain("d", head_dim),
            IterationDomain("k", capacity),
        ),
        reduction_axes=("k",),
        loads=loads,
        init=zero,
        combine=zero,
        post=zero,
        accumulator_dtype="torch.float32",
        store=TensorStore(
            output.id, output.shape, output.dtype, output.strides, True
        ),
        predicates=(
            AxisPredicate("bh", batch * query_heads),
            AxisPredicate("s", query_length),
            AxisPredicate("d", head_dim),
        ),
        state_accesses=(state_access,),
    )


def attention_access_axes(
    head_axis: int, batch_axis: AxisAffine, head: AxisAffine, sequence_axis: str
) -> tuple[AxisExpression, ...]:
    if head_axis == 2:
        return (batch_axis, sequence_axis, head, "d")
    return (batch_axis, head, sequence_axis, "d")


def attention_tensor_program(
    output_id: str,
    q_root: str,
    k_root: str,
    v_root: str,
    q: Any,
    k: Any,
    v: Any,
    axes: tuple[int, int, int],
    mask_value: str | None,
    graph: ComputeGraph,
) -> TensorProgram | None:
    output = graph.values[output_id]
    if output.shape is None or output.dtype is None or output.strides is None:
        return None
    if q.shape is None or k.shape is None or v.shape is None:
        return None
    q_head_axis, k_head_axis, v_head_axis = axes
    batch = q.shape[0]
    query_heads = q.shape[q_head_axis]
    query_length = q.shape[3 - q_head_axis]
    kv_heads = k.shape[k_head_axis]
    kv_length = k.shape[3 - k_head_axis]
    head_dim = q.shape[-1]
    group_size = query_heads // kv_heads
    batch_head = AxisAffine((AxisTerm("bh", divisor=query_heads),))
    query_head = AxisAffine((AxisTerm("bh", modulus=query_heads),))
    kv_head = AxisAffine(
        (AxisTerm("bh", divisor=query_heads, modulus=group_size),)
    )

    loads = (
        IndexedLoad(
            "attention:q",
            q_root,
            attention_access_axes(q_head_axis, batch_head, query_head, "s"),
            q.strides or (),
            q.dtype or "",
            q.shape,
        ),
        IndexedLoad(
            "attention:k",
            k_root,
            attention_access_axes(k_head_axis, batch_head, kv_head, "k"),
            k.strides or (),
            k.dtype or "",
            k.shape,
        ),
        IndexedLoad(
            "attention:v",
            v_root,
            attention_access_axes(v_head_axis, batch_head, kv_head, "k"),
            v.strides or (),
            v.dtype or "",
            v.shape,
        ),
    )
    if mask_value is not None:
        mask = graph.values[mask_value]
        if mask.shape is None or mask.strides is None or mask.dtype is None:
            return None
        mask_head = 0 if mask.shape[1] == 1 else query_head
        loads = (
            *loads,
            IndexedLoad(
                "attention:mask",
                mask_value,
                (batch_head, mask_head, "s", "k"),
                mask.strides,
                mask.dtype,
                mask.shape,
            ),
        )
    zero = Constant(0.0, "torch.float32")
    return TensorProgram(
        family="attention",
        source_target="megabake.attention",
        decomposition="stable softmax attention with grouped heads and typed masks",
        iteration_domains=(
            IterationDomain("bh", batch * query_heads),
            IterationDomain("s", query_length),
            IterationDomain("d", head_dim),
            IterationDomain("k", kv_length),
        ),
        reduction_axes=("k",),
        loads=loads,
        init=zero,
        combine=zero,
        post=zero,
        accumulator_dtype="torch.float32",
        store=TensorStore(
            output.id,
            output.shape,
            output.dtype,
            output.strides,
            True,
        ),
        predicates=(
            AxisPredicate("bh", batch * query_heads),
            AxisPredicate("s", query_length),
            AxisPredicate("d", head_dim),
        ),
    )


def _attention_program(node: torch.fx.Node, op: Op, graph: ComputeGraph) -> TensorProgram:
    query, key, value = (_node_value(argument, graph) for argument in node.args[:3])
    mask = node.args[3]
    if mask is not None and not isinstance(mask, torch.fx.Node):
        raise UnsupportedLowering("attention mask must be a tensor or None")
    if any(
        operand.shape is None or len(operand.shape) != 4 or operand.dtype != "torch.bfloat16"
        for operand in (query, key, value)
    ):
        raise UnsupportedLowering("attention supports four-dimensional BF16 operands")
    if query.shape[2] % key.shape[2] or key.shape != value.shape:
        raise UnsupportedLowering("attention needs whole KV-head groups and matching K and V")
    # Operands are [batch, sequence, heads, dim] by the op contract, so the
    # head axis is known and no stride heuristic is needed.
    program = attention_tensor_program(
        _result_value(op, graph).id,
        query.id,
        key.id,
        value.id,
        query,
        key,
        value,
        (2, 2, 2),
        None if mask is None else _node_value(mask, graph).id,
        graph,
    )
    if program is None:
        raise UnsupportedLowering("attention operands lack static layouts")
    return program


LOWERERS: dict[Any, LoweringRule] = {
    KV_ATTENTION_OP: _kv_attention_program,
    ATTENTION_OP: _attention_program,
    torch.ops.aten.add.Tensor: lambda node, op, graph: _binary_program(node, op, graph, "add"),
    torch.ops.aten.add.Scalar: lambda node, op, graph: _binary_program(node, op, graph, "add"),
    torch.ops.aten.sub.Tensor: lambda node, op, graph: _binary_program(node, op, graph, "sub"),
    torch.ops.aten.mul.Tensor: lambda node, op, graph: _binary_program(node, op, graph, "mul"),
    torch.ops.aten.mul.Scalar: lambda node, op, graph: _binary_program(node, op, graph, "mul"),
    torch.ops.aten.div.Tensor: lambda node, op, graph: _binary_program(node, op, graph, "div"),
    torch.ops.aten.pow.Tensor_Scalar: lambda node, op, graph: _binary_program(
        node, op, graph, "pow"
    ),
    torch.ops.aten.bitwise_and.Tensor: lambda node, op, graph: _binary_program(
        node, op, graph, "bitwise_and"
    ),
    torch.ops.aten.eq.Tensor: lambda node, op, graph: _binary_program(node, op, graph, "eq"),
    torch.ops.aten.ne.Tensor: lambda node, op, graph: _binary_program(node, op, graph, "ne"),
    torch.ops.aten.lt.Tensor: lambda node, op, graph: _binary_program(node, op, graph, "lt"),
    torch.ops.aten.le.Tensor: lambda node, op, graph: _binary_program(node, op, graph, "le"),
    torch.ops.aten.gt.Tensor: lambda node, op, graph: _binary_program(node, op, graph, "gt"),
    torch.ops.aten.ge.Tensor: lambda node, op, graph: _binary_program(node, op, graph, "ge"),
    torch.ops.aten.eq.Scalar: lambda node, op, graph: _binary_program(node, op, graph, "eq"),
    torch.ops.aten.ne.Scalar: lambda node, op, graph: _binary_program(node, op, graph, "ne"),
    torch.ops.aten.lt.Scalar: lambda node, op, graph: _binary_program(node, op, graph, "lt"),
    torch.ops.aten.le.Scalar: lambda node, op, graph: _binary_program(node, op, graph, "le"),
    torch.ops.aten.gt.Scalar: lambda node, op, graph: _binary_program(node, op, graph, "gt"),
    torch.ops.aten.ge.Scalar: lambda node, op, graph: _binary_program(node, op, graph, "ge"),
    torch.ops.aten.where.self: _where_program,
    torch.ops.aten.relu.default: lambda node, op, graph: _unary_program(node, op, graph, "relu"),
    torch.ops.aten.neg.default: lambda node, op, graph: _unary_program(node, op, graph, "neg"),
    torch.ops.aten.logical_not.default: lambda node, op, graph: _unary_program(
        node, op, graph, "logical_not"
    ),
    torch.ops.aten.abs.default: lambda node, op, graph: _unary_program(node, op, graph, "abs"),
    torch.ops.aten.sigmoid.default: lambda node, op, graph: _unary_program(node, op, graph, "sigmoid"),
    torch.ops.aten.tanh.default: lambda node, op, graph: _unary_program(node, op, graph, "tanh"),
    torch.ops.aten.rsqrt.default: lambda node, op, graph: _unary_program(node, op, graph, "rsqrt"),
    torch.ops.aten.exp.default: lambda node, op, graph: _unary_program(node, op, graph, "exp"),
    torch.ops.aten.cos.default: lambda node, op, graph: _unary_program(node, op, graph, "cos"),
    **{
        getattr(torch.ops.aten, name).default: (
            lambda node, op, graph, name=name: _unary_program(node, op, graph, name)
        )
        for name in (
            "sqrt", "reciprocal", "exp2", "expm1", "log", "log1p", "log2",
            "erf", "floor", "ceil", "trunc",
        )
    },
    **{
        getattr(torch.ops.aten, name).default: (
            lambda node, op, graph, name=name: _logical_program(node, op, graph, name)
        )
        for name in ("logical_and", "logical_or", "logical_xor")
    },
    torch.ops.aten.bitwise_or.Tensor: lambda node, op, graph: _binary_program(
        node, op, graph, "bitwise_or"
    ),
    torch.ops.aten.bitwise_xor.Tensor: lambda node, op, graph: _binary_program(
        node, op, graph, "bitwise_xor"
    ),
    torch.ops.aten.maximum.default: lambda node, op, graph: _binary_program(
        node, op, graph, "maximum"
    ),
    torch.ops.aten.minimum.default: lambda node, op, graph: _binary_program(
        node, op, graph, "minimum"
    ),
    torch.ops.aten.clamp_min.default: lambda node, op, graph: _binary_program(
        node, op, graph, "maximum"
    ),
    torch.ops.aten.clamp_max.default: lambda node, op, graph: _binary_program(
        node, op, graph, "minimum"
    ),
    torch.ops.aten.select.int: _view_program,
    torch.ops.aten.slice_scatter.default: _slice_scatter_program,
    torch.ops.aten.select_scatter.default: _select_scatter_program,
    torch.ops.aten.constant_pad_nd.default: _constant_pad_program,
    torch.ops.aten.copy.default: _functional_copy_program,
    torch.ops.aten.copy_.default: _state_write_program,
    torch.ops.aten.squeeze.dim: _view_program,
    torch.ops.aten.squeeze.dims: _view_program,
    torch.ops.aten.squeeze.default: _view_program,
    torch.ops.aten.alias.default: _view_program,
    torch.ops.aten.sin.default: lambda node, op, graph: _unary_program(node, op, graph, "sin"),
    torch.ops.aten._to_copy.default: _cast_program,
    torch.ops.aten.to.dtype: _cast_program,
    torch.ops.prims.convert_element_type.default: _cast_program,
    torch.ops.aten.view.dtype: _view_program,
    torch.ops.aten.reshape.default: _view_program,
    torch.ops.aten.expand.default: _view_program,
    torch.ops.aten.unsqueeze.default: _view_program,
    torch.ops.aten.slice.Tensor: _view_program,
    torch.ops.aten.clone.default: _copy_program,
    torch.ops.aten.lift_fresh_copy.default: _copy_program,
    torch.ops.aten.t.default: _view_program,
    torch.ops.aten.transpose.int: _view_program,
    torch.ops.aten.permute.default: _view_program,
    torch.ops.aten.mm.default: _mm_program,
    torch.ops.aten.bmm.default: _bmm_program,
    torch.ops.aten.addmm.default: _addmm_program,
    torch.ops.aten.linear.default: _linear_program,
    torch.ops.aten.convolution.default: _convolution_program,
    torch.ops.aten.conv2d.default: _conv2d_program,
    torch.ops.aten.avg_pool2d.default: _avg_pool2d_program,
    torch.ops.aten.gather.default: _gather_program,
    torch.ops.aten.sum.dim_IntList: lambda node, op, graph: _reduction_program(node, op, graph, False),
    torch.ops.aten.mean.dim: lambda node, op, graph: _reduction_program(node, op, graph, True),
    torch.ops.aten.sum.default: lambda node, op, graph: _reduction_program(node, op, graph, False),
    torch.ops.aten.mean.default: lambda node, op, graph: _reduction_program(node, op, graph, True),
    torch.ops.aten.amax.default: _amax_program,
    torch.ops.aten.any.dim: _any_program,
    operator.getitem: _tuple_projection_program,
    torch.ops.aten.cat.default: _concat_program,
    torch.ops.aten.embedding.default: _embedding_program,
    torch.ops.aten.index.Tensor: _index_program,
    torch.ops.aten.full.default: _constant_program,
    torch.ops.aten.scalar_tensor.default: _constant_program,
    torch.ops.prims.iota.default: _iota_program,
    "prims.prepare_softmax_online.default": _softmax_statistics_program,
}


def _expression_loads(expression: ScalarExpression) -> set[str]:
    if isinstance(expression, Load):
        return {expression.load_id}
    if isinstance(expression, (Constant, Accumulator, AxisIndex)):
        return set()
    if isinstance(expression, Unary):
        return _expression_loads(expression.operand)
    if isinstance(expression, Binary):
        return _expression_loads(expression.lhs) | _expression_loads(expression.rhs)
    if isinstance(expression, Where):
        return (
            _expression_loads(expression.condition)
            | _expression_loads(expression.true_value)
            | _expression_loads(expression.false_value)
        )
    if isinstance(expression, Cast):
        return _expression_loads(expression.operand)
    raise UnsupportedLowering(f"unknown scalar expression {expression!r}")


def _expression_has_accumulator(expression: ScalarExpression) -> bool:
    if isinstance(expression, Accumulator):
        return True
    if isinstance(expression, (Constant, Load, AxisIndex)):
        return False
    if isinstance(expression, Unary):
        return _expression_has_accumulator(expression.operand)
    if isinstance(expression, Binary):
        return _expression_has_accumulator(expression.lhs) or _expression_has_accumulator(expression.rhs)
    if isinstance(expression, Where):
        return any(
            _expression_has_accumulator(value)
            for value in (
                expression.condition,
                expression.true_value,
                expression.false_value,
            )
        )
    if isinstance(expression, Cast):
        return _expression_has_accumulator(expression.operand)
    raise UnsupportedLowering(f"unknown scalar expression {expression!r}")


def _verify_tensor_program(program: TensorProgram, op: Op, graph: ComputeGraph) -> None:
    axes = {domain.axis: domain.size for domain in program.iteration_domains}
    if len(axes) != len(program.iteration_domains):
        raise UnsupportedLowering("tensor program has duplicate iteration axes")
    if set(program.reduction_axes) - set(axes):
        raise UnsupportedLowering("tensor program has an unbound reduction axis")
    if program.store.value_id not in op.results:
        raise UnsupportedLowering("tensor program stores a non-result value")
    result = graph.values[program.store.value_id]
    if (
        program.store.shape != result.shape
        or program.store.dtype != result.dtype
        or program.store.strides != result.strides
    ):
        raise UnsupportedLowering("tensor program store does not match its result metadata")

    load_ids = {load.id for load in program.loads}
    if len(load_ids) != len(program.loads):
        raise UnsupportedLowering("tensor program has duplicate load IDs")
    for load in program.loads:
        if load.value_id not in graph.values:
            raise UnsupportedLowering(f"tensor program loads missing value {load.value_id}")
        if load.value_id not in op.operands:
            raise UnsupportedLowering(f"load {load.id} does not come from an operation operand")
        source = graph.values[load.value_id]
        if len(load.axes) != len(load.strides):
            raise UnsupportedLowering(f"load {load.id} rank does not match its value")
        if program.family != "view" and len(load.axes) != len(source.shape or ()):
            raise UnsupportedLowering(f"load {load.id} rank does not match its source value")
        if source.shape is not None and source.strides is None:
            raise UnsupportedLowering(f"load {load.id} source lacks strides")
        maximum_address = 0
        if load.source_shape is not None and source.shape != load.source_shape:
            raise UnsupportedLowering(f"load {load.id} source shape does not match its value")
        for axis, stride in zip(load.axes, load.strides):
            if isinstance(axis, DataIndex):
                if axis.load_id not in load_ids:
                    raise UnsupportedLowering(
                        f"load {load.id} uses an unknown data index"
                    )
                continue
            if isinstance(axis, AxisAffine):
                if set(axis_references(axis)) - set(axes):
                    raise UnsupportedLowering(f"load {load.id} uses an unbound affine axis")
                continue
            if isinstance(axis, str):
                if axis not in axes:
                    raise UnsupportedLowering(f"load {load.id} uses unbound axis {axis}")
                maximum_address += max(axes[axis] - 1, 0) * abs(stride)
            elif axis >= 0:
                maximum_address += axis * abs(stride)
            else:
                raise UnsupportedLowering(f"load {load.id} has a negative constant index")
        source_extent = 1
        for size, stride in zip(source.shape or (), source.strides or ()):
            source_extent += max(size - 1, 0) * abs(stride)
        if (
            program.family != "concat"
            and source_extent
            and maximum_address >= source_extent
        ):
            raise UnsupportedLowering(f"load {load.id} may read outside its source storage")
    if _expression_loads(program.combine) - load_ids:
        raise UnsupportedLowering("combine expression uses an unknown load")
    if isinstance(program.combine, AxisIndex) and program.combine.axis not in axes:
        raise UnsupportedLowering("axis index uses an unbound iteration axis")
    if program.post is not None and _expression_loads(program.post) - load_ids:
        raise UnsupportedLowering("post expression uses an unknown load")
    if isinstance(program.post, AxisIndex) and program.post.axis not in axes:
        raise UnsupportedLowering("axis index uses an unbound iteration axis")
    if program.family in {"reduction", "contraction"}:
        if not program.reduction_axes or program.init is None or program.post is None:
            raise UnsupportedLowering("reduction lacks an accumulator or result expression")
        if program.accumulator_dtype is None:
            raise UnsupportedLowering("reduction lacks an accumulator dtype")
        if program.init.dtype != program.accumulator_dtype:
            raise UnsupportedLowering("reduction initializer and accumulator dtypes differ")
        if not _expression_has_accumulator(program.combine):
            raise UnsupportedLowering("reduction combine expression does not use the accumulator")
        if program.post is not None:
            loads_by_id = {load.id: load for load in program.loads}
            for load_id in _expression_loads(program.post):
                if any(
                    set(axis_references(axis)) & set(program.reduction_axes)
                    for axis in loads_by_id[load_id].axes
                ):
                    raise UnsupportedLowering("post expression depends on a reduction coordinate")
    elif program.family == "attention":
        if program.reduction_axes != ("k",) or program.init is None:
            raise UnsupportedLowering("attention lacks its reduction state")
        _verify_state_accesses(program, op, graph, load_ids)
    elif program.reduction_axes or program.init is not None or program.post is not None:
        raise UnsupportedLowering("elementwise program unexpectedly has reduction state")
    if program.family not in {"reduction", "contraction"} and _expression_has_accumulator(
        program.combine
    ):
        raise UnsupportedLowering("elementwise combine expression uses the accumulator")


def _verify_state_accesses(
    program: TensorProgram, op: Op, graph: ComputeGraph, load_ids: set[str]
) -> None:
    if len(program.state_accesses) > 1:
        raise UnsupportedLowering("an attention program may own at most one KV cache")
    loads_by_id = {load.id: load for load in program.loads}
    for access in program.state_accesses:
        required_load_ids = {
            access.key_cache_load_id,
            access.value_cache_load_id,
            access.new_key_load_id,
            access.new_value_load_id,
            access.valid_length_load_id,
        }
        if not required_load_ids <= load_ids:
            raise UnsupportedLowering("KV state access names an unknown load")
        value_ids = {
            access.key_cache_value_id,
            access.value_cache_value_id,
            access.valid_length_value_id,
        }
        if not value_ids <= set(op.operands):
            raise UnsupportedLowering("KV state access names a non-operand value")
        key = graph.values[access.key_cache_value_id]
        value = graph.values[access.value_cache_value_id]
        if (
            access.capacity <= 0
            or access.query_length <= 0
            or access.capacity < access.query_length
            or key.shape != access.key_layout.shape
            or key.strides != access.key_layout.strides
            or value.shape != access.value_layout.shape
            or value.strides != access.value_layout.strides
            or access.storage_class != "bounded_contiguous"
            or access.valid_length_specialized
        ):
            raise UnsupportedLowering("KV state access has an invalid ownership contract")
        if loads_by_id[access.valid_length_load_id].value_id != access.valid_length_value_id:
            raise UnsupportedLowering("KV valid-length load is bound to the wrong value")


def _verify_canonical_program(
    program: CanonicalProgram, op: Op, graph: ComputeGraph
) -> None:
    if len(program.result_roles) != len(op.results):
        raise UnsupportedLowering("canonical result roles do not match results")
    if len(set(program.result_roles)) != len(program.result_roles):
        raise UnsupportedLowering("canonical program has duplicate result roles")
    if program.family == "concat":
        operands = program.semantics.get("operands")
        if (
            not isinstance(operands, tuple)
            or not operands
            or not set(operands) <= set(op.operands)
        ):
            raise UnsupportedLowering("concat semantics do not match its operands")
    elif program.family == "gather":
        operand_ids = set(op.operands)
        referenced_ids = {
            value_id
            for key in ("data", "indices", "source")
            for value_id in (
                program.semantics[key]
                if isinstance(program.semantics.get(key), tuple)
                else (program.semantics.get(key),)
            )
            if value_id is not None
        }
        if not referenced_ids or not referenced_ids <= operand_ids:
            raise UnsupportedLowering("gather semantics reference non-operand values")
    elif program.family == "softmax_statistics":
        if program.semantics.get("input") not in op.operands:
            raise UnsupportedLowering("softmax semantics reference a non-operand value")
    else:
        raise UnsupportedLowering(f"unknown canonical family {program.family}")


def _verify_program(
    program: TensorProgram | CanonicalProgram, op: Op, graph: ComputeGraph
) -> None:
    if isinstance(program, TensorProgram):
        _verify_tensor_program(program, op, graph)
    else:
        _verify_canonical_program(program, op, graph)


def lower_semantics(graph: ComputeGraph) -> ComputeGraph:
    lowered_ops: dict[str, Op] = {}
    for op_id, operation in graph.ops.items():
        source_id = operation.source_fx_ids[0]
        node = graph.source_nodes[source_id]
        target = node.target
        rule = (
            _attribute_constant_program
            if node.op == "get_attr"
            else LOWERERS.get(target) or LOWERERS.get(str(target))
        )
        declared_effects = target in (KV_ATTENTION_OP, torch.ops.aten.copy_.default)
        unsupported_effects = not declared_effects and bool(
            operation.effects.get("unknown")
            or operation.effects.get("mutates_inputs")
            or operation.effects.get("may_mutate_inputs")
        )
        reasons: list[str] = []
        if rule is None:
            reasons.append(f"missing lowering rule for FX target {target}")
        if unsupported_effects:
            reasons.append("unsupported effects; device planning is blocked")
        if reasons:
            reason = "; ".join(reasons)
            tensor_program = None
        else:
            try:
                tensor_program = rule(node, operation, graph)
                _verify_program(tensor_program, operation, graph)
                reason = ""
            except UnsupportedLowering as error:
                tensor_program = None
                reason = f"unsupported lowering for FX target {target}: {error}"
            except Exception as error:
                tensor_program = None
                reason = (
                    f"lowering rule failed for FX target {target}: "
                    f"{type(error).__name__}: {error}"
                )
        known = tensor_program is not None
        lowered_ops[op_id] = replace(
            operation,
            tensor_program=tensor_program,
            numerical_rules={
                "known": known,
                **({"reason": reason} if reason else {}),
            },
        )
    lowered = replace(graph, ops=lowered_ops)
    from .graph import verify_compute_graph

    verify_compute_graph(lowered)
    return lowered


CANONICAL_TARGETS = frozenset(
    {
        operator.getitem,
        torch.ops.aten.bitwise_and.Tensor,
        torch.ops.aten.bmm.default,
        torch.ops.aten.cat.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.cos.default,
        torch.ops.aten.exp.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.full.default,
        torch.ops.prims.iota.default,
        torch.ops.aten.pow.Tensor_Scalar,
        "prims.prepare_softmax_online.default",
        torch.ops.aten.rsqrt.default,
        torch.ops.aten.scalar_tensor.default,
        torch.ops.aten.sin.default,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.unsqueeze.default,
    }
)


def canonicalize_semantics(graph: ComputeGraph) -> ComputeGraph:
    lowered = lower_semantics(graph)
    canonical_ops: dict[str, Op] = {}
    for op_id, operation in lowered.ops.items():
        target = lowered.source_nodes[operation.source_fx_ids[0]].target
        program = operation.tensor_program
        canonical = program is not None and (
            target in CANONICAL_TARGETS
            or str(target) in CANONICAL_TARGETS
            or isinstance(program, CanonicalProgram)
        )
        result_roles = (
            program.result_roles
            if isinstance(program, CanonicalProgram)
            else ("result",)
            if program is not None
            else ()
        )
        numerical_rules = {
            **operation.numerical_rules,
            "canonical": canonical,
            "result_roles": result_roles,
        }
        canonical_ops[op_id] = replace(
            operation, numerical_rules=numerical_rules
        )
    canonicalized = replace(lowered, ops=canonical_ops)
    from .graph import verify_compute_graph

    verify_compute_graph(canonicalized)
    return canonicalized


def semantic_coverage(graph: ComputeGraph) -> dict[str, int]:
    canonical = sum(
        bool(operation.numerical_rules.get("canonical"))
        for operation in graph.ops.values()
    )
    known = sum(operation.tensor_program is not None for operation in graph.ops.values())
    return {
        "canonical": canonical,
        "generic": known - canonical,
        "unresolved": len(graph.ops) - known,
    }


def dump_lowering_graph(graph: ComputeGraph, path: str | Path) -> Path:
    coverage = semantic_coverage(graph)
    known = coverage["canonical"] + coverage["generic"]
    lines = [
        "# LoweringGraph",
        f"values={len(graph.values)} operations={len(graph.ops)} "
        f"known={known} unknown={len(graph.ops) - known} "
        f"canonical={coverage['canonical']} generic={coverage['generic']} "
        f"unresolved={coverage['unresolved']} "
        f"inputs={len(graph.inputs)} outputs={len(graph.outputs)}",
        "",
        "## Operations",
    ]
    for operation in graph.ops.values():
        program = operation.tensor_program
        lines.append(
            f"{operation.id} source={list(operation.source_fx_ids)} "
            f"target={operation.attributes.get('target')} "
            f"operands={list(operation.operands)} "
            f"results={list(operation.results)} "
            f"family={program.family if program is not None else None} "
            f"known={program is not None} "
            f"canonical={bool(operation.numerical_rules.get('canonical'))} "
            f"reason={operation.numerical_rules.get('reason', '')!r}"
        )

    lines.extend(["", "## Tensor Programs"])
    for operation in graph.ops.values():
        program = operation.tensor_program
        if program is None:
            continue
        if isinstance(program, CanonicalProgram):
            lines.extend(
                [
                    f"### {operation.id}",
                    f"source_target={program.source_target}",
                    f"decomposition={program.decomposition}",
                    f"result_roles={list(program.result_roles)}",
                    f"semantics={dict(program.semantics)!r}",
                    "",
                ]
            )
            continue
        lines.extend(
            [
                f"### {operation.id}",
                f"source_target={program.source_target}",
                f"decomposition={program.decomposition}",
                f"iteration_domains={list(program.iteration_domains)!r}",
                f"reduction_axes={list(program.reduction_axes)}",
                f"loads={list(program.loads)!r}",
                f"init={program.init!r}",
                f"combine={program.combine!r}",
                f"post={program.post!r}",
                f"accumulator_dtype={program.accumulator_dtype}",
                f"store={program.store!r}",
                f"predicates={list(program.predicates)!r}",
                "",
            ]
        )

    lines.extend(
        [
            "## Inputs",
            *[str(value_id) for value_id in graph.inputs],
            "",
            "## Outputs",
            *[str(value_id) for value_id in graph.outputs],
            "",
            "## Output Reconstruction",
            repr(graph.output_reconstruction),
            "",
        ]
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
