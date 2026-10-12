from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..lowering import (
    AxisIndex,
    Binary,
    Cast,
    Constant,
    Load,
    ScalarExpression,
    TensorProgram,
    Unary,
    Where,
)
from .interface import BodyConfig, validate_body_config


@dataclass(frozen=True)
class GeneratedMapBody:
    """A device-callable map body and its required work-iterator encoding."""

    source: str
    function_name: str = "run"
    threads_per_cta: int = 128
    tile_elements: int = 128
    tile_count: int = 1
    work_iterator_dtype: str = "torch.int64"
    work_iterator_encoding: str = "one flat output-tile base index per tile"
    arguments: tuple[str, ...] = (
        "work_iterator",
        "tensor_bindings",
        "scratch",
        "epilogue",
    )


@dataclass(frozen=True)
class _TypedExpression:
    code: str
    dtype: str


class UnsupportedMapBody(ValueError):
    pass


_CUTLASS_TYPES = {
    "torch.bool": "cutlass.Boolean",
    "torch.int8": "cutlass.Int8",
    "torch.int16": "cutlass.Int16",
    "torch.int32": "cutlass.Int32",
    "torch.int64": "cutlass.Int64",
    "torch.float16": "cutlass.Float16",
    "torch.bfloat16": "cutlass.BFloat16",
    "torch.float32": "cutlass.Float32",
    "torch.float64": "cutlass.Float64",
}

_BINARY_SYMBOLS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "bitwise_and": "&",
    "eq": "==",
    "ne": "!=",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
}


def emit_map_body(program: TensorProgram, config: BodyConfig) -> GeneratedMapBody:
    _validate_map_body(program, config)

    threads = config.participants.participating_threads
    axis_coordinates = {
        domain.axis: f"o{index}" for index, domain in enumerate(program.iteration_domains)
    }
    load_indices = {load.id: index for index, load in enumerate(program.loads)}
    value = _emit_expression(
        program.combine, load_indices, axis_coordinates, program.loads
    )
    value = _cast_expression(value, program.store.dtype)

    coordinate_lines = _coordinate_lines(program.store.shape)
    output_coordinates = ", ".join(
        axis_coordinates[domain.axis] for domain in program.iteration_domains
    )
    binding_type = ", ".join(
        "cute.Tensor" for _ in range(len(program.loads) + 1)
    )
    source = f'''import cutlass
import cutlass.cute as cute


@cute.kernel
def run(
    work_iterator: cute.Tensor,
    tensor_bindings: tuple[{binding_type}],
    scratch,
    epilogue,
):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    workers, _, _ = cute.arch.grid_dim()
    output = tensor_bindings[{len(program.loads)}]
    output_count = cute.size(output)
    for tile in cutlass.range(block, cute.size(work_iterator), workers):
        linear_index = work_iterator[tile] + thread
        if linear_index < output_count:
{coordinate_lines}            value = {value.code}
            output[{output_coordinates}] = value
'''
    element_count = 1
    for size in program.store.shape:
        element_count *= size
    tile_count = (element_count + threads - 1) // threads
    return GeneratedMapBody(
        source=source,
        threads_per_cta=threads,
        tile_elements=threads,
        tile_count=tile_count,
    )


def _validate_map_body(program: TensorProgram, config: BodyConfig) -> None:
    validate_body_config(config)
    if program.family not in {"elementwise", "copy"}:
        raise UnsupportedMapBody(
            f"map body generator supports elementwise and copy, not {program.family}"
        )
    if config.semantic.operation_family != program.family:
        raise UnsupportedMapBody("body config and tensor program families differ")
    if program.reduction_axes or program.init is not None or program.post is not None:
        raise UnsupportedMapBody("map body generator does not support reduction state")
    if program.accumulator_dtype is not None:
        raise UnsupportedMapBody("map body generator does not support an accumulator")

    threads = config.participants.participating_threads
    if threads <= 0:
        raise UnsupportedMapBody("map body requires participating threads")
    if config.tiles.cta_shape != (threads,) + (1,) * (
        len(program.iteration_domains) - 1
    ):
        raise UnsupportedMapBody("map body requires a flat one-element-per-thread tile")
    if config.tiles.work_partition != "one output element per thread":
        raise UnsupportedMapBody("map body requires one output element per thread")
    if config.scratch.owner != "none" or config.scratch.size_bytes != 0:
        raise UnsupportedMapBody("map body does not use scratch")

    if not program.store.shape or any(size <= 0 for size in program.store.shape):
        raise UnsupportedMapBody("map body requires a nonempty positive-rank output")
    if any(stride < 0 for stride in program.store.strides):
        raise UnsupportedMapBody("negative output strides are not supported")

    expected_input_ids = {load.value_id for load in program.loads}
    if set(config.layouts.inputs) != expected_input_ids:
        raise UnsupportedMapBody("body input layouts do not match tensor program loads")
    if set(config.layouts.outputs) != {program.store.value_id}:
        raise UnsupportedMapBody("body output layouts do not match tensor program store")

    expected_output_axes = tuple(
        domain.axis for domain in program.iteration_domains
    )
    output_layout = config.layouts.outputs[program.store.value_id]
    if (
        output_layout.axes != expected_output_axes
        or output_layout.strides != tuple(program.store.strides)
        or output_layout.dtype != program.store.dtype
    ):
        raise UnsupportedMapBody("body output layout does not match tensor program")

    for load in program.loads:
        layout = config.layouts.inputs[load.value_id]
        if (
            layout.axes != load.axes
            or layout.strides != tuple(load.strides)
            or layout.dtype != load.dtype
        ):
            raise UnsupportedMapBody("body input layout does not match tensor program")
        if any(stride < 0 for stride in load.strides):
            raise UnsupportedMapBody("negative input strides are not supported")


def _coordinate_lines(shape: tuple[int, ...]) -> str:
    lines: list[str] = []
    remainder = "linear_index"
    for index, size in enumerate(shape):
        coordinate = f"o{index}"
        lines.append(f"            {coordinate} = {remainder} % {size}\n")
        if index < len(shape) - 1:
            lines.append(f"            {remainder} = {remainder} // {size}\n")
    return "".join(lines)


def _emit_expression(
    expression: ScalarExpression,
    load_indices: Mapping[str, int],
    axis_coordinates: Mapping[str, str],
    loads: tuple,
) -> _TypedExpression:
    if isinstance(expression, Load):
        index = load_indices.get(expression.load_id)
        if index is None:
            raise UnsupportedMapBody(f"unknown load {expression.load_id}")
        load = loads[index]
        coordinates = []
        for axis in load.axes:
            coordinates.append(
                axis_coordinates[axis] if isinstance(axis, str) else str(axis)
            )
        return _TypedExpression(
            f"tensor_bindings[{index}][{', '.join(coordinates)}]", load.dtype
        )
    if isinstance(expression, Constant):
        return _TypedExpression(
            f"{_cutlass_type(expression.dtype)}({expression.value!r})",
            expression.dtype,
        )
    if isinstance(expression, AxisIndex):
        coordinate = axis_coordinates.get(expression.axis)
        if coordinate is None:
            raise UnsupportedMapBody(f"unknown axis {expression.axis}")
        return _cast_expression(
            _TypedExpression(coordinate, "torch.int64"), expression.dtype
        )
    if isinstance(expression, Unary):
        operand = _cast_expression(
            _emit_expression(
                expression.operand, load_indices, axis_coordinates, loads
            ),
            expression.dtype,
        )
        return _TypedExpression(
            _unary_code(expression.operator, operand), expression.dtype
        )
    if isinstance(expression, Binary):
        left = _emit_expression(expression.lhs, load_indices, axis_coordinates, loads)
        right = _emit_expression(expression.rhs, load_indices, axis_coordinates, loads)
        if expression.operator in {"eq", "ne", "lt", "le", "gt", "ge"}:
            comparison_dtype = _comparison_dtype(left.dtype, right.dtype)
            left = _cast_expression(left, comparison_dtype)
            right = _cast_expression(right, comparison_dtype)
        else:
            left = _cast_expression(left, expression.dtype)
            right = _cast_expression(right, expression.dtype)
        symbol = _BINARY_SYMBOLS.get(expression.operator)
        if symbol is None:
            raise UnsupportedMapBody(f"unsupported binary operator {expression.operator}")
        code = (
            f"cute.pow({left.code}, {right.code})"
            if expression.operator == "pow"
            else f"({left.code} {symbol} {right.code})"
        )
        return _TypedExpression(code, expression.dtype)
    if isinstance(expression, Where):
        condition = _emit_expression(
            expression.condition, load_indices, axis_coordinates, loads
        )
        condition = _cast_expression(condition, "torch.bool")
        true_value = _cast_expression(
            _emit_expression(
                expression.true_value, load_indices, axis_coordinates, loads
            ),
            expression.dtype,
        )
        false_value = _cast_expression(
            _emit_expression(
                expression.false_value, load_indices, axis_coordinates, loads
            ),
            expression.dtype,
        )
        return _TypedExpression(
            f"cute.where({condition.code}, {true_value.code}, {false_value.code})",
            expression.dtype,
        )
    if isinstance(expression, Cast):
        return _cast_expression(
            _emit_expression(
                expression.operand, load_indices, axis_coordinates, loads
            ),
            expression.dtype,
        )
    raise UnsupportedMapBody(f"unsupported scalar expression {expression!r}")


def _unary_code(operator: str, operand: _TypedExpression) -> str:
    if operator == "relu":
        zero = f"{_cutlass_type(operand.dtype)}(0)"
        return f"cute.where(({operand.code}) > {zero}, ({operand.code}), {zero})"
    if operator == "neg":
        return f"(-({operand.code}))"
    functions = {
        "abs": "cute.abs",
        "sigmoid": "sigmoid",
        "tanh": "cute.tanh",
        "rsqrt": "cute.rsqrt",
        "exp": "cute.exp",
        "cos": "cute.cos",
        "sin": "cute.sin",
    }
    function = functions.get(operator)
    if function is None:
        raise UnsupportedMapBody(f"unsupported unary operator {operator}")
    if function == "sigmoid":
        one = f"{_cutlass_type(operand.dtype)}(1.0)"
        return f"({one} / ({one} + cute.exp(-({operand.code}))))"
    return f"{function}({operand.code})"


def _cast_expression(value: _TypedExpression, dtype: str) -> _TypedExpression:
    cutlass_type = _cutlass_type(dtype)
    if value.dtype == dtype:
        return value
    return _TypedExpression(f"({value.code}).to({cutlass_type})", dtype)


def _cutlass_type(dtype: str) -> str:
    try:
        return _CUTLASS_TYPES[dtype]
    except KeyError as error:
        raise UnsupportedMapBody(f"unsupported dtype {dtype}") from error


def _comparison_dtype(left: str, right: str) -> str:
    if left == right:
        return left
    if _is_float(left) and _is_float(right):
        if {left, right} == {"torch.float16", "torch.bfloat16"}:
            return "torch.float32"
        return _larger_float(left, right)
    if _is_float(left):
        return left
    if _is_float(right):
        return right
    if left == "torch.bool":
        return right
    if right == "torch.bool":
        return left
    return _larger_integer(left, right)


def _is_float(dtype: str) -> bool:
    return dtype in {
        "torch.float16",
        "torch.bfloat16",
        "torch.float32",
        "torch.float64",
    }


def _float_rank(dtype: str) -> int:
    return ["torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"].index(
        dtype
    )


def _integer_rank(dtype: str) -> int:
    return ["torch.bool", "torch.int8", "torch.int16", "torch.int32", "torch.int64"].index(
        dtype
    )


def _larger_float(left: str, right: str) -> str:
    return left if _float_rank(left) > _float_rank(right) else right


def _larger_integer(left: str, right: str) -> str:
    return left if _integer_rank(left) > _integer_rank(right) else right
