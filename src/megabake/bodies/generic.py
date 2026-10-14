from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ..lowering import (
    AxisAffine,
    AxisIndex,
    DataIndex,
    Accumulator,
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
class GeneratedReductionBody:
    """A device-callable reduction body and its work-iterator encoding."""

    source: str
    function_name: str = "run"
    threads_per_cta: int = 128
    reduction_tile: int = 128
    tile_count: int = 1
    scratch_elements: int = 4
    work_iterator_dtype: str = "torch.int64"
    work_iterator_encoding: str = "one flat output index per CTA"
    scratch_encoding: str = "scratch[block, warp] holds one accumulator partial per warp"
    phases: tuple[str, ...] = ("thread_partial", "warp_partial", "cta_merge", "epilogue")
    phase_descriptions: Mapping[str, str] = field(
        default_factory=lambda: {
            "thread_partial": "each thread accumulates bounded reduction strips",
            "warp_partial": "lane zero stores one deterministic warp partial",
            "cta_merge": "all threads merge warp partials in warp order",
            "epilogue": "thread zero applies the post expression and stores",
        }
    )
    arguments: tuple[str, ...] = (
        "work_iterator",
        "tensor_bindings",
        "scratch",
        "epilogue",
    )


@dataclass(frozen=True)
class GeneratedContractionBody:
    """A device-callable baseline contraction and its work-iterator encoding."""

    source: str
    function_name: str = "run"
    threads_per_cta: int = 128
    output_tile_elements: int = 128
    reduction_chunk: int = 32
    tile_count: int = 1
    work_iterator_dtype: str = "torch.int64"
    work_iterator_encoding: str = (
        "one flat output-tile base index per CTA; thread adds its lane"
    )
    phases: tuple[str, ...] = (
        "output_tile",
        "bounded_reduction",
        "epilogue",
    )
    phase_descriptions: Mapping[str, str] = field(
        default_factory=lambda: {
            "output_tile": "each thread owns one predicated output coordinate",
            "bounded_reduction": (
                "the owning thread accumulates indexed products over "
                "fixed-size reduction chunks with a tail predicate"
            ),
            "epilogue": "the post expression applies the final cast and store",
        }
    )
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


UnsupportedReductionBody = UnsupportedMapBody
UnsupportedContractionBody = UnsupportedMapBody


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
    input_value_ids = tuple(dict.fromkeys(load.value_id for load in program.loads))
    binding_indices = {value_id: index for index, value_id in enumerate(input_value_ids)}
    load_indices = {load.id: binding_indices[load.value_id] for load in program.loads}
    value = _emit_expression(
        program.combine, load_indices, axis_coordinates, program.loads
    )
    value = _cast_expression(value, program.store.dtype)

    coordinate_lines = _coordinate_lines(program.store.shape, program.store.strides)
    output_coordinates = ", ".join(
        axis_coordinates[domain.axis] for domain in program.iteration_domains
    )
    if program.source_target == "rope":
        return _emit_rope_map_body(
            program,
            threads,
            axis_coordinates,
            coordinate_lines,
            output_coordinates,
        )
    binding_type = ", ".join(
        "cute.Tensor" for _ in range(len(input_value_ids) + 1)
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
    output = tensor_bindings[{len(input_value_ids)}]
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


def _emit_rope_map_body(
    program: TensorProgram,
    threads: int,
    axis_coordinates: Mapping[str, str],
    coordinate_lines: str,
    output_coordinates: str,
) -> GeneratedMapBody:
    final_axis = program.iteration_domains[-1].axis
    final_coordinate = axis_coordinates[final_axis]
    rotary_width = program.loads[3].source_shape[-1] if program.loads[3].source_shape else None
    if rotary_width is None:
        domain_sizes = {
            domain.axis: domain.size for domain in program.iteration_domains
        }
        rotary_width = domain_sizes[final_axis]
    half_width = rotary_width // 2
    current = _load_access(program.loads[0], 0, program.loads, axis_coordinates)
    plus = _load_access(program.loads[1], 0, program.loads, axis_coordinates)
    minus = _load_access(program.loads[2], 0, program.loads, axis_coordinates)
    cosine = _load_access(program.loads[3], 1, program.loads, axis_coordinates)
    sine = _load_access(program.loads[4], 2, program.loads, axis_coordinates)
    output_type = _cutlass_type(program.store.dtype)
    round_call = (
        "cvt_f32_bf16"
        if program.store.dtype == "torch.bfloat16"
        else "cutlass.Float32"
    )
    conversion_import = (
        "from cutlass.cute.arch.nvvm_wrappers import cvt_f32_bf16\n"
        if program.store.dtype == "torch.bfloat16"
        else ""
    )
    source = f'''import cutlass
import cutlass.cute as cute
{conversion_import}


@cute.kernel
def run(
    work_iterator: cute.Tensor,
    tensor_bindings: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
    scratch,
    epilogue,
):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    workers, _, _ = cute.arch.grid_dim()
    output = tensor_bindings[3]
    output_count = cute.size(output)
    for tile in cutlass.range(block, cute.size(work_iterator), workers):
        linear_index = work_iterator[tile] + thread
        if linear_index < output_count:
{coordinate_lines}            current_value = {current.code}
            plus_value = {plus.code}
            minus_value = {minus.code}
            cosine_value = {cosine.code}
            sine_value = {sine.code}
            value = current_value
            if {final_coordinate} < {half_width}:
                left_product = {round_call}(cutlass.Float32(current_value * cosine_value).ir_value())
                right_product = {round_call}(cutlass.Float32(plus_value * sine_value).ir_value())
                value = {round_call}(cutlass.Float32(left_product - right_product).ir_value())
            elif {final_coordinate} < {rotary_width}:
                left_product = {round_call}(cutlass.Float32(current_value * sine_value).ir_value())
                right_product = {round_call}(cutlass.Float32(minus_value * cosine_value).ir_value())
                value = {round_call}(cutlass.Float32(left_product + right_product).ir_value())
            else:
                value = current_value
            output[{output_coordinates}] = {output_type}(value)
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


def emit_reduction_body(
    program: TensorProgram, config: BodyConfig
) -> GeneratedReductionBody:
    _validate_reduction_body(program, config)

    threads = config.participants.participating_threads
    warps = threads // 32
    output_domains = [
        domain
        for domain in program.iteration_domains
        if domain.axis not in program.reduction_axes
    ]
    reduction_domain = next(
        domain for domain in program.iteration_domains if domain.axis in program.reduction_axes
    )
    axis_coordinates = {
        domain.axis: f"o{index}" for index, domain in enumerate(output_domains)
    }
    reduction_coordinate = f"r{program.reduction_axes.index(reduction_domain.axis)}"
    input_value_ids = tuple(dict.fromkeys(load.value_id for load in program.loads))
    binding_indices = {value_id: index for index, value_id in enumerate(input_value_ids)}
    load_indices = {load.id: binding_indices[load.value_id] for load in program.loads}

    accumulator = _emit_expression(program.init, load_indices, axis_coordinates, program.loads)
    combine = _emit_expression(
        program.combine,
        load_indices,
        {**axis_coordinates, reduction_domain.axis: reduction_coordinate},
        program.loads,
    )
    assert program.post is not None
    post = _emit_expression(program.post, load_indices, axis_coordinates, program.loads)

    coordinate_lines = _coordinate_lines(
        program.store.shape, program.store.strides, "output_index"
    )
    output_coordinates = ", ".join(
        axis_coordinates[domain.axis] for domain in output_domains
    )
    binding_type = ", ".join(
        "cute.Tensor" for _ in range(len(input_value_ids) + 1)
    )
    accumulator_type = _cutlass_type(program.accumulator_dtype or "")
    source = f'''import cutlass
import cutlass.cute as cute


@cute.kernel
def run(
    work_iterator: cute.Tensor,
    tensor_bindings: tuple[{binding_type}],
    scratch: cute.Tensor,
    epilogue,
):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    workers, _, _ = cute.arch.grid_dim()
    warp = thread // 32
    lane = thread % 32
    warps = {warps}
    output = tensor_bindings[{len(input_value_ids)}]
    output_count = cute.size(output)
    for tile in cutlass.range(block, cute.size(work_iterator), workers):
        output_index = work_iterator[tile]
        if output_index < output_count:
{coordinate_lines}            accumulator = {accumulator_type}({accumulator.code})
            for reduction_index in cutlass.range(thread, {reduction_domain.size}, {threads}):
                {reduction_coordinate} = reduction_index
                accumulator = {combine.code}
            warp_partial = cute.arch.warp_reduction_sum(accumulator)
            if lane == 0:
                scratch[block, warp] = warp_partial
            cute.arch.sync_threads()
            accumulator = {accumulator_type}({accumulator.code})
            for warp_index in cutlass.range(warps):
                accumulator = accumulator + scratch[block, warp_index]
            value = {post.code}
            if thread == 0:
                output[{output_coordinates}] = value
            cute.arch.sync_threads()
'''
    output_count = 1
    for size in program.store.shape:
        output_count *= size
    return GeneratedReductionBody(
        source=source,
        threads_per_cta=threads,
        reduction_tile=threads,
        tile_count=output_count,
        scratch_elements=warps,
    )


def emit_contraction_body(
    program: TensorProgram, config: BodyConfig
) -> GeneratedContractionBody:
    _validate_contraction_body(program, config)

    threads = config.participants.participating_threads
    reduction_chunk = config.tiles.reduction_tile[0]
    output_domains = [
        domain
        for domain in program.iteration_domains
        if domain.axis not in program.reduction_axes
    ]
    reduction_domain = next(
        domain
        for domain in program.iteration_domains
        if domain.axis in program.reduction_axes
    )
    axis_coordinates = {
        domain.axis: f"o{index}" for index, domain in enumerate(output_domains)
    }
    reduction_coordinate = f"r{program.reduction_axes.index(reduction_domain.axis)}"
    load_indices = {load.id: index for index, load in enumerate(program.loads)}

    accumulator = _emit_expression(
        program.init, load_indices, axis_coordinates, program.loads
    )
    combine = _emit_expression(
        program.combine,
        load_indices,
        {**axis_coordinates, reduction_domain.axis: reduction_coordinate},
        program.loads,
    )
    assert program.post is not None
    post = _emit_expression(
        program.post, load_indices, axis_coordinates, program.loads
    )

    coordinate_lines = _coordinate_lines(
        program.store.shape, program.store.strides
    )
    output_coordinates = ", ".join(
        axis_coordinates[domain.axis] for domain in output_domains
    )
    binding_type = ", ".join(
        "cute.Tensor" for _ in range(len(program.loads) + 1)
    )
    accumulator_type = _cutlass_type(program.accumulator_dtype or "")
    output_cutlass_type = _cutlass_type(program.store.dtype)
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
{coordinate_lines}            accumulator = {accumulator_type}({accumulator.code})
            for reduction_base in cutlass.range(0, {reduction_domain.size}, {reduction_chunk}):
                for reduction_offset in cutlass.range({reduction_chunk}):
                    reduction_index = reduction_base + reduction_offset
                    if reduction_index < {reduction_domain.size}:
                        {reduction_coordinate} = reduction_index
                        accumulator = {combine.code}
            stored_value = {output_cutlass_type}({post.code})
            output[{output_coordinates}] = stored_value
'''
    element_count = 1
    for size in program.store.shape:
        element_count *= size
    tile_count = (element_count + threads - 1) // threads
    return GeneratedContractionBody(
        source=source,
        threads_per_cta=threads,
        output_tile_elements=threads,
        reduction_chunk=reduction_chunk,
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

    _validate_body_layouts(program, config, program.iteration_domains)


def _validate_reduction_body(program: TensorProgram, config: BodyConfig) -> None:
    validate_body_config(config)
    # shortcut: one reduction axis and ordered addition only, add more when a captured fixture needs them.
    if program.family != "reduction":
        raise UnsupportedReductionBody(
            f"reduction body generator supports reduction, not {program.family}"
        )
    if config.semantic.operation_family != program.family:
        raise UnsupportedReductionBody("body config and tensor program families differ")
    if len(program.reduction_axes) != 1:
        raise UnsupportedReductionBody(
            "reduction body generator supports one reduction axis"
        )
    if program.init is None or program.post is None or program.accumulator_dtype is None:
        raise UnsupportedReductionBody("reduction body requires complete reduction state")
    if not _is_add_reduction(program.combine):
        raise UnsupportedReductionBody("reduction body generator supports ordered addition")
    if program.init.dtype != program.accumulator_dtype:
        raise UnsupportedReductionBody("initializer and accumulator dtypes differ")

    threads = config.participants.participating_threads
    warps = threads // 32
    if threads <= 0 or threads % 32:
        raise UnsupportedReductionBody("reduction body requires whole warps")
    if config.tiles.cta_shape != (threads,) + (1,) * (
        len(program.iteration_domains) - len(program.reduction_axes) - 1
    ):
        raise UnsupportedReductionBody("reduction body requires one CTA per output row")
    if config.tiles.work_partition != "one output row per CTA":
        raise UnsupportedReductionBody("reduction body requires one output row per CTA")
    if config.tiles.reduction_tile != (threads,):
        raise UnsupportedReductionBody("reduction body requires one reduction strip per thread")
    if config.scratch.owner != "cta" or config.scratch.size_bytes < warps * _dtype_bytes(
        program.accumulator_dtype
    ):
        raise UnsupportedReductionBody("reduction body lacks warp-partial scratch")

    output_domains = tuple(
        domain
        for domain in program.iteration_domains
        if domain.axis not in program.reduction_axes
    )
    if not output_domains or any(domain.size <= 0 for domain in output_domains):
        raise UnsupportedReductionBody("reduction body requires a nonempty output row")
    if any(domain.size <= 0 for domain in program.iteration_domains):
        raise UnsupportedReductionBody("reduction body requires a nonempty reduction extent")
    if any(stride < 0 for stride in program.store.strides):
        raise UnsupportedReductionBody("negative output strides are not supported")

    _validate_body_layouts(program, config, output_domains)


def _validate_contraction_body(program: TensorProgram, config: BodyConfig) -> None:
    validate_body_config(config)
    if program.family != "contraction":
        raise UnsupportedContractionBody(
            f"contraction body generator supports contraction, not {program.family}"
        )
    if config.semantic.operation_family != program.family:
        raise UnsupportedContractionBody(
            "body config and tensor program families differ"
        )
    if len(program.reduction_axes) != 1:
        raise UnsupportedContractionBody(
            "contraction body generator supports one reduction axis"
        )
    if program.init is None or program.post is None or program.accumulator_dtype is None:
        raise UnsupportedContractionBody(
            "contraction body requires complete reduction state"
        )
    if not _is_add_reduction(program.combine):
        raise UnsupportedContractionBody(
            "contraction body generator supports ordered multiply-accumulate"
        )
    if program.init.dtype != program.accumulator_dtype:
        raise UnsupportedContractionBody(
            "initializer and accumulator dtypes differ"
        )

    threads = config.participants.participating_threads
    if threads <= 0:
        raise UnsupportedContractionBody("contraction body requires participating threads")
    output_domains = tuple(
        domain
        for domain in program.iteration_domains
        if domain.axis not in program.reduction_axes
    )
    if config.tiles.cta_shape != (threads,) + (1,) * (len(output_domains) - 1):
        raise UnsupportedContractionBody(
            "contraction body requires a flat one-output-element-per-thread tile"
        )
    if config.tiles.work_partition != "one output element per thread":
        raise UnsupportedContractionBody(
            "contraction body requires one output element per thread"
        )
    if (
        not config.tiles.reduction_tile
        or config.tiles.reduction_tile[0] <= 0
        or len(config.tiles.reduction_tile) != 1
    ):
        raise UnsupportedContractionBody(
            "contraction body requires one positive reduction chunk"
        )
    if config.scratch.owner != "none" or config.scratch.size_bytes != 0:
        raise UnsupportedContractionBody("contraction body does not use scratch")

    if not program.store.shape or any(size <= 0 for size in program.store.shape):
        raise UnsupportedContractionBody(
            "contraction body requires a nonempty positive-rank output"
        )
    if any(domain.size <= 0 for domain in program.iteration_domains):
        raise UnsupportedContractionBody(
            "contraction body requires a nonempty positive reduction extent"
        )
    if any(stride < 0 for stride in program.store.strides):
        raise UnsupportedContractionBody(
            "negative output strides are not supported"
        )

    _validate_body_layouts(program, config, output_domains)


def _validate_body_layouts(
    program: TensorProgram,
    config: BodyConfig,
    output_domains: tuple,
) -> None:
    expected_input_ids = {load.value_id for load in program.loads}
    if set(config.layouts.inputs) != expected_input_ids:
        raise UnsupportedMapBody("body input layouts do not match tensor program loads")
    if set(config.layouts.outputs) != {program.store.value_id}:
        raise UnsupportedMapBody("body output layouts do not match tensor program store")

    output_layout = config.layouts.outputs[program.store.value_id]
    if (
        output_layout.axes != tuple(domain.axis for domain in output_domains)
        or output_layout.strides != tuple(program.store.strides)
        or output_layout.dtype != program.store.dtype
    ):
        raise UnsupportedMapBody("body output layout does not match tensor program")

    loads_by_value: dict[str, list[Any]] = {}
    for load in program.loads:
        loads_by_value.setdefault(load.value_id, []).append(load)
    for load in program.loads:
        layout = config.layouts.inputs[load.value_id]
        if (
            layout.strides != tuple(load.strides)
            or layout.dtype != load.dtype
        ):
            raise UnsupportedMapBody("body input layout does not match tensor program")
        primary_load = loads_by_value[load.value_id][0]
        if load is primary_load and layout.axes != primary_load.axes:
            raise UnsupportedMapBody("body input layout does not match tensor program")
        if any(
            other.source_shape != load.source_shape
            or other.strides != load.strides
            or other.dtype != load.dtype
            for other in loads_by_value[load.value_id]
        ):
            raise UnsupportedMapBody(
                "repeated loads of one value must share shape, strides, and dtype"
            )
        if any(stride < 0 for stride in load.strides):
            raise UnsupportedMapBody("negative input strides are not supported")


def _is_add_reduction(expression: ScalarExpression) -> bool:
    if not isinstance(expression, Binary) or expression.operator != "add":
        return False
    left = _expression_has_accumulator(expression.lhs)
    right = _expression_has_accumulator(expression.rhs)
    return left != right


def _expression_has_accumulator(expression: ScalarExpression) -> bool:
    if isinstance(expression, Accumulator):
        return True
    if isinstance(expression, Binary):
        return _expression_has_accumulator(expression.lhs) or _expression_has_accumulator(
            expression.rhs
        )
    if isinstance(expression, Unary):
        return _expression_has_accumulator(expression.operand)
    if isinstance(expression, Cast):
        return _expression_has_accumulator(expression.operand)
    if isinstance(expression, Where):
        return any(
            _expression_has_accumulator(value)
            for value in (expression.condition, expression.true_value, expression.false_value)
        )
    return False


def _coordinate_lines(
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    value: str = "linear_index",
) -> str:
    if len(shape) != len(strides):
        raise UnsupportedMapBody("output shape and strides do not match")
    lines: list[str] = []
    remainder = value
    axes = sorted(
        range(len(shape)),
        key=lambda index: (strides[index] == 0, strides[index], index),
    )
    for position, index in enumerate(axes):
        size = shape[index]
        coordinate = f"o{index}"
        lines.append(f"            {coordinate} = {remainder} % {size}\n")
        if position < len(axes) - 1:
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
        load = next(
            candidate for candidate in loads if candidate.id == expression.load_id
        )
        return _load_access(load, index, loads, axis_coordinates)

    if isinstance(expression, Constant):
        return _TypedExpression(
            f"{_cutlass_type(expression.dtype)}({expression.value!r})",
            expression.dtype,
        )
    if isinstance(expression, Accumulator):
        return _TypedExpression("accumulator", expression.dtype)
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
        if expression.operator == "pow":
            code = f"cute.pow({left.code}, {right.code})"
        else:
            symbol = _BINARY_SYMBOLS.get(expression.operator)
            if symbol is None:
                raise UnsupportedMapBody(
                    f"unsupported binary operator {expression.operator}"
                )
            code = f"({left.code} {symbol} {right.code})"
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
            f"({true_value.code}) if ({condition.code}) else ({false_value.code})",
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


def _load_access(
    load: Any,
    index: int,
    loads: tuple,
    axis_coordinates: Mapping[str, str],
) -> _TypedExpression:
    coordinates = []
    bounds = []
    for axis in load.axes:
        coordinate = _axis_code(axis, axis_coordinates, loads)
        coordinates.append(coordinate)
        if load.source_shape is not None and not isinstance(axis, DataIndex):
            size = load.source_shape[len(coordinates) - 1]
            bounds.extend((f"({coordinate}) >= 0", f"({coordinate}) < {size}"))
    access = f"tensor_bindings[{index}][{', '.join(coordinates)}]"
    if bounds:
        zero = f"{_cutlass_type(load.dtype)}(0)"
        access = f"({access} if {' and '.join(bounds)} else {zero})"
    return _TypedExpression(access, load.dtype)


def _axis_code(
    axis: Any, axis_coordinates: Mapping[str, str], loads: tuple
) -> str:
    if isinstance(axis, str):
        return axis_coordinates[axis]
    if isinstance(axis, int):
        return str(axis)
    if isinstance(axis, DataIndex):
        for index, candidate in enumerate(loads):
            if candidate.id == axis.load_id:
                return _load_access(
                    candidate, index, loads, axis_coordinates
                ).code
        raise UnsupportedMapBody(f"unknown data index {axis.load_id}")
    if not isinstance(axis, AxisAffine):
        raise UnsupportedMapBody(f"unsupported indexed axis {axis!r}")
    code = str(axis.offset)
    for term in axis.terms:
        coordinate = axis_coordinates[term.axis]
        if term.divisor != 1:
            coordinate = f"({coordinate} // {term.divisor})"
        if term.modulus:
            coordinate = f"({coordinate} % {term.modulus})"
        if term.scale != 1:
            coordinate = f"({coordinate} * {term.scale})"
        code = f"{code} + {coordinate}"
    return f"({code})"


def _unary_code(operator: str, operand: _TypedExpression) -> str:
    if operator == "relu":
        zero = f"{_cutlass_type(operand.dtype)}(0)"
        return f"(({operand.code}) if (({operand.code}) > {zero}) else {zero})"
    if operator == "neg":
        return f"(-({operand.code}))"
    functions = {
        "abs": "cute.abs",
        "sigmoid": "sigmoid",
        "tanh": "cute.tanh",
        "rsqrt": "cute.rsqrt",
        "sqrt": "cute.sqrt",
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


def _dtype_bytes(dtype: str) -> int:
    try:
        return _DTYPE_BYTES[dtype]
    except KeyError as error:
        raise UnsupportedReductionBody(f"unsupported dtype {dtype}") from error
