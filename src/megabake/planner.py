from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from itertools import product
from math import prod
from typing import Any

from .bodies.catalog import get_candidates
from .bodies.interface import BodyConfig, SelectionPolicy, TargetFacts
from .graph import ComputeGraph
from .lowering import TensorProgram
from .plan import (
    BodySelectionPlan,
    ConversionBody,
    CuTeLayout,
    InvalidTileMap,
    LogicalTile,
    ResourceRequirements,
    RejectionRecord,
    SelectedBody,
    TileAccess,
    TileGrid,
    TileMap,
    UnsupportedRegion,
)


def select_bodies(
    graph: ComputeGraph,
    target: TargetFacts,
    policy: SelectionPolicy | None = None,
) -> BodySelectionPlan:
    selected_policy = policy or SelectionPolicy()
    selections: list[SelectedBody] = []
    rejections: list[RejectionRecord] = []
    unsupported: list[UnsupportedRegion] = []

    for region in graph.regions:
        if not region.tensor_programs:
            unsupported.append(UnsupportedRegion(region.id, "region has no tensor program"))
            continue
        if all(
            isinstance(program, TensorProgram) and program.family == "view"
            for program in region.tensor_programs
        ):
            continue

        candidates = get_candidates(region, target, selected_policy)
        rejections.extend(
            RejectionRecord(region.id, rejection.provider, rejection.reason)
            for rejection in candidates.rejections
        )
        if not candidates.configurations:
            unsupported.append(
                UnsupportedRegion(region.id, "no valid provider candidate")
            )
            continue

        body = _choose_body(candidates.configurations)
        program = region.tensor_programs[0]
        if not isinstance(program, TensorProgram):
            unsupported.append(UnsupportedRegion(region.id, "tensor program is unknown"))
            continue
        selections.append(
            SelectedBody(
                region_id=region.id,
                source_op_ids=region.source_op_ids,
                body=body,
                program=program,
                inputs=tuple(load.value_id for load in program.loads),
                outputs=(program.store.value_id,),
                dependencies=(),
                participants=body.participants,
                resources=_resource_requirements(body),
            )
        )

    region_ids_by_operation = {
        operation_id: region.id
        for region in graph.regions
        for operation_id in region.source_op_ids
    }
    selections = [
        replace(
            selection,
            dependencies=tuple(
                dict.fromkeys(
                    region_ids_by_operation[graph.values[value_id].producer]
                    for value_id in selection.inputs
                    if graph.values[value_id].producer in region_ids_by_operation
                )
            ),
        )
        for selection in selections
    ]
    conversions = _insert_layout_conversions(tuple(selections))
    return BodySelectionPlan(
        tuple(selections),
        tuple(rejections),
        tuple(unsupported),
        conversions,
        graph,
        target,
    )


def make_tile_maps(selection: BodySelectionPlan) -> tuple[TileMap, ...]:
    return tuple(_make_tile_map(selected, selection.conversions) for selected in selection.selections)


def _choose_body(configurations: Sequence[BodyConfig]) -> BodyConfig:
    measured = [config for config in configurations if config.evidence.measured]
    available = measured or list(configurations)
    return min(available, key=_body_cost)


def _body_cost(config: BodyConfig) -> tuple[int, int]:
    return (
        config.participants.participating_threads
        * config.participants.registers_per_thread
        + config.scratch.size_bytes,
        -len(config.evidence.measurements),
    )


def _resource_requirements(config: BodyConfig) -> ResourceRequirements:
    active_blocks = config.evidence.resources.get("active_blocks_per_sm")
    return ResourceRequirements(
        shared_memory_bytes=config.scratch.size_bytes,
        pipeline_stages=config.pipeline.stage_count,
        barriers=config.pipeline.barriers,
        maximum_active_blocks=active_blocks if isinstance(active_blocks, int) else None,
    )


def _insert_layout_conversions(
    selections: Sequence[SelectedBody],
) -> tuple[ConversionBody, ...]:
    output_layouts: dict[str, tuple[str, CuTeLayout]] = {}
    for selected in selections:
        value_id = selected.program.store.value_id
        contract = selected.body.layouts.outputs.get(value_id)
        if contract is None:
            raise InvalidTileMap(f"{selected.region_id} has no output layout for {value_id}")
        output_layouts[value_id] = (selected.region_id, CuTeLayout.from_contract(contract))

    conversions: list[ConversionBody] = []
    for selected in selections:
        for value_id in selected.inputs:
            contract = selected.body.layouts.inputs.get(value_id)
            if contract is None:
                raise InvalidTileMap(f"{selected.region_id} has no input layout for {value_id}")
            produced = output_layouts.get(value_id)
            if produced is None:
                continue
            source_region_id, source_layout = produced
            target_layout = CuTeLayout.from_contract(contract)
            if not source_layout.needs_conversion_to(target_layout):
                continue
            conversions.append(
                ConversionBody(
                    value_id=value_id,
                    source_region_id=source_region_id,
                    target_region_id=selected.region_id,
                    source_layout=source_layout,
                    target_layout=target_layout,
                    cost_bytes=_conversion_cost(target_layout),
                    dependencies=(source_region_id,),
                )
            )
    return tuple(conversions)


def _make_tile_map(
    selected: SelectedBody,
    conversions: Sequence[ConversionBody],
) -> TileMap:
    program = selected.program
    body = selected.body
    output_contract = body.layouts.outputs.get(program.store.value_id)
    if output_contract is None:
        raise InvalidTileMap(f"{selected.region_id} has no output layout")
    output_layout = CuTeLayout.from_contract(output_contract)
    output_domains = tuple(
        domain
        for domain in program.iteration_domains
        if domain.axis not in program.reduction_axes
    )
    output_axes = tuple(domain.axis for domain in output_domains)
    if tuple(output_contract.axes) != output_axes:
        raise InvalidTileMap(f"{selected.region_id} output axes do not match its domains")

    input_layouts: list[tuple[str, CuTeLayout]] = []
    for load in program.loads:
        contract = body.layouts.inputs.get(load.value_id)
        if contract is None:
            raise InvalidTileMap(f"{selected.region_id} has no layout for {load.value_id}")
        if tuple(contract.axes) != tuple(load.axes):
            raise InvalidTileMap(f"{selected.region_id} input axes do not match {load.value_id}")
        input_layouts.append((load.value_id, CuTeLayout.from_contract(contract)))

    grid = TileGrid(program.store.shape, _output_tile_shape(program, body))
    reduction_size = prod(
        domain.size for domain in program.iteration_domains
        if domain.axis in program.reduction_axes
    )
    reduction_range = (0, reduction_size) if program.reduction_axes else None
    accesses = tuple(
        TileAccess(
            output=_logical_tile(
                program.store.value_id,
                output_axes,
                output_layout,
                origin,
                grid.tile_shape,
            ),
            inputs=tuple(
                _input_tile(load, layout, origin, grid.tile_shape, output_axes, program)
                for load, (_, layout) in zip(program.loads, input_layouts, strict=True)
            ),
            reduction_range=reduction_range,
        )
        for origin in grid.origins()
    )
    return TileMap(
        region_id=selected.region_id,
        body=body,
        participants=selected.participants,
        resources=selected.resources,
        output_grid=grid,
        output_layout=output_layout,
        input_layouts=tuple(input_layouts),
        accesses=accesses,
        conversion_bodies=tuple(
            conversion
            for conversion in conversions
            if conversion.target_region_id == selected.region_id
        ),
    )


def _output_tile_shape(program: TensorProgram, body: BodyConfig) -> tuple[int, ...]:
    output_shape = program.store.shape
    if program.family == "reduction":
        return (1,) * len(output_shape)
    if body.tiles.work_partition == "one output element per thread":
        return _row_major_tile_shape(output_shape, body.participants.participating_threads)
    if len(body.tiles.cta_shape) != len(output_shape):
        raise InvalidTileMap("CTA tile rank does not match the output rank")
    return body.tiles.cta_shape


def _row_major_tile_shape(
    shape: Sequence[int], element_limit: int
) -> tuple[int, ...]:
    tile_shape = [1] * len(shape)
    remaining = element_limit
    for dimension in reversed(range(len(shape))):
        extent = min(shape[dimension], remaining)
        tile_shape[dimension] = extent
        remaining //= extent
        if extent < shape[dimension]:
            break
    return tuple(tile_shape)


def _input_tile(
    load: Any,
    layout: CuTeLayout,
    output_origin: tuple[int, ...],
    output_tile_shape: tuple[int, ...],
    output_axes: tuple[str, ...],
    program: TensorProgram,
) -> LogicalTile:
    origin: list[int] = []
    shape: list[int] = []
    reduction_domains = {
        domain.axis: domain.size
        for domain in program.iteration_domains
        if domain.axis in program.reduction_axes
    }
    for axis in load.axes:
        if isinstance(axis, int):
            origin.append(axis)
            shape.append(1)
            continue
        if axis in output_axes:
            output_dimension = output_axes.index(axis)
            origin.append(output_origin[output_dimension])
            shape.append(output_tile_shape[output_dimension])
            continue
        if axis in reduction_domains:
            origin.append(0)
            shape.append(reduction_domains[axis])
            continue
        raise InvalidTileMap(f"input axis {axis} is not connected to the iteration domains")
    return _logical_tile(load.value_id, tuple(load.axes), layout, tuple(origin), tuple(shape))


def _logical_tile(
    value_id: str,
    axes: tuple[str | int, ...],
    layout: CuTeLayout,
    origin: tuple[int, ...],
    tile_shape: tuple[int, ...],
) -> LogicalTile:
    clipped_shape = tuple(
        min(extent, dimension - begin)
        for extent, dimension, begin in zip(
            tile_shape, layout.shape, origin, strict=True
        )
    )
    if any(extent <= 0 for extent in clipped_shape):
        raise InvalidTileMap(f"tile for {value_id} is empty")
    return LogicalTile(value_id, axes, layout, origin, clipped_shape)


def _conversion_cost(layout: CuTeLayout) -> int:
    return 2 * prod(layout.shape) * _DTYPE_BYTES.get(layout.dtype, 0)


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
