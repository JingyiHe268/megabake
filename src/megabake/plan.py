from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import product
from math import prod
from typing import Any

from .bodies.catalog import get_candidates
from .bodies.interface import (
    BodyConfig,
    ParticipantContract,
    SelectionPolicy,
    TargetFacts,
    TensorLayout,
    WorkItem,
)
from .graph import ComputeGraph
from .lowering import TensorProgram


class InvalidTileMap(ValueError):
    pass


@dataclass(frozen=True)
class CuTeSwizzle:
    mask: int
    shift: int

    def apply(self, offset: int) -> int:
        return offset ^ (((offset >> self.shift) & self.mask) << self.shift)


@dataclass(frozen=True)
class CuTeLayout:
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    alignment_bytes: int
    swizzle: CuTeSwizzle | None = None

    def __post_init__(self) -> None:
        if len(self.shape) != len(self.strides):
            raise InvalidTileMap("CuTe shape and stride ranks differ")
        if any(dimension <= 0 for dimension in self.shape):
            raise InvalidTileMap("CuTe layout dimensions must be positive")
        if self.alignment_bytes <= 0 or self.alignment_bytes & (
            self.alignment_bytes - 1
        ):
            raise InvalidTileMap("CuTe layout alignment must be a power of two")
        if self.swizzle is not None and (
            self.swizzle.mask <= 0
            or self.swizzle.mask & (self.swizzle.mask - 1)
            or self.swizzle.shift < 0
        ):
            raise InvalidTileMap("CuTe XOR swizzle is invalid")

    @classmethod
    def from_contract(cls, layout: TensorLayout) -> CuTeLayout:
        shape = layout.shape or (1,) * len(layout.axes)
        if len(shape) != len(layout.axes) or len(shape) != len(layout.strides):
            raise InvalidTileMap(
                f"layout {layout.dtype} has mismatched axes, shape, and strides"
            )
        return cls(layout.dtype, shape, layout.strides, layout.alignment_bytes)

    @property
    def cute_shape(self) -> tuple[int, ...]:
        return tuple(reversed(self.shape))

    @property
    def cute_stride(self) -> tuple[int, ...]:
        return tuple(reversed(self.strides))

    @property
    def cute_layout(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self.cute_shape, self.cute_stride

    def offset(self, coordinates: Sequence[int]) -> int:
        if len(coordinates) != len(self.shape):
            raise InvalidTileMap("coordinates do not match the CuTe layout rank")
        for coordinate, size in zip(coordinates, self.shape, strict=True):
            if not 0 <= coordinate < size:
                raise InvalidTileMap("coordinates are outside the CuTe layout")
        offset = sum(
            coordinate * stride
            for coordinate, stride in zip(coordinates, self.strides, strict=True)
        )
        return self.swizzle.apply(offset) if self.swizzle is not None else offset

    def needs_conversion_to(self, other: CuTeLayout) -> bool:
        if self.dtype != other.dtype or self.shape != other.shape:
            raise InvalidTileMap("conversion layouts must have the same value domain")
        normalize = lambda layout: tuple(
            stride if size != 1 else 0
            for stride, size in zip(layout.strides, layout.shape, strict=True)
        )
        return (normalize(self), self.swizzle) != (normalize(other), other.swizzle)


@dataclass(frozen=True)
class LogicalTile:
    value_id: str
    axes: tuple[str | int, ...]
    layout: CuTeLayout
    origin: tuple[int, ...]
    shape: tuple[int, ...]

    def element_offset(self, coordinates: Sequence[int]) -> int:
        if len(coordinates) != len(self.origin):
            raise InvalidTileMap("coordinates do not match the logical tile rank")
        for coordinate, begin, extent, size in zip(
            coordinates,
            self.origin,
            self.shape,
            self.layout.shape,
            strict=True,
        ):
            if not begin <= coordinate < begin + extent or not 0 <= coordinate < size:
                raise InvalidTileMap("coordinates are outside the logical tile")
        return self.layout.offset(coordinates)


@dataclass(frozen=True)
class TileAccess:
    output: LogicalTile
    inputs: tuple[LogicalTile, ...]
    reduction_range: tuple[int, int] | None

    @property
    def work_item(self) -> WorkItem:
        return WorkItem(self.output.origin, self.reduction_range)


@dataclass(frozen=True)
class TileGrid:
    shape: tuple[int, ...]
    tile_shape: tuple[int, ...]

    @property
    def counts(self) -> tuple[int, ...]:
        if len(self.shape) != len(self.tile_shape) or any(
            dimension <= 0 for dimension in self.tile_shape
        ):
            raise InvalidTileMap("tile shape does not match the output rank")
        return tuple(
            (dimension + tile - 1) // tile
            for dimension, tile in zip(self.shape, self.tile_shape, strict=True)
        )

    def origins(self):
        return product(*(range(0, dimension, tile) for dimension, tile in zip(
            self.shape, self.tile_shape, strict=True
        )))


@dataclass(frozen=True)
class ResourceRequirements:
    shared_memory_bytes: int
    pipeline_stages: int
    barriers: tuple[str, ...]
    maximum_active_blocks: int | None = None


@dataclass(frozen=True)
class RejectionRecord:
    region_id: str
    provider: str
    reason: str


@dataclass(frozen=True)
class UnsupportedRegion:
    region_id: str
    reason: str


@dataclass(frozen=True)
class ConversionBody:
    value_id: str
    source_region_id: str
    target_region_id: str
    source_layout: CuTeLayout
    target_layout: CuTeLayout
    cost_bytes: int
    dependencies: tuple[str, ...]


@dataclass(frozen=True)
class SelectedBody:
    region_id: str
    source_op_ids: tuple[str, ...]
    body: BodyConfig
    program: TensorProgram
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    dependencies: tuple[str, ...]
    participants: ParticipantContract
    resources: ResourceRequirements


@dataclass(frozen=True)
class PlanReport:
    selected_region_ids: tuple[str, ...]
    rejections: tuple[RejectionRecord, ...]
    unsupported_regions: tuple[UnsupportedRegion, ...]
    conversions: tuple[ConversionBody, ...]

    @property
    def complete(self) -> bool:
        return not self.unsupported_regions


@dataclass(frozen=True)
class BodySelectionPlan:
    selections: tuple[SelectedBody, ...]
    rejections: tuple[RejectionRecord, ...]
    unsupported_regions: tuple[UnsupportedRegion, ...]
    conversions: tuple[ConversionBody, ...]
    graph: ComputeGraph | None = None
    target: TargetFacts | None = None

    @property
    def report(self) -> PlanReport:
        return PlanReport(
            tuple(selection.region_id for selection in self.selections),
            self.rejections,
            self.unsupported_regions,
            self.conversions,
        )


@dataclass(frozen=True)
class TileMap:
    region_id: str
    body: BodyConfig
    participants: ParticipantContract
    resources: ResourceRequirements
    output_grid: TileGrid
    output_layout: CuTeLayout
    input_layouts: tuple[tuple[str, CuTeLayout], ...]
    accesses: tuple[TileAccess, ...]
    conversion_bodies: tuple[ConversionBody, ...] = ()

    @property
    def work_items(self) -> tuple[WorkItem, ...]:
        return tuple(access.work_item for access in self.accesses)


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
