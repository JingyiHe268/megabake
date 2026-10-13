from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

from .bodies.interface import (
    BodyConfig,
    ParticipantContract,
    TargetFacts,
    TensorLayout,
    WorkItem,
)
from .graph import ComputeGraph
from .lowering import TensorProgram

if TYPE_CHECKING:
    from .memory import LaunchRequirements, StoragePlan
    from .schedule import DependencyPlan, ScheduledPhase


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


@dataclass(frozen=True)
class KernelPlan:
    selection: BodySelectionPlan
    dependencies: DependencyPlan
    phases: tuple[ScheduledPhase, ...]
    tile_maps: tuple[TileMap, ...]
    storage: StoragePlan
    launch: LaunchRequirements

    @property
    def bodies(self) -> tuple[BodyConfig, ...]:
        return tuple(phase.body for phase in self.phases)

    @property
    def target(self) -> TargetFacts | None:
        return self.selection.target

    @property
    def graph(self) -> ComputeGraph | None:
        return self.selection.graph
