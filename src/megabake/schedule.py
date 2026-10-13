from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from bisect import bisect_left, bisect_right
import heapq

from .bodies.interface import BodyConfig, ParticipantContract, WorkItem
from .graph import ComputeGraph
from .plan import (
    BodySelectionPlan,
    ConversionBody,
    ResourceRequirements,
    SelectedBody,
    TileMap,
)
from .planner import make_tile_maps


class InvalidSchedule(ValueError):
    pass


class DependencyKind(StrEnum):
    PRODUCER_TO_CONSUMER = "producer_to_consumer"
    LAYOUT_CONVERSION = "layout_conversion"
    WRITE_AFTER_READ = "write_after_read"
    WRITE_AFTER_WRITE = "write_after_write"


@dataclass(frozen=True)
class RegionDependency:
    producer_region_id: str
    consumer_region_id: str
    value_id: str
    kind: DependencyKind
    producer_tiles: tuple[tuple[int, ...], ...]
    consumer_tiles: tuple[tuple[int, ...], ...]
    conservative: bool = False
    reason: str = ""


@dataclass(frozen=True)
class PrerequisiteSet:
    region_id: str
    prerequisites: tuple[str, ...]


@dataclass(frozen=True)
class DependencyPlan:
    dependencies: tuple[RegionDependency, ...]
    prerequisite_sets: tuple[PrerequisiteSet, ...]

    @property
    def conservative(self) -> bool:
        return any(dependency.conservative for dependency in self.dependencies)

    def prerequisites_for(self, region_id: str) -> frozenset[str]:
        for record in self.prerequisite_sets:
            if record.region_id == region_id:
                return frozenset(record.prerequisites)
        raise InvalidSchedule(f"region {region_id} has no prerequisite set")


@dataclass(frozen=True)
class PhaseBarrier:
    id: str
    participating_workers: tuple[int, ...]
    kind: str = "grid"


@dataclass(frozen=True)
class WorkerAssignment:
    worker_id: int
    region_id: str
    work_items: tuple[WorkItem, ...]
    participates_in_barrier: bool = True


@dataclass(frozen=True)
class ScheduledPhase:
    index: int
    region_id: str
    body: BodyConfig
    participants: ParticipantContract
    resources: ResourceRequirements
    work_items: tuple[WorkItem, ...]
    workers: tuple[WorkerAssignment, ...]
    dependencies: tuple[str, ...]
    conversion_bodies: tuple[ConversionBody, ...] = ()
    barrier: PhaseBarrier | None = None


@dataclass(frozen=True)
class PhaseSchedule:
    phases: tuple[ScheduledPhase, ...]
    worker_count: int
    prerequisite_sets: tuple[PrerequisiteSet, ...]
    selection: BodySelectionPlan | None = None
    dependency_plan: DependencyPlan | None = None

    @property
    def region_ids(self) -> tuple[str, ...]:
        return tuple(phase.region_id for phase in self.phases)

    @property
    def barriers(self) -> tuple[PhaseBarrier, ...]:
        return tuple(phase.barrier for phase in self.phases if phase.barrier)

    def prerequisites_for(self, region_id: str) -> frozenset[str]:
        for record in self.prerequisite_sets:
            if record.region_id == region_id:
                return frozenset(record.prerequisites)
        raise InvalidSchedule(f"region {region_id} has no prerequisite set")

    def phase_for(self, region_id: str) -> ScheduledPhase:
        for phase in self.phases:
            if phase.region_id == region_id:
                return phase
        raise InvalidSchedule(f"region {region_id} is not scheduled")


@dataclass(frozen=True)
class ScheduleSimulation:
    phase_order: tuple[str, ...]
    barrier_order: tuple[str, ...]
    work_item_counts: Mapping[str, int]
    idle_worker_counts: Mapping[str, int]


def derive_dependencies(selection: BodySelectionPlan) -> DependencyPlan:
    if not selection.report.complete:
        raise InvalidSchedule("cannot schedule a plan with unsupported regions")
    if selection.graph is None:
        raise InvalidSchedule("body selection is missing its source graph")

    selected_by_region = {
        selected.region_id: selected for selected in selection.selections
    }
    tile_maps = {
        tile_map.region_id: tile_map for tile_map in make_tile_maps(selection)
    }
    dependencies: list[RegionDependency] = []

    for consumer in selection.selections:
        for producer_region_id in consumer.dependencies:
            producer = selected_by_region.get(producer_region_id)
            if producer is None:
                continue
            for value_id in consumer.inputs:
                if value_id not in producer.outputs:
                    continue
                dependencies.append(
                    _tile_dependency(
                        producer_region_id,
                        consumer.region_id,
                        value_id,
                        tile_maps[producer_region_id],
                        tile_maps[consumer.region_id],
                    )
                )

    for conversion in selection.conversions:
        dependencies.append(
            RegionDependency(
                producer_region_id=conversion.source_region_id,
                consumer_region_id=conversion.target_region_id,
                value_id=conversion.value_id,
                kind=DependencyKind.LAYOUT_CONVERSION,
                producer_tiles=_output_tiles(tile_maps[conversion.source_region_id]),
                consumer_tiles=_consumer_tiles(
                    tile_maps[conversion.target_region_id], conversion.value_id
                ),
            )
        )

    dependencies.extend(_storage_order_dependencies(selection, tile_maps))
    plan = _dependency_plan(tuple(dependencies), selected_by_region)
    _topological_order(selection.selections, plan)
    return plan


def make_phase_schedule(
    selection: BodySelectionPlan,
    dependencies: DependencyPlan,
    *,
    worker_count: int | None = None,
) -> PhaseSchedule:
    if not selection.report.complete:
        raise InvalidSchedule("cannot schedule a plan with unsupported regions")
    if not isinstance(dependencies, DependencyPlan):
        raise InvalidSchedule("dependencies must come from derive_dependencies")

    selected_by_region = {
        selected.region_id: selected for selected in selection.selections
    }
    for dependency in dependencies.dependencies:
        if dependency.producer_region_id not in selected_by_region:
            raise InvalidSchedule(
                f"dependency names unscheduled producer {dependency.producer_region_id}"
            )
        if dependency.consumer_region_id not in selected_by_region:
            raise InvalidSchedule(
                f"dependency names unscheduled consumer {dependency.consumer_region_id}"
            )

    order = _topological_order(selection.selections, dependencies)
    tile_maps = {
        tile_map.region_id: tile_map for tile_map in make_tile_maps(selection)
    }
    default_worker_count = _worker_count(selection, tile_maps)
    if worker_count is None:
        worker_count = default_worker_count
    elif not 1 <= worker_count <= default_worker_count:
        raise InvalidSchedule(
            f"worker count {worker_count} is outside 1..{default_worker_count}"
        )
    phases: list[ScheduledPhase] = []

    for phase_index, region_id in enumerate(order):
        selected = selected_by_region[region_id]
        tile_map = tile_maps[region_id]
        work_items = tile_map.work_items
        barrier = (
            PhaseBarrier(
                f"phase:{phase_index}:grid",
                tuple(range(worker_count)),
            )
            if phase_index + 1 < len(order)
            else None
        )
        phases.append(
            ScheduledPhase(
                index=phase_index,
                region_id=region_id,
                body=selected.body,
                participants=selected.participants,
                resources=selected.resources,
                work_items=work_items,
                workers=_worker_assignments(region_id, work_items, worker_count),
                dependencies=tuple(
                    sorted(dependencies.prerequisites_for(region_id))
                ),
                conversion_bodies=tuple(
                    conversion
                    for conversion in selection.conversions
                    if conversion.source_region_id == region_id
                ),
                barrier=barrier,
            )
        )

    return PhaseSchedule(
        tuple(phases),
        worker_count,
        dependencies.prerequisite_sets,
        selection,
        dependencies,
    )


def simulate_phase_schedule(schedule: PhaseSchedule) -> ScheduleSimulation:
    completed: set[str] = set()
    phase_order: list[str] = []
    barrier_order: list[str] = []
    work_counts: dict[str, int] = {}
    idle_counts: dict[str, int] = {}

    for index, phase in enumerate(schedule.phases):
        missing = schedule.prerequisites_for(phase.region_id) - completed
        if missing:
            raise InvalidSchedule(
                f"phase {phase.index} has unfinished prerequisites {sorted(missing)}"
            )
        if phase.index != index:
            raise InvalidSchedule("phase indices are not contiguous")
        if len(phase.workers) != schedule.worker_count:
            raise InvalidSchedule(f"phase {phase.index} does not cover every worker")

        worker_ids = [worker.worker_id for worker in phase.workers]
        if worker_ids != list(range(schedule.worker_count)):
            raise InvalidSchedule(f"phase {phase.index} has invalid worker IDs")
        assigned = Counter(
            work_item
            for worker in phase.workers
            for work_item in worker.work_items
        )
        if assigned != Counter(phase.work_items):
            raise InvalidSchedule(f"phase {phase.index} has invalid work assignments")
        if any(not worker.participates_in_barrier for worker in phase.workers):
            raise InvalidSchedule(f"phase {phase.index} has a worker outside its barrier")

        is_final = index + 1 == len(schedule.phases)
        if phase.barrier is None:
            if not is_final:
                raise InvalidSchedule(f"phase {phase.index} is missing its grid barrier")
        else:
            if is_final:
                raise InvalidSchedule("the final phase has an unnecessary barrier")
            if phase.barrier.kind != "grid":
                raise InvalidSchedule(f"phase {phase.index} has an invalid barrier")
            if phase.barrier.participating_workers != tuple(
                range(schedule.worker_count)
            ):
                raise InvalidSchedule(f"phase {phase.index} has an incomplete barrier")
            barrier_order.append(phase.barrier.id)

        phase_order.append(phase.region_id)
        work_counts[phase.region_id] = len(phase.work_items)
        idle_counts[phase.region_id] = sum(
            not worker.work_items for worker in phase.workers
        )
        completed.add(phase.region_id)

    return ScheduleSimulation(
        tuple(phase_order),
        tuple(barrier_order),
        work_counts,
        idle_counts,
    )


def _tile_dependency(
    producer_region_id: str,
    consumer_region_id: str,
    value_id: str,
    producer_map: TileMap,
    consumer_map: TileMap,
) -> RegionDependency:
    producer_output_tiles = tuple(
        access.output for access in producer_map.accesses
    )
    consumer_accesses = tuple(
        access
        for access in consumer_map.accesses
        if any(input_tile.value_id == value_id for input_tile in access.inputs)
    )
    if not consumer_accesses:
        raise InvalidSchedule(
            f"{consumer_region_id} has no mapped access to {value_id}"
        )

    producer_origins: set[tuple[int, ...]] = set()
    consumer_tiles: set[tuple[int, ...]] = set()
    exact = True
    producer_by_first_origin: dict[int, list[Any]] = {}
    for tile in producer_output_tiles:
        producer_by_first_origin.setdefault(tile.origin[0], []).append(tile)
    first_origins = tuple(sorted(producer_by_first_origin))
    maximum_first_extent = max(tile.shape[0] for tile in producer_output_tiles)
    for access in consumer_accesses:
        input_tile = next(
            input_tile
            for input_tile in access.inputs
            if input_tile.value_id == value_id
        )
        matched = False
        consumer_begin = input_tile.origin[0]
        consumer_end = consumer_begin + input_tile.shape[0]
        lower = bisect_left(
            first_origins, consumer_begin - maximum_first_extent + 1
        )
        upper = bisect_right(first_origins, consumer_end - 1)
        candidates = (
            output_tile
            for first_origin in first_origins[lower:upper]
            for output_tile in producer_by_first_origin[first_origin]
        )
        for output_tile in candidates:
            overlaps, is_exact = _logical_tiles_overlap(output_tile, input_tile)
            exact = exact and is_exact
            if not is_exact:
                break
            if overlaps:
                producer_origins.add(output_tile.origin)
                consumer_tiles.add(access.output.origin)
                matched = True
        if exact and not matched:
            raise InvalidSchedule(
                f"{consumer_region_id} reads no producer tile for {value_id}"
            )

    if not exact:
        return RegionDependency(
            producer_region_id,
            consumer_region_id,
            value_id,
            DependencyKind.PRODUCER_TO_CONSUMER,
            _output_tiles(producer_map),
            tuple(access.output.origin for access in consumer_accesses),
            True,
            "input axes do not permit exact tile-overlap analysis",
        )
    return RegionDependency(
        producer_region_id,
        consumer_region_id,
        value_id,
        DependencyKind.PRODUCER_TO_CONSUMER,
            tuple(sorted(producer_origins)),
        tuple(sorted(consumer_tiles)),
    )


def _logical_tiles_overlap(
    producer_tile: object, consumer_tile: object
) -> tuple[bool, bool]:
    if (
        len(producer_tile.origin) != len(consumer_tile.origin)
        or producer_tile.layout.shape != consumer_tile.layout.shape
    ):
        return False, False
    for producer_origin, consumer_origin, producer_extent, consumer_extent in zip(
        producer_tile.origin,
        consumer_tile.origin,
        producer_tile.shape,
        consumer_tile.shape,
        strict=True,
    ):
        producer_end = producer_origin + producer_extent
        consumer_end = consumer_origin + consumer_extent
        if max(producer_origin, consumer_origin) >= min(
            producer_end, consumer_end
        ):
            return False, True
    return True, True


def _output_tiles(tile_map: TileMap) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(access.output.origin for access in tile_map.accesses))


def _consumer_tiles(tile_map: TileMap, value_id: str) -> tuple[tuple[int, ...], ...]:
    return tuple(
        sorted(
            access.output.origin
            for access in tile_map.accesses
            if any(input_tile.value_id == value_id for input_tile in access.inputs)
        )
    )


def _storage_order_dependencies(
    selection: BodySelectionPlan, tile_maps: Mapping[str, TileMap]
) -> tuple[RegionDependency, ...]:
    graph = selection.graph
    assert graph is not None
    unknown_effects = any(
        graph.ops[operation_id].effects.get("unknown")
        for selected in selection.selections
        for operation_id in selected.source_op_ids
    )
    reads: dict[str, frozenset[str]] = {}
    writes: dict[str, frozenset[str]] = {}
    for selected in selection.selections:
        reads[selected.region_id] = _storage_ids(graph, selected.inputs)
        writes[selected.region_id] = _storage_ids(graph, selected.outputs)
        for operation_id in selected.source_op_ids:
            operation = graph.ops[operation_id]
            if operation.effects.get("mutates_inputs") or operation.effects.get(
                "may_mutate_inputs"
            ):
                operand_storage = _storage_ids(graph, operation.operands)
                reads[selected.region_id] = reads[selected.region_id] | operand_storage
                writes[selected.region_id] = writes[selected.region_id] | operand_storage
        if unknown_effects:
            reads[selected.region_id] = reads[selected.region_id] | {
                "unknown:global-storage"
            }
            writes[selected.region_id] = writes[selected.region_id] | {
                "unknown:global-storage"
            }

    dependencies: list[RegionDependency] = []
    selections = selection.selections
    for later_index, later in enumerate(selections[1:], start=1):
        for earlier in selections[:later_index]:
            write_after_write = writes[earlier.region_id] & writes[later.region_id]
            if write_after_write:
                dependencies.append(
                    _storage_dependency(
                        earlier,
                        later,
                        DependencyKind.WRITE_AFTER_WRITE,
                        write_after_write,
                        "regions write the same storage",
                        tile_maps[earlier.region_id],
                        tile_maps[later.region_id],
                    )
                )
            write_after_read = writes[later.region_id] & reads[earlier.region_id]
            if write_after_read:
                dependencies.append(
                    _storage_dependency(
                        earlier,
                        later,
                        DependencyKind.WRITE_AFTER_READ,
                        write_after_read,
                        "later write overlaps an earlier read",
                        tile_maps[earlier.region_id],
                        tile_maps[later.region_id],
                    )
                )
    return tuple(dependencies)


def _storage_dependency(
    earlier: SelectedBody,
    later: SelectedBody,
    kind: DependencyKind,
    storage_ids: frozenset[str],
    reason: str,
    producer_map: TileMap,
    consumer_map: TileMap,
) -> RegionDependency:
    return RegionDependency(
        earlier.region_id,
        later.region_id,
        ",".join(sorted(storage_ids)),
        kind,
        _output_tiles(producer_map),
        _output_tiles(consumer_map),
        True,
        reason,
    )


def _storage_ids(graph: ComputeGraph, value_ids: tuple[str, ...]) -> frozenset[str]:
    storage_ids: set[str] = set()
    for value_id in value_ids:
        if value_id not in graph.values:
            raise InvalidSchedule(f"body selection references unknown value {value_id}")
        relation = graph.values[value_id].storage_relation
        if relation.storage_id is not None:
            storage_ids.add(relation.storage_id)
        storage_ids.update(relation.aliases)
        if relation.unknown or relation.storage_id is None:
            storage_ids.add(f"unknown:{value_id}")
    return frozenset(storage_ids)


def _dependency_plan(
    dependencies: tuple[RegionDependency, ...],
    selected_by_region: Mapping[str, SelectedBody],
) -> DependencyPlan:
    prerequisites = {
        region_id: set() for region_id in selected_by_region
    }
    for dependency in dependencies:
        if dependency.producer_region_id == dependency.consumer_region_id:
            raise InvalidSchedule(f"region {dependency.producer_region_id} depends on itself")
        if dependency.producer_region_id not in selected_by_region:
            raise InvalidSchedule(
                f"dependency names unscheduled producer {dependency.producer_region_id}"
            )
        if dependency.consumer_region_id not in selected_by_region:
            raise InvalidSchedule(
                f"dependency names unscheduled consumer {dependency.consumer_region_id}"
            )
        prerequisites[dependency.consumer_region_id].add(dependency.producer_region_id)
    return DependencyPlan(
        dependencies,
        tuple(
            PrerequisiteSet(region_id, tuple(sorted(values)))
            for region_id, values in prerequisites.items()
        ),
    )


def _topological_order(
    selections: tuple[SelectedBody, ...], dependencies: DependencyPlan
) -> tuple[str, ...]:
    positions = {
        selected.region_id: position for position, selected in enumerate(selections)
    }
    consumers = {region_id: [] for region_id in positions}
    indegree = {region_id: 0 for region_id in positions}
    for dependency in dependencies.dependencies:
        producer = dependency.producer_region_id
        consumer = dependency.consumer_region_id
        if producer not in positions or consumer not in positions:
            raise InvalidSchedule("dependency references an unscheduled region")
        consumers[producer].append(consumer)
        indegree[consumer] += 1

    ready = [positions[region_id] for region_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        position = heapq.heappop(ready)
        region_id = selections[position].region_id
        order.append(region_id)
        for consumer in consumers[region_id]:
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                heapq.heappush(ready, positions[consumer])
    if len(order) != len(selections):
        raise InvalidSchedule("body dependencies contain a cycle")
    return tuple(order)


def _worker_count(
    selection: BodySelectionPlan, tile_maps: Mapping[str, TileMap]
) -> int:
    maximum_work = max(
        (len(tile_map.accesses) for tile_map in tile_maps.values()), default=0
    )
    if maximum_work == 0:
        return 1
    if selection.target is None:
        return maximum_work
    active_blocks = min(
        selected.resources.maximum_active_blocks or 1
        for selected in selection.selections
    )
    return min(maximum_work, selection.target.multiprocessor_count * active_blocks)


def _worker_assignments(
    region_id: str, work_items: tuple[WorkItem, ...], worker_count: int
) -> tuple[WorkerAssignment, ...]:
    quotient, remainder = divmod(len(work_items), worker_count)
    start = 0
    assignments: list[WorkerAssignment] = []
    for worker_id in range(worker_count):
        count = quotient + (1 if worker_id < remainder else 0)
        assignments.append(
            WorkerAssignment(
                worker_id,
                region_id,
                work_items[start : start + count],
            )
        )
        start += count
    return tuple(assignments)
