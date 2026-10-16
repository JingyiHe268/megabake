from __future__ import annotations

from dataclasses import dataclass
import uuid
from typing import Protocol

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda

from .memory import verify_plan
from .plan import KernelPlan
from .schedule import EventSchedule, simulate_event_schedule


class InvalidRuntime(ValueError):
    pass


_MAX_INT32 = 2**31 - 1


def ready_slots(workers: int) -> int:
    """Barrier state: per-worker slots, the release flag, and the arrival counter."""
    return workers + 2


def make_ready(workers: int, device: torch.device | str) -> torch.Tensor:
    ready = torch.full((ready_slots(workers),), -1, dtype=torch.int32, device=device)
    ready[-1] = 0
    return ready


def reset_ready(ready: torch.Tensor) -> None:
    ready.fill_(-1)
    ready[-1] = 0


@cute.jit
def grid_barrier(ready: cute.Tensor, workers: cutlass.Int32, epoch: cutlass.Int32):
    # One arrival atomic per CTA. The last arrival clears the counter before
    # releasing the epoch flag, so the counter is zero between barriers and no
    # CTA can arrive at the next barrier before the clear. The previous design
    # polled every worker's flag serially from one thread, which cost one
    # acquire round trip per worker (13 us per barrier at 60 workers).
    thread, _, _ = cute.arch.thread_idx()
    cute.arch.sync_threads()
    if thread == 0:
        arrived = cute.arch.atomic_add(
            ready.iterator + workers + 1,
            cutlass.Int32(1),
            sem="acq_rel",
            scope="gpu",
        )
        if arrived == workers - 1:
            cute.arch.atomic_exch(
                ready.iterator + workers + 1,
                cutlass.Int32(0),
                sem="relaxed",
                scope="gpu",
            )
            cute.arch.atomic_exch(
                ready.iterator + workers,
                epoch,
                sem="release",
                scope="gpu",
            )
        else:
            released = False
            while not released:
                seen = cute.arch.atomic_cas(
                    ready.iterator + workers,
                    cmp=epoch,
                    val=epoch,
                    sem="acquire",
                    scope="gpu",
                )
                released = seen == epoch
    cute.arch.sync_threads()


@cute.jit
def cooperative_worker_loop(
    ready: cute.Tensor,
    workers: cutlass.Int32,
    epoch_base: cutlass.Int32,
    phases: cutlass.Constexpr,
):
    for phase_index, run_phase, drain, publish in phases:
        run_phase(phase_index)
        drain()
        publish(epoch_base)


@cute.jit
def cooperative_event_worker_loop(
    events: cute.Tensor,
    workers: cutlass.Int32,
    epoch: cutlass.Int32,
    phases: cutlass.Constexpr,
):
    for phase_index, wait, run, publish in phases:
        wait(epoch)
        run(phase_index)
        publish()


class PhaseKernel(Protocol):
    def __call__(
        self,
        ready: torch.Tensor,
        epoch: int,
        stream: torch.cuda.Stream,
    ) -> None: ...


class EventKernel(Protocol):
    def __call__(
        self,
        events: torch.Tensor,
        epoch: int,
        stream: torch.cuda.Stream,
    ) -> None: ...


@dataclass(frozen=True)
class RuntimeAdmission:
    plan: KernelPlan
    barrier_count: int
    max_workers: int


@dataclass(frozen=True)
class RuntimeInvocation:
    invocation_id: uuid.UUID
    epoch: int
    barrier_count: int


@dataclass(frozen=True)
class EventRuntimeAdmission:
    schedule: EventSchedule
    event_count: int
    max_invocations: int


@dataclass(frozen=True)
class EventRuntimeInvocation:
    invocation_id: uuid.UUID
    epoch: int
    event_count: int


@cute.jit
def publish_event(
    events: cute.Tensor,
    event_id: cutlass.Int32,
):
    """Publish one producer CTA after the caller has drained async body work."""
    cute.arch.sync_threads()
    thread, _, _ = cute.arch.thread_idx()
    if thread == 0:
        cute.arch.atomic_add(
            events.iterator + event_id,
            1,
            sem="release",
            scope="gpu",
        )
    cute.arch.sync_threads()


@cute.jit
def wait_for_event(
    events: cute.Tensor,
    event_id: cutlass.Int32,
    trigger_count: cutlass.Int32,
    epoch: cutlass.Int32,
):
    thread, _, _ = cute.arch.thread_idx()
    cute.arch.sync_threads()
    if thread == 0:
        target = (epoch + 1) * trigger_count
        count = 0
        while count < target:
            count = cute.arch.atomic_cas(
                events.iterator + event_id,
                cmp=target,
                val=target,
                sem="acquire",
                scope="gpu",
            )
    cute.arch.sync_threads()


def admit_runtime(plan: KernelPlan) -> RuntimeAdmission:
    verify_plan(plan)
    if not plan.launch.cooperative:
        raise InvalidRuntime("phase plans require a cooperative launch")

    target = plan.target
    if target is None:
        raise InvalidRuntime("phase plan is missing target facts")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if _architecture_number(plan.launch.architecture) != _architecture_number(
        f"sm{properties.major}{properties.minor}"
    ):
        raise InvalidRuntime(
            f"plan architecture {plan.launch.architecture} does not match the current device"
        )
    if properties.multi_processor_count != target.multiprocessor_count:
        raise InvalidRuntime("plan multiprocessor count does not match the current device")
    if not _cooperative_launch_supported():
        raise InvalidRuntime("the current device does not support cooperative launch")

    if plan.launch.max_active_blocks_per_sm is None:
        raise InvalidRuntime("plan has no compiled active-block capacity")
    max_workers = (
        plan.launch.max_active_blocks_per_sm * properties.multi_processor_count
    )
    if plan.launch.grid_size > max_workers:
        raise InvalidRuntime(
            f"cooperative grid {plan.launch.grid_size} exceeds capacity {max_workers}"
        )

    barrier_count = sum(phase.barrier is not None for phase in plan.phases)
    if barrier_count > _MAX_INT32:
        raise InvalidRuntime("phase epoch values would overflow int32")
    return RuntimeAdmission(plan, barrier_count, max_workers)


def execute_phase_plan(
    plan: KernelPlan,
    kernel: PhaseKernel,
    stream: torch.cuda.Stream | None = None,
    *,
    admission: RuntimeAdmission | None = None,
    ready: torch.Tensor | None = None,
    epoch: int | None = None,
) -> RuntimeInvocation:
    active_stream = stream or torch.cuda.current_stream()
    device = active_stream.device
    with torch.cuda.device(device):
        selected_admission = admission or admit_runtime(plan)
    invocation_id = uuid.uuid4()

    if _stream_is_capturing(active_stream):
        raise InvalidRuntime("CUDA Graph capture is not supported for phase plans")
    with torch.cuda.stream(active_stream):
        selected_ready = (
            ready if ready is not None else make_ready(plan.launch.grid_size, device)
        )
        selected_epoch = 0 if epoch is None else epoch
        kernel(selected_ready, selected_epoch, active_stream)
        active_stream.synchronize()

    return RuntimeInvocation(
        invocation_id, selected_epoch, selected_admission.barrier_count
    )


def admit_event_runtime(schedule: EventSchedule) -> EventRuntimeAdmission:
    if not isinstance(schedule, EventSchedule):
        raise InvalidRuntime("admit_event_runtime expects an EventSchedule")
    simulate_event_schedule(schedule)
    event_count = len(schedule.events)
    if event_count > _MAX_INT32:
        raise InvalidRuntime("event IDs would overflow int32")
    if any(event.trigger_count < 1 for event in schedule.events):
        raise InvalidRuntime("event plans require at least one producer per event")
    max_invocations = min(
        (
            _MAX_INT32 // event.trigger_count
            for event in schedule.events
        ),
        default=_MAX_INT32,
    )
    if max_invocations < 1:
        raise InvalidRuntime("event counters would overflow int32")
    return EventRuntimeAdmission(schedule, event_count, max_invocations)


def execute_event_plan(
    schedule: EventSchedule,
    kernel: EventKernel,
    stream: torch.cuda.Stream | None = None,
    *,
    admission: EventRuntimeAdmission | None = None,
    events: torch.Tensor | None = None,
    epoch: int | None = None,
    validate_counters: bool = True,
) -> EventRuntimeInvocation:
    active_stream = stream or torch.cuda.current_stream()
    device = active_stream.device
    with torch.cuda.device(device):
        selected_admission = admission or admit_event_runtime(schedule)
    invocation_id = uuid.uuid4()

    if _stream_is_capturing(active_stream):
        raise InvalidRuntime("CUDA Graph capture is not supported for event plans")
    with torch.cuda.stream(active_stream):
        selected_events = events if events is not None else torch.zeros(
            selected_admission.event_count,
            dtype=torch.int32,
            device=device,
        )
        if tuple(selected_events.shape) != (selected_admission.event_count,):
            raise InvalidRuntime("event tensor shape does not match the event plan")
        if selected_events.dtype != torch.int32 or selected_events.device != device:
            raise InvalidRuntime("event tensor must be an int32 tensor on the stream device")
        if (
            epoch is not None
            and not 0 <= epoch < selected_admission.max_invocations
        ):
            raise InvalidRuntime("event epoch would overflow int32 counters")
        selected_epoch = 0 if epoch is None else epoch
        if events is not None and validate_counters:
            if selected_epoch == 0:
                if selected_events.count_nonzero():
                    raise InvalidRuntime(
                        "the first event invocation requires zeroed counters"
                    )
            else:
                for event in schedule.events:
                    if selected_events[event.id].item() != selected_epoch * event.trigger_count:
                        raise InvalidRuntime(
                            f"event {event.id} counters do not match epoch {selected_epoch}"
                        )
        kernel(selected_events, selected_epoch, active_stream)
        active_stream.synchronize()

    return EventRuntimeInvocation(
        invocation_id,
        selected_epoch,
        selected_admission.event_count,
    )


def _architecture_number(architecture: str) -> str:
    return architecture.removeprefix("sm").removesuffix("a")


def _cooperative_launch_supported() -> bool:
    device = cuda.cuDeviceGet(torch.cuda.current_device())[1]
    status, supported = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
        device,
    )
    return int(status) == 0 and int(supported) == 1


def _stream_is_capturing(stream: torch.cuda.Stream) -> bool:
    status, capture = cuda.cuStreamIsCapturing(cuda.CUstream(stream.cuda_stream))
    return int(status) == 0 and int(
        capture
    ) == int(cuda.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE)
