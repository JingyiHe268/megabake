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


class InvalidRuntime(ValueError):
    pass


_MAX_INT32 = 2**31 - 1


@cute.jit
def grid_barrier(ready: cute.Tensor, workers: cutlass.Int32, epoch: cutlass.Int32):
    _, _, block = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    cute.arch.sync_threads()
    if thread == 0:
        cute.arch.atomic_exch(
            ready.iterator + block,
            epoch,
            sem="release",
            scope="gpu",
        )
        if block == 0:
            all_ready = False
            while not all_ready:
                all_ready = True
                for worker in cutlass.range(workers):
                    seen = cute.arch.atomic_cas(
                        ready.iterator + worker,
                        cmp=epoch,
                        val=epoch,
                        sem="acquire",
                        scope="gpu",
                    )
                    if seen != epoch:
                        all_ready = False
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
        run_phase()
        drain()
        publish(epoch_base)


class PhaseKernel(Protocol):
    def __call__(
        self,
        ready: torch.Tensor,
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
        selected_ready = ready if ready is not None else torch.full(
            (plan.launch.grid_size + 1,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        selected_epoch = 0 if epoch is None else epoch
        kernel(selected_ready, selected_epoch, active_stream)
        active_stream.synchronize()

    return RuntimeInvocation(
        invocation_id, selected_epoch, selected_admission.barrier_count
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
