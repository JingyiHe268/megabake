from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..lowering import AxisAffine, DataIndex

AxisExpression = str | int | AxisAffine | DataIndex


@dataclass(frozen=True)
class TargetFacts:
    architecture: str
    threads_per_warp: int
    max_threads_per_cta: int
    max_shared_memory_per_cta: int
    max_registers_per_thread: int
    multiprocessor_count: int
    supports_tma: bool
    supports_wgmma: bool


@dataclass(frozen=True)
class SelectionPolicy:
    mode: str = "prefill"
    allow_generic: bool = True
    allow_tuned: bool = True
    allow_tma_gemm: bool = True
    # Large graphs use a table-driven launch: tensors come from a device
    # address table and TMA GEMM descriptors are written per phase in the
    # kernel prologue, so bodies must share one GEMM shared-memory layout.
    table_launch: bool = False
    # Table-launched GEMMs with at most 16 rows put the weight on WGMMA's M
    # side (Mirage's swapAB). One orientation serves the whole launch because
    # its GEMMs share one shared-memory layout.
    table_gemm_swap_ab: bool = False
    # The token tile (WGMMA N) of swapAB launches: 16, 32, or 64 rows.
    table_gemm_swap_ab_rows: int = 16
    # Split the K range of swapAB table GEMMs that have fewer weight tiles
    # than SMs (see bodies.gemm.split_k_count).
    table_gemm_split_k: bool = False
    numerical_policy: str = "source_semantics"
    require_measured: bool = False


@dataclass(frozen=True)
class WorkItem:
    output_tile: tuple[int, ...]
    reduction_range: tuple[int, int] | None = None


@runtime_checkable
class WorkIterator(Protocol):
    def __next__(self) -> WorkItem:
        ...


@runtime_checkable
class DeviceBody(Protocol):
    def run(
        self,
        work_iterator: WorkIterator,
        tensor_bindings: Mapping[str, Any],
        scratch: Any,
        epilogue: Any,
    ) -> None:
        ...


@dataclass(frozen=True)
class DeviceEntry:
    name: str = "run"
    arguments: tuple[str, ...] = (
        "work_iterator",
        "tensor_bindings",
        "scratch",
        "epilogue",
    )
    whole_kernel_launch: bool = False


@dataclass(frozen=True)
class SemanticContract:
    operation_family: str
    source_op_ids: tuple[str, ...]
    numerical_policy: str
    expression_hooks: tuple[str, ...]
    expression_hook_restrictions: tuple[str, ...]
    access_effects: Mapping[str, Any]
    access_restrictions: tuple[str, ...]


@dataclass(frozen=True)
class TensorLayout:
    dtype: str
    axes: tuple[AxisExpression, ...]
    strides: tuple[int, ...]
    alignment_bytes: int
    shape: tuple[int, ...] | None = None
    tail_policy: str = "predicated"


@dataclass(frozen=True)
class LayoutContract:
    inputs: Mapping[str, TensorLayout]
    outputs: Mapping[str, TensorLayout]


@dataclass(frozen=True)
class TileContract:
    cta_shape: tuple[int, ...]
    cluster_shape: tuple[int, ...]
    work_partition: str
    mma_family: str | None
    reduction_tile: tuple[int, ...]
    tail_policy: str = "predicated"


@dataclass(frozen=True)
class ParticipantContract:
    participating_threads: int
    waiting_threads: int
    warp_roles: tuple[str, ...]
    registers_per_thread: int


@dataclass(frozen=True)
class PipelineContract:
    stage_count: int
    barriers: tuple[str, ...]
    async_operations: tuple[str, ...]
    shared_memory_layouts: Mapping[str, str]
    drain_rule: str
    descriptors: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ScratchContract:
    owner: str
    size_bytes: int
    alignment_bytes: int
    initial_state: str
    reuse_rule: str


@dataclass(frozen=True)
class CompletionContract:
    outputs_ready: str
    output_tiles_ready: str
    async_work_complete: str
    scratch_reusable: str
    register_fragments: str
    warp_roles_on_return: str


@dataclass(frozen=True)
class ResourceEvidence:
    measured: bool
    source: str
    revision: str
    resources: Mapping[str, Any]
    measurements: Mapping[str, Any]


@dataclass(frozen=True)
class BodyConfig:
    provider: str
    name: str
    semantic: SemanticContract
    layouts: LayoutContract
    tiles: TileContract
    participants: ParticipantContract
    pipeline: PipelineContract
    scratch: ScratchContract
    completion: CompletionContract
    evidence: ResourceEvidence
    entrypoint: DeviceEntry = DeviceEntry()


@dataclass(frozen=True)
class ProviderRejection:
    provider: str
    reason: str


@dataclass(frozen=True)
class CandidateSet:
    configurations: tuple[BodyConfig, ...]
    rejections: tuple[ProviderRejection, ...]


@runtime_checkable
class BodyProvider(Protocol):
    def get_candidates(
        self, region: Any, target: TargetFacts, policy: SelectionPolicy
    ) -> CandidateSet:
        ...


class InvalidBodyConfig(ValueError):
    pass


def validate_body_config(
    config: BodyConfig,
    target: TargetFacts | None = None,
    policy: SelectionPolicy | None = None,
) -> None:
    if policy is not None and policy.mode not in {"prefill", "decode"}:
        raise InvalidBodyConfig(
            f"selection mode {policy.mode!r} must be 'prefill' or 'decode'"
        )
    if config.entrypoint.name != "run" or config.entrypoint.arguments != (
        "work_iterator",
        "tensor_bindings",
        "scratch",
        "epilogue",
    ):
        raise InvalidBodyConfig(
            "device entry must be run(work_iterator, tensor_bindings, scratch, epilogue)"
        )
    if config.entrypoint.whole_kernel_launch:
        raise InvalidBodyConfig("whole-kernel launch wrappers are not device bodies")

    for label, layouts in (
        ("input", config.layouts.inputs),
        ("output", config.layouts.outputs),
    ):
        if not layouts and label == "output":
            raise InvalidBodyConfig(f"body has no {label} layouts")
        for value_id, layout in layouts.items():
            if not value_id or not layout.dtype:
                raise InvalidBodyConfig(f"body has an incomplete {label} layout")
            if len(layout.axes) != len(layout.strides):
                raise InvalidBodyConfig(
                    f"{label} layout {value_id} has mismatched axes and strides"
                )
            if layout.shape is not None and len(layout.shape) != len(layout.axes):
                raise InvalidBodyConfig(
                    f"{label} layout {value_id} has mismatched shape and axes"
                )
            if layout.alignment_bytes <= 0 or layout.alignment_bytes & (
                layout.alignment_bytes - 1
            ):
                raise InvalidBodyConfig(f"{label} layout {value_id} has invalid alignment")

    for dimensions in (config.tiles.cta_shape, config.tiles.cluster_shape):
        if not dimensions or any(dimension <= 0 for dimension in dimensions):
            raise InvalidBodyConfig("body tile dimensions must be positive")

    participants = config.participants
    if participants.participating_threads <= 0:
        raise InvalidBodyConfig("body must have participating threads")
    if (
        participants.waiting_threads < 0
        or participants.waiting_threads > participants.participating_threads
    ):
        raise InvalidBodyConfig("body waiting threads are outside the participant set")
    if participants.registers_per_thread <= 0:
        raise InvalidBodyConfig("body register requirement must be positive")
    if not participants.warp_roles:
        raise InvalidBodyConfig("body must declare warp roles")
    if config.pipeline.stage_count <= 0:
        raise InvalidBodyConfig("body must declare at least one pipeline stage")
    if config.scratch.size_bytes < 0 or config.scratch.alignment_bytes <= 0:
        raise InvalidBodyConfig("body scratch requirements are invalid")
    if not config.completion.outputs_ready or not config.completion.scratch_reusable:
        raise InvalidBodyConfig("body must declare output and scratch completion rules")

    if target is not None:
        if participants.participating_threads > target.max_threads_per_cta:
            raise InvalidBodyConfig("body requires more threads than the target allows")
        if participants.registers_per_thread > target.max_registers_per_thread:
            raise InvalidBodyConfig("body requires more registers than the target allows")
        if config.scratch.size_bytes > target.max_shared_memory_per_cta:
            raise InvalidBodyConfig("body requires more shared memory than the target allows")
        if len(participants.warp_roles) != participants.participating_threads // target.threads_per_warp:
            raise InvalidBodyConfig("body warp roles do not match participating threads")

    if policy is not None and config.semantic.numerical_policy != policy.numerical_policy:
        raise InvalidBodyConfig("body numerical policy does not match the selection policy")
    if policy is not None and policy.require_measured and not config.evidence.measured:
        raise InvalidBodyConfig("body lacks required measured evidence")
