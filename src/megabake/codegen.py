from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
from typing import Any
from collections.abc import Mapping, Sequence

from .bodies.gemm import HopperWgmmaGemmPersistentKernel, is_swap_ab, kernel_tile, split_k
from .bodies.attention import emit_attention_body
from .bodies.blackwell import emit_blackwell_gemm
from .bodies.generic import (
    emit_contraction_body,
    emit_map_body,
    emit_reduction_body,
)
from .bodies.interface import (
    BodyConfig,
    LayoutContract,
    TensorLayout,
)
from .lowering import (
    AxisPredicate,
    IndexedLoad,
    IterationDomain,
    Load,
    TensorProgram,
    TensorStore,
)
from .memory import verify_plan
from .plan import ConversionBody, CuTeLayout, KernelPlan, SelectedBody
from .runtime import DYNAMIC_ADDRESS_RING, PHASE_TIMER_SLOTS, call_counter_offset
from .schedule import EventSchedule


class UnsupportedCodegen(ValueError):
    pass


@dataclass(frozen=True)
class WorkIteratorBinding:
    name: str
    phase_index: int
    region_id: str
    dtype: str
    length: int
    start: int
    step: int
    encoding: str


@dataclass(frozen=True)
class TensorArgument:
    name: str
    value_id: str
    storage_id: str
    kind: str
    layout: Any
    binding_name: str | None
    region_ids: tuple[str, ...]


@dataclass(frozen=True)
class PhaseEmission:
    index: int
    region_id: str
    body_symbol: str
    body_variant_id: int
    provider: str
    body_name: str
    work_item_count: int
    device_tile_count: int
    conversion_symbols: tuple[str, ...]
    barrier: str | None


@dataclass(frozen=True)
class LaunchMetadata:
    kernel_name: str
    launch_function_name: str
    grid_size: int
    threads_per_block: int
    cooperative: bool
    shared_memory_bytes: int
    architecture: str
    arguments: tuple[str, ...]
    work_iterators: tuple[WorkIteratorBinding, ...]
    tensors: tuple[TensorArgument, ...]
    phases: tuple[PhaseEmission, ...]
    launch_policy: str = "megabake_cooperative"
    body_variant_count: int = 0
    phase_runner_count: int = 0
    source_bytes: int = 0
    # Table-driven launches read tensor addresses and phase records from
    # device memory instead of passing one kernel parameter per tensor.
    phase_table: tuple[tuple[int, ...], ...] = ()
    dynamic_slots: tuple[int, ...] = ()
    barrier_epochs: int = 0
    # Table-launched TMA GEMMs: one row per GEMM phase (variant, A, B, and C
    # address slots), and the phase whose tensors build each variant's
    # descriptor templates on the host.
    gemm_table: tuple[tuple[int, ...], ...] = ()
    gemm_template_phases: tuple[int, ...] = ()
    # K splits per GEMM variant (split-K).
    gemm_splits: tuple[int, ...] = ()
    # Phase scratch: FP32 and Int32 global buffers for bodies whose CTAs
    # cooperate within a phase (split-K GEMMs, split-context attention).
    # Counters start at zero and each phase leaves them at zero. Phases of
    # one barrier interval get disjoint regions; columns 4 and 5 of a phase
    # row hold its float and counter offsets.
    phase_float_elements: int = 0
    phase_counter_count: int = 0


@dataclass(frozen=True)
class SourceMapEntry:
    symbol: str
    kind: str
    line_start: int
    line_end: int
    phase_index: int | None
    region_id: str | None
    body_provider: str | None
    body_name: str | None
    source_op_ids: tuple[str, ...] = ()
    source_path: str = "module.py"


@dataclass(frozen=True)
class GeneratedCuTeModule:
    source: str
    module_name: str
    source_path: Path
    source_map_path: Path
    source_map: tuple[SourceMapEntry, ...]
    launch: LaunchMetadata


_CUTLASS_TYPES = {
    "torch.bool": "cutlass.Boolean",
    "torch.uint8": "cutlass.UInt8",
    "torch.int8": "cutlass.Int8",
    "torch.int16": "cutlass.Int16",
    "torch.int32": "cutlass.Int32",
    "torch.int64": "cutlass.Int64",
    "torch.float16": "cutlass.Float16",
    "torch.bfloat16": "cutlass.BFloat16",
    "torch.float32": "cutlass.Float32",
    "torch.float64": "cutlass.Float64",
}

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


def emit_cute(
    plan: KernelPlan,
    output_dir: Path | str | None = Path("artifacts/generated"),
    *,
    event_schedule: EventSchedule | None = None,
    table: bool = False,
) -> GeneratedCuTeModule:
    """Emit one CuTe module for ``plan``.

    ``table`` selects the table-driven launch even when the plan contains TMA
    GEMM bodies; it is required once per-phase kernel parameters would exceed
    CUDA's parameter limit. Without it, plans with TMA GEMMs pass their
    descriptors as kernel parameters.
    """
    verify_plan(plan)
    if plan.launch.architecture not in {"sm90a", "sm100", "sm103", "sm120", "sm121"}:
        raise UnsupportedCodegen("codegen supports SM90 and Blackwell plans only")

    if event_schedule is None and (
        table or not any(phase.body.provider == "g0_range" for phase in plan.phases)
    ):
        source, launch, source_map = _emit_table_module(plan)
    else:
        source, launch, source_map = _emit_generic_module(plan, event_schedule)

    module_name = (
        f"megabake_{hashlib.sha256((source + repr(launch)).encode()).hexdigest()[:16]}"
    )
    source_path = Path(output_dir or Path(".")) / module_name / "module.py"
    source_map_path = source_path.with_name("source_map.json")
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(source, encoding="utf-8")
    source_map_path.write_text(
        json.dumps(
            {
                "module_name": module_name,
                "source_file": source_path.name,
                "launch": launch,
                "entries": source_map,
            },
            indent=2,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    return GeneratedCuTeModule(
        source=source,
        module_name=module_name,
        source_path=source_path,
        source_map_path=source_map_path,
        source_map=source_map,
        launch=launch,
    )


def _emit_generic_module(
    plan: KernelPlan,
    event_schedule: EventSchedule | None = None,
) -> tuple[str, LaunchMetadata, tuple[SourceMapEntry, ...]]:
    selected = {
        selection.region_id: selection for selection in plan.selection.selections
    }
    has_g0 = any(phase.body.provider == "g0_range" for phase in plan.phases)
    normalized = (
        {
            region_id: _normalize_generic_selection(
                selection, plan.launch.threads_per_block
            )
            for region_id, selection in selected.items()
        }
        if has_g0
        else selected
    )
    tensor_arguments = tuple(
        TensorArgument(
            f"tensor_{index}",
            binding.value_id,
            binding.storage_id,
            binding.kind,
            binding.layout,
            binding.binding_name,
            binding.region_ids,
        )
        for index, binding in enumerate(plan.storage.bindings)
    )
    work_iterators: list[WorkIteratorBinding] = []
    phase_records: list[PhaseEmission] = []
    variants: dict[tuple[str, str, str], tuple[str, int, Any]] = {}
    body_definitions: list[str] = []
    g0_definitions: list[str] = []
    conversion_definitions: list[str] = []
    phase_runners: list[str] = []
    phase_runner_calls: dict[int, list[tuple[int, list[str]]]] = {}
    phase_runner_symbols: dict[int, str] = {}
    phase_publishers: list[str] = []
    g0_tensor_arguments: list[TensorArgument] = []
    g0_prepared: list[tuple[str, str, str, str, str, str, str]] = []
    g0_scratch: list[str] = []
    g0_scratch_by_body: dict[str, str] = {}
    scratch_elements: dict[int, int] = {}
    g0_shared_bytes = 0

    for phase in plan.phases:
        selection = normalized[phase.region_id]
        if phase.body.provider == "g0_range":
            variant_key = (
                phase.body.provider,
                phase.body.tiles.cta_shape[:2],
                phase.body.tiles.cluster_shape[:2],
                phase.body.scratch.size_bytes,
            )
            if variant_key not in variants:
                variant_id = len(variants)
                body_symbol = f"_g0_body_{variant_id}"
                variants[variant_key] = (body_symbol, variant_id, phase.body)
                tile = phase.body.tiles.cta_shape[:2]
                cluster = phase.body.tiles.cluster_shape[:2]
                g0_definitions.append(
                    f"{body_symbol} = HopperWgmmaGemmPersistentKernel(\n"
                    "    cutlass.Float32,\n"
                    f"    {tile!r},\n"
                    f"    {cluster!r},\n"
                    "    1,\n"
                    "    True,\n"
                    ")"
                )
                g0_shared_bytes += phase.body.scratch.size_bytes
            body_symbol, variant_id, _ = variants[variant_key]

            g0_group_index = len(g0_prepared)
            g0_indices = []
            for value_id in (*selection.inputs, selection.outputs[0]):
                g0_indices.append(len(g0_tensor_arguments))
                argument_name = f"g0_tensor_{len(g0_tensor_arguments)}"
                layout = (
                    phase.body.layouts.inputs.get(value_id)
                    if value_id in selection.inputs
                    else phase.body.layouts.outputs[value_id]
                )
                g0_tensor_arguments.append(
                    TensorArgument(
                        argument_name,
                        value_id,
                        value_id,
                        "g0_input" if value_id in selection.inputs else "g0_output",
                        layout,
                        None,
                        (phase.region_id,),
                    )
                )

            scratch = g0_scratch_by_body.get(body_symbol)
            if scratch is None:
                scratch = f"_g0_scratch_{len(g0_scratch)}"
                g0_scratch_by_body[body_symbol] = scratch
                g0_scratch.append(
                    f"    {scratch} = cutlass.utils.SmemAllocator().allocate"
                    f"({body_symbol}.shared_storage)"
                )
            g0_prepared.append(
                (
                    body_symbol,
                    *g0_indices,
                )
            )
            runner_lines = [
                f"        {body_symbol}.run(\n"
                f"            g0_values[{g0_group_index}][2],\n"
                f"            g0_values[{g0_group_index}][0],\n"
                f"            {scratch},\n"
                f"            g0_values[{g0_group_index}][1],\n"
                "        )"
            ]
            phase_runner_symbols[phase.index] = f"_run_variant_{variant_id}"
            phase_runner_calls.setdefault(variant_id, []).append(
                (phase.index, runner_lines)
            )
            phase_records.append(
                PhaseEmission(
                    phase.index,
                    phase.region_id,
                    f"{body_symbol}.run",
                    variant_id,
                    phase.body.provider,
                    phase.body.name,
                    len(phase.work_items),
                    len(phase.work_items),
                    (),
                    phase.barrier.id if phase.barrier else None,
                )
            )
            if phase.barrier is not None:
                phase_publishers.append(
                    f"    @cute.jit\n"
                    f"    def _publish_phase_{phase.index}(epoch: cutlass.Int32):\n"
                    f"        grid_barrier(ready, workers, epoch + {phase.index})\n"
                )
            continue

        generated = _generate_device_body(selection)
        if generated.__dataclass_fields__.get("scratch_elements") is not None:
            scratch_elements[phase.index] = generated.scratch_elements
        variant_key = (phase.body.provider, phase.body.name, generated.source)
        if variant_key not in variants:
            variant_id = len(variants)
            body_symbol = f"_body_{variant_id}"
            variants[variant_key] = (body_symbol, variant_id, generated)
            body_definitions.append(_nested_body_source(generated.source, body_symbol))
        body_symbol, variant_id, generated = variants[variant_key]

        iterator_index = len(work_iterators)
        iterator_name = f"work_iterator_{phase.index}"
        step = (
            1
            if selection.program.family in {"reduction", "attention"}
            or selection.body.provider == "blackwell_mma"
            else generated.threads_per_cta
        )
        work_iterators.append(
            WorkIteratorBinding(
                iterator_name,
                phase.index,
                phase.region_id,
                "torch.int64",
                generated.tile_count,
                0,
                step,
                generated.work_iterator_encoding,
            )
        )

        program_input_ids = tuple(
            dict.fromkeys(load.value_id for load in selection.program.loads)
        )
        tensor_names = tuple(
            _tensor_name(
                tensor_arguments,
                value_id,
                phase.region_id,
                "input",
                prefer_conversion=any(
                    conversion.value_id == value_id
                    and conversion.target_region_id == phase.region_id
                    for conversion in phase.conversion_bodies
                ),
            )
            for value_id in program_input_ids
        ) + (
            _tensor_name(
                tensor_arguments,
                selection.outputs[0],
                phase.region_id,
                "output",
            ),
        )
        tensor_indices = tuple(
            _tensor_argument_index(tensor_arguments, tensor_name)
            for tensor_name in tensor_names
        )
        scratch_name = (
            f"_scratch_{phase.index}"
            if phase.body.scratch.size_bytes
            else "None"
        )
        runner_lines = [
            f"        {body_symbol}(\n"
            f"            work_iterators[{iterator_index}],\n"
            "            _block,\n"
            "            _thread,\n"
            "            workers,\n"
            f"            ({', '.join(f'tensors[{index}]' for index in tensor_indices)},),\n"
            f"            {scratch_name},\n"
            "            None,\n"
            "        )",
        ]

        conversion_symbols: list[str] = []
        for conversion_index, conversion in enumerate(phase.conversion_bodies):
            conversion_symbol = f"_conversion_{phase.index}_{conversion_index}"
            target_selection = normalized[conversion.target_region_id]
            conversion_generated = _generate_conversion_body(
                conversion, target_selection.body, target_selection
            )
            conversion_key = (
                "conversion",
                f"{conversion.source_layout}:{conversion.target_layout}",
                conversion_generated.source,
            )
            if conversion_key not in variants:
                variants[conversion_key] = (conversion_symbol, len(variants), conversion_generated)
                conversion_definitions.append(
                    _nested_body_source(conversion_generated.source, conversion_symbol)
                )
            else:
                conversion_symbol = variants[conversion_key][0]
            conversion_symbols.append(conversion_symbol)
            conversion_iterator_index = len(work_iterators)
            conversion_iterator_name = (
                f"work_iterator_{phase.index}_conversion_{conversion_index}"
            )
            work_iterators.append(
                WorkIteratorBinding(
                    conversion_iterator_name,
                    phase.index,
                    phase.region_id,
                    "torch.int64",
                    conversion_generated.tile_count,
                    0,
                    conversion_generated.threads_per_cta,
                    conversion_generated.work_iterator_encoding,
                )
            )
            source_tensor = _tensor_name(
                tensor_arguments,
                conversion.value_id,
                conversion.source_region_id,
                "output",
            )
            target_tensor = _tensor_name(
                tensor_arguments,
                conversion.value_id,
                conversion.target_region_id,
                "input",
                prefer_conversion=True,
            )
            runner_lines.append(
            f"        {conversion_symbol}(\n"
            f"            work_iterators[{conversion_iterator_index}],\n"
            "            _block,\n"
            "            _thread,\n"
            "            workers,\n"
            f"            (tensors[{_tensor_argument_index(tensor_arguments, source_tensor)}], tensors[{_tensor_argument_index(tensor_arguments, target_tensor)}],),\n"
                "            None,\n"
                "            None,\n"
                "        )"
            )

        if phase.barrier is not None:
            phase_publishers.append(
                f"    @cute.jit\n"
                f"    def _publish_phase_{phase.index}(epoch: cutlass.Int32):\n"
                f"        grid_barrier(ready, workers, epoch + {phase.index})\n"
            )
        phase_runner_symbols[phase.index] = f"_run_variant_{variant_id}"
        phase_runner_calls.setdefault(variant_id, []).append(
            (phase.index, runner_lines)
        )
        phase_records.append(
            PhaseEmission(
                phase.index,
                phase.region_id,
                body_symbol,
                variant_id,
                phase.body.provider,
                phase.body.name,
                len(phase.work_items),
                generated.tile_count,
                tuple(conversion_symbols),
                phase.barrier.id if phase.barrier else None,
            )
        )

    for variant_id, calls in phase_runner_calls.items():
        phase_runners.append(_phase_runner_source(variant_id, calls))

    if event_schedule is not None:
        event_by_region: dict[str, int] = {}
        for event in event_schedule.events:
            producer_regions = {
                event_schedule.task_for(task_id).region_id
                for task_id in event.producer_task_ids
            }
            if len(producer_regions) != 1:
                raise UnsupportedCodegen("event producers must share one region")
            event_by_region[producer_regions.pop()] = event.id
        phase_publishers.clear()
        for phase in plan.phases:
            wait_lines: list[str] = []
            for producer_region_id in phase.dependencies:
                if producer_region_id not in event_by_region:
                    raise UnsupportedCodegen(
                        f"region {phase.region_id} waits on an unmapped event"
                    )
                event_id = event_by_region[producer_region_id]
                wait_lines.extend(
                    (
                        "        wait_for_event(",
                        "            ready,",
                        f"            cutlass.Int32({event_id}),",
                        f"            cutlass.Int32({plan.launch.grid_size}),",
                        "            epoch,",
                        "        )",
                    )
                )
            if wait_lines:
                phase_publishers.append(
                    f"    @cute.jit\n"
                    f"    def _wait_phase_{phase.index}(epoch: cutlass.Int32):\n"
                    + "\n".join(wait_lines)
                    + "\n"
                )
            if phase.region_id in event_by_region:
                event_id = event_by_region[phase.region_id]
                phase_publishers.append(
                    f"    @cute.jit\n"
                    f"    def _publish_phase_{phase.index}():\n"
                    f"        publish_event(ready, cutlass.Int32({event_id}))\n"
                )
        phase_records = [
            replace(record, barrier=None) for record in phase_records
        ]

    shared_source = _shared_scratch_source(plan, scratch_elements)
    scratch_prelude = _scratch_prelude(plan, scratch_elements)
    shared_memory_bytes = (
        g0_shared_bytes + _generic_scratch_bytes(plan, scratch_elements)
        if has_g0
        else plan.launch.dynamic_shared_memory_bytes
    )
    target = plan.selection.target
    if target is not None and shared_memory_bytes > target.max_shared_memory_per_cta:
        raise UnsupportedCodegen(
            f"generated kernel requires {shared_memory_bytes} shared-memory bytes"
        )
    g0_prepare_lines = [
        f"        {body_symbol}.prepare(\n"
        f"            g0_tensors[{group_index}][0],\n"
        f"            g0_tensors[{group_index}][1],\n"
        f"            g0_tensors[{group_index}][2],\n"
        "            max_active_clusters,\n"
        "        ),"
        for group_index, (body_symbol, *rest) in enumerate(g0_prepared)
    ]
    g0_values_source = (
        "    g0_values = (\n" + "\n".join(g0_prepare_lines) + "\n    )\n"
        if g0_prepare_lines
        else ""
    )
    launch_smem = "" if has_g0 else f"        smem={shared_memory_bytes},\n"
    runtime_import = (
        "from megabake.runtime import cooperative_event_worker_loop, publish_event, wait_for_event"
        if event_schedule is not None
        else "from megabake.runtime import cooperative_worker_loop, grid_barrier"
    )
    kernel_arguments = [
        "ready: cute.Tensor",
        "workers: cutlass.Int32",
        "epoch: cutlass.Int32",
        "work_iterators: tuple[cute.Tensor, ...]",
        "tensors: tuple[cute.Tensor, ...]",
        *(("g0_values: tuple[tuple[cute.Tensor, cute.Tensor, cute.Tensor], ...]",) if has_g0 else ()),
    ]
    launch_arguments = [
        "ready: cute.Tensor",
        "workers: cutlass.Int32",
        "epoch: cutlass.Int32",
        "work_iterators: tuple[cute.Tensor, ...]",
        "tensors: tuple[cute.Tensor, ...]",
        *(("g0_tensors: tuple[tuple[cute.Tensor, cute.Tensor, cute.Tensor], ...]",) if has_g0 else ()),
        "stream: cuda.CUstream",
    ]
    if has_g0:
        launch_arguments.insert(-1, "max_active_clusters: cutlass.Constexpr")
    source = f'''import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass._mlir.dialects import llvm

{runtime_import}
from megabake.bodies.gemm import HopperWgmmaGemmPersistentKernel

{"\n\n".join(g0_definitions)}


{shared_source}

@cute.kernel
def megabake_kernel(
{("," + chr(10)).join(f"    {argument}" for argument in kernel_arguments)}
):
    _, _, _block = cute.arch.block_idx()
    _thread, _, _ = cute.arch.thread_idx()
{scratch_prelude}
{chr(10).join(g0_scratch)}
{"".join(body_definitions)}
{"".join(conversion_definitions)}
    @cute.jit
    def _no_op():
        pass

    @cute.jit
    def _ignore_epoch(epoch: cutlass.Int32):
        pass

    @cute.jit
    def _cta_sync(epoch: cutlass.Int32):
        cute.arch.sync_threads()

    @cute.jit
    def _run_terminal(phase_index: cutlass.Int32):
        pass

{"".join(phase_runners)}
{"".join(phase_publishers)}
{_event_worker_loop(plan, phase_runner_symbols, event_schedule)}


@cute.jit
def launch(
{("," + chr(10)).join(f"    {argument}" for argument in launch_arguments)}
):
{g0_values_source}    megabake_kernel(
        ready,
        workers,
        epoch,
        work_iterators,
        tensors{"," if has_g0 else ""}
{f"        g0_values{chr(10)}" if has_g0 else ""}    ).launch(
        grid=[1, 1, workers],
        block=[{plan.launch.threads_per_block}, 1, 1],
        cooperative=True,
{launch_smem}        stream=stream,
    )
'''
    launch_metadata = LaunchMetadata(
        "megabake_kernel",
        "launch",
        plan.launch.grid_size,
        plan.launch.threads_per_block,
        True,
        shared_memory_bytes,
        plan.launch.architecture,
        ("ready", "workers", "epoch", "work_iterators", "tensors")
        + (("g0_tensors",) if has_g0 else ())
        + (("max_active_clusters",) if has_g0 else ())
        + ("stream",),
        tuple(work_iterators),
        tuple((*tensor_arguments, *g0_tensor_arguments)),
        tuple(phase_records),
        launch_policy=(
            "megabake_cooperative_events"
            if event_schedule is not None
            else "megabake_cooperative"
        ),
        body_variant_count=len(g0_definitions) + len(body_definitions),
        phase_runner_count=len(phase_runner_calls),
        source_bytes=len(source.encode("utf-8")),
    )
    return source, launch_metadata, _source_map(source, plan, phase_records)


_DYNAMIC_BINDING_KINDS = frozenset({"input", "state", "output", "alias"})


def _emit_table_module(
    plan: KernelPlan,
) -> tuple[str, LaunchMetadata, tuple[SourceMapEntry, ...]]:
    """Emit one body per variant and drive phases from a device-side table.

    Every tensor becomes an address-table slot rebuilt in the kernel from its
    static layout, so kernel parameters and generated code stay bounded by the
    number of distinct bodies rather than by the number of layers.
    """
    has_gemm = any(phase.body.provider == "g0_range" for phase in plan.phases)
    selected = {
        selection.region_id: (
            _normalize_generic_selection(selection, plan.launch.threads_per_block)
            if has_gemm
            else selection
        )
        for selection in plan.selection.selections
    }
    tensor_arguments = tuple(
        TensorArgument(
            f"tensor_{index}",
            binding.value_id,
            binding.storage_id,
            binding.kind,
            binding.layout or _value_layout(plan, binding.value_id),
            binding.binding_name,
            binding.region_ids,
        )
        for index, binding in enumerate(plan.storage.bindings)
    )
    gemm_variants: dict[tuple[Any, ...], int] = {}
    gemm_definitions: list[str] = []
    gemm_shapes: list[tuple[int, int, int, bool]] = []
    gemm_template_phases: list[int] = []
    gemm_rows: list[tuple[int, ...]] = []
    gemm_splits: list[int] = []
    # Phase index -> (FP32 elements, Int32 counters) of phase scratch.
    phase_scratch: dict[int, tuple[int, int]] = {}
    gemm_scratch_bytes: set[int] = set()
    variants: dict[tuple[Any, ...], tuple[str, int, Any]] = {}
    body_definitions: list[str] = []
    runner_variants: dict[tuple[Any, ...], tuple[int, int, list[str]]] = {}
    work_iterators: list[WorkIteratorBinding] = []
    phase_records: list[PhaseEmission] = []
    phase_rows: list[tuple[int, ...]] = []
    # Work tiles per phase, for spreading concurrent phases across CTAs.
    phase_tiles: list[int] = []
    # Stateful attention phase -> address slot of its KV valid length.
    length_publishes: dict[int, int] = {}
    # Stateful attention phases whose interval ends in a grid barrier that no
    # other length publish uses.
    deferred_publishes: set[int] = set()
    closing_barriers: set[int] = set()
    for phase in plan.phases:
        if not selected[phase.region_id].program.state_accesses:
            continue
        closing = next(
            (
                position
                for position in range(phase.index, len(plan.phases))
                if plan.phases[position].barrier is not None
            ),
            None,
        )
        if closing is not None and closing not in closing_barriers:
            closing_barriers.add(closing)
            deferred_publishes.add(phase.index)
    representative_phases: list[Any] = []
    scratch_elements: dict[int, int] = {}

    def body_variant(
        generated: Any, key: tuple[Any, ...], symbol: str | None = None
    ) -> tuple[str, int]:
        if key not in variants:
            symbol = symbol or f"_body_{len(variants)}"
            variants[key] = (symbol, len(variants), generated)
            body_definitions.append(_nested_body_source(generated.source, symbol))
        symbol, variant_id, _ = variants[key]
        return symbol, variant_id

    def iterator_index(name: str, phase: Any, generated: Any, step: int) -> int:
        work_iterators.append(
            WorkIteratorBinding(
                name,
                phase.index,
                phase.region_id,
                "torch.int64",
                generated.tile_count,
                0,
                step,
                generated.work_iterator_encoding,
            )
        )
        return len(work_iterators) - 1

    for phase in plan.phases:
        selection = selected[phase.region_id]
        if phase.body.provider == "g0_range":
            if phase.conversion_bodies:
                raise UnsupportedCodegen(
                    f"table GEMM phase {phase.region_id} needs a layout conversion"
                )
            program = selection.program
            rows, columns = program.store.shape
            reduction = next(
                domain.size
                for domain in program.iteration_domains
                if domain.axis in program.reduction_axes
            )
            tile = kernel_tile(phase.body)
            splits = split_k(phase.body)
            gemm_scratch_bytes.add(phase.body.scratch.size_bytes)
            gemm_key = (
                tile, phase.body.scratch.size_bytes, rows, columns, reduction, splits
            )
            if gemm_key not in gemm_variants:
                gemm_variant = len(gemm_variants)
                gemm_variants[gemm_key] = gemm_variant
                # CUTLASS problem shape; swapAB computes the transposed product.
                gemm_shapes.append(
                    (columns, rows, reduction, True)
                    if is_swap_ab(phase.body)
                    else (rows, columns, reduction, False)
                )
                gemm_splits.append(splits)
                gemm_template_phases.append(phase.index)
                symbol = f"_gemm_{gemm_variant}"
                variants[("g0_range", gemm_key)] = (symbol, len(variants), None)
                gemm_definitions.append(
                    f"{symbol} = HopperWgmmaGemmPersistentKernel(\n"
                    "    cutlass.Float32,\n"
                    f"    {tile!r},\n"
                    "    (1, 1),\n"
                    "    1,\n"
                    "    True,\n"
                    f"    smem_capacity={phase.body.scratch.size_bytes},\n"
                    "    reallocate_registers=False,\n"
                    f"    splits={splits},\n"
                    ")"
                )
            gemm_variant = gemm_variants[gemm_key]
            body_symbol, body_variant_id, _ = variants[("g0_range", gemm_key)]
            gemm_slots = [
                _tensor_argument_index(
                    tensor_arguments,
                    _tensor_name(tensor_arguments, load.value_id, phase.region_id, "input"),
                )
                for load in program.loads
            ]
            if is_swap_ab(phase.body):
                # CUTLASS A is the weight and B the activation.
                gemm_slots = [gemm_slots[1], gemm_slots[0], *gemm_slots[2:]]
            gemm_slots.append(
                _tensor_argument_index(
                    tensor_arguments,
                    _tensor_name(
                        tensor_arguments, selection.outputs[0], phase.region_id, "output"
                    ),
                )
            )
            gemm_index = len(gemm_rows)
            gemm_rows.append((gemm_variant, *gemm_slots))
            if splits > 1:
                output_tiles = -(-gemm_shapes[gemm_variant][0] // tile[0]) * -(
                    -gemm_shapes[gemm_variant][1] // tile[1]
                )
                # FP32 partials of every split of every tile, one counter per tile.
                phase_scratch[phase.index] = (
                    output_tiles * splits * tile[0] * tile[1],
                    output_tiles,
                )
            runner_key = ("g0_range", gemm_variant)
            if runner_key not in runner_variants:
                representative_phases.append(phase)
                split_state = (
                    [
                        "                (",
                        "                    phase_floats,",
                        "                    phase_counters,",
                        "                    phase_table[phase, 4],",
                        "                    phase_table[phase, 5],",
                        "                ),",
                    ]
                    if splits > 1
                    else []
                )
                runner_variants[runner_key] = (
                    len(runner_variants),
                    0,
                    [
                        f"            {body_symbol}.run_table(",
                        f"                gemm_values[{gemm_variant}],",
                        "                _gemm_scratch,",
                        "                table_descriptors(tma_descriptors, phase_table[phase, 6]),",
                        "                _rotated,",
                        *split_state,
                        "            )",
                    ],
                )
            runner_id = runner_variants[runner_key][0]
            phase_rows.append(
                (runner_id, int(phase.barrier is not None), 0, 0, 0, 0, gemm_index)
            )
            phase_tiles.append(len(phase.work_items) * splits)
            phase_records.append(
                PhaseEmission(
                    phase.index,
                    phase.region_id,
                    f"{body_symbol}.run_table",
                    body_variant_id,
                    phase.body.provider,
                    phase.body.name,
                    len(phase.work_items),
                    len(phase.work_items),
                    (),
                    phase.barrier.id if phase.barrier else None,
                )
            )
            continue
        # The table kernel publishes KV valid lengths after the barrier that
        # ends the attention's interval (see the phase loop below); without
        # such a barrier the body publishes from a single CTA itself.
        generated = _generate_device_body(
            selection,
            defer_length_publish=phase.index in deferred_publishes,
            workers=plan.launch.grid_size,
        )
        scratch_floats = getattr(generated, "phase_floats", 0)
        scratch_counters = getattr(generated, "phase_counters", 0)
        if scratch_floats or scratch_counters:
            phase_scratch[phase.index] = (scratch_floats, scratch_counters)
        if generated.__dataclass_fields__.get("scratch_elements") is not None:
            scratch_elements[phase.index] = generated.scratch_elements
        body_symbol, body_variant_id = body_variant(
            generated, (phase.body.provider, phase.body.name, generated.source)
        )
        program_input_ids = tuple(
            dict.fromkeys(load.value_id for load in selection.program.loads)
        )
        slots = [
            _tensor_argument_index(
                tensor_arguments,
                _tensor_name(
                    tensor_arguments,
                    value_id,
                    phase.region_id,
                    "input",
                    prefer_conversion=any(
                        conversion.value_id == value_id
                        and conversion.target_region_id == phase.region_id
                        for conversion in phase.conversion_bodies
                    ),
                ),
            )
            for value_id in program_input_ids
        ]
        slots.append(
            _tensor_argument_index(
                tensor_arguments,
                _tensor_name(
                    tensor_arguments, selection.outputs[0], phase.region_id, "output"
                ),
            )
        )
        if phase.index in deferred_publishes:
            length_value = next(
                load.value_id
                for load in selection.program.loads
                if load.id == "attention:valid_length"
            )
            length_publishes[phase.index] = slots[program_input_ids.index(length_value)]
        calls: list[tuple[str, tuple[int, ...], Any, int]] = [
            (
                body_symbol,
                tuple(slots),
                generated,
                1
                if selection.program.family in {"reduction", "attention"}
                or selection.body.provider == "blackwell_mma"
                else generated.threads_per_cta,
            )
        ]
        conversion_symbols: list[str] = []
        for conversion_index, conversion in enumerate(phase.conversion_bodies):
            target_selection = selected[conversion.target_region_id]
            conversion_generated = _generate_conversion_body(
                conversion, target_selection.body, target_selection
            )
            conversion_symbol, _ = body_variant(
                conversion_generated,
                (
                    "conversion",
                    f"{conversion.source_layout}:{conversion.target_layout}",
                    conversion_generated.source,
                ),
                f"_conversion_{phase.index}_{conversion_index}",
            )
            conversion_symbols.append(conversion_symbol)
            source_slot = _tensor_argument_index(
                tensor_arguments,
                _tensor_name(
                    tensor_arguments,
                    conversion.value_id,
                    conversion.source_region_id,
                    "output",
                ),
            )
            target_slot = _tensor_argument_index(
                tensor_arguments,
                _tensor_name(
                    tensor_arguments,
                    conversion.value_id,
                    conversion.target_region_id,
                    "input",
                    prefer_conversion=True,
                ),
            )
            calls.append(
                (
                    conversion_symbol,
                    (source_slot, target_slot),
                    conversion_generated,
                    conversion_generated.threads_per_cta,
                )
            )

        # Phases share a runner when their bodies, slot layouts, and scratch
        # agree; only the slot numbers in the phase table differ.
        runner_key = (
            tuple(
                (
                    symbol,
                    tuple(_layout_key(tensor_arguments[slot].layout) for slot in call_slots),
                    step,
                )
                for symbol, call_slots, _, step in calls
            ),
            phase.body.scratch.size_bytes,
            scratch_elements.get(phase.index),
        )
        if runner_key not in runner_variants:
            runner_id = len(runner_variants)
            representative_phases.append(phase)
            scratch_name = (
                f"_scratch_{phase.index}" if phase.body.scratch.size_bytes else "None"
            )
            lines: list[str] = []
            column = 0
            for call_index, (symbol, call_slots, call_generated, step) in enumerate(calls):
                iterator = iterator_index(
                    f"work_iterator_variant_{runner_id}_{call_index}",
                    phase,
                    call_generated,
                    step,
                )
                tensors = ", ".join(
                    _slot_tensor_code(column + offset, tensor_arguments[slot].layout)
                    for offset, slot in enumerate(call_slots)
                )
                column += len(call_slots)
                lines.extend(
                    (
                        f"            {symbol}(",
                        f"                work_iterators[{iterator}],",
                        "                _rotated,",
                        "                _thread,",
                        "                workers,",
                        f"                ({tensors},),",
                        f"                {scratch_name if call_index == 0 else 'None'},",
                        (
                            "                (phase_floats, phase_counters, "
                            "phase_table[phase, 4], phase_table[phase, 5]),"
                            if getattr(call_generated, "phase_floats", 0)
                            or getattr(call_generated, "phase_counters", 0)
                            else "                None,"
                        ),
                        "            )",
                    )
                )
            runner_variants[runner_key] = (runner_id, column, lines)
        runner_id, _, _ = runner_variants[runner_key]
        phase_rows.append(
            (
                runner_id,
                int(phase.barrier is not None),
                0,
                0,
                0,
                0,
                *(slot for _, call_slots, _, _ in calls for slot in call_slots),
            )
        )
        phase_tiles.append(max(call[2].tile_count for call in calls))
        phase_records.append(
            PhaseEmission(
                phase.index,
                phase.region_id,
                body_symbol,
                body_variant_id,
                phase.body.provider,
                phase.body.name,
                len(phase.work_items),
                generated.tile_count,
                tuple(conversion_symbols),
                phase.barrier.id if phase.barrier else None,
            )
        )

    # Phases of one barrier interval run concurrently, so each takes its own
    # phase-scratch region; later intervals reuse the regions.
    float_elements = counter_count = 0
    float_offset = counter_offset = 0
    for index, phase in enumerate(plan.phases):
        floats, counters = phase_scratch.get(phase.index, (0, 0))
        if floats or counters:
            row = phase_rows[index]
            phase_rows[index] = (*row[:4], float_offset, counter_offset, *row[6:])
            float_offset += floats
            counter_offset += counters
            float_elements = max(float_elements, float_offset)
            counter_count = max(counter_count, counter_offset)
        if phase.barrier is not None:
            float_offset = counter_offset = 0

    # Every body starts its tiles at CTA 0. Phases in one barrier interval run
    # concurrently, so each starts where the previous phase's tiles ended.
    offset = 0
    for index, phase in enumerate(plan.phases):
        if index == 0 or plan.phases[index - 1].barrier is not None:
            offset = 0
        row = phase_rows[index]
        phase_rows[index] = (*row[:2], offset % plan.launch.grid_size, *row[3:])
        offset += phase_tiles[index]
    for index, length_slot in length_publishes.items():
        closing = next(
            (
                position
                for position in range(index, len(plan.phases))
                if plan.phases[position].barrier is not None
            ),
            None,
        )
        if closing is None or phase_rows[closing][3] != 0:
            raise UnsupportedCodegen(
                f"KV length of phase {index} has no grid barrier of its own to follow"
            )
        row = phase_rows[closing]
        phase_rows[closing] = (*row[:3], length_slot + 1, *row[4:])
    width = max(len(row) for row in phase_rows)
    # Barriers of one call get dense ordinals: the address barrier, the GEMM
    # descriptor barrier, then each phase barrier in order. Column 1 of a
    # phase row holds its barrier's ordinal (0: no barrier).
    phase_epoch_offset = 2 if has_gemm else 1
    ordinal = phase_epoch_offset
    for index, phase in enumerate(plan.phases):
        if phase.barrier is not None:
            row = phase_rows[index]
            phase_rows[index] = (row[0], ordinal, *row[2:])
            ordinal += 1
    barrier_epochs = ordinal
    phase_table = tuple(row + (0,) * (width - len(row)) for row in phase_rows)
    dynamic_slots = tuple(
        index
        for index, argument in enumerate(tensor_arguments)
        if argument.kind in _DYNAMIC_BINDING_KINDS and argument.layout is not None
    )


    scratch_plan = _representative_scratch_plan(plan, representative_phases)
    shared_source = _shared_scratch_source(scratch_plan, scratch_elements)
    scratch_prelude = _scratch_prelude(scratch_plan, scratch_elements)
    shared_memory_bytes = plan.launch.dynamic_shared_memory_bytes
    if has_gemm:
        if len(gemm_scratch_bytes) != 1 or len({key[0] for key in gemm_variants}) != 1:
            raise UnsupportedCodegen(
                "table-launched GEMMs must share one tile and shared-memory layout"
            )
        # The allocator sizes dynamic shared memory from the GEMM stages and
        # the generic union; the 1 KiB covers the GEMM buffers' alignment.
        shared_memory_bytes = (
            next(iter(gemm_scratch_bytes))
            + _generic_scratch_bytes(scratch_plan, scratch_elements)
            + 1024
        )
    target = plan.selection.target
    if target is not None and shared_memory_bytes > target.max_shared_memory_per_cta:
        raise UnsupportedCodegen(
            f"generated kernel requires {shared_memory_bytes} shared-memory bytes"
        )

    dispatch = "\n".join(
        f"        if variant == {runner_id}:\n" + "\n".join(lines)
        for runner_id, _, lines in runner_variants.values()
    )
    # Per-call boundary addresses arrive through a host-mapped ring of
    # buffers whose base never changes; the call counter selects this call's
    # entry, whose first word repeats the counter. The first CTA checks it and
    # copies the addresses into the device table before the first barrier.
    ring_words = len(dynamic_slots) + 1
    dynamic_writes = (
        "        _dynamic = cute.make_tensor(\n"
        "            cute.make_ptr(\n"
        "                cutlass.Int64,\n"
        f"                dynamic_addresses + cutlass.Int64(_call % {DYNAMIC_ADDRESS_RING}) * {ring_words * 8},\n"
        "                cute.AddressSpace.gmem,\n"
        "                assumed_align=8,\n"
        "            ),\n"
        f"            cute.make_layout({ring_words}),\n"
        "        )\n"
        "        if _dynamic[0] != cutlass.Int64(_call):\n"
        "            llvm.intr_trap()\n"
        + (
            f"        for _position in cutlass.range(_thread, {len(dynamic_slots)}, {plan.launch.threads_per_block}, unroll=1):\n"
            "            addresses[dynamic_slot_table[_position]] = _dynamic[_position + 1]\n"
            if dynamic_slots
            else ""
        )

    )
    # Phase barriers follow the address barrier and, with GEMMs, the
    # descriptor barrier.

    gemm_kernel_arguments = (
        "    gemm_table: cute.Tensor,\n"
        "    tma_descriptors: cute.Tensor,\n"
        "    gemm_values,\n"
        if has_gemm
        else ""
    )
    gemm_launch_arguments = (
        "    gemm_table: cute.Tensor,\n"
        "    tma_descriptors: cute.Tensor,\n"
        "    gemm_tensors: tuple[tuple[cute.Tensor, cute.Tensor, cute.Tensor], ...],\n"
        "    max_active_clusters: cutlass.Constexpr,\n"
        if has_gemm
        else ""
    )
    gemm_prepare = (
        "    gemm_values = (\n"
        + "".join(
            f"        _gemm_{variant}.prepare(\n"
            f"            gemm_tensors[{variant}][0],\n"
            f"            gemm_tensors[{variant}][1],\n"
            f"            gemm_tensors[{variant}][2],\n"
            "            max_active_clusters,\n"
            "        ),\n"
            for variant in range(len(gemm_shapes))
        )
        + "    )\n"
        if has_gemm
        else ""
    )
    gemm_scratch = (
        "    _gemm_scratch = cutlass.utils.SmemAllocator().allocate(_gemm_0.shared_storage)\n"
        if has_gemm
        else ""
    )
    gemm_builds = "\n".join(
        f"            if gemm_variant == {variant}:\n"
        f"                build_table_descriptors(\n"
        f"                    gemm_values[{variant}],\n"
        "                    tma_descriptors,\n"
        "                    gemm,\n"
        "                    addresses[gemm_table[gemm, 1]],\n"
        "                    addresses[gemm_table[gemm, 2]],\n"
        "                    addresses[gemm_table[gemm, 3]],\n"
        f"                    {shape[:3]!r},\n"
        f"                    {shape[3]!r},\n"
        f"                    {gemm_splits[variant]!r},\n"
        "                )"
        for variant, shape in enumerate(gemm_shapes)
    )
    # One warp per CTA writes the descriptors of the GEMM phases it owns from
    # this call's addresses, then a barrier publishes them to every TMA user.
    gemm_prologue = (
        "    _warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())\n"
        "    if _warp == 0:\n"
        f"        for gemm in cutlass.range(_block, {len(gemm_rows)}, workers, unroll=1):\n"
        "            gemm_variant = gemm_table[gemm, 0]\n"
        f"{gemm_builds}\n"
        "        cute.nvgpu.cpasync.fence_tma_desc_release()\n"
        "    counting_grid_barrier(ready, workers, _ordinal + cutlass.Int32(1))\n"
        if has_gemm
        else ""
    )
    gemm_kernel_call = (
        "        gemm_table,\n        tma_descriptors,\n        gemm_values,\n"
        if has_gemm
        else ""
    )
    launch_smem = "" if has_gemm else f"        smem={shared_memory_bytes},\n"
    gemm_import = (
        "from megabake.bodies.gemm import (\n"
        "    HopperWgmmaGemmPersistentKernel,\n"
        "    build_table_descriptors,\n"
        "    table_descriptors,\n"
        ")\n"
        if has_gemm
        else ""
    )
    source = f"""import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass._mlir.dialects import llvm
from cutlass.cute.tensor import _Tensor

from megabake.runtime import counting_grid_barrier
{gemm_import}
{chr(10).join(gemm_definitions)}


@cute.jit
def _slot(
    addresses: cute.Tensor,
    phase_table: cute.Tensor,
    phase: cutlass.Int32,
    column: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
    layout: cutlass.Constexpr,
    alignment: cutlass.Constexpr,
):
    address = addresses[phase_table[phase, 6 + column]]
    if cutlass.const_expr(dtype is cutlass.Boolean):
        # Boolean tensors are byte-backed, matching DLPack-bound tensors.
        storage = cute.make_tensor(
            cute.make_ptr(
                cutlass.Int8, address, cute.AddressSpace.gmem, assumed_align=alignment
            ),
            layout,
        )
        return _Tensor(storage, dtype=cutlass.Boolean)
    return cute.make_tensor(
        cute.make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=alignment),
        layout,
    )


{shared_source}

@cute.kernel
def megabake_kernel(
    ready: cute.Tensor,
    workers: cutlass.Int32,
    addresses: cute.Tensor,
    phase_table: cute.Tensor,
    work_iterators: tuple[cute.Tensor, ...],
    dynamic_addresses: cutlass.Int64,
    dynamic_slot_table: cute.Tensor,
    phase_floats: cute.Tensor,
    phase_counters: cute.Tensor,
{gemm_kernel_arguments}):
    _, _, _block = cute.arch.block_idx()
    _thread, _, _ = cute.arch.thread_idx()
    # Every CTA reads the call counter before the first barrier; block 0
    # advances it after that barrier, for the next launch on this stream.
    _call = ready[workers + {call_counter_offset()}]
    _ordinal = _call * cutlass.Int32({barrier_epochs})
{scratch_prelude}
{gemm_scratch}
{"".join(body_definitions)}
    # Publish this call's boundary addresses before any phase reads them.
    if _block == 0:
{dynamic_writes}
    counting_grid_barrier(ready, workers, _ordinal)
    if _block == 0:
        if _thread == 0:
            ready[workers + {call_counter_offset()}] = _call + cutlass.Int32(1)
{gemm_prologue}{_phase_timer_stamp(len(plan.phases), "    ")}
    for phase in cutlass.range({len(plan.phases)}, unroll=1):
        variant = phase_table[phase, 0]
        _rotated = (_block + workers - phase_table[phase, 2]) % workers
{dispatch}
        if phase_table[phase, 1] != 0:
            counting_grid_barrier(ready, workers, _ordinal + phase_table[phase, 1])
{_phase_timer_stamp("phase", "            ")}            # Every reader of this interval's KV valid length has finished.
            if phase_table[phase, 3] != 0:
                if _block == 0:
                    if _thread == 0:
                        _length = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Int32,
                                addresses[phase_table[phase, 3] - 1],
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            ),
                            cute.make_layout(1),
                        )
                        _length[0] = _length[0] + cutlass.Int32(1)
        else:
            # Independent phases share a barrier interval; the CTA still
            # synchronizes before the next body reuses its shared scratch.
            cute.arch.sync_threads()


@cute.jit
def launch(
    ready: cute.Tensor,
    workers: cutlass.Int32,
    addresses: cute.Tensor,
    phase_table: cute.Tensor,
    work_iterators: tuple[cute.Tensor, ...],
    dynamic_addresses: cutlass.Int64,
    dynamic_slot_table: cute.Tensor,
    phase_floats: cute.Tensor,
    phase_counters: cute.Tensor,
{gemm_launch_arguments}    stream: cuda.CUstream,
):
{gemm_prepare}    megabake_kernel(
        ready,
        workers,
        addresses,
        phase_table,
        work_iterators,
        dynamic_addresses,
        dynamic_slot_table,
        phase_floats,
        phase_counters,
{gemm_kernel_call}    ).launch(
        grid=[1, 1, workers],
        block=[{plan.launch.threads_per_block}, 1, 1],
        cooperative=True,
{launch_smem}        stream=stream,
    )
"""
    launch_metadata = LaunchMetadata(
        "megabake_kernel",
        "launch",
        plan.launch.grid_size,
        plan.launch.threads_per_block,
        True,
        shared_memory_bytes,
        plan.launch.architecture,
        (
            "ready",
            "workers",
            "addresses",
            "phase_table",
            "work_iterators",
            "dynamic_addresses",
            "dynamic_slot_table",
            "phase_floats",
            "phase_counters",
            *(
                ("gemm_table", "tma_descriptors", "gemm_tensors", "max_active_clusters")
                if has_gemm
                else ()
            ),
            "stream",
        ),
        tuple(work_iterators),
        tensor_arguments,
        tuple(phase_records),
        launch_policy="megabake_cooperative_table",
        body_variant_count=len(variants),
        phase_runner_count=len(runner_variants),
        source_bytes=len(source.encode("utf-8")),
        phase_table=phase_table,
        dynamic_slots=dynamic_slots,
        barrier_epochs=barrier_epochs,
        gemm_table=tuple(gemm_rows),
        gemm_template_phases=tuple(gemm_template_phases),
        gemm_splits=tuple(gemm_splits),
        phase_float_elements=float_elements,
        phase_counter_count=counter_count,

    )
    return source, launch_metadata, _source_map(source, plan, phase_records)


def _phase_timer_stamp(slot: int | str, indent: str) -> str:
    """Block 0 records when every CTA has passed this barrier."""
    if not PHASE_TIMER_SLOTS:
        return ""
    if isinstance(slot, int) and slot >= PHASE_TIMER_SLOTS:
        raise UnsupportedCodegen(
            f"phase timer holds {PHASE_TIMER_SLOTS} stamps, plan needs {slot + 1}"
        )
    return (
        f"{indent}if _block == 0:\n"
        f"{indent}    if _thread == 0:\n"
        f"{indent}        ready[workers + 2 + {slot}] = cute.arch.globaltimer_lo()\n"
    )


def _value_layout(plan: KernelPlan, value_id: str) -> CuTeLayout | None:
    value = plan.graph.values[value_id]
    if value.shape is None or value.strides is None or value.dtype is None:
        if value.users:
            raise UnsupportedCodegen(
                f"table launch needs a static layout for {value_id}"
            )
        # Unused non-tensor inputs such as flags keep an empty slot.
        return None
    return CuTeLayout(
        value.dtype, value.shape, value.strides, _DTYPE_BYTES[value.dtype]
    )


def _layout_key(layout: Any) -> tuple[Any, ...]:
    return (layout.dtype, tuple(layout.shape), tuple(layout.strides))


def _slot_tensor_code(column: int, layout: Any) -> str:
    dtype = _CUTLASS_TYPES[layout.dtype]
    shape = tuple(layout.shape)
    strides = tuple(layout.strides)
    layout_code = (
        "cute.make_layout(1)"
        if not shape
        else f"cute.make_layout({shape!r}, stride={strides!r})"
    )
    alignment = _DTYPE_BYTES[layout.dtype]
    return (
        f"_slot(addresses, phase_table, phase, {column}, {dtype}, "
        f"{layout_code}, {alignment})"
    )


def _representative_scratch_plan(plan: KernelPlan, phases: Sequence[Any]) -> KernelPlan:
    indices = {phase.index for phase in phases}
    worker = plan.storage.worker_scratch[0]
    representative_worker = replace(
        worker,
        phases=tuple(
            layout
            for layout, phase in zip(worker.phases, plan.phases, strict=True)
            if phase.index in indices
        ),
    )
    return replace(
        plan,
        phases=tuple(phase for phase in plan.phases if phase.index in indices),
        storage=replace(
            plan.storage,
            worker_scratch=(representative_worker, *plan.storage.worker_scratch[1:]),
        ),
    )


def _phase_runner_source(
    variant_id: int, calls: list[tuple[int, list[str]]]
) -> str:
    ordered_calls = sorted(calls)

    def dispatch(
        entries: list[tuple[int, list[str]]], indent: str
    ) -> list[str]:
        if len(entries) == 1:
            return [f"{indent}if phase_index == {entries[0][0]}:", *(
                f"{indent}    {line.strip()}" for line in entries[0][1]
            )]
        midpoint = len(entries) // 2
        return [
            f"{indent}if phase_index < {entries[midpoint][0]}:",
            *dispatch(entries[:midpoint], indent + "    "),
            f"{indent}else:",
            *dispatch(entries[midpoint:], indent + "    "),
        ]

    lines = [
        "    @cute.jit",
        f"    def _run_variant_{variant_id}(phase_index: cutlass.Int32):",
        *(
            ordered_calls[0][1]
            if len(ordered_calls) == 1
            else dispatch(ordered_calls, "        ")
        ),
    ]
    return "\n".join(lines) + "\n"


def _event_worker_loop(
    plan: KernelPlan,
    phase_runner_symbols: Mapping[int, str],
    event_schedule: EventSchedule | None,
) -> str:
    if event_schedule is None:
        return f'''    cooperative_worker_loop(
        ready,
        workers,
        epoch,
        (
{("," + chr(10)).join([f"            ({phase.index}, {phase_runner_symbols[phase.index]}, _no_op, {f'_publish_phase_{phase.index}' if phase.barrier is not None else ('_cta_sync' if phase.index + 1 < len(plan.phases) else '_ignore_epoch')})" for phase in plan.phases] + [f"            ({len(plan.phases)}, _run_terminal, _no_op, _ignore_epoch)"])}
        ),
    )
'''
    event_regions = {
        event_schedule.task_for(task_id).region_id
        for event in event_schedule.events
        for task_id in event.producer_task_ids
    }
    entries = tuple(
        (
            phase.index,
            f"_wait_phase_{phase.index}" if phase.dependencies else "_ignore_epoch",
            phase_runner_symbols[phase.index],
            f"_publish_phase_{phase.index}"
            if phase.region_id in event_regions
            else "_no_op",
        )
        for phase in plan.phases
    )
    return f'''    cooperative_event_worker_loop(
        ready,
        workers,
        epoch,
        (
{("," + chr(10)).join(f"            ({entry[0]}, {entry[1]}, {entry[2]}, {entry[3]})" for entry in entries)}
        ),
    )
'''


def _normalize_generic_selection(
    selection: SelectedBody, threads: int
) -> SelectedBody:
    body = selection.body
    old_threads = body.participants.participating_threads
    if body.provider == "g0_range" or old_threads == threads:
        return selection
    if threads % old_threads:
        raise UnsupportedCodegen(
            f"generic body uses {old_threads} threads; cannot share {threads}"
        )

    factor = threads // old_threads
    participants = replace(
        body.participants,
        participating_threads=threads,
        warp_roles=body.participants.warp_roles * factor,
    )
    family = selection.program.family
    if family in {"elementwise", "copy", "gather", "concat", "contraction"}:
        output_domains = tuple(
            domain
            for domain in selection.program.iteration_domains
            if domain.axis not in selection.program.reduction_axes
        )
        tiles = replace(
            body.tiles,
            cta_shape=(threads,) + (1,) * max(len(output_domains) - 1, 0),
        )
    elif family == "reduction":
        output_domains = tuple(
            domain
            for domain in selection.program.iteration_domains
            if domain.axis not in selection.program.reduction_axes
        )
        accumulator_size = _DTYPE_BYTES[selection.program.accumulator_dtype or ""]
        tiles = replace(
            body.tiles,
            cta_shape=(threads,) + (1,) * max(len(output_domains) - 1, 0),
            reduction_tile=(threads,),
        )
        scratch = replace(
            body.scratch,
            size_bytes=max(
                body.scratch.size_bytes,
                (threads // 32) * accumulator_size,
            ),
        )
        return replace(
            selection,
            body=replace(body, participants=participants, tiles=tiles, scratch=scratch),
        )
    elif family == "attention":
        return replace(selection, body=replace(body, participants=participants))
    else:
        raise UnsupportedCodegen(f"cannot normalize generic body family {family}")

    return replace(
        selection,
        body=replace(body, participants=participants, tiles=tiles),
    )


def _generate_device_body(
    selection: SelectedBody,
    *,
    defer_length_publish: bool = False,
    workers: int | None = None,
):
    """``workers`` is given for table launches, whose bodies may split work
    across CTAs through phase scratch."""
    program = selection.program
    if selection.body.provider == "blackwell_mma":
        return emit_blackwell_gemm(program, selection.body)
    if program.family in {"elementwise", "copy", "gather", "concat"}:
        return emit_map_body(program, selection.body)
    if program.family == "reduction":
        return emit_reduction_body(program, selection.body)
    if program.family == "contraction":
        return emit_contraction_body(program, selection.body)
    if program.family == "attention":
        return emit_attention_body(
            program,
            selection.body,
            defer_length_publish=defer_length_publish,
            workers=workers,
        )
    raise UnsupportedCodegen(f"unsupported device body family {program.family}")


def _generate_conversion_body(
    conversion: ConversionBody, base_body: BodyConfig, selection: SelectedBody
):
    axes = base_body.layouts.inputs[conversion.value_id].axes
    if any(isinstance(axis, int) for axis in axes):
        raise UnsupportedCodegen(
            "layout conversion with constant broadcast axes is not supported"
        )
    if conversion.source_layout.dtype != conversion.target_layout.dtype:
        raise UnsupportedCodegen("layout conversion cannot change dtype")

    domains = tuple(
        IterationDomain(axis, size)
        for axis, size in zip(
            axes,
            conversion.source_layout.shape,
            strict=True,
        )
    )
    load = IndexedLoad(
        "load:0",
        conversion.value_id,
        axes,
        conversion.source_layout.strides,
        conversion.source_layout.dtype,
    )
    converted_id = (
        f"{conversion.value_id}:converted:{conversion.target_region_id}"
    )
    program = TensorProgram(
        family="copy",
        source_target="megabake.layout_conversion",
        decomposition="explicit",
        iteration_domains=domains,
        reduction_axes=(),
        loads=(load,),
        init=None,
        combine=Load(load.id, load.dtype),
        post=None,
        accumulator_dtype=None,
        store=TensorStore(
            converted_id,
            conversion.target_layout.shape,
            conversion.target_layout.dtype,
            conversion.target_layout.strides,
            True,
        ),
        predicates=tuple(
            AxisPredicate(domain.axis, domain.size) for domain in domains
        ),
    )
    output_axes = tuple(domain.axis for domain in domains)
    threads = base_body.participants.participating_threads
    config = replace(
        base_body,
        semantic=replace(
            base_body.semantic,
            operation_family="copy",
            source_op_ids=(f"conversion:{conversion.value_id}",),
            access_effects={
                "reads": (conversion.value_id,),
                "writes": (converted_id,),
                "owns_storage": True,
            },
            access_restrictions=("copy source coordinates exactly",),
        ),
        layouts=LayoutContract(
            {
                conversion.value_id: TensorLayout(
                    conversion.source_layout.dtype,
                    axes,
                    conversion.source_layout.strides,
                    conversion.source_layout.alignment_bytes,
                    conversion.source_layout.shape,
                )
            },
            {
                converted_id: TensorLayout(
                    conversion.target_layout.dtype,
                    axes,
                    conversion.target_layout.strides,
                    conversion.target_layout.alignment_bytes,
                    conversion.target_layout.shape,
                )
            },
        ),
        tiles=replace(
            base_body.tiles,
            cta_shape=(threads,) + (1,) * max(len(output_axes) - 1, 0),
            work_partition="one output element per thread",
            reduction_tile=(),
        ),
        pipeline=replace(
            base_body.pipeline,
            barriers=(),
            async_operations=(),
            shared_memory_layouts={},
        ),
        scratch=replace(
            base_body.scratch,
            owner="none",
            size_bytes=0,
            initial_state="empty",
        ),
        completion=replace(
            base_body.completion,
            outputs_ready="copied elements are stored when run returns",
            output_tiles_ready="each copied element is ready after its store",
            scratch_reusable="scratch is reusable when run returns",
        ),
    )
    return emit_map_body(program, config)


def _nested_body_source(source: str, symbol: str) -> str:
    parsed = ast.parse(source)
    function = next(
        node
        for node in parsed.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run"
    )
    source_lines = source.splitlines(keepends=True)
    decorator_line = min(
        (decorator.lineno for decorator in function.decorator_list),
        default=function.lineno,
    )
    body_source = "".join(
        source_lines[decorator_line - 1 : function.end_lineno]
    )
    body_source = body_source.replace("@cute.kernel", "@cute.jit", 1)
    body_source = body_source.replace("def run(", f"def {symbol}(", 1)
    body_source = body_source.replace(
        "    work_iterator: cute.Tensor,",
        (
            "        work_iterator: cute.Tensor,\n"
            "        block: cutlass.Int32,\n"
            "        thread: cutlass.Int32,"
            "\n        workers: cutlass.Int32,"
        ),
        1,
    )
    body_source = body_source.replace(
        "    block, _, _ = cute.arch.block_idx()\n", "", 1
    )
    body_source = body_source.replace(
        "    thread, _, _ = cute.arch.thread_idx()\n", "", 1
    )
    body_source = body_source.replace(
        "    workers, _, _ = cute.arch.grid_dim()\n", "", 1
    )
    body_source = body_source.replace("    scratch: cute.Tensor,", "    scratch,", 1)
    body_source = body_source.replace("scratch[block,", "scratch[")
    return textwrap.indent(body_source, "    ") + "\n"


def _generic_scratch_bytes(
    plan: KernelPlan, scratch_elements: Mapping[int, int]
) -> int:
    sizes = []
    for phase in plan.phases:
        if phase.body.provider == "g0_range":
            continue
        offset = 0
        for _ in phase.body.pipeline.barriers:
            offset = _round_up(offset + 8, 8)
        for _ in phase.body.pipeline.descriptors:
            offset = _round_up(offset + 128, 128)
        alignment = max(1, phase.body.scratch.alignment_bytes)
        body_bytes = _phase_scratch_bytes(plan, phase, scratch_elements)
        offset = _round_up(offset, alignment) + body_bytes
        sizes.append(_round_up(offset, alignment))
    return max(sizes, default=0)


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _shared_scratch_source(
    plan: KernelPlan, scratch_elements: Mapping[int, int]
) -> str:
    worker = plan.storage.worker_scratch[0]
    phase_structs: list[str] = []
    scratch_phase_indices: list[int] = []
    generic_phases = tuple(
        phase for phase in plan.phases if phase.body.provider != "g0_range"
    )
    if not any(_phase_has_scratch_fields(phase) for phase in generic_phases):
        return ""
    for phase_layout, phase in zip(worker.phases, plan.phases, strict=True):
        if phase.body.provider == "g0_range":
            continue
        fields: list[str] = []
        for index, name in enumerate(phase.body.pipeline.barriers):
            fields.append(
                f"    barrier_{index}: cute.struct.Align[\n"
                f"        cute.struct.MemRange[cutlass.Int64, 1], 8\n"
                f"    ]"
            )
        descriptor_offsets = sorted(
            phase_layout.descriptor_offsets.items(), key=lambda item: item[1]
        )
        for index, (name, offset) in enumerate(descriptor_offsets):
            next_offset = (
                descriptor_offsets[index + 1][1]
                if index + 1 < len(descriptor_offsets)
                else phase_layout.body_data_offset_bytes
            )
            size = max(0, next_offset - offset)
            if size:
                fields.append(
                    f"    descriptor_{index}: cute.struct.Align[\n"
                    f"        cute.struct.MemRange[cutlass.UInt8, {size}], 128\n"
                    f"    ]"
                )
        if phase.index in scratch_elements or phase.body.scratch.size_bytes:
            body_scratch_elements = _phase_scratch_elements(
                plan, phase, scratch_elements
            )
        else:
            body_scratch_elements = 0
        if body_scratch_elements:
            accumulator_dtype = next(
                selection.program.accumulator_dtype
                for selection in plan.selection.selections
                if selection.region_id == phase.region_id
            )
            cutlass_type = _CUTLASS_TYPES[accumulator_dtype]
            fields.append(
                f"    body: cute.struct.Align[\n"
                f"        cute.struct.MemRange[{cutlass_type}, {body_scratch_elements}],\n"
                f"        {phase.body.scratch.alignment_bytes}\n"
                f"    ]"
            )
        if fields:
            scratch_phase_indices.append(phase.index)
            phase_structs.append(
                f"@cute.struct\nclass _Phase{phase.index}Scratch:\n"
                + "\n".join(fields)
            )
    if not phase_structs:
        return ""
    union_fields = "\n".join(
        f"    phase_{index}: _Phase{index}Scratch"
        for index in scratch_phase_indices
    )
    return (
        "\n\n".join(phase_structs)
        + "\n\n\n@cute.union\nclass _WorkerScratch:\n"
        + union_fields
    )


def _scratch_prelude(
    plan: KernelPlan, scratch_elements: Mapping[int, int]
) -> str:
    generic_phases = tuple(
        phase for phase in plan.phases if phase.body.provider != "g0_range"
    )
    if not any(_phase_has_scratch_fields(phase) for phase in generic_phases):
        return ""
    lines = [
        "    _worker_scratch = cutlass.utils.SmemAllocator().allocate(_WorkerScratch)"
    ]
    for phase in plan.phases:
        if phase.body.provider == "g0_range":
            continue
        if _phase_has_scratch_fields(phase):
            elements = _phase_scratch_elements(plan, phase, scratch_elements)
            lines.append(
                f"    _scratch_{phase.index} = _worker_scratch.phase_{phase.index}.body"
                f".get_tensor(cute.make_layout({elements}))"
            )
    for phase in plan.phases:
        if phase.body.provider == "g0_range":
            continue
        if _phase_has_scratch_fields(phase):
            for index in range(len(phase.body.pipeline.barriers)):
                lines.extend(
                    (
                        "    if _thread == 0:",
                        f"        _worker_scratch.phase_{phase.index}.barrier_{index}[0] = 0",
                        "    cute.arch.sync_threads()",
                    )
                )
    return "\n".join(lines) + "\n"


def _phase_scratch_elements(
    plan: KernelPlan, phase: Any, scratch_elements: Mapping[int, int]
) -> int:
    accumulator_dtype = next(
        selection.program.accumulator_dtype
        for selection in plan.selection.selections
        if selection.region_id == phase.region_id
    )
    return _phase_scratch_bytes(
        plan, phase, scratch_elements
    ) // _DTYPE_BYTES[accumulator_dtype]


def _phase_scratch_bytes(
    plan: KernelPlan, phase: Any, scratch_elements: Mapping[int, int]
) -> int:
    if phase.index in scratch_elements:
        accumulator_dtype = next(
            selection.program.accumulator_dtype
            for selection in plan.selection.selections
            if selection.region_id == phase.region_id
        )
        return scratch_elements[phase.index] * _DTYPE_BYTES[accumulator_dtype]
    return phase.body.scratch.size_bytes


def _phase_has_scratch_fields(phase: Any) -> bool:
    return bool(
        phase.body.pipeline.barriers
        or phase.body.pipeline.descriptors
        or phase.body.scratch.size_bytes
    )


def _tensor_name(
    arguments: tuple[TensorArgument, ...],
    value_id: str,
    region_id: str,
    role: str,
    prefer_conversion: bool = False,
) -> str:
    candidates = [
        argument
        for argument in arguments
        if argument.value_id == value_id and region_id in argument.region_ids
    ]
    if not candidates:
        raise UnsupportedCodegen(
            f"region {region_id} has no tensor argument for {value_id}"
        )
    if prefer_conversion:
        conversions = [argument for argument in candidates if argument.kind == "conversion"]
        if conversions:
            return conversions[0].name
    if role == "output":
        outputs = [
            argument
            for argument in candidates
            if argument.kind in {"workspace", "output"}
        ]
        if outputs:
            return outputs[0].name
    return candidates[0].name


def _tensor_argument_index(
    arguments: tuple[TensorArgument, ...], name: str
) -> int:
    return next(
        index for index, argument in enumerate(arguments) if argument.name == name
    )


def _source_map(
    source: str,
    plan: KernelPlan,
    phase_records: tuple[PhaseEmission, ...],
) -> tuple[SourceMapEntry, ...]:
    selected = {
        selection.region_id: selection for selection in plan.selection.selections
    }
    entries: list[SourceMapEntry] = []
    for record in phase_records:
        if record.provider == "g0_range":
            source_path = Path(inspect.getsourcefile(HopperWgmmaGemmPersistentKernel))
            lines, line_start = inspect.getsourcelines(
                HopperWgmmaGemmPersistentKernel.run
            )
            entries.append(
                SourceMapEntry(
                    record.body_symbol,
                    "device_body",
                    line_start,
                    line_start + len(lines) - 1,
                    record.index,
                    record.region_id,
                    record.provider,
                    record.body_name,
                    selected[record.region_id].source_op_ids,
                    str(source_path),
                )
            )
        else:
            lines = _symbol_lines(source, record.body_symbol)
            entries.append(
                SourceMapEntry(
                    record.body_symbol,
                    "device_body",
                    lines[0],
                    lines[1],
                    record.index,
                    record.region_id,
                    record.provider,
                    record.body_name,
                    selected[record.region_id].source_op_ids,
                )
            )
        for symbol in record.conversion_symbols:
            lines = _symbol_lines(source, symbol)
            entries.append(
                SourceMapEntry(
                    symbol,
                    "layout_conversion",
                    lines[0],
                    lines[1],
                    record.index,
                    record.region_id,
                    "generic",
                    "layout_conversion",
                    (f"conversion:{record.region_id}",),
                )
            )
        if record.barrier:
            source_lines = source.splitlines()
            line = next(
                (
                    index
                    for index, line in enumerate(source_lines, 1)
                    if "grid_barrier(ready, workers, epoch + " in line
                    and f"epoch + {record.index})" in line
                ),
                None,
            ) or next(
                index
                for index, line in enumerate(source_lines, 1)
                if "grid_barrier(ready, workers, epoch + phase + " in line
                or "counting_grid_barrier(ready, workers, _ordinal + phase_table[phase, 1])" in line
            )
            entries.append(
                SourceMapEntry(
                    record.barrier,
                    "phase_barrier",
                    line,
                    line,
                    record.index,
                    record.region_id,
                    record.provider,
                    record.body_name,
                    selected[record.region_id].source_op_ids,
                )
            )
    kernel_lines = _symbol_lines(source, "megabake_kernel")
    entries.append(
        SourceMapEntry(
            "megabake_kernel",
            "enclosing_kernel",
            kernel_lines[0],
            kernel_lines[1],
            None,
            None,
            None,
            None,
        )
    )
    launch_lines = _symbol_lines(source, "launch")
    entries.append(
        SourceMapEntry(
            "launch",
            "host_launch_wrapper",
            launch_lines[0],
            launch_lines[1],
            None,
            None,
            None,
            None,
        )
    )
    return tuple(entries)


def _symbol_lines(source: str, symbol: str) -> tuple[int, int]:
    for node in ast.walk(ast.parse(source)):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and node.name == symbol:
            return node.lineno, node.end_lineno
    raise UnsupportedCodegen(f"generated symbol {symbol} is missing")


def _json_default(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        result = {}
        for field in value.__dataclass_fields__:
            result[field] = getattr(value, field)
        return result
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return str(value)
