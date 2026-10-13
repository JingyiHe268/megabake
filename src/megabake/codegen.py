from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
from typing import Any
from collections.abc import Mapping

from .bodies.gemm import HopperWgmmaGemmPersistentKernel
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
from .plan import ConversionBody, KernelPlan, SelectedBody
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
) -> GeneratedCuTeModule:
    verify_plan(plan)
    if plan.launch.architecture != "sm90a":
        raise UnsupportedCodegen("codegen currently supports sm90a plans only")

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

            g0_names = []
            for value_id in (*selection.inputs, selection.outputs[0]):
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
                g0_names.append(argument_name)

            tensor_bindings = f"_g0_bindings_{phase.index}"
            epilogue = f"_g0_epilogue_{phase.index}"
            work_iterator = f"_g0_work_iterator_{phase.index}"
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
                    g0_names[0],
                    g0_names[1],
                    g0_names[2],
                    tensor_bindings,
                    epilogue,
                    work_iterator,
                )
            )
            runner_lines = [
                f"        {body_symbol}.run(\n"
                f"            {work_iterator},\n"
                f"            {tensor_bindings},\n"
                f"            {scratch},\n"
                f"            {epilogue},\n"
                "        )"
            ]
            phase_runners.append(
                f"    @cute.jit\n"
                f"    def _run_phase_{phase.index}():\n"
                + "\n".join(runner_lines)
                + "\n"
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

        iterator_name = f"work_iterator_{phase.index}"
        step = (
            1
            if selection.program.family == "reduction"
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
            for value_id in selection.inputs
        ) + (
            _tensor_name(
                tensor_arguments,
                selection.outputs[0],
                phase.region_id,
                "output",
            ),
        )
        scratch_name = (
            f"_scratch_{phase.index}"
            if phase.body.scratch.size_bytes
            else "None"
        )
        runner_lines = [
            f"        {body_symbol}(\n"
            f"            {iterator_name},\n"
            f"            ({', '.join(tensor_names)},),\n"
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
                f"            {conversion_iterator_name},\n"
                f"            ({source_tensor}, {target_tensor},),\n"
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
        phase_runners.append(
            f"    @cute.jit\n"
            f"    def _run_phase_{phase.index}():\n"
            + "\n".join(runner_lines)
            + "\n"
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
    g0_kernel_values = tuple(
        argument
        for prepared in g0_prepared
        for argument in prepared[4:]
    )
    prepare_lines = [
        f"    {bindings}, {epilogue}, {work_iterator} = {body_symbol}.prepare(\n"
        f"        {tensor_a},\n"
        f"        {tensor_b},\n"
        f"        {tensor_c},\n"
        "        max_active_clusters,\n"
        "    )\n"
        for body_symbol, tensor_a, tensor_b, tensor_c, bindings, epilogue, work_iterator in g0_prepared
    ]
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
        *(iterator.name + ": cute.Tensor" for iterator in work_iterators),
        *(argument.name + ": cute.Tensor" for argument in tensor_arguments),
        *g0_kernel_values,
    ]
    launch_arguments = [
        "ready: cute.Tensor",
        "workers: cutlass.Int32",
        "epoch: cutlass.Int32",
        *(iterator.name + ": cute.Tensor" for iterator in work_iterators),
        *(argument.name + ": cute.Tensor" for argument in tensor_arguments),
        *(argument.name + ": cute.Tensor" for argument in g0_tensor_arguments),
        "stream: cuda.CUstream",
    ]
    if has_g0:
        launch_arguments.insert(-1, "max_active_clusters: cutlass.Constexpr")
    source = f'''import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda

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
    def _run_terminal():
        pass

{"".join(phase_runners)}
{"".join(phase_publishers)}
{_event_worker_loop(plan, event_schedule)}


@cute.jit
def launch(
{("," + chr(10)).join(f"    {argument}" for argument in launch_arguments)}
):
{"".join(prepare_lines)}
    megabake_kernel(
        ready,
        workers,
        epoch,
        {("," + chr(10)).join([f"        {argument.name}" for argument in (*work_iterators, *tensor_arguments)] + [f"        {argument}" for argument in g0_kernel_values])}
    ).launch(
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
        ("ready", "workers", "epoch")
        + tuple(argument.name for argument in (*work_iterators, *tensor_arguments))
        + tuple(argument.name for argument in g0_tensor_arguments)
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
    )
    return source, launch_metadata, _source_map(source, plan, phase_records)


def _event_worker_loop(
    plan: KernelPlan, event_schedule: EventSchedule | None
) -> str:
    if event_schedule is None:
        return f'''    cooperative_worker_loop(
        ready,
        workers,
        epoch,
        (
{("," + chr(10)).join([f"            ({phase.index}, _run_phase_{phase.index}, _no_op, {f'_publish_phase_{phase.index}' if phase.barrier is not None else '_ignore_epoch'})" for phase in plan.phases] + [f"            ({len(plan.phases)}, _run_terminal, _no_op, _ignore_epoch)"])}
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
            f"_wait_phase_{phase.index}" if phase.dependencies else "_ignore_epoch",
            f"_run_phase_{phase.index}",
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
{("," + chr(10)).join(f"            ({entry[0]}, {entry[1]}, {entry[2]})" for entry in entries)}
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
    if family in {"elementwise", "copy", "contraction"}:
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
    else:
        raise UnsupportedCodegen(f"cannot normalize generic body family {family}")

    return replace(
        selection,
        body=replace(body, participants=participants, tiles=tiles),
    )


def _generate_device_body(selection: SelectedBody):
    program = selection.program
    if program.family in {"elementwise", "copy"}:
        return emit_map_body(program, selection.body)
    if program.family == "reduction":
        return emit_reduction_body(program, selection.body)
    if program.family == "contraction":
        return emit_contraction_body(program, selection.body)
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
        "    block, _, _ = cute.arch.block_idx()", "    block = _block", 1
    )
    body_source = body_source.replace(
        "    thread, _, _ = cute.arch.thread_idx()", "    thread = _thread", 1
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
            line = next(
                index
                for index, line in enumerate(source.splitlines(), 1)
                if "grid_barrier(ready, workers, epoch + " in line
                and f"epoch + {record.index}" in line
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
