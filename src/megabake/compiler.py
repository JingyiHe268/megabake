from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from importlib.util import module_from_spec, spec_from_file_location
from statistics import median
import threading
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import uuid

import cutlass
import cutlass.cute as cute
import numpy as np
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda
import torch
from torch.utils import _pytree as pytree

from .bodies import SelectionPolicy, TargetFacts
from .bodies.gemm import cute_gemm_tensors
from .cache import PlanCache, make_plan_key
from .codegen import GeneratedCuTeModule, emit_cute
from .diagnostics import CompilerDiagnostic, CompilerError
from .frontend import CaptureContext, PreparedFX
from .graph import ComputeGraph, import_fx
from .lowering import DataIndex, TensorProgram, canonicalize_semantics, semantic_coverage
from .memory import verify_plan
from .plan import (
    BodySelectionPlan,
    KernelPlan,
    PlanMeasurement,
    PlanSearchResult,
    SearchBudget,
    search_plans,
)
from .planner import enumerate_body_selections
from .schedule import EventSchedule, make_region_event_schedule
from .regions import form_regions
from .runtime import (
    EventRuntimeAdmission,
    RuntimeAdmission,
    admit_event_runtime,
    admit_runtime,
    execute_event_plan,
    execute_phase_plan,
    make_ready,
    reset_ready,
)


class UnsupportedCompiler(RuntimeError):
    pass


def _source_location(operation: Any) -> str:
    location = (operation.source_location or "").strip()
    lines = location.splitlines()
    return next(
        (line.strip() for line in reversed(lines) if line.lstrip().startswith("File ")),
        lines[-1] if lines else "unknown source location",
    )


def _semantic_diagnostic(graph: ComputeGraph) -> CompilerDiagnostic | None:
    unresolved = tuple(
        operation for operation in graph.ops.values() if operation.tensor_program is None
    )
    if not unresolved:
        return None
    messages = [
        (
            f"{operation.id} ({operation.attributes.get('target')}) "
            f"at {_source_location(operation)}: "
            f"{operation.numerical_rules.get('reason', 'missing lowering rule')}"
        )
        for operation in unresolved
    ]
    return CompilerDiagnostic(
        "semantic-lowering",
        "; ".join(messages),
        tuple(operation.id for operation in unresolved),
        tuple(region.id for region in graph.regions),
        "complete MegaBake semantic coverage",
    )


def _structural_plan_exists(
    graph: ComputeGraph,
    target: TargetFacts,
    policy: SelectionPolicy | None,
    scheduler: str,
) -> bool:
    result = search_plans(
        graph,
        target,
        policy,
        SearchBudget(evaluator=None),
    )
    return any(
        candidate.plan is not None and candidate.status == "unmeasured"
        for candidate in result.candidates
    )


def _preflight_complete_plan(
    graph: ComputeGraph,
    target: TargetFacts,
    policy: SelectionPolicy | None,
    scheduler: str,
) -> None:
    schedulers = ("phase", "region-events") if scheduler == "auto" else (scheduler,)
    if any(
        _structural_plan_exists(graph, target, policy, scheduler_name)
        for scheduler_name in schedulers
    ):
        return
    semantic = _semantic_diagnostic(graph)
    if semantic is not None:
        raise CompilerError(semantic, safe_for_fallback=True)
    reasons: list[str] = []
    for scheduler_name in schedulers:
        result = search_plans(
            graph,
            target,
            policy,
            SearchBudget(evaluator=None),
        )
        reasons.extend(
            f"{candidate.description}: {candidate.reason or candidate.status}"
            for candidate in result.candidates
            if candidate.status != "measured"
        )
        reasons.extend(
            f"{rejection.region_id}/{rejection.provider}: {rejection.reason}"
            for rejection in result.rejections
        )
    raise CompilerError(
        CompilerDiagnostic(
            "plan-search",
            "; ".join(dict.fromkeys(reasons)) or "no complete plan was produced",
            _source_op_ids_from_graph(graph),
            tuple(region.id for region in graph.regions),
            "one structurally complete plan",
        ),
        safe_for_fallback=True,
    )


@dataclass(frozen=True)
class CompiledResources:
    kernel_name: str
    registers_per_thread: int
    static_shared_memory_bytes: int
    local_memory_bytes_per_thread: int
    dynamic_shared_memory_bytes: int
    active_blocks_per_sm: int
    max_cooperative_workers: int
    workspace_bytes: int
    peak_live_workspace_bytes: int
    body_variant_count: int
    phase_runner_count: int
    compiled_code_bytes: int
    generated_code_bytes: int


@dataclass(frozen=True)
class CompiledKernel:
    plan: KernelPlan
    generated: GeneratedCuTeModule
    compiled: Any
    resources: CompiledResources
    admission: RuntimeAdmission | EventRuntimeAdmission
    event_schedule: EventSchedule | None
    input_order: tuple[str, ...]
    parameter_names: frozenset[str]
    work_iterators: Mapping[str, torch.Tensor]
    max_active_clusters: Any | None = None
    search: Any | None = None


@dataclass(frozen=True)
class CallableTrace:
    invocation_id: uuid.UUID
    epoch: int
    stream: torch.cuda.Stream
    input_data_ptrs: tuple[int, ...]
    output_data_ptrs: tuple[int, ...]
    workload_kernel_launches: int
    stream_synchronizations: int
    recurring_gpu_operations: tuple[tuple[str, int], ...]


# Above this size, per-phase kernel parameters exceed CUDA's limit, so plans
# use the table-driven launch: tensors come from a device address table and
# TMA GEMM descriptors are written per phase in the kernel prologue.
_TABLE_LAUNCH_OPS = 1_000


def _uses_table_launch(graph: ComputeGraph) -> bool:
    return len(graph.ops) > _TABLE_LAUNCH_OPS
# A measured candidate must match the captured reference within this
# relative L2 error before it can be selected. Deep table-launched graphs
# accumulate bf16 rounding differences: on the 30-layer pinned SmolLM the
# generic bodies differ from eager bf16 by 1.6% while both sit equally far
# from an fp32 reference, so those graphs use the 2% model-level tolerance.
_PLAN_RELATIVE_L2_LIMIT = 0.01
_TABLE_PLAN_RELATIVE_L2_LIMIT = 0.02


def _plan_relative_l2_limit(plan: KernelPlan) -> float:
    if plan.graph is not None and _uses_table_launch(plan.graph):
        return _TABLE_PLAN_RELATIVE_L2_LIMIT
    return _PLAN_RELATIVE_L2_LIMIT


def compile_post_grad(
    prepared: PreparedFX,
    *,
    target: TargetFacts | None = None,
    policy: SelectionPolicy | None = None,
    output_dir: Path | str | None = None,
    search_budget: SearchBudget | int | None = None,
    scheduler: str = "phase",
    cache: PlanCache | None = None,
) -> Callable[..., Any]:
    compile_started = time.perf_counter()
    if scheduler not in {"phase", "region-events", "auto"}:
        raise ValueError(
            "scheduler must be 'phase', 'region-events', or 'auto'"
        )
    graph, selected_target = _post_grad_graph(prepared, target)
    if policy is not None and policy.mode != prepared.context.mode:
        raise ValueError(
            f"selection mode {policy.mode!r} does not match backend mode "
            f"{prepared.context.mode!r}"
        )
    if policy is None:
        policy = SelectionPolicy(
            mode=prepared.context.mode,
            table_launch=_uses_table_launch(graph),
        )
    if isinstance(search_budget, int):
        if search_budget < 1:
            raise ValueError("search budget must be positive")
        budget = SearchBudget(max_plans=search_budget)
    else:
        budget = search_budget or SearchBudget()
        if search_budget is None and _uses_table_launch(graph):
            budget = SearchBudget(max_plans=1, max_compilations=1)

    plan_key: str | None = None
    if cache is not None:
        selections = enumerate_body_selections(
            graph, selected_target, policy, max_plans=1
        )
        key_selection = (
            selections[0]
            if selections
            else BodySelectionPlan((), (), (), (), graph, selected_target)
        )
        body_revisions = tuple(
            dict.fromkeys(
                (
                    selected.body.provider,
                    selected.body.evidence.revision,
                    selected.body.semantic.numerical_policy,
                )
                for selection in selections
                for selected in selection.selections
            )
        )
        plan_key = make_plan_key(
            key_selection,
            prepared.context,
            scheduler=scheduler,
            body_revisions=body_revisions,
        )

    def compile_selected() -> CompiledKernel:
        _preflight_complete_plan(graph, selected_target, policy, scheduler)
        state_snapshot = _StateSnapshot.create(graph, prepared.example_inputs)
        evaluation = replace(prepared, example_inputs=state_snapshot.inputs)
        with torch.no_grad():
            state_snapshot.reset()
            reference = pytree.tree_map_only(
                torch.Tensor,
                torch.clone,
                prepared.forward(*evaluation.example_inputs),
            )
        if scheduler == "auto":
            phase_result = _search_scheduler(
                graph,
                selected_target,
                policy,
                budget,
                evaluation,
                reference,
                output_dir,
                "phase",
                state_snapshot.reset,
            )
            event_result = _search_scheduler(
                graph,
                selected_target,
                policy,
                budget,
                evaluation,
                reference,
                output_dir,
                "region-events",
                state_snapshot.reset,
            )
            choices = (phase_result, event_result)
            chosen = min(
                (result for result in choices if result.selected is not None),
                key=_selected_median,
                default=phase_result,
            )
            result = PlanSearchResult(
                chosen.selected,
                chosen.selected_compiled,
                tuple(
                    candidate
                    for labeled in choices
                    for candidate in labeled.candidates
                ),
                tuple(
                    rejection
                    for labeled in choices
                    for rejection in labeled.rejections
                ),
                any(labeled.budget_exhausted for labeled in choices),
                chosen.selected_description,
            )
        else:
            result = _search_scheduler(
                graph,
                selected_target,
                policy,
                budget,
                evaluation,
                reference,
                output_dir,
                scheduler,
                state_snapshot.reset,
            )
        if result.selected is None or result.selected_compiled is None:
            reasons = "; ".join(
                f"{candidate.description}: {candidate.reason or candidate.status}"
                for candidate in result.candidates
                if candidate.status != "measured"
            )
            raise CompilerError(
                CompilerDiagnostic(
                    "plan-search",
                    reasons or "no measured complete plan was produced",
                    _source_op_ids_from_graph(graph),
                    tuple(region.id for region in graph.regions),
                    "one correct measured complete plan",
                )
            )
        result = replace(result, selected=result.selected_compiled.plan)
        return replace(result.selected_compiled, search=result)

    cache_hit = False
    if cache is not None:
        assert plan_key is not None
        cache_result = cache.get_or_compile(plan_key, compile_selected)
        compiled = cache_result.compiled
        cache_hit = cache_result.hit
    else:
        compiled = compile_selected()
    selected_candidate = next(
        (
            candidate
            for candidate in compiled.search.candidates
            if candidate.description == compiled.search.selected_description
            and candidate.measurement is not None
        ),
        None,
    )
    prepared.context.requirements["generic_coverage"] = {
        "semantic_support": semantic_coverage(graph),
        "body_names": tuple(
            dict.fromkeys(
                f"{phase.body.provider}/{phase.body.name}"
                for phase in compiled.plan.phases
                if phase.body.provider == "generic"
            )
        ),
        "workspace_bytes": compiled.plan.storage.workspace_bytes,
        "performance": {
            "median_isolated_launch_us": (
                selected_candidate.measurement.median_isolated_launch_us
                if selected_candidate is not None
                and selected_candidate.measurement is not None
                else None
            ),
            "median_device_event_span_us": (
                selected_candidate.measurement.median_device_event_span_us
                if selected_candidate is not None
                and selected_candidate.measurement is not None
                else None
            ),
            "median_complete_call_us": (
                selected_candidate.measurement.median_complete_call_us
                if selected_candidate is not None
                and selected_candidate.measurement is not None
                else None
            ),
            "median_device_event_span_us": (
                selected_candidate.measurement.median_device_event_span_us
                if selected_candidate is not None
                and selected_candidate.measurement is not None
                else None
            ),
        },
    }
    parameter_tensors = _current_parameter_tensors(
        compiled, prepared.context, prepared.example_inputs
    )
    forward = bind_callable(
        compiled, prepared.context, parameter_tensors=parameter_tensors
    )
    forward.megabake_compile_us = (time.perf_counter() - compile_started) * 1_000_000
    forward.megabake_cache_hit = cache_hit
    return forward


def compile_inductor_fallback(
    prepared: PreparedFX, reason: str
) -> Callable[..., Any]:
    from torch._inductor.compile_fx import compile_fx

    graph = copy.deepcopy(prepared.graph)
    compiled = compile_fx(graph, prepared.example_inputs)
    context = prepared.context
    parameter_positions = {
        index
        for index, name in enumerate(context.input_order)
        if name in context.parameter_names
    }
    runtime_positions = [
        index
        for index in range(len(context.input_order))
        if index not in parameter_positions
    ]
    current_parameters = {
        context.input_order[index]: prepared.example_inputs[index]
        for index in parameter_positions
        if index < len(prepared.example_inputs)
    }
    launch_count = 0

    def forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal launch_count
        if kwargs:
            raise CompilerError(
                CompilerDiagnostic(
                    "fallback-binding",
                    "keyword arguments are not supported by the fallback callable",
                    requirement="positional runtime arguments",
                )
            )
        runtime_values = pytree.tree_flatten(args)[0]
        if len(runtime_values) == len(context.input_order):
            full_inputs = list(runtime_values)
        elif len(runtime_values) == len(runtime_positions):
            full_inputs = [None] * len(context.input_order)
            for position, value in zip(runtime_positions, runtime_values, strict=True):
                full_inputs[position] = value
            for position in parameter_positions:
                name = context.input_order[position]
                value = current_parameters.get(name)
                if value is None:
                    raise CompilerError(
                        CompilerDiagnostic(
                            "fallback-binding",
                            f"missing current parameter {name}",
                            requirement="current parameter binding",
                        )
                    )
                full_inputs[position] = value
        else:
            raise CompilerError(
                CompilerDiagnostic(
                    "fallback-binding",
                    (
                        f"expected {len(runtime_positions)} runtime arguments "
                        f"or {len(context.input_order)} graph arguments, "
                        f"got {len(runtime_values)}"
                    ),
                    requirement="runtime argument count",
                )
            )

        output = compiled(*full_inputs)
        launch_count += 1
        forward.megabake_launch_count = launch_count
        context.requirements["fallback_launch_count"] = launch_count
        return pytree.tree_unflatten(
            output,
            context.runtime_output_tree_spec or context.output_tree_spec,
        )

    context.requirements["execution"] = "inductor_fallback"
    context.requirements["fallback_reason"] = reason
    context.requirements["fallback_launch_count"] = launch_count
    forward.megabake_execution = "inductor_fallback"
    forward.megabake_fallback_reason = reason
    forward.megabake_launch_count = launch_count
    forward.megabake_context = context
    return forward


def compile_plan(
    plan: KernelPlan,
    example_inputs: Sequence[Any],
    *,
    output_dir: Path | str | None = None,
    scheduler: str = "phase",
) -> CompiledKernel:
    if scheduler not in {"phase", "region-events"}:
        raise ValueError("compile_plan scheduler must be 'phase' or 'region-events'")
    _reject_invalid_plan(plan)
    if plan.graph is None:
        raise CompilerError(
            CompilerDiagnostic(
                "plan-validation",
                "kernel plan is missing its source graph",
                requirement="source graph",
            )
        )
    device = _common_cuda_device(example_inputs)
    selected_target = plan.target
    if selected_target is None:
        raise CompilerError(
            CompilerDiagnostic(
                "plan-validation",
                "kernel plan is missing target facts",
                requirement="target facts",
            )
        )
    with torch.cuda.device(device):
        artifact_dir = Path(output_dir) if output_dir is not None else Path("artifacts/generated")
        input_tensors = _input_tensors(plan.graph, example_inputs)
        event_schedule = (
            make_region_event_schedule(
                plan.selection,
                plan.dependencies,
                worker_count=plan.launch.grid_size,
            )
            if scheduler == "region-events"
            else None
        )
        current_plan = plan
        seen_workers = {current_plan.launch.grid_size}
        for _ in range(4):
            if (
                event_schedule is not None
                and event_schedule.worker_count != current_plan.launch.grid_size
            ):
                event_schedule = make_region_event_schedule(
                    current_plan.selection,
                    current_plan.dependencies,
                    worker_count=current_plan.launch.grid_size,
                )
            generated = emit_cute(
                current_plan,
                artifact_dir,
                event_schedule=event_schedule,
                table=_uses_table_launch(current_plan.graph),
            )
            module = _load_generated_module(generated)
            max_active_clusters = _max_active_clusters(generated)
            state = _RuntimeState.create(
                current_plan,
                generated,
                input_tensors,
                max_active_clusters=max_active_clusters,
                ready=(
                    torch.zeros(
                        len(event_schedule.events),
                        dtype=torch.int32,
                        device=device,
                    )
                    if event_schedule is not None
                    else None
                ),
            )
            compiled = cute.compile(module.launch, *state.compile_arguments)
            resources = _query_resources(
                compiled,
                current_plan,
                generated,
                current_plan.launch.threads_per_block,
                generated.launch.shared_memory_bytes,
            )
            max_workers = (
                resources.active_blocks_per_sm
                * torch.cuda.get_device_properties(device).multi_processor_count
            )
            legal_workers = min(current_plan.launch.grid_size, max_workers)
            if legal_workers < 1:
                raise CompilerError(
                    CompilerDiagnostic(
                        "resource-admission",
                        "compiled kernel cannot sustain a cooperative worker",
                        _source_op_ids(current_plan),
                        _region_ids(current_plan),
                        "at least one co-resident cooperative worker",
                    )
                )
            if legal_workers != current_plan.launch.grid_size:
                if legal_workers in seen_workers:
                    raise CompilerError(
                        CompilerDiagnostic(
                            "resource-admission",
                            "worker admission did not converge",
                            _source_op_ids(current_plan),
                            _region_ids(current_plan),
                            "stable legal worker count",
                        )
                    )
                seen_workers.add(legal_workers)
                current_plan = _with_worker_count(
                    current_plan, legal_workers, resources.active_blocks_per_sm
                )
                continue

            current_plan = replace(
                current_plan,
                launch=replace(
                    current_plan.launch,
                    registers_per_thread=resources.registers_per_thread,
                    max_active_blocks_per_sm=resources.active_blocks_per_sm,
                    required_active_blocks=legal_workers,
                ),
            )
            try:
                admission = (
                    admit_event_runtime(event_schedule)
                    if event_schedule is not None
                    else admit_runtime(current_plan)
                )
            except Exception as error:
                raise CompilerError(
                    CompilerDiagnostic(
                        "resource-admission",
                        str(error),
                        _source_op_ids(current_plan),
                        _region_ids(current_plan),
                        "legal cooperative launch",
                    )
                ) from error
            input_order = tuple(
                current_plan.graph.values[value_id].binding_name or value_id
                for value_id in current_plan.graph.inputs
            )
            parameter_names = frozenset(
                name
                for name, value in zip(input_order, example_inputs, strict=True)
                if isinstance(value, torch.nn.Parameter)
            )
            torch.cuda.synchronize(device)
            return CompiledKernel(
                current_plan,
                generated,
                compiled,
                resources,
                admission,
                event_schedule,
                input_order,
                parameter_names,
                state.work_iterators,
                max_active_clusters,
            )

    raise CompilerError(
        CompilerDiagnostic(
            "resource-admission",
            "worker admission exceeded its retry budget",
            _source_op_ids(plan),
            _region_ids(plan),
            "stable legal worker count",
        )
    )


def _current_parameter_tensors(
    compiled_kernel: CompiledKernel,
    context: CaptureContext,
    example_inputs: Sequence[Any],
) -> dict[str, torch.Tensor]:
    positions = {
        spec.name: position for position, spec in enumerate(context.input_specs)
    }
    values: dict[str, torch.Tensor] = {}
    for graph_name in compiled_kernel.input_order:
        if graph_name not in compiled_kernel.parameter_names:
            continue
        candidates = (graph_name, graph_name.removeprefix("l_").removesuffix("_"))
        position = next(
            (positions[name] for name in candidates if name in positions), None
        )
        if position is None:
            raise CompilerError(
                CompilerDiagnostic(
                    "callable-binding",
                    f"cannot map parameter {graph_name!r} to a captured input",
                    requirement="current parameter binding",
                )
            )
        values[graph_name] = example_inputs[position]
    return values


def _initial_ready(
    compiled_kernel: CompiledKernel, device: torch.device
) -> torch.Tensor | None:
    if compiled_kernel.event_schedule is None:
        return None
    return torch.zeros(
        len(compiled_kernel.event_schedule.events),
        dtype=torch.int32,
        device=device,
    )


def _max_active_clusters(generated: GeneratedCuTeModule) -> Any | None:
    if "max_active_clusters" not in generated.launch.arguments:
        return None
    return cutlass.utils.HardwareInfo().get_max_active_clusters(1)


def bind_callable(
    compiled_kernel: CompiledKernel,
    context: CaptureContext,
    *,
    parameter_tensors: Mapping[str, torch.Tensor] | None = None,
) -> Callable[..., Any]:
    if not context.input_order:
        raise CompilerError(
            CompilerDiagnostic(
                "callable-binding",
                "capture context has no input order",
                requirement="runtime argument order",
            )
        )
    graph = compiled_kernel.plan.graph
    if graph is None:
        raise CompilerError(
            CompilerDiagnostic(
                "callable-binding",
                "compiled plan is missing its source graph",
                requirement="source graph",
            )
        )
    parameter_positions = {
        index
        for index, name in enumerate(context.input_order)
        if name in compiled_kernel.parameter_names
    }
    runtime_positions = [
        index for index in range(len(context.input_order)) if index not in parameter_positions
    ]
    current_parameters = dict(parameter_tensors or {})
    trace: list[CallableTrace] = []
    runtime_lock = threading.Lock()
    state: _RuntimeState | None = None
    next_epoch = 0
    epoch_stride = _epoch_stride(compiled_kernel)

    def forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal state, next_epoch
        if kwargs:
            raise CompilerError(
                CompilerDiagnostic(
                    "callable-binding",
                    "keyword arguments are not supported by generated callables",
                    requirement="positional runtime arguments",
                )
            )
        runtime_values = pytree.tree_flatten(args)[0]
        if len(runtime_values) == len(context.input_order):
            full_inputs = list(runtime_values)
        elif len(runtime_values) == len(runtime_positions):
            full_inputs = [None] * len(context.input_order)
            for position, value in zip(runtime_positions, runtime_values, strict=True):
                full_inputs[position] = value
            for position in parameter_positions:
                name = context.input_order[position]
                value = current_parameters.get(name)
                if value is None:
                    raise CompilerError(
                        CompilerDiagnostic(
                            "callable-binding",
                            f"missing current parameter {name}",
                            requirement="current parameter binding",
                        )
                    )
                full_inputs[position] = value
        else:
            raise CompilerError(
                CompilerDiagnostic(
                    "callable-binding",
                    "expected "
                    f"{len(runtime_positions)} runtime arguments "
                    f"or {len(context.input_order)} graph arguments, "
                    f"got {len(runtime_values)}",
                    requirement="runtime argument count",
                )
            )

        with runtime_lock:
            input_tensors = _input_tensors(graph, full_inputs)
            if state is None:
                # Structural: every graph input is bound on every call.
                _validate_data_indices(compiled_kernel.plan, input_tensors)
            active_stream = torch.cuda.current_stream(
                next(iter(input_tensors.values())).device
            )

            if state is None:
                ready = _initial_ready(compiled_kernel, active_stream.device)
                state = _RuntimeState.create(
                    compiled_kernel.plan,
                    compiled_kernel.generated,
                    input_tensors,
                    max_active_clusters=compiled_kernel.max_active_clusters,
                    ready=ready,
                    work_iterators=compiled_kernel.work_iterators,
                )
                forward.megabake_runtime_state = state
            else:
                state.rebind(input_tensors)

            launches = 0
            uploads_before = state.address_table_uploads

            def kernel(
                ready: torch.Tensor, epoch: int, stream: torch.cuda.Stream
            ) -> None:
                nonlocal launches
                arguments = state.arguments(epoch, cuda.CUstream(stream.cuda_stream))
                compiled_kernel.compiled(*arguments)
                state.record_launch(stream)
                launches += 1

            if compiled_kernel.event_schedule is None:
                invocation = execute_phase_plan(
                    compiled_kernel.plan,
                    kernel,
                    active_stream,
                    admission=compiled_kernel.admission,
                    ready=state.ready,
                    epoch=next_epoch,
                )
            else:
                invocation = execute_event_plan(
                    compiled_kernel.event_schedule,
                    kernel,
                    active_stream,
                    admission=compiled_kernel.admission,
                    events=state.ready,
                    epoch=next_epoch,
                    validate_counters=False,
                )
            outputs = tuple(
                state.tensor_for_value(value_id) for value_id in graph.outputs
            )
            trace.append(
                CallableTrace(
                    invocation.invocation_id,
                    invocation.epoch,
                    active_stream,
                    tuple(tensor.data_ptr() for tensor in input_tensors.values()),
                    tuple(
                        output.data_ptr()
                        for output in outputs
                        if isinstance(output, torch.Tensor)
                    ),
                    workload_kernel_launches=launches,
                    # The phase and event runtimes synchronize once per launch.
                    stream_synchronizations=launches,
                    recurring_gpu_operations=_recurring_operations(
                        launches, state.address_table_uploads - uploads_before
                    ),
                )
            )
            next_epoch += epoch_stride
            if next_epoch > 2**31 - 1 - epoch_stride:
                if compiled_kernel.event_schedule is None:
                    state.reset_ready()
                else:
                    state.ready.zero_()
                next_epoch = 0
            return pytree.tree_unflatten(
                outputs,
                context.runtime_output_tree_spec or context.output_tree_spec,
            )

    forward.megabake_compiled = compiled_kernel
    forward.megabake_trace = trace
    forward.megabake_context = context
    forward.megabake_runtime_state = state
    return forward


def _post_grad_graph(
    prepared: PreparedFX, target: TargetFacts | None
) -> tuple[ComputeGraph, TargetFacts]:
    try:
        graph = import_fx(prepared.graph, prepared.context)
        graph = form_regions(canonicalize_semantics(graph))
        semantic = _semantic_diagnostic(graph)
        if semantic is not None:
            raise CompilerError(semantic, safe_for_fallback=True)
        selected_target = target or _target_facts(
            _common_cuda_device(prepared.example_inputs)
        )
        return graph, selected_target
    except CompilerError:
        raise
    except Exception as error:
        raise CompilerError(
            CompilerDiagnostic(
                "planning",
                str(error),
                requirement="valid complete KernelPlan",
            )
        ) from error


@dataclass(frozen=True)
class _StateSnapshot:
    """Private copies of mutable state inputs used while searching plans."""

    inputs: tuple[Any, ...]
    pairs: tuple[tuple[torch.Tensor, torch.Tensor], ...]

    @classmethod
    def create(
        cls, graph: ComputeGraph, example_inputs: Sequence[Any]
    ) -> "_StateSnapshot":
        state_ids = {
            value_id
            for operation in graph.ops.values()
            if isinstance(operation.tensor_program, TensorProgram)
            for access in operation.tensor_program.state_accesses
            for value_id in (
                access.key_cache_value_id,
                access.value_cache_value_id,
                access.valid_length_value_id,
            )
        }
        inputs = list(example_inputs)
        pairs = []
        for position, value_id in enumerate(graph.inputs):
            if value_id in state_ids and isinstance(inputs[position], torch.Tensor):
                original = inputs[position].detach().clone()
                inputs[position] = original.clone()
                pairs.append((inputs[position], original))
        return cls(tuple(inputs), tuple(pairs))

    def reset(self) -> None:
        for private, original in self.pairs:
            private.copy_(original)


def _make_plan_evaluator(
    prepared: PreparedFX,
    reference: Any,
    output_dir: Path | str | None,
    scheduler: str = "phase",
    reset_state: Callable[[], None] | None = None,
):
    def evaluate(plan: KernelPlan, budget: SearchBudget) -> PlanMeasurement:
        try:
            compiled = compile_plan(
                plan,
                prepared.example_inputs,
                output_dir=output_dir,
                scheduler=scheduler,
            )
            bound = bind_callable(
                compiled,
                prepared.context,
                parameter_tensors=_current_parameter_tensors(
                    compiled, prepared.context, prepared.example_inputs
                ),
            )

            def executable(*arguments: Any) -> Any:
                # Stateful candidates start every call from the captured state.
                if reset_state is not None:
                    reset_state()
                return bound(*arguments)

            with torch.no_grad():
                actual = executable(*prepared.example_inputs)
            relative_error = _output_relative_l2(reference, actual)
            if relative_error >= _plan_relative_l2_limit(plan):
                return PlanMeasurement(
                    compiled=compiled,
                    resources=compiled.resources,
                    correct=False,
                    error=f"relative L2 error {relative_error:.6g}",
                )
            complete_samples, device_samples = _complete_call_samples(
                executable,
                prepared.example_inputs,
                budget.timing_samples,
            )
            verification_samples, _ = _complete_call_samples(
                executable, prepared.example_inputs, 1
            )
            isolated_samples = _isolated_kernel_samples(
                compiled,
                prepared.context,
                prepared.example_inputs,
                budget.timing_samples,
                reset_state,
            )
            return PlanMeasurement(
                compiled=compiled,
                resources=compiled.resources,
                correct=True,
                complete_call_samples_us=complete_samples,
                device_event_span_samples_us=device_samples,
                median_complete_call_us=median(complete_samples),
                median_device_event_span_us=median(device_samples),
                isolated_launch_samples_us=isolated_samples,
                median_isolated_launch_us=median(isolated_samples),
                verification_complete_call_us=verification_samples[0],
            )
        except Exception as error:
            return PlanMeasurement(error=f"{type(error).__name__}: {error}")

    return evaluate


def _search_scheduler(
    graph: ComputeGraph,
    target: TargetFacts,
    policy: Any | None,
    budget: SearchBudget,
    prepared: PreparedFX,
    reference: Any,
    output_dir: Path | str | None,
    scheduler: str,
    reset_state: Callable[[], None] | None = None,
):
    evaluator = (
        budget.evaluator
        if scheduler == "phase" and budget.evaluator is not None
        else _make_plan_evaluator(
            prepared, reference, output_dir, scheduler, reset_state
        )
    )
    scheduler_budget = replace(
        budget,
        evaluator=evaluator,
    )
    result = search_plans(graph, target, policy, scheduler_budget)
    prefix = f"scheduler={scheduler}; "
    return replace(
        result,
        candidates=tuple(
            replace(candidate, description=prefix + candidate.description)
            for candidate in result.candidates
        ),
        selected_description=(
            None
            if result.selected_description is None
            else prefix + result.selected_description
        ),
    )


def _selected_median(result: Any) -> float:
    for candidate in result.candidates:
        if candidate.plan is result.selected and candidate.measurement is not None:
            median_value = (
                candidate.measurement.median_isolated_launch_us
                if candidate.measurement.median_isolated_launch_us is not None
                else candidate.measurement.median_device_event_span_us
                if candidate.measurement.median_device_event_span_us is not None
                else candidate.measurement.median_complete_call_us
            )
            if median_value is not None:
                return median_value
    return float("inf")


def _isolated_kernel_samples(
    compiled_kernel: CompiledKernel,
    context: CaptureContext,
    example_inputs: Sequence[Any],
    count: int,
    reset_state: Callable[[], None] | None = None,
) -> tuple[float, ...]:
    graph_inputs = _graph_ordered_inputs(compiled_kernel, context, example_inputs)
    graph = compiled_kernel.plan.graph
    if graph is None:
        raise CompilerError(
            CompilerDiagnostic(
                "runtime-binding",
                "compiled plan is missing its source graph",
                requirement="source graph",
            )
        )
    state = _RuntimeState.create(
        compiled_kernel.plan,
        compiled_kernel.generated,
        _input_tensors(graph, graph_inputs),
        max_active_clusters=compiled_kernel.max_active_clusters,
        ready=_initial_ready(compiled_kernel, torch.cuda.current_stream().device),
        work_iterators=compiled_kernel.work_iterators,
    )
    if compiled_kernel.event_schedule is not None:
        state.ready.zero_()
    else:
        reset_ready(state.ready)
    epoch_stride = _epoch_stride(compiled_kernel)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fast_path = hasattr(
        compiled_kernel.compiled, "generate_execution_args"
    ) and hasattr(compiled_kernel.compiled, "run_compiled_program")
    epoch = 0

    def prepare() -> tuple[tuple[Any, ...], Any, tuple[Any, ...]]:
        nonlocal epoch
        arguments = state.arguments(epoch, stream)
        epoch += epoch_stride
        if not fast_path:
            return arguments, arguments, arguments
        packed = compiled_kernel.compiled.generate_execution_args(*arguments)
        execution = (
            packed[0]
            if isinstance(packed, tuple) and len(packed) == 2
            else packed
        )
        return execution, packed, arguments

    def launch(token: tuple[tuple[Any, ...], Any, tuple[Any, ...]]) -> None:
        if fast_path:
            compiled_kernel.compiled.run_compiled_program(token[0])
        else:
            compiled_kernel.compiled(*token[0])

    samples: list[float] = []
    # A stateful launch advances its state, so time one launch per reset.
    launches_per_sample = count if reset_state is None else 1
    for _ in range(count):
        if reset_state is not None:
            reset_state()
        tokens = [prepare() for _ in range(launches_per_sample)]
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for token in tokens:
            launch(token)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1_000 / launches_per_sample)
    return tuple(samples)


def _graph_ordered_inputs(
    compiled_kernel: CompiledKernel,
    context: CaptureContext,
    example_inputs: Sequence[Any],
) -> tuple[Any, ...]:
    positions = {
        spec.name: position
        for position, spec in enumerate(context.input_specs)
    }
    ordered: list[Any] = [None] * len(example_inputs)
    for graph_name in compiled_kernel.input_order:
        candidates = (graph_name, graph_name.removeprefix("l_").removesuffix("_"))
        position = next(
            (positions[name] for name in candidates if name in positions), None
        )
        if position is None:
            raise CompilerError(
                CompilerDiagnostic(
                    "runtime-binding",
                    f"cannot map graph input {graph_name!r} to a captured input",
                    requirement="complete runtime binding",
                )
            )
        ordered[position] = example_inputs[position]
    if any(value is None for value in ordered):
        raise CompilerError(
            CompilerDiagnostic(
                "runtime-binding",
                "could not map every captured input to the compiled graph",
                requirement="complete runtime binding",
            )
        )
    return tuple(ordered)


def _complete_call_samples(
    function: Callable[..., Any], arguments: Sequence[Any], count: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    host_samples: list[float] = []
    device_samples: list[float] = []
    with torch.no_grad():
        for _ in range(count):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_host = time.perf_counter()
            start_event.record()
            function(*arguments)
            end_event.record()
            end_event.synchronize()
            host_samples.append((time.perf_counter() - start_host) * 1_000_000)
            device_samples.append(start_event.elapsed_time(end_event) * 1_000)
    return tuple(host_samples), tuple(device_samples)


def _output_relative_l2(expected: Any, actual: Any) -> float:
    expected_values, _ = pytree.tree_flatten(expected)
    actual_values, _ = pytree.tree_flatten(actual)
    if len(expected_values) != len(actual_values):
        return float("inf")
    numerator = 0.0
    denominator = 0.0
    for expected_value, actual_value in zip(expected_values, actual_values, strict=True):
        if type(expected_value) is not type(actual_value):
            return float("inf")
        if not isinstance(expected_value, torch.Tensor):
            if expected_value != actual_value:
                return float("inf")
            continue
        if (
            not isinstance(actual_value, torch.Tensor)
            or expected_value.shape != actual_value.shape
            or expected_value.dtype != actual_value.dtype
        ):
            return float("inf")
        difference = (expected_value.float() - actual_value.float()).square().sum()
        denominator_value = expected_value.float().square().sum()
        numerator += float(difference.item())
        denominator += float(denominator_value.item())
    if denominator == 0:
        return numerator**0.5
    return (numerator / denominator) ** 0.5


def _source_op_ids_from_graph(graph: ComputeGraph) -> tuple[str, ...]:
    return tuple(graph.ops)


def _reject_invalid_plan(plan: KernelPlan) -> None:
    try:
        verify_plan(plan)
    except Exception as error:
        raise CompilerError(
            CompilerDiagnostic(
                "plan-validation",
                str(error),
                _source_op_ids(plan),
                _region_ids(plan),
                "valid complete KernelPlan",
            )
        ) from error


def _target_facts(device: torch.device) -> TargetFacts:
    if device.type != "cuda":
        raise UnsupportedCompiler("MegaBake generated kernels require CUDA")
    properties = torch.cuda.get_device_properties(device)
    architecture = f"sm{properties.major}{properties.minor}"
    if architecture != "sm90":
        raise UnsupportedCompiler(
            f"generated CuTe kernels require SM90; found {architecture}"
        )
    return TargetFacts(
        architecture="sm90a",
        threads_per_warp=properties.warp_size,
        max_threads_per_cta=properties.max_threads_per_block,
        max_shared_memory_per_cta=properties.shared_memory_per_block_optin,
        max_registers_per_thread=255,
        multiprocessor_count=properties.multi_processor_count,
        supports_tma=True,
        supports_wgmma=True,
    )


def _common_cuda_device(example_inputs: Sequence[Any]) -> torch.device:
    devices = {
        value.device
        for value in example_inputs
        if isinstance(value, torch.Tensor)
    }
    if not devices or any(device.type != "cuda" for device in devices):
        raise UnsupportedCompiler("all captured tensor inputs must be CUDA tensors")
    if len(devices) != 1:
        raise UnsupportedCompiler(f"captured inputs span multiple devices: {devices}")
    return next(iter(devices))


def _input_tensors(
    graph: ComputeGraph, example_inputs: Sequence[Any]
) -> dict[str, torch.Tensor]:
    if len(example_inputs) != len(graph.inputs):
        raise CompilerError(
            CompilerDiagnostic(
                "input-binding",
                f"expected {len(graph.inputs)} graph inputs, got {len(example_inputs)}",
                requirement="graph argument count",
            )
        )
    result: dict[str, torch.Tensor] = {}
    for value_id, value in zip(graph.inputs, example_inputs, strict=True):
        if not isinstance(value, torch.Tensor):
            raise CompilerError(
                CompilerDiagnostic(
                    "input-binding",
                    f"graph input {value_id} is not a tensor",
                    (value_id,),
                    requirement="tensor input",
                )
            )
        if value.device.type != "cuda":
            raise CompilerError(
                CompilerDiagnostic(
                    "input-binding",
                    f"graph input {value_id} is not on CUDA",
                    (value_id,),
                    requirement="CUDA input",
                )
            )
        _validate_input_metadata(value_id, graph.values[value_id], value)
        result[value_id] = value
    return result


def _validate_input_metadata(value_id: str, value: Any, tensor: torch.Tensor) -> None:
    expected = (value.shape, str(value.dtype), value.strides)
    actual = (tensor.shape, str(tensor.dtype), tuple(tensor.stride()))
    if actual != expected:
        raise CompilerError(
            CompilerDiagnostic(
                "input-binding",
                f"input {value_id} metadata changed: expected {expected}, got {actual}",
                (value_id,),
                requirement="guarded input shape, dtype, and strides",
            )
        )


def _gather_index_is_static(graph: ComputeGraph, value_id: str) -> bool:
    pending = [value_id]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        if current in graph.inputs:
            return False
        value = graph.values.get(current)
        operation = graph.ops.get(value.producer) if value is not None else None
        if operation is None:
            return False
        target = operation.attributes.get("target")
        if target == "aten.add.Tensor":
            arguments = operation.attributes.get("fx_args", ())
            if len(arguments) != 2 or not isinstance(arguments[1], int):
                return False
        elif target not in {
            "prims.iota.default",
            "aten.unsqueeze.default",
            "aten.reshape.default",
            "aten.permute.default",
        }:
            return False
        pending.extend(operation.operands)
    return True


def _recurring_operations(
    launches: int, address_uploads: int
) -> tuple[tuple[str, int], ...]:
    operations = [("workload_kernel", launches)]
    if address_uploads:
        operations.append(("address_table_upload", address_uploads))
    return tuple(operations)


def _epoch_stride(compiled_kernel: CompiledKernel) -> int:
    if compiled_kernel.event_schedule is not None:
        return 1
    if compiled_kernel.generated.launch.barrier_epochs:
        return compiled_kernel.generated.launch.barrier_epochs
    return max(
        max(
            (
                phase.index
                for phase in compiled_kernel.plan.phases
                if phase.barrier is not None
            ),
            default=-1,
        )
        + 1,
        1,
    )


def _validate_data_indices(plan: KernelPlan, inputs: Mapping[str, torch.Tensor]) -> None:
    """Check gather-index bindings; index ranges trap inside the generated body."""
    for selected in plan.selection.selections:
        for load in selected.program.loads:
            for axis in load.axes:
                if not isinstance(axis, DataIndex):
                    continue
                index_load = next(
                    candidate for candidate in selected.program.loads
                    if candidate.id == axis.load_id
                )
                index_tensor = inputs.get(index_load.value_id)
                if index_tensor is None:
                    if _gather_index_is_static(plan.graph, index_load.value_id):
                        continue
                    raise CompilerError(
                        CompilerDiagnostic(
                            "runtime-binding",
                            f"gather index {index_load.value_id} is not a runtime input",
                            selected.source_op_ids,
                            requirement="runtime gather index",
                        )
                    )


def _load_generated_module(generated: GeneratedCuTeModule) -> Any:
    spec = spec_from_file_location(generated.module_name, generated.source_path)
    if spec is None or spec.loader is None:
        raise CompilerError(
            CompilerDiagnostic(
                "codegen",
                f"cannot load generated module {generated.source_path}",
                requirement="loadable generated module",
            )
        )
    module = module_from_spec(spec)
    sys.modules[generated.module_name] = module
    spec.loader.exec_module(module)
    return module


def _query_resources(
    compiled: Any,
    plan: KernelPlan,
    generated: GeneratedCuTeModule,
    threads: int,
    dynamic_shared_bytes: int,
) -> CompiledResources:
    kernel_names = tuple(compiled.kernel_info)
    if len(kernel_names) != 1:
        raise CompilerError(
            CompilerDiagnostic(
                "resource-query",
                f"expected one complete kernel, found {len(kernel_names)}",
                requirement="one enclosing workload kernel",
            )
        )
    kernel_name = kernel_names[0]
    status, kernel = cuda.cuLibraryGetKernel(
        cuda.CUlibrary(int(compiled.library)), kernel_name.encode()
    )
    if int(status) != 0:
        raise CompilerError(
            CompilerDiagnostic(
                "resource-query",
                f"failed to open compiled kernel {kernel_name}: {status}",
                requirement="compiled CUDA kernel",
            )
        )
    attributes = {
        "registers": cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,
        "static_shared": cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
        "local": cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
    }
    values = {}
    for label, attribute in attributes.items():
        status, value = cuda.cuKernelGetAttribute(attribute, kernel, 0)
        if int(status) != 0:
            raise CompilerError(
                CompilerDiagnostic(
                    "resource-query",
                    f"failed to query {label}: {status}",
                    requirement="compiled resource attributes",
                )
            )
        values[label] = int(value)
    status, function = cuda.cuKernelGetFunction(kernel)
    if int(status) != 0:
        raise CompilerError(
            CompilerDiagnostic(
                "resource-query",
                f"failed to get kernel function: {status}",
                requirement="CUDA function handle",
            )
        )
    status, active_blocks = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        function, threads, dynamic_shared_bytes
    )
    if int(status) != 0:
        raise CompilerError(
            CompilerDiagnostic(
                "resource-query",
                f"failed to query occupancy: {status}",
                requirement="compiled active-block limit",
            )
        )
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return CompiledResources(
        kernel_name,
        values["registers"],
        values["static_shared"],
        values["local"],
        dynamic_shared_bytes,
        int(active_blocks),
        int(active_blocks) * properties.multi_processor_count,
        plan.storage.workspace_bytes,
        plan.storage.peak_live_workspace_bytes,
        generated.launch.body_variant_count,
        generated.launch.phase_runner_count,
        _compiled_code_bytes(compiled),
        generated.launch.source_bytes,
    )


def _compiled_code_bytes(compiled: Any) -> int:
    dump_to_object = getattr(compiled, "dump_to_object", None)
    if callable(dump_to_object):
        return len(dump_to_object("megabake_kernel"))
    cubin = getattr(compiled, "__cubin__", None)
    if isinstance(cubin, bytes):
        return len(cubin)
    if cubin is None:
        return 0
    cubin_path = Path(str(cubin))
    if cubin_path.is_file():
        return cubin_path.stat().st_size
    return len(str(cubin).encode("utf-8"))


def _with_worker_count(
    plan: KernelPlan, worker_count: int, active_blocks_per_sm: int
) -> KernelPlan:
    phases = tuple(
        replace(
            phase,
            workers=_worker_assignments(phase.region_id, phase.work_items, worker_count),
            barrier=None
            if phase.barrier is None
            else replace(
                phase.barrier, participating_workers=tuple(range(worker_count))
            ),
        )
        for phase in plan.phases
    )
    prototype = plan.storage.worker_scratch[0]
    worker_scratch = tuple(
        replace(prototype, worker_id=worker_id) for worker_id in range(worker_count)
    )
    storage = replace(plan.storage, worker_scratch=worker_scratch)
    launch = replace(
        plan.launch,
        grid_size=worker_count,
        max_active_blocks_per_sm=active_blocks_per_sm,
        required_active_blocks=worker_count,
    )
    return replace(plan, phases=phases, storage=storage, launch=launch)


def _worker_assignments(region_id: str, work_items: Sequence[Any], worker_count: int):
    from .schedule import WorkerAssignment

    quotient, remainder = divmod(len(work_items), worker_count)
    start = 0
    assignments = []
    for worker_id in range(worker_count):
        count = quotient + (1 if worker_id < remainder else 0)
        assignments.append(
            WorkerAssignment(
                worker_id, region_id, tuple(work_items[start : start + count])
            )
        )
        start += count
    return tuple(assignments)


_DYNAMIC_ADDRESS_BUFFERS = 4


@dataclass
class _TableRebind:
    """Slots and graph outputs as offsets from per-call root tensors."""

    inputs: list[torch.Tensor]
    output_layouts: list[Any]
    output_ids: list[str]
    slot_roots: np.ndarray
    slot_offsets: np.ndarray
    empty_slots: np.ndarray
    # (root, shape, strides, element offset, is the root itself) per output.
    graph_outputs: tuple[tuple[int, tuple[int, ...], tuple[int, ...], int, bool], ...]

    def addresses(self, roots: Sequence[torch.Tensor]) -> np.ndarray:
        pointers = np.fromiter(
            (tensor.data_ptr() for tensor in roots), dtype=np.int64, count=len(roots)
        )
        addresses = pointers[self.slot_roots] + self.slot_offsets
        addresses[self.empty_slots] = 0
        return addresses


class _RuntimeState:
    def __init__(
        self,
        plan: KernelPlan,
        generated: GeneratedCuTeModule,
        input_tensors: Mapping[str, torch.Tensor],
        ready: torch.Tensor,
        work_iterators: Mapping[str, torch.Tensor],
        max_active_clusters: Any | None,
        workspace: Mapping[str, torch.Tensor],
        outputs: Mapping[str, torch.Tensor],
        values: Mapping[str, torch.Tensor],
        g0_tensors: tuple[tuple[torch.Tensor, ...], ...],
        workspace_base: torch.Tensor,
    ):
        self.plan = plan
        self.generated = generated
        self.input_tensors = input_tensors
        self.ready = ready
        self.work_iterators = work_iterators
        self.workspace = workspace
        self.workspace_base = workspace_base
        self.outputs = outputs
        self._values = values
        self._values_stale = False
        self.g0_tensors = g0_tensors
        self._normal_view_cache: dict[str, tuple[tuple[Any, ...], Any]] = {}
        self._work_iterator_view_cache: dict[
            str, tuple[tuple[Any, ...], Any]
        ] = {}
        self._g0_view_cache: dict[
            int, tuple[tuple[tuple[Any, ...], ...], tuple[Any, ...]]
        ] = {}
        self._ready_view: tuple[torch.Tensor, Any] | None = None
        self._max_active_clusters = max_active_clusters
        self._address_table: torch.Tensor | None = None
        self._address_mirror: np.ndarray | None = None
        self._phase_table: tuple[torch.Tensor, Any] | None = None
        self._gemm_table: tuple[torch.Tensor, Any] | None = None
        # Per-invocation TMA descriptors (three 128-byte maps per table GEMM),
        # rewritten by every call's kernel prologue.
        self._tma_descriptors: tuple[torch.Tensor, Any] | None = None
        self._checked_layouts: dict[int, tuple[Any, ...]] = {}
        self.address_table_uploads = 0
        # Outputs this state allocates per call, by value id.
        self._allocated_outputs: dict[str, Any] = {}
        # Table launches: per-call rebinding from root pointers (see
        # _prepare_table_rebind); None until the first full binding.
        self._table_rebind: _TableRebind | None = None
        self._slot_addresses: np.ndarray | None = None
        self._dynamic_slot_table: tuple[torch.Tensor, Any] | None = None
        # [pinned buffer, its NumPy view, event of the launch that read it]
        self._dynamic_buffers: list[list[Any]] = []
        self._dynamic_buffer_index = 0
        self._work_iterator_views: tuple[Any, ...] | None = None

    @classmethod
    def create(
        cls,
        plan: KernelPlan,
        generated: GeneratedCuTeModule,
        input_tensors: Mapping[str, torch.Tensor],
        max_active_clusters: Any | None = None,
        ready: torch.Tensor | None = None,
        work_iterators: Mapping[str, torch.Tensor] | None = None,
    ) -> _RuntimeState:
        if plan.graph is None:
            raise CompilerError(
                CompilerDiagnostic(
                    "runtime-binding",
                    "plan is missing its source graph",
                    requirement="source graph",
                )
            )
        _validate_state_capacities(plan, input_tensors)
        device = next(iter(input_tensors.values())).device
        if ready is None:
            ready = make_ready(plan.launch.grid_size, device)
        if work_iterators is None:
            work_iterators = {
                iterator.name: torch.arange(
                    0,
                    iterator.length * iterator.step,
                    iterator.step,
                    device=device,
                    dtype=torch.int64,
                )
                for iterator in generated.launch.work_iterators
            }
        workspace_base = torch.empty(
            plan.storage.workspace_bytes, dtype=torch.uint8, device=device
        )
        workspace: dict[str, torch.Tensor] = {
            assignment.storage_id: _workspace_view(workspace_base, assignment)
            for assignment in plan.storage.workspace
        }
        state = cls(
            plan,
            generated,
            input_tensors,
            ready,
            work_iterators,
            max_active_clusters,
            workspace,
            {},
            {},
            (),
            workspace_base,
        )
        state._refresh_values(input_tensors)
        return state

    def rebind(self, input_tensors: Mapping[str, torch.Tensor]) -> None:
        if self._table_rebind is not None and self._fast_table_rebind(input_tensors):
            return
        self._refresh_values(input_tensors)

    def reset_ready(self) -> None:
        reset_ready(self.ready)

    def poison_workspace(self, value: int = 0xE5) -> None:
        self.workspace_base.fill_(value)

    @property
    def values(self) -> dict[str, torch.Tensor]:
        """Tensor bound to each value for the current call.

        The table fast path updates only outputs; other bindings are derived
        again here, keeping this call's outputs, when something reads them."""
        if self._values_stale:
            self._values_stale = False
            outputs = {value_id: self._values[value_id] for value_id in self._allocated_outputs}
            self._refresh_values(self.input_tensors, reuse_outputs=outputs)
        return self._values

    def _refresh_values(
        self,
        input_tensors: Mapping[str, torch.Tensor],
        reuse_outputs: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        plan = self.plan
        if plan.graph is None:
            raise CompilerError(
                CompilerDiagnostic(
                    "runtime-binding",
                    "plan is missing its source graph",
                    requirement="source graph",
                )
            )
        device = next(iter(input_tensors.values())).device
        workspace = self.workspace
        values: dict[str, torch.Tensor] = {}
        self._allocated_outputs = {}
        for binding in plan.storage.bindings:
            if binding.storage_id in workspace:
                values[binding.value_id] = workspace[binding.storage_id]
                continue
            if binding.kind in {"input", "parameter", "state"}:
                values[binding.value_id] = input_tensors[binding.value_id]
                continue
            if binding.kind in {"alias", "output"}:
                internal_alias = _internal_alias(
                    binding.value_id, plan.graph, values
                )
                if internal_alias is not None and binding.layout is not None:
                    alias_id, alias_tensor = internal_alias
                    values[binding.value_id] = alias_tensor.as_strided(
                        binding.layout.shape,
                        binding.layout.strides,
                        _alias_storage_offset(
                            binding.value_id, alias_id, alias_tensor, plan.graph
                        ),
                    )
                    continue
                alias = _boundary_alias(binding.value_id, plan.graph, input_tensors)
                if alias is not None:
                    values[binding.value_id] = alias
                    continue
            if binding.kind == "output":
                if binding.layout is None:
                    raise CompilerError(
                        CompilerDiagnostic(
                            "runtime-binding",
                            f"output {binding.value_id} has no layout",
                            (binding.value_id,),
                            requirement="output layout",
                        )
                    )
                values[binding.value_id] = (
                    reuse_outputs[binding.value_id]
                    if reuse_outputs is not None and binding.value_id in reuse_outputs
                    else _allocate(binding.layout, device)
                )
                self._allocated_outputs[binding.value_id] = binding.layout
                continue
            if binding.kind == "alias":
                raise CompilerError(
                    CompilerDiagnostic(
                        "runtime-binding",
                        f"cannot resolve alias {binding.value_id}",
                        (binding.value_id,),
                        requirement="boundary alias",
                    )
                )
        # Table launches bind only one template group per GEMM variant; the
        # kernel writes every phase's descriptors from the address table.
        template_phases = set(self.generated.launch.gemm_template_phases)
        table_launch = (
            self.generated.launch.launch_policy == "megabake_cooperative_table"
        )
        g0_tensors = tuple(
            (
                _g0_activation_view(
                    selected.program.loads[0].value_id,
                    _g0_source_tensor(
                        selected.program.loads[0].value_id,
                        plan.graph,
                        input_tensors,
                        values,
                    ),
                    plan.graph,
                ),
                *(
                    _g0_source_tensor(load.value_id, plan.graph, input_tensors, values)
                    for load in selected.program.loads[1:]
                ),
                values[selected.program.store.value_id],
            )
            for phase in plan.phases
            if phase.body.provider == "g0_range"
            and (not table_launch or phase.index in template_phases)
            for selected in plan.selection.selections
            if selected.region_id == phase.region_id
        )
        self.input_tensors = input_tensors
        self._values = values
        self._values_stale = False
        self.outputs = {
            value_id: values[value_id] for value_id in plan.graph.outputs
        }
        if reuse_outputs is not None:
            # Diagnostic re-derivation keeps the fast path and its templates.
            return
        self.g0_tensors = g0_tensors
        self._slot_addresses = None
        self._table_rebind = None

    def _prepare_table_rebind(self) -> None:
        """Express every slot and graph output as (root, byte offset).

        Roots are the graph inputs, the workspace, and the outputs allocated
        per call. After one full binding, later calls only re-check inputs
        whose tensor object changed and gather root pointers, instead of
        re-deriving thousands of bindings in Python. A value whose storage
        cannot be traced to a root keeps the full path.
        """
        graph = self.plan.graph
        launch = self.generated.launch
        inputs = [self.input_tensors[value_id] for value_id in graph.inputs]
        outputs = list(self._allocated_outputs)
        roots = [*inputs, self.workspace_base, *(self.values[v] for v in outputs)]
        index = {value_id: position for position, value_id in enumerate(graph.inputs)}
        index.update(
            {value_id: len(inputs) + 1 + position for position, value_id in enumerate(outputs)}
        )
        workspace_ids = {
            binding.value_id
            for binding in self.plan.storage.bindings
            if binding.storage_id in self.workspace
        }

        def locate(value_id: str) -> tuple[int, int] | None:
            tensor = self.values.get(value_id)
            if tensor is None:
                return None
            root_id = _storage_root(value_id, graph)
            if value_id in workspace_ids or root_id in workspace_ids:
                position = len(inputs)
            elif value_id in index:
                position = index[value_id]
            elif root_id in index:
                position = index[root_id]
            else:
                return None
            root = roots[position]
            if tensor.untyped_storage().data_ptr() != root.untyped_storage().data_ptr():
                return None
            return position, tensor.data_ptr() - root.data_ptr()

        slot_roots = np.full(len(launch.tensors), -1, dtype=np.int64)
        slot_offsets = np.zeros(len(launch.tensors), dtype=np.int64)
        for slot, argument in enumerate(launch.tensors):
            if argument.layout is None:
                continue
            located = locate(argument.value_id)
            if located is None:
                return
            slot_roots[slot], slot_offsets[slot] = located
        graph_outputs = []
        for value_id in graph.outputs:
            located = locate(value_id)
            if located is None:
                return
            tensor = self.values[value_id]
            position, offset = located
            element = tensor.element_size()
            if offset % element:
                return
            graph_outputs.append(
                (
                    position,
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    offset // element,
                    tensor is roots[position],
                )
            )
        self._table_rebind = _TableRebind(
            inputs,
            [self._allocated_outputs[value_id] for value_id in outputs],
            outputs,
            np.where(slot_roots < 0, 0, slot_roots),
            np.where(slot_roots < 0, 0, slot_offsets),
            slot_roots < 0,
            tuple(graph_outputs),
        )
        self._slot_addresses = self._table_rebind.addresses(roots)

    def _fast_table_rebind(self, input_tensors: Mapping[str, torch.Tensor]) -> bool:
        rebind = self._table_rebind
        graph = self.plan.graph
        inputs = [input_tensors[value_id] for value_id in graph.inputs]
        for position, (tensor, previous) in enumerate(zip(inputs, rebind.inputs)):
            if tensor is previous:
                continue
            if (
                tensor.shape != previous.shape
                or tensor.stride() != previous.stride()
                or tensor.dtype != previous.dtype
                or tensor.device != previous.device
            ):
                return False
        device = self.workspace_base.device
        outputs = [_allocate(layout, device) for layout in rebind.output_layouts]
        roots = [*inputs, self.workspace_base, *outputs]
        self._slot_addresses = rebind.addresses(roots)
        rebind.inputs = inputs
        self.input_tensors = input_tensors
        values = self._values
        for value_id, tensor in zip(rebind.output_ids, outputs, strict=True):
            values[value_id] = tensor
        graph_outputs = {}
        for value_id, (position, shape, stride, offset, is_root) in zip(
            graph.outputs, rebind.graph_outputs, strict=True
        ):
            root = roots[position]
            graph_outputs[value_id] = (
                root
                if is_root
                else root.as_strided(shape, stride, root.storage_offset() + offset)
            )
        values.update(graph_outputs)
        self.outputs = graph_outputs
        self._values_stale = True
        return True

    def tensor_for_value(self, value_id: str) -> torch.Tensor:
        if value_id in self.outputs:
            return self.outputs[value_id]
        return self.values[value_id]

    @property
    def compile_arguments(self) -> tuple[Any, ...]:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        return self.arguments(0, stream, include_constants=True)

    def arguments(
        self, epoch: int, stream: cuda.CUstream, *, include_constants: bool = False
    ) -> tuple[Any, ...]:
        if self.generated.launch.launch_policy == "megabake_cooperative_table":
            return self._table_arguments(
                epoch, stream, include_constants=include_constants
            )
        normal_views = {
            argument.name: self._cached_view(
                self._normal_view_cache,
                argument.name,
                self.values[argument.value_id],
            )
            for argument in self.generated.launch.tensors
            if not argument.kind.startswith("g0_")
        }
        g0_view_groups = tuple(
            self._cached_gemm_tensors(group_index, tensors)
            for group_index, tensors in enumerate(self.g0_tensors)
        )
        g0_views = tuple(
            view
            for group in g0_view_groups
            for view in group
        )
        g0_names = [
            argument.name
            for argument in self.generated.launch.tensors
            if argument.kind.startswith("g0_")
        ]
        g0_by_name = dict(zip(g0_names, g0_views, strict=True))
        max_active_clusters = self._max_active_clusters
        arguments = []
        for name in self.generated.launch.arguments:
            if name == "ready":
                if (
                    self._ready_view is None
                    or self._ready_view[0] is not self.ready
                ):
                    self._ready_view = (self.ready, _cute_view(self.ready))
                arguments.append(self._ready_view[1])
            elif name == "workers":
                arguments.append(cutlass.Int32(self.plan.launch.grid_size))
            elif name == "epoch":
                arguments.append(cutlass.Int32(epoch))
            elif name == "work_iterators":
                arguments.append(
                    tuple(
                        self._cached_view(
                            self._work_iterator_view_cache,
                            iterator.name,
                            self.work_iterators[iterator.name],
                        )
                        for iterator in self.generated.launch.work_iterators
                    )
                )
            elif name == "tensors":
                arguments.append((*normal_views.values(), *g0_views))
            elif name == "g0_tensors":
                arguments.append(g0_view_groups)
            elif name in normal_views:
                arguments.append(normal_views[name])
            elif name in g0_by_name:
                arguments.append(g0_by_name[name])
            elif name == "max_active_clusters":
                if include_constants:
                    arguments.append(max_active_clusters)
            elif name == "stream":
                arguments.append(stream)
            else:
                raise CompilerError(
                    CompilerDiagnostic(
                        "runtime-binding",
                        f"generated launch has unknown argument {name}",
                        requirement="complete launch binding",
                    )
                )
        return tuple(arguments)

    def _table_arguments(
        self, epoch: int, stream: cuda.CUstream, *, include_constants: bool = False
    ) -> tuple[Any, ...]:
        launch = self.generated.launch
        device = self.ready.device
        if self._slot_addresses is None:
            for slot, argument in enumerate(launch.tensors):
                if argument.layout is not None:
                    self._check_slot_layout(
                        slot, argument, self.values[argument.value_id]
                    )
            self._prepare_table_rebind()
        if self._slot_addresses is None:
            addresses = np.array(
                [
                    0
                    if argument.layout is None
                    else self.values[argument.value_id].data_ptr()
                    for argument in launch.tensors
                ]
                or [0],
                dtype=np.int64,
            )
        else:
            addresses = self._slot_addresses
        if launch.gemm_table:
            self._check_gemm_alignment(addresses)
        dynamic_slots = np.asarray(launch.dynamic_slots, dtype=np.int64)
        static_addresses = addresses.copy()
        static_addresses[dynamic_slots] = 0
        if self._address_table is None or not np.array_equal(
            self._address_mirror, static_addresses
        ):
            # Static slots (weights and workspace) change only on rebinding;
            # the kernel prologue writes the per-call boundary slots itself.
            table = torch.from_numpy(static_addresses)
            if self._address_table is None:
                self._address_table = table.to(device)
            else:
                self._address_table.copy_(table)
            self._address_mirror = static_addresses
            self.address_table_uploads += 1
        if self._phase_table is None:
            phase_table = torch.tensor(
                launch.phase_table, dtype=torch.int32, device=device
            )
            self._phase_table = (phase_table, _cute_view(phase_table))
        if self._ready_view is None or self._ready_view[0] is not self.ready:
            self._ready_view = (self.ready, _cute_view(self.ready))
        if self._work_iterator_views is None:
            self._work_iterator_views = tuple(
                _cute_view(self.work_iterators[iterator.name])
                for iterator in launch.work_iterators
            )
        if self._dynamic_slot_table is None:
            slot_table = torch.tensor(
                list(launch.dynamic_slots) or [0], dtype=torch.int32, device=device
            )
            self._dynamic_slot_table = (slot_table, _cute_view(slot_table))
        arguments: list[Any] = [
            self._ready_view[1],
            cutlass.Int32(self.plan.launch.grid_size),
            cutlass.Int32(epoch),
            self._cached_view(
                self._normal_view_cache, "addresses", self._address_table
            ),
            self._phase_table[1],
            self._work_iterator_views,
            cutlass.Int64(self._stage_dynamic_addresses(addresses[dynamic_slots])),
            self._dynamic_slot_table[1],
        ]
        if launch.gemm_table:
            if self._gemm_table is None:
                gemm_table = torch.tensor(
                    launch.gemm_table, dtype=torch.int32, device=device
                )
                self._gemm_table = (gemm_table, _cute_view(gemm_table))
            if self._tma_descriptors is None:
                descriptors = torch.zeros(
                    len(launch.gemm_table) * 3 * 16, dtype=torch.int64, device=device
                )
                self._tma_descriptors = (
                    descriptors,
                    from_dlpack(descriptors, assumed_align=128),
                )
            arguments.extend(
                (
                    self._gemm_table[1],
                    self._tma_descriptors[1],
                    tuple(
                        self._cached_gemm_tensors(index, tensors)
                        for index, tensors in enumerate(self.g0_tensors)
                    ),
                )
            )
            if include_constants:
                arguments.append(self._max_active_clusters)
        arguments.append(stream)
        return tuple(arguments)

    def _check_gemm_alignment(self, addresses: np.ndarray) -> None:
        """TMA descriptors need 16-byte-aligned operand base addresses."""
        gemm_slots = getattr(self, "_gemm_slots", None)
        if gemm_slots is None:
            gemm_slots = np.unique(
                np.asarray(self.generated.launch.gemm_table, dtype=np.int64)[:, 1:]
            )
            self._gemm_slots = gemm_slots
        misaligned = gemm_slots[addresses[gemm_slots] % 16 != 0]
        if misaligned.size:
            argument = self.generated.launch.tensors[int(misaligned[0])]
            raise CompilerError(
                CompilerDiagnostic(
                    "runtime-binding",
                    f"{argument.value_id} is not 16-byte aligned for its TMA GEMM",
                    (argument.value_id,),
                    requirement="16-byte-aligned GEMM operand",
                )
            )

    def _stage_dynamic_addresses(self, addresses: np.ndarray) -> int:
        """Write this call's boundary addresses to a host-mapped pinned buffer
        and return its address, which the kernel prologue reads directly.

        A small ring keeps a buffer untouched until the launch that read it
        has finished, without adding GPU work.
        """
        if not self._dynamic_buffers:
            for _ in range(_DYNAMIC_ADDRESS_BUFFERS):
                host = torch.zeros(max(len(addresses), 1), dtype=torch.int64).pin_memory()
                self._dynamic_buffers.append([host, host.numpy(), None])
        entry = self._dynamic_buffers[self._dynamic_buffer_index]
        self._dynamic_buffer_index = (
            self._dynamic_buffer_index + 1
        ) % _DYNAMIC_ADDRESS_BUFFERS
        if entry[2] is not None:
            entry[2].synchronize()
        entry[1][: len(addresses)] = addresses
        self._pending_dynamic_buffer = entry
        return entry[0].data_ptr()

    def record_launch(self, stream: torch.cuda.Stream) -> None:
        """Mark the dynamic-address buffer of the launch just enqueued."""
        entry = getattr(self, "_pending_dynamic_buffer", None)
        if entry is None:
            return
        event = entry[2] or torch.cuda.Event()
        event.record(stream)
        entry[2] = event
        self._pending_dynamic_buffer = None

    def _check_slot_layout(
        self, slot: int, argument: Any, tensor: torch.Tensor
    ) -> None:
        identity = _tensor_identity(tensor)
        if self._checked_layouts.get(slot) == identity:
            return
        layout = argument.layout
        shape = tuple(layout.shape)
        if tuple(tensor.shape) != shape or any(
            size > 1 and actual != expected
            for size, actual, expected in zip(
                shape, tensor.stride(), layout.strides, strict=True
            )
        ):
            raise CompilerError(
                CompilerDiagnostic(
                    "runtime-binding",
                    f"{argument.value_id} has shape {tuple(tensor.shape)} and strides "
                    f"{tuple(tensor.stride())}; the kernel was compiled for "
                    f"{shape} and {tuple(layout.strides)}",
                    (argument.value_id,),
                    requirement="compiled slot layout",
                )
            )
        self._checked_layouts[slot] = identity

    @staticmethod
    def _cached_view(
        cache: dict[Any, tuple[tuple[Any, ...], Any]], key: Any, tensor: torch.Tensor
    ) -> Any:
        entry = cache.get(key)
        identity = _tensor_identity(tensor)
        if entry is None or entry[0] != identity:
            entry = (identity, _cute_view(tensor))
            cache[key] = entry
        return entry[1]

    def _cached_gemm_tensors(
        self, group_index: int, tensors: tuple[torch.Tensor, ...]
    ) -> tuple[Any, ...]:
        entry = self._g0_view_cache.get(group_index)
        identities = tuple(_tensor_identity(tensor) for tensor in tensors)
        if (
            entry is None
            or len(entry[0]) != len(tensors)
            or any(old != new for old, new in zip(entry[0], identities))
        ):
            entry = (identities, cute_gemm_tensors(*tensors))
            self._g0_view_cache[group_index] = entry
        return entry[1]


def _tensor_identity(tensor: torch.Tensor) -> tuple[Any, ...]:
    return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)


def _validate_state_capacities(
    plan: KernelPlan, input_tensors: Mapping[str, torch.Tensor]
) -> None:
    for state in plan.storage.state:
        valid_length = input_tensors[state.valid_length_value_id]
        length = int(valid_length.item())
        if length < 0 or length + state.query_length > state.capacity:
            raise CompilerError(
                CompilerDiagnostic(
                    "state-capacity",
                    (
                        f"KV valid length {length} plus {state.query_length} exceeds "
                        f"capacity {state.capacity}"
                    ),
                    (state.valid_length_value_id,),
                    requirement="bounded KV state",
                )
            )


def _workspace_view(base: torch.Tensor, assignment: Any) -> torch.Tensor:
    dtype = getattr(torch, assignment.layout.dtype.removeprefix("torch."))
    element_size = dtype.itemsize
    if assignment.required_bytes % element_size:
        raise CompilerError(
            CompilerDiagnostic(
                "workspace-binding",
                f"workspace slice {assignment.storage_id} is not element-aligned",
                requirement="dtype-aligned workspace slice",
            )
        )
    storage = base.narrow(
        0, assignment.offset_bytes, assignment.required_bytes
    ).view(dtype)
    return storage.as_strided(assignment.layout.shape, assignment.layout.strides)


def _allocate(layout: Any, device: torch.device) -> torch.Tensor:
    dtype = getattr(torch, layout.dtype.removeprefix("torch."))
    return torch.empty_strided(
        layout.shape, layout.strides, dtype=dtype, device=device
    )


def _boundary_alias(
    value_id: str,
    graph: ComputeGraph,
    input_tensors: Mapping[str, torch.Tensor],
    seen: set[str] | None = None,
) -> torch.Tensor | None:
    seen = seen or set()
    if value_id in seen:
        return None
    seen.add(value_id)
    if value_id in input_tensors:
        return input_tensors[value_id]
    value = graph.values.get(value_id)
    if value is None:
        return None
    for alias in value.storage_relation.aliases:
        base = _boundary_alias(alias, graph, input_tensors, seen)
        if base is not None:
            return base.as_strided(
                base.shape if value.shape is None else value.shape,
                base.stride() if value.strides is None else value.strides,
                _alias_storage_offset(value_id, alias, base, graph),
            )
    storage_id = value.storage_relation.storage_id
    if storage_id in input_tensors:
        base = input_tensors[storage_id]
        return base.as_strided(
            base.shape if value.shape is None else value.shape,
            base.stride() if value.strides is None else value.strides,
            _alias_storage_offset(value_id, storage_id, base, graph),
        )
    return None


def _alias_storage_offset(
    value_id: str, base_id: str, base: torch.Tensor, graph: ComputeGraph
) -> int:
    # FX offsets are relative to the shared source storage; rebase them onto
    # the runtime tensor that backs the aliased value.
    value = graph.values.get(value_id)
    base_value = graph.values.get(base_id)
    value_offset = 0 if value is None else value.storage_offset
    base_offset = 0 if base_value is None else base_value.storage_offset
    return base.storage_offset() + value_offset - base_offset


def _internal_alias(
    value_id: str,
    graph: ComputeGraph,
    values: Mapping[str, torch.Tensor],
) -> tuple[str, torch.Tensor] | None:
    candidates = [value_id]
    seen: set[str] = set()
    while candidates:
        candidate = candidates.pop()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate != value_id and candidate in values:
            return candidate, values[candidate]
        value = graph.values.get(candidate)
        if value is None:
            continue
        relation = value.storage_relation
        candidates.extend((*relation.aliases, relation.storage_id))
    return None


def _g0_source_tensor(
    value_id: str,
    graph: ComputeGraph,
    input_tensors: Mapping[str, torch.Tensor],
    value_tensors: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    if value_id in input_tensors:
        return input_tensors[value_id].detach()
    value = graph.values.get(value_id)
    if value is None:
        raise CompilerError(
            CompilerDiagnostic(
                "runtime-binding",
                f"GEMM input {value_id} is missing",
                (value_id,),
                requirement="GEMM input binding",
            )
        )
    for alias in value.storage_relation.aliases:
        if alias in input_tensors:
            return input_tensors[alias].detach()
        return _g0_source_tensor(alias, graph, input_tensors, value_tensors)
    return _g0_value_source_tensor(value_id, graph, value_tensors)


def _g0_activation_view(
    value_id: str, source: torch.Tensor, graph: ComputeGraph
) -> torch.Tensor:
    # The G0 body flattens leading axes of A, which is only valid when the
    # source root already ends in the GEMM K axis; otherwise bind the 2-D view.
    value = graph.values.get(value_id)
    if (
        value is None
        or value.shape is None
        or value.strides is None
        or len(value.shape) != 2
        or tuple(source.shape[-1:]) == tuple(value.shape[-1:])
    ):
        return source
    root = graph.values.get(_storage_root(value_id, graph))
    root_offset = 0 if root is None else root.storage_offset
    return source.as_strided(
        value.shape,
        value.strides,
        source.storage_offset() + value.storage_offset - root_offset,
    )


def _storage_root(value_id: str, graph: ComputeGraph) -> str:
    seen: set[str] = set()
    while value_id not in seen:
        seen.add(value_id)
        value = graph.values.get(value_id)
        if value is None or not value.storage_relation.aliases:
            return value_id
        value_id = value.storage_relation.aliases[0]
    return value_id


def _g0_value_source_tensor(
    value_id: str,
    graph: ComputeGraph,
    value_tensors: Mapping[str, torch.Tensor],
    seen: set[str] | None = None,
) -> torch.Tensor:
    seen = seen or set()
    if value_id in seen:
        raise CompilerError(
            CompilerDiagnostic(
                "runtime-binding",
                f"GEMM input {value_id} has a cyclic alias",
                (value_id,),
                requirement="acyclic boundary alias",
            )
        )
    seen.add(value_id)
    if value_id in value_tensors:
        return value_tensors[value_id].detach()
    value = graph.values.get(value_id)
    if value is None:
        raise CompilerError(
            CompilerDiagnostic(
                "runtime-binding",
                f"GEMM input {value_id} is missing",
                (value_id,),
                requirement="GEMM input binding",
            )
        )
    for alias in value.storage_relation.aliases:
        if alias in value_tensors:
            return value_tensors[alias].detach()
        return _g0_value_source_tensor(alias, graph, value_tensors, seen)
    storage_id = value.storage_relation.storage_id
    if storage_id in value_tensors:
        return value_tensors[storage_id].detach()
    raise CompilerError(
        CompilerDiagnostic(
            "runtime-binding",
            f"GEMM input {value_id} does not alias a boundary tensor",
            (value_id,),
            requirement="GEMM boundary input",
        )
    )


def _cute_view(tensor: torch.Tensor) -> Any:
    if tensor.data_ptr() % 16:
        raise CompilerError(
            CompilerDiagnostic(
                "runtime-binding",
                f"tensor at {tensor.data_ptr():#x} is not 16-byte aligned",
                requirement="16-byte tensor alignment",
            )
        )
    result = from_dlpack(tensor.detach(), assumed_align=16)
    leading_dimension = next(
        (index for index, stride in enumerate(tensor.stride()) if stride == 1),
        None,
    )
    if leading_dimension is None:
        return result
    return result.mark_layout_dynamic(leading_dim=leading_dimension)


def _source_op_ids(plan: KernelPlan) -> tuple[str, ...]:
    if plan.graph is None:
        return ()
    return tuple(plan.graph.ops)


def _region_ids(plan: KernelPlan) -> tuple[str, ...]:
    if plan.graph is None:
        return ()
    return tuple(region.id for region in plan.graph.regions)
