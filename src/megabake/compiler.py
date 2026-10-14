from __future__ import annotations

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
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda
import torch
from torch.utils import _pytree as pytree

from .bodies import SelectionPolicy, TargetFacts
from .bodies.gemm import cute_gemm_tensors
from .codegen import GeneratedCuTeModule, emit_cute
from .diagnostics import CompilerDiagnostic, CompilerError
from .frontend import CaptureContext, PreparedFX
from .graph import ComputeGraph, import_fx
from .lowering import canonicalize_semantics
from .memory import verify_plan
from .plan import (
    KernelPlan,
    PlanMeasurement,
    PlanSearchResult,
    SearchBudget,
    search_plans,
)
from .schedule import EventSchedule, make_region_event_schedule
from .regions import form_regions
from .runtime import (
    EventRuntimeAdmission,
    RuntimeAdmission,
    admit_event_runtime,
    admit_runtime,
    execute_event_plan,
    execute_phase_plan,
)


class UnsupportedCompiler(RuntimeError):
    pass


@dataclass(frozen=True)
class CompiledResources:
    kernel_name: str
    registers_per_thread: int
    static_shared_memory_bytes: int
    local_memory_bytes_per_thread: int
    dynamic_shared_memory_bytes: int
    active_blocks_per_sm: int
    max_cooperative_workers: int


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
    parameter_tensors: Mapping[str, torch.Tensor]
    ready: torch.Tensor
    work_iterators: Mapping[str, torch.Tensor]
    search: Any | None = None


@dataclass(frozen=True)
class CallableTrace:
    invocation_id: uuid.UUID
    epoch: int
    stream: torch.cuda.Stream
    input_data_ptrs: tuple[int, ...]
    output_data_ptrs: tuple[int, ...]


def compile_post_grad(
    prepared: PreparedFX,
    *,
    target: TargetFacts | None = None,
    policy: SelectionPolicy | None = None,
    output_dir: Path | str | None = None,
    search_budget: SearchBudget | int | None = None,
    scheduler: str = "phase",
) -> Callable[..., Any]:
    compile_started = time.perf_counter()
    if scheduler not in {"phase", "region-events", "auto"}:
        raise ValueError(
            "scheduler must be 'phase', 'region-events', or 'auto'"
        )
    graph, selected_target = _post_grad_graph(prepared, target)
    if isinstance(search_budget, int):
        if search_budget < 1:
            raise ValueError("search budget must be positive")
        budget = SearchBudget(max_plans=search_budget)
    else:
        budget = search_budget or SearchBudget()
    with torch.no_grad():
        reference = prepared.forward(*prepared.example_inputs)
    if scheduler == "auto":
        phase_result = _search_scheduler(
            graph,
            selected_target,
            policy,
            budget,
            prepared,
            reference,
            output_dir,
            "phase",
        )
        event_result = _search_scheduler(
            graph,
            selected_target,
            policy,
            budget,
            prepared,
            reference,
            output_dir,
            "region-events",
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
            prepared,
            reference,
            output_dir,
            scheduler,
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
    compiled = result.selected_compiled
    result = replace(result, selected=compiled.plan)
    compiled = replace(compiled, search=result)
    if compiled.event_schedule is not None:
        compiled.ready.zero_()
    forward = bind_callable(compiled, prepared.context)
    forward.megabake_compile_us = (time.perf_counter() - compile_started) * 1_000_000
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
                current_plan, artifact_dir, event_schedule=event_schedule
            )
            module = _load_generated_module(generated)
            state = _RuntimeState.create(
                current_plan,
                generated,
                input_tensors,
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
            parameter_tensors = {
                name: value
                for name, value in zip(input_order, example_inputs, strict=True)
                if isinstance(value, torch.nn.Parameter)
            }
            parameter_names = frozenset(parameter_tensors)
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
                parameter_tensors,
                state.ready,
                state.work_iterators,
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


def bind_callable(
    compiled_kernel: CompiledKernel, context: CaptureContext
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
    trace: list[CallableTrace] = []
    runtime_lock = threading.Lock()
    state: _RuntimeState | None = None
    next_epoch = 0
    epoch_stride = (
        1
        if compiled_kernel.event_schedule is not None
        else max(
            (phase.index for phase in compiled_kernel.plan.phases if phase.barrier is not None),
            default=-1,
        )
        + 1
    )
    epoch_stride = max(epoch_stride, 1)

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
                value = compiled_kernel.parameter_tensors.get(name)
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
            active_stream = torch.cuda.current_stream(
                next(iter(input_tensors.values())).device
            )

            if state is None:
                state = _RuntimeState.create(
                    compiled_kernel.plan,
                    compiled_kernel.generated,
                    input_tensors,
                    ready=compiled_kernel.ready,
                    work_iterators=compiled_kernel.work_iterators,
                )
            else:
                state.rebind(input_tensors)

            def kernel(
                ready: torch.Tensor, epoch: int, stream: torch.cuda.Stream
            ) -> None:
                arguments = state.arguments(epoch, cuda.CUstream(stream.cuda_stream))
                compiled_kernel.compiled(*arguments)

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
    return forward


def _post_grad_graph(
    prepared: PreparedFX, target: TargetFacts | None
) -> tuple[ComputeGraph, TargetFacts]:
    try:
        selected_target = target or _target_facts(_common_cuda_device(prepared.example_inputs))
        graph = import_fx(prepared.graph, prepared.context)
        return form_regions(canonicalize_semantics(graph)), selected_target
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


def _make_plan_evaluator(
    prepared: PreparedFX,
    reference: Any,
    output_dir: Path | str | None,
    scheduler: str = "phase",
):
    def evaluate(plan: KernelPlan, budget: SearchBudget) -> PlanMeasurement:
        try:
            compiled = compile_plan(
                plan,
                prepared.example_inputs,
                output_dir=output_dir,
                scheduler=scheduler,
            )
            executable = bind_callable(compiled, prepared.context)
            with torch.no_grad():
                actual = executable(*prepared.example_inputs)
            relative_error = _output_relative_l2(reference, actual)
            if relative_error >= 0.01:
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
):
    evaluator = (
        budget.evaluator
        if scheduler == "phase" and budget.evaluator is not None
        else _make_plan_evaluator(prepared, reference, output_dir, scheduler)
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
        ready=compiled_kernel.ready,
        work_iterators=compiled_kernel.work_iterators,
    )
    if compiled_kernel.event_schedule is not None:
        state.ready.zero_()
        epoch_stride = 1
    else:
        state.ready.fill_(-1)
        epoch_stride = max(
            (
                phase.index
                for phase in compiled_kernel.plan.phases
                if phase.barrier is not None
            ),
            default=-1,
        ) + 1
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
    for _ in range(count):
        tokens = [prepare() for _ in range(count)]
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for token in tokens:
            launch(token)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1_000 / count)
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
    compiled: Any, threads: int, dynamic_shared_bytes: int
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
    )


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


class _RuntimeState:
    def __init__(
        self,
        plan: KernelPlan,
        generated: GeneratedCuTeModule,
        input_tensors: Mapping[str, torch.Tensor],
        ready: torch.Tensor,
        work_iterators: Mapping[str, torch.Tensor],
        workspace: Mapping[str, torch.Tensor],
        outputs: Mapping[str, torch.Tensor],
        values: Mapping[str, torch.Tensor],
        g0_tensors: tuple[tuple[torch.Tensor, ...], ...],
    ):
        self.plan = plan
        self.generated = generated
        self.input_tensors = input_tensors
        self.ready = ready
        self.work_iterators = work_iterators
        self.workspace = workspace
        self.outputs = outputs
        self.values = values
        self.g0_tensors = g0_tensors
        self._normal_view_cache: dict[str, tuple[tuple[Any, ...], Any]] = {}
        self._work_iterator_view_cache: dict[
            str, tuple[tuple[Any, ...], Any]
        ] = {}
        self._g0_view_cache: dict[
            int, tuple[tuple[tuple[Any, ...], ...], tuple[Any, ...]]
        ] = {}
        self._ready_view: tuple[torch.Tensor, Any] | None = None
        self._max_active_clusters = (
            cutlass.utils.HardwareInfo().get_max_active_clusters(1)
            if "max_active_clusters" in generated.launch.arguments
            else None
        )

    @classmethod
    def create(
        cls,
        plan: KernelPlan,
        generated: GeneratedCuTeModule,
        input_tensors: Mapping[str, torch.Tensor],
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
        device = next(iter(input_tensors.values())).device
        if ready is None:
            ready = torch.full(
                (plan.launch.grid_size + 1,),
                -1,
                dtype=torch.int32,
                device=device,
            )
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
        workspace: dict[str, torch.Tensor] = {}
        for assignment in plan.storage.workspace:
            if assignment.storage_id in workspace:
                continue
            workspace[assignment.storage_id] = _allocate(assignment.layout, device)
        state = cls(
            plan,
            generated,
            input_tensors,
            ready,
            work_iterators,
            workspace,
            {},
            {},
            (),
        )
        state._refresh_values(input_tensors)
        return state

    def rebind(self, input_tensors: Mapping[str, torch.Tensor]) -> None:
        self._refresh_values(input_tensors)

    def reset_ready(self) -> None:
        self.ready.fill_(-1)

    def _refresh_values(self, input_tensors: Mapping[str, torch.Tensor]) -> None:
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
        for binding in plan.storage.bindings:
            if binding.storage_id in workspace:
                values[binding.value_id] = workspace[binding.storage_id]
                continue
            if binding.kind in {"input", "parameter", "state"}:
                values[binding.value_id] = input_tensors[binding.value_id]
                continue
            if binding.kind in {"alias", "output"}:
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
                values[binding.value_id] = _allocate(binding.layout, device)
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
        g0_tensors = tuple(
            (
                *(
                    _g0_source_tensor(load.value_id, plan.graph, input_tensors, values)
                    for load in selected.program.loads
                ),
                values[selected.program.store.value_id],
            )
            for phase in plan.phases
            if phase.body.provider == "g0_range"
            for selected in plan.selection.selections
            if selected.region_id == phase.region_id
        )
        self.input_tensors = input_tensors
        self.values = values
        self.outputs = {
            value_id: values[value_id] for value_id in plan.graph.outputs
        }
        self.g0_tensors = g0_tensors

    def tensor_for_value(self, value_id: str) -> torch.Tensor:
        return self.values[value_id]

    @property
    def compile_arguments(self) -> tuple[Any, ...]:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        return self.arguments(0, stream, include_constants=True)

    def arguments(
        self, epoch: int, stream: cuda.CUstream, *, include_constants: bool = False
    ) -> tuple[Any, ...]:
        normal_views = {
            argument.name: self._cached_view(
                self._normal_view_cache,
                argument.name,
                self.values[argument.value_id],
            )
            for argument in self.generated.launch.tensors
            if not argument.kind.startswith("g0_")
        }
        g0_views = tuple(
            view
            for group_index, tensors in enumerate(self.g0_tensors)
            for view in self._cached_gemm_tensors(group_index, tensors)
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
            elif name.startswith("work_iterator_"):
                arguments.append(
                    self._cached_view(
                        self._work_iterator_view_cache,
                        name,
                        self.work_iterators[name],
                    )
                )
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
            )
    storage_id = value.storage_relation.storage_id
    if storage_id in input_tensors:
        base = input_tensors[storage_id]
        return base.as_strided(
            base.shape if value.shape is None else value.shape,
            base.stride() if value.strides is None else value.strides,
        )
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
