from __future__ import annotations

import textwrap

from dataclasses import dataclass
from typing import Any, Mapping

from ..graph import Region
from .interface import (
    BodyConfig,
    CandidateSet,
    CompletionContract,
    InvalidBodyConfig,
    LayoutContract,
    ParticipantContract,
    PipelineContract,
    ProviderRejection,
    ResourceEvidence,
    ScratchContract,
    SelectionPolicy,
    SemanticContract,
    TargetFacts,
    TensorLayout,
    TileContract,
    validate_body_config,
)


class UnsupportedAttentionBody(ValueError):
    pass


def _attention_sequence_length(load: Any) -> int | None:
    if load.source_shape is None:
        return None
    for index, axis in enumerate(load.axes):
        if axis in {"s", "k"}:
            return load.source_shape[index]
    return None


def _attention_shape_is_valid(
    load: Any, batch: int, length: int, heads: int, head_dim: int
) -> bool:
    return load.source_shape in {
        (batch, length, heads, head_dim),
        (batch, heads, length, head_dim),
    }


def _attention_access(
    load: Any, head_variable: str, binding_index: int, dimension_variable: str = "d"
) -> str:
    indices = []
    for axis in load.axes:
        if isinstance(axis, str):
            if axis == "k":
                indices.append("key")
            elif axis == "d":
                indices.append(dimension_variable)
            else:
                indices.append(axis)
        elif isinstance(axis, int):
            indices.append(str(axis))
        elif axis.terms[0].modulus == 0:
            indices.append("b")
        else:
            indices.append(head_variable)
    return f"tensor_bindings[{binding_index}][{', '.join(indices)}]"


@dataclass(frozen=True)
class GeneratedAttentionBody:
    source: str
    function_name: str = "run"
    threads_per_cta: int = 128
    tile_count: int = 1
    work_iterator_dtype: str = "torch.int64"
    work_iterator_encoding: str = "one row-block base per CTA tile"
    # Phase scratch (FP32 elements, Int32 counters) the body takes as its
    # epilogue argument: (floats, counters, float offset, counter offset).
    phase_floats: int = 0
    phase_counters: int = 0


class AttentionBodyProvider:
    provider_name = "attention"

    def get_candidates(
        self, region: Region, target: TargetFacts, policy: SelectionPolicy
    ) -> CandidateSet:
        def reject(reason: str) -> CandidateSet:
            return CandidateSet((), (ProviderRejection(self.provider_name, reason),))

        if not policy.allow_tuned:
            return reject("tuned providers disabled")
        if not region.tensor_programs or len(region.tensor_programs) != 1:
            return reject("region has no single tensor program")

        program = region.tensor_programs[0]
        if program.family != "attention":
            return reject("provider supports attention only")
        facts = region.semantic_facts
        if facts.get("family") != "attention":
            return reject("region semantics are not attention")

        numerical_policy = facts.get("numerical_policy", {})
        if facts.get("algorithm") not in {"stable_softmax", "online_softmax"}:
            return reject("provider supports stable or online softmax only")
        if numerical_policy != {
            "score_dtype": "torch.bfloat16",
            "statistics_dtype": "torch.float32",
            "probability_dtype": "torch.bfloat16",
            "accumulator_dtype": "torch.float32",
        }:
            return reject("provider preserves only the captured BF16/float32 policy")
        if program.accumulator_dtype != "torch.float32":
            return reject("attention accumulator must be float32")
        if program.store.dtype != "torch.bfloat16":
            return reject("attention output must be BF16")
        requested_outputs = tuple(facts.get("requested_outputs", ()))
        boundary_output = facts.get("boundary_output")
        allowed_outputs = {
            (program.store.value_id,),
        }
        if isinstance(boundary_output, str):
            allowed_outputs.update(
                ((boundary_output,), (program.store.value_id, boundary_output))
            )
        if requested_outputs not in allowed_outputs:
            return reject("materialized scores and extra attention outputs are unsupported")

        loads = {load.id: load for load in program.loads}
        if not {"attention:q", "attention:k", "attention:v"} <= set(loads):
            return reject("attention program lacks Q/K/V accesses")
        if any(
            loads[load_id].dtype != "torch.bfloat16"
            for load_id in ("attention:q", "attention:k", "attention:v")
        ):
            return reject("provider supports BF16 Q/K/V only")

        grouping = facts.get("head_grouping", {})
        if not isinstance(grouping, Mapping):
            return reject("attention head grouping is invalid")
        batch = grouping.get("batch")
        query_heads = grouping.get("query_heads")
        kv_heads = grouping.get("kv_heads")
        group_size = grouping.get("group_size")
        head_dim = grouping.get("head_dim")
        if (
            not all(isinstance(value, int) for value in (batch, query_heads, kv_heads, group_size, head_dim))
            or batch <= 0
            or query_heads <= 0
            or kv_heads <= 0
            or group_size <= 0
            or query_heads != kv_heads * group_size
            or head_dim <= 0
        ):
            return reject("attention head grouping is invalid")

        q_load = loads["attention:q"]
        k_load = loads["attention:k"]
        v_load = loads["attention:v"]
        query_length = _attention_sequence_length(q_load)
        kv_length = _attention_sequence_length(k_load)
        if (
            query_length is None
            or kv_length is None
            or not _attention_shape_is_valid(
                q_load, batch, query_length, query_heads, head_dim
            )
            or not _attention_shape_is_valid(
                k_load, batch, kv_length, kv_heads, head_dim
            )
            or not _attention_shape_is_valid(
                v_load, batch, kv_length, kv_heads, head_dim
            )
            or program.store.shape
            != (batch * query_heads, query_length, head_dim)
        ):
            return reject("attention shapes do not match the captured access maps")

        mask = facts.get("mask", {})
        if not isinstance(mask, Mapping):
            return reject("attention mask semantics are invalid")
        if mask.get("kind") == "additive_tensor":
            if "attention:mask" not in loads:
                return reject("additive mask is not bound")
        elif mask.get("kind") == "stateful_causal":
            if not program.state_accesses:
                return reject("stateful causal mask has no KV state contract")
        elif mask.get("kind") == "none":
            pass
        elif mask.get("kind") == "typed_where":
            if (
                mask.get("window_offsets") is not None
                or not isinstance(mask.get("false_value"), (int, float))
            ):
                return reject("provider supports additive tensors and causal masks only")
            if mask.get("causal_offset") is None and (
                "attention:mask" not in loads or mask.get("true_value") not in (0, 0.0)
            ):
                return reject("typed mask lacks a bound boolean condition and zero true value")
        else:
            return reject("unsupported attention mask hook")

        cache_effects = facts.get("cache_effects", {})
        stateful = bool(program.state_accesses)
        if (
            not isinstance(cache_effects, Mapping)
            or cache_effects.get("policy") != ("stateful" if stateful else "read_only")
            or (stateful and cache_effects.get("updates") == ())
            or (not stateful and cache_effects.get("appends") != ())
            or (not stateful and cache_effects.get("updates") != ())
        ):
            return reject("provider supports read-only or bounded stateful KV caches only")
        if stateful and query_length != 1:
            return reject("stateful KV attention supports one-token decode only")

        if target.architecture != "sm90a" or not target.supports_wgmma:
            return reject("provider targets WGMMA-capable sm90a only")
        if attention_scratch_rows(kv_length) < 1:
            return reject(
                f"a {kv_length}-key FP32 score row exceeds {ATTENTION_SCRATCH_BYTES} scratch bytes"
            )
        body_name = (
            "attention_stateful_decode_rows"
            if stateful
            else "attention_decode_rows"
            if query_length == 1
            else "attention_prefill_rows"
        )
        return CandidateSet(
            (_attention_config(region, program, body_name),),
            (),
        )


def attention_candidates(
    region: Region,
    target: TargetFacts,
    policy: SelectionPolicy | None = None,
) -> CandidateSet:
    selected_policy = policy or SelectionPolicy()
    result = AttentionBodyProvider().get_candidates(region, target, selected_policy)
    configurations = []
    rejections = list(result.rejections)
    for config in result.configurations:
        try:
            validate_body_config(config, target, selected_policy)
        except InvalidBodyConfig as error:
            rejections.append(ProviderRejection(config.provider, str(error)))
        else:
            configurations.append(config)
    return CandidateSet(tuple(configurations), tuple(rejections))


# Each warp owns one query row at a time and keeps that row's FP32 scores in
# shared scratch; the scratch holds at most this many rows and this many bytes.
ATTENTION_SCRATCH_ROWS = 16
ATTENTION_SCRATCH_BYTES = 32 * 1024


# Value-sum keys loaded together before accumulation in the decode body.
_VALUE_BLOCK = 8
# Keys per warp iteration of the split-context decode value sum.
_VALUE_UNROLL = 8


def attention_scratch_rows(kv_length: int) -> int:
    return min(ATTENTION_SCRATCH_ROWS, ATTENTION_SCRATCH_BYTES // (kv_length * 4))


def _attention_config(region: Region, program: Any, name: str) -> BodyConfig:
    facts = region.semantic_facts
    stateful = region.semantic_facts.get("kv_state") is not None
    threads = 128
    warp_roles = ("one query row at a time: lanes split keys, then head dimensions",) * 4
    kv_capacity = next(
        length
        for load in program.loads
        if load.id == "attention:k"
        for length in (_attention_sequence_length(load),)
        if length is not None
    )
    scratch_bytes = attention_scratch_rows(kv_capacity) * kv_capacity * 4
    layouts = {
        load.value_id: TensorLayout(
            load.dtype,
            load.axes,
            load.strides,
            16,
            load.source_shape,
        )
        for load in program.loads
    }
    layouts[program.store.value_id] = TensorLayout(
        program.store.dtype,
        ("bh", "s", "d"),
        program.store.strides,
        16,
        program.store.shape,
    )
    return BodyConfig(
        provider="attention",
        name=name,
        semantic=SemanticContract(
            operation_family="attention",
            source_op_ids=region.source_op_ids,
            numerical_policy="source_semantics",
            expression_hooks=("score", "mask"),
            expression_hook_restrictions=(
                "score hooks are row-local over query and key coordinates",
                "mask hooks are additive tensors or causal selections",
            ),
            access_effects={
                "q": facts["q_access"],
                "k": facts["k_access"],
                "v": facts["v_access"],
                "mask": facts["mask"],
                "head_grouping": facts["head_grouping"],
                "scale": facts["scale"],
                "numerical_policy": facts["numerical_policy"],
                "cache_effects": facts["cache_effects"],
                "kv_state": facts.get("kv_state"),
            },
            access_restrictions=(
                (
                    "KV state is bounded, contiguous, and updated in source order"
                    if stateful
                    else "KV state is read-only"
                ),
            ),
        ),
        layouts=LayoutContract(layouts, {program.store.value_id: layouts[program.store.value_id]}),
        tiles=TileContract(
            cta_shape=(1, 1, next(load.source_shape[3] for load in program.loads if load.id == "attention:q")),
            cluster_shape=(1, 1, 1),
            work_partition=(
                "one query row per warp in one CTA" if stateful else "one query row per warp"
            ),
            mma_family=None,
            reduction_tile=(
                next(
                    length
                    for load in program.loads
                    if load.id == "attention:k"
                    for length in (_attention_sequence_length(load),)
                    if length is not None
                ),
            ),
            tail_policy="predicated",
        ),
        participants=ParticipantContract(
            participating_threads=threads,
            waiting_threads=0,
            warp_roles=warp_roles,
            registers_per_thread=48,
        ),
        pipeline=PipelineContract(
            stage_count=3,
            barriers=(),
            async_operations=(),
            shared_memory_layouts={},
            drain_rule=(
                "the score-maximum, softmax-denominator, and value-accumulation "
                "passes complete before return"
            ),
        ),
        scratch=ScratchContract(
            owner="cta",
            size_bytes=scratch_bytes,
            alignment_bytes=16,
            initial_state="unspecified; each row writes its scores first",
            reuse_rule="each warp's score row is reused after its sync_warp",
        ),
        completion=CompletionContract(
            outputs_ready=(
                "assigned rows and all KV state writes are stored when run returns"
                if stateful
                else "assigned rows are stored when run returns"
            ),
            output_tiles_ready="each assigned output element is ready after its store",
            async_work_complete="run performs no asynchronous work",
            scratch_reusable="scratch is reusable when run returns",
            register_fragments="row score, softmax maximum and total, and value accumulators",
            warp_roles_on_return="unchanged",
        ),
        evidence=ResourceEvidence(
            measured=not stateful,
            source=(
                "stateful decode extension of the CuTe row pipeline adapted from "
                "Mirage multitoken Hopper attention"
                if stateful
                else "CuTe row pipeline adapted from Mirage multitoken Hopper attention"
            ),
            revision=(
                "mirage@f9eb70c254acefc9f3667b2a973d0dcf25471fce; task-28"
                if stateful
                else "mirage@f9eb70c254acefc9f3667b2a973d0dcf25471fce; task-26"
            ),
            resources={
                "threads": threads,
                "shared_bytes": scratch_bytes,
                "registers_per_thread": 48,
                "active_blocks_per_sm": 10,
            },
            measurements={
                "target": "NVIDIA H200 MIG 3g.71GB",
                "workload": "batch=2, query=5, kv=7, query_heads=4, kv_heads=2, head_dim=8",
                "samples": 8,
                "median_isolated_launch_us": 16.794,
                "median_device_event_span_us": 137.040,
                "median_complete_call_us": 150.397,
            },
        ),
    )


# Split-context decode: a query row's visible keys split into up to this many
# chunks, each one CTA work item, with at least this many keys per chunk.
DECODE_MAX_CHUNKS = 16
DECODE_MIN_CHUNK_KEYS = 64


def emit_attention_body(
    program: Any,
    config: BodyConfig,
    *,
    defer_length_publish: bool = False,
    workers: int | None = None,
) -> GeneratedAttentionBody:
    """Emit an attention device body.

    With ``defer_length_publish`` the stateful decode body leaves the KV
    valid-length increment to the enclosing kernel, which must perform it after
    the grid barrier that ends this phase's barrier interval. Its query rows can
    then spread across CTAs, because no CTA can observe the new length while
    another is still reading the old one.
    """
    if program.family != "attention":
        raise UnsupportedAttentionBody("program is not attention")
    facts = config.semantic.access_effects
    grouping = facts.get("head_grouping")
    if not isinstance(grouping, Mapping):
        raise UnsupportedAttentionBody("attention body lacks head grouping")
    batch = grouping.get("batch")
    query_heads = grouping.get("query_heads")
    group_size = grouping.get("group_size")
    head_dim = grouping.get("head_dim")
    if not all(isinstance(value, int) for value in (batch, query_heads, group_size, head_dim)):
        raise UnsupportedAttentionBody("attention body has invalid head grouping")

    loads = {load.id: index for index, load in enumerate(program.loads)}
    required_loads = ("attention:q", "attention:k", "attention:v")
    if not set(required_loads) <= set(loads):
        raise UnsupportedAttentionBody("attention body lacks Q/K/V bindings")
    query_load = next(load for load in program.loads if load.id == "attention:q")
    key_load = next(load for load in program.loads if load.id == "attention:k")
    query_length = _attention_sequence_length(query_load)
    kv_length = _attention_sequence_length(key_load)
    if not isinstance(query_length, int) or not isinstance(kv_length, int):
        raise UnsupportedAttentionBody("attention sequence lengths must be static")

    if program.state_accesses:
        state = program.state_accesses[0]
        required_state_loads = (
            "attention:new_k",
            "attention:new_v",
            "attention:valid_length",
        )
        if not set(required_state_loads) <= set(loads):
            raise UnsupportedAttentionBody("stateful attention lacks K/V and length bindings")
        if query_length != 1 or state.query_length != 1:
            raise UnsupportedAttentionBody("stateful attention supports one-token decode only")
        threads = config.participants.participating_threads
        row_count = batch * query_heads
        scale = facts.get("scale")
        if not isinstance(scale, (int, float)):
            raise UnsupportedAttentionBody("attention scale must be static")
        warps = min(threads // 32, config.scratch.size_bytes // (kv_length * 4))
        if warps < 1:
            raise UnsupportedAttentionBody(
                "stateful attention scratch must hold one score row per warp"
            )
        if defer_length_publish and workers is not None and head_dim % 32 == 0:
            return _split_decode_attention_body(
                program, loads, row_count, query_heads, group_size, head_dim,
                kv_length, scale, threads, workers, config.scratch.size_bytes,
            )
        if defer_length_publish:
            # One warp-sized group of rows per CTA tile.
            tile_count = (row_count + warps - 1) // warps
            # A single iteration: this warp's row of the tile, if it exists.
            row_loop = (
                f"for bh in cutlass.range(cutlass.Int32(work_iterator[tile]) * {warps} + warp, "
                f"row_count, {warps * row_count}):"
            )
            length_publish = ""
        else:
            # One CTA owns every row so the length publish can follow a CTA barrier.
            tile_count = (row_count + threads - 1) // threads
            if tile_count != 1:
                raise UnsupportedAttentionBody(
                    "stateful attention requires every query row in one CTA"
                )
            row_loop = f"for bh in cutlass.range(warp, row_count, {warps}):"
            length_publish = (
                "        cute.arch.sync_threads()\n"
                "        if thread == 0:\n"
                f"            tensor_bindings[{loads['attention:valid_length']}][0] = current_length + cutlass.Int32(1)\n"
            )
        binding_type = ", ".join("cute.Tensor" for _ in range(len(program.loads) + 1))
        VALUE_BLOCK = _VALUE_BLOCK
        value_loads = "\n                        ".join(
            f"value_{index} = tensor_bindings[{loads['attention:v']}][b, block_start + {index}, kv_head, d]"
            for index in range(VALUE_BLOCK)
        )
        value_sums = "\n                        ".join(
            f"value = value + scratch[base + block_start + {index}] * value_{index}.to(cutlass.Float32)"
            for index in range(VALUE_BLOCK)
        )
        source = f'''import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm


@cute.kernel
def run(
    work_iterator: cute.Tensor,
    tensor_bindings: tuple[{binding_type}],
    scratch: cute.Tensor,
    epilogue,
):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    workers, _, _ = cute.arch.grid_dim()
    row_count = {row_count}
    current_length = tensor_bindings[{loads['attention:valid_length']}][0].to(cutlass.Int32)
    if current_length < 0:
        llvm.intr_trap()
    if current_length + cutlass.Int32(1) > cutlass.Int32({kv_length}):
        llvm.intr_trap()
    # Each of the first {warps} warps takes whole query rows: lanes split the
    # visible keys for scores and the head dimensions for the value sum, and
    # the row's scores and probabilities stay in that warp's scratch slice.
    for tile in cutlass.range(block, cute.size(work_iterator), workers):
        warp = thread // 32
        lane = thread % 32
        if warp < {warps}:
            {row_loop}
                s = 0
                b = bh // {query_heads}
                h = bh % {query_heads}
                kv_head = h // {group_size}
                base = warp * {kv_length}
                maximum = cutlass.Float32(-1.0e30)
                for key in cutlass.range(lane, current_length + cutlass.Int32(1), 32):
                    score = cutlass.Float32(0.0)
                    if key < current_length:
                        for d in cutlass.range_constexpr({head_dim}):
                            score = score + tensor_bindings[{loads['attention:q']}][b, s, h, d].to(cutlass.Float32) * tensor_bindings[{loads['attention:k']}][b, key, kv_head, d].to(cutlass.Float32)
                    else:
                        for d in cutlass.range_constexpr({head_dim}):
                            score = score + tensor_bindings[{loads['attention:q']}][b, s, h, d].to(cutlass.Float32) * tensor_bindings[{loads['attention:new_k']}][b, 0, kv_head, d].to(cutlass.Float32)
                    score = score.to(cutlass.BFloat16).to(cutlass.Float32)
                    score = score * cutlass.Float32({scale!r})
                    scratch[base + key] = score
                    maximum = cute.arch.fmax(maximum, score, nan=True)
                maximum = cute.arch.warp_reduction_max(maximum)
                total = cutlass.Float32(0.0)
                for key in cutlass.range(lane, current_length + cutlass.Int32(1), 32):
                    probability = cute.exp(scratch[base + key] - maximum)
                    scratch[base + key] = probability
                    total = total + probability
                total = cute.arch.warp_reduction_sum(total)
                denominator = total
                if total <= cutlass.Float32(0.0):
                    denominator = cutlass.Float32(1.0)
                for key in cutlass.range(lane, current_length + cutlass.Int32(1), 32):
                    scratch[base + key] = (scratch[base + key] / denominator).to(cutlass.BFloat16).to(cutlass.Float32)
                cute.arch.sync_warp()
                for d in cutlass.range(lane, {head_dim}, 32):
                    value = cutlass.Float32(0.0)
                    # Cached keys first, then the new key: the source key order.
                    # Eight loads are issued before they are accumulated, so
                    # their latencies overlap; the sum order is unchanged.
                    blocked = current_length - current_length % {VALUE_BLOCK}
                    for block_start in cutlass.range(0, blocked, {VALUE_BLOCK}):
                        {value_loads}
                        {value_sums}
                    for key in cutlass.range(blocked, current_length):
                        value = value + scratch[base + key] * tensor_bindings[{loads['attention:v']}][b, key, kv_head, d].to(cutlass.Float32)
                    value = value + scratch[base + current_length] * tensor_bindings[{loads['attention:new_v']}][b, 0, kv_head, d].to(cutlass.Float32)
                    tensor_bindings[{len(program.loads)}][bh, s, d] = value.to(cutlass.BFloat16)
                # Keys before current_length come from the cache and the new
                # key from new_k, so appending here cannot race with readers.
                if h % {group_size} == 0:
                    for d in cutlass.range(lane, {head_dim}, 32):
                        tensor_bindings[{loads['attention:k']}][b, current_length, kv_head, d] = tensor_bindings[{loads['attention:new_k']}][b, 0, kv_head, d]
                        tensor_bindings[{loads['attention:v']}][b, current_length, kv_head, d] = tensor_bindings[{loads['attention:new_v']}][b, 0, kv_head, d]
                cute.arch.sync_warp()
{length_publish}'''
        return GeneratedAttentionBody(
            source=source,
            threads_per_cta=threads,
            tile_count=tile_count,
            work_iterator_encoding="one stateful query-row block base per CTA tile",
        )

    mask = facts.get("mask")
    if not isinstance(mask, Mapping):
        raise UnsupportedAttentionBody("attention body lacks mask semantics")
    if mask.get("kind") == "additive_tensor":
        if "attention:mask" not in loads:
            raise UnsupportedAttentionBody("additive mask is not bound")
        mask_load = next(load for load in program.loads if load.id == "attention:mask")
        if mask_load.source_shape is None or len(mask_load.source_shape) != 4:
            raise UnsupportedAttentionBody("additive mask must be four-dimensional")
        mask_head_is_broadcast = mask_load.source_shape[1] == 1
        mask_access = (
            f"tensor_bindings[{loads['attention:mask']}][b, 0, s, key]"
            if mask_head_is_broadcast
            else f"tensor_bindings[{loads['attention:mask']}][b, h, s, key]"
        )
        mask_code = f"score = (score.to(cutlass.Float32) + ({mask_access}).to(cutlass.Float32)).to(cutlass.BFloat16)"
    elif mask.get("kind") == "typed_where":
        causal_offset = mask.get("causal_offset")
        masked_value = mask.get("false_value")
        if not isinstance(masked_value, (int, float)):
            raise UnsupportedAttentionBody("attention mask lacks a fill value")
        if isinstance(causal_offset, int):
            mask_code = (
                f"score = (score.to(cutlass.Float32) + cutlass.Float32({masked_value!r})).to(cutlass.BFloat16) if key > s + {causal_offset} else score"
            )
        else:
            if "attention:mask" not in loads:
                raise UnsupportedAttentionBody("typed attention mask is not bound")
            mask_load = next(
                load for load in program.loads if load.id == "attention:mask"
            )
            mask_access = _attention_access(
                mask_load, "h", loads["attention:mask"]
            )
            mask_code = (
                f"score = (score.to(cutlass.Float32) + cutlass.Float32({masked_value!r})).to(cutlass.BFloat16) if not {mask_access} else score"
            )
    elif mask.get("kind") == "none":
        mask_code = "pass"
    else:
        raise UnsupportedAttentionBody("unsupported attention mask hook")

    threads = config.participants.participating_threads
    row_count = batch * query_heads * query_length
    scale = facts.get("scale")
    if not isinstance(scale, (int, float)):
        raise UnsupportedAttentionBody("attention scale must be static")
    output_type = "cutlass.BFloat16"
    binding_type = ", ".join("cute.Tensor" for _ in range(len(program.loads) + 1))
    q_access = _attention_access(query_load, "h", loads["attention:q"])
    k_access = _attention_access(key_load, "kv_head", loads["attention:k"])
    value_load = next(load for load in program.loads if load.id == "attention:v")
    v_access = _attention_access(value_load, "kv_head", loads["attention:v"])
    warps = min(threads // 32, config.scratch.size_bytes // (kv_length * 4))
    if warps < 1:
        raise UnsupportedAttentionBody("attention scratch must hold one score row per warp")
    tile_count = (row_count + warps - 1) // warps
    source = f'''import cutlass
import cutlass.cute as cute


@cute.kernel
def run(
    work_iterator: cute.Tensor,
    tensor_bindings: tuple[{binding_type}],
    scratch: cute.Tensor,
    epilogue,
):
    # Each CTA tile covers {warps} query rows, one per warp: lanes split the keys
    # for scores and the head dimensions for the value sum, with the row's
    # scores and probabilities in that warp's scratch slice. Rounding points
    # match the captured program; the value sum runs in key order.
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    workers, _, _ = cute.arch.grid_dim()
    warp = thread // 32
    lane = thread % 32
    row_count = {row_count}
    for tile in cutlass.range(block, cute.size(work_iterator), workers):
        row = work_iterator[tile] * {warps} + warp
        if warp < {warps}:
            if row < row_count:
                bh = row // {query_length}
                s = row % {query_length}
                b = bh // {query_heads}
                h = bh % {query_heads}
                kv_head = h // {group_size}
                base = warp * {kv_length}
                maximum = cutlass.Float32(-1.0e30)
                for key in cutlass.range(lane, {kv_length}, 32):
                    score = cutlass.Float32(0.0)
                    for d in cutlass.range_constexpr({head_dim}):
                        score = score + {q_access}.to(cutlass.Float32) * {k_access}.to(cutlass.Float32)
                    score = score.to(cutlass.BFloat16).to(cutlass.Float32)
                    score = (score * cutlass.Float32({scale!r})).to(cutlass.BFloat16)
                    {mask_code}
                    score = score.to(cutlass.Float32)
                    scratch[base + key] = score
                    maximum = cute.arch.fmax(maximum, score, nan=True)
                maximum = cute.arch.warp_reduction_max(maximum)
                total = cutlass.Float32(0.0)
                for key in cutlass.range(lane, {kv_length}, 32):
                    probability = cute.exp(scratch[base + key] - maximum)
                    scratch[base + key] = probability
                    total = total + probability
                total = cute.arch.warp_reduction_sum(total)
                denominator = total
                if total <= cutlass.Float32(0.0):
                    denominator = cutlass.Float32(1.0)
                for key in cutlass.range(lane, {kv_length}, 32):
                    scratch[base + key] = (scratch[base + key] / denominator).to(cutlass.BFloat16).to(cutlass.Float32)
                cute.arch.sync_warp()
                for d in cutlass.range(lane, {head_dim}, 32):
                    value = cutlass.Float32(0.0)
                    for key in cutlass.range({kv_length}, unroll=8):
                        value = value + scratch[base + key] * {v_access}.to(cutlass.Float32)
                    tensor_bindings[{len(program.loads)}][bh, s, d] = value.to({output_type})
                cute.arch.sync_warp()
'''
    return GeneratedAttentionBody(
        source=source,
        threads_per_cta=threads,
        tile_count=tile_count,
        work_iterator_encoding="one query-row block base per CTA tile",
    )


def decode_chunks(rows: int, workers: int) -> int:
    """Chunks per query row for split-context decode at long contexts.

    Time per key scales with ``passes / chunks``, where ``passes`` is how
    many rounds of ``rows * chunks`` items ``workers`` CTAs take; every pass
    after the first also waits on the chunk exchanges, so it costs extra.
    """
    best, best_cost = 1, float("inf")
    for chunks in range(1, min(DECODE_MAX_CHUNKS, workers) + 1):
        passes = -(-rows * chunks // workers)
        cost = passes / chunks * (1 + 0.15 * (passes - 1))
        if cost < best_cost - 1e-9:
            best, best_cost = chunks, cost
    return best


def _split_decode_attention_body(
    program: Any,
    loads: Mapping[str, int],
    row_count: int,
    query_heads: int,
    group_size: int,
    head_dim: int,
    kv_length: int,
    scale: float,
    threads: int,
    workers: int,
    scratch_bytes: int = 0,
) -> GeneratedAttentionBody:
    """One-token decode with the visible keys of each query row split into
    chunks that run on different CTAs (flash decoding), keeping the captured
    rounding points: BF16 scores, FP32 softmax statistics, BF16 normalized
    probabilities, and an FP32 value sum.

    Work item ``row * CHUNKS + chunk`` covers one chunk. All warps of the CTA
    share it: warps take keys and lanes the head dimensions. Chunks of a row
    exchange their maxima and exponential sums through phase scratch: each
    publishes, then waits until every chunk in use has. The waits cannot
    deadlock because a row's items are consecutive and there are at most as
    many chunks as CTAs, so a waiting CTA only waits on items that are some
    CTA's current or next item. The last chunk to finish sums the chunks'
    FP32 value partials in chunk order, so the result does not depend on
    which CTA finishes last, and resets the row's counters.

    A chunk's scores are read only by the CTA that computes them, so they stay
    in shared scratch after the warp partials whenever a full KV row fits
    there; otherwise they go through phase scratch. A single chunk in use
    skips the statistics exchange and its fences.
    """
    warps = threads // 32
    chunks = decode_chunks(row_count, workers)
    # Shared scratch: warp maxima or sums, then warp value partials, then scores.
    shared_scores = warps + warps * head_dim
    scores_in_shared = (shared_scores + kv_length) * 4 <= scratch_bytes
    q_index = loads["attention:q"]
    k_index = loads["attention:k"]
    v_index = loads["attention:v"]
    new_k_index = loads["attention:new_k"]
    new_v_index = loads["attention:new_v"]
    length_index = loads["attention:valid_length"]
    output_index = len(program.loads)
    # Phase scratch: scores (rows x keys), per-chunk max and sum, and
    # per-chunk value partials; three counters per row.
    stats_base = row_count * kv_length
    partial_base = stats_base + row_count * chunks * 2
    floats = partial_base + row_count * chunks * head_dim
    binding_type = ", ".join("cute.Tensor" for _ in range(len(program.loads) + 1))
    dims_per_lane = head_dim // 32

    def lines(template: str, indent: str, **values: object) -> str:
        return "\n".join(
            indent + template.format(j=j, d=f"lane + {32 * j}", **values)
            for j in range(dims_per_lane)
        )

    # Value sum: warps take keys, lanes the head dimensions. Each iteration
    # issues VALUE_UNROLL keys' loads before accumulating them in key order;
    # keys past the cached range read the last cached key with probability 0.
    unroll = _VALUE_UNROLL
    def score_load(key: str) -> str:
        if scores_in_shared:
            return f"scratch[{shared_scores} + {key}]"
        return f"cute.arch.load(floats.iterator + scores + {key}, cutlass.Float32, cop=\"cg\")"

    def score_store(key: str, value: str) -> str:
        if scores_in_shared:
            return f"scratch[{shared_scores} + {key}] = {value}"
        return f"cute.arch.store(floats.iterator + scores + {key}, {value}, cop=\"cg\")"

    value_loads = "\n".join(
        f"                    key_{u} = base + {u * warps}\n"
        f"                    probability_{u} = cutlass.Float32(0.0)\n"
        f"                    if key_{u} < cached_stop:\n"
        f"                        probability_{u} = (\n"
        f"                            {score_load(f'key_{u}')} / denominator\n"
        f"                        ).to(cutlass.BFloat16).to(cutlass.Float32)\n"
        f"                    else:\n"
        f"                        key_{u} = cached_stop - cutlass.Int32(1)\n"
        + lines(
            "row_{u}_{j} = tensor_bindings[" + str(v_index) + "][b, key_{u}, kv_head, {d}].to(cutlass.Float32)",
            "                    ",
            u=u,
        )
        for u in range(unroll)
    )
    value_sums = "\n".join(
        lines("value_{j} = value_{j} + probability_{u} * row_{u}_{j}", "                    ", u=u)
        for u in range(unroll)
    )
    value_pass = (
        lines("value_{j} = cutlass.Float32(0.0)", "                ")
        + f"\n                for base in cutlass.range(start + warp, cached_stop, {unroll * warps}):\n"
        + value_loads
        + "\n"
        + value_sums
        + "\n                if stop == keys:\n"
        "                    if warp == 0:\n"
        "                        probability = (\n"
        f"                            {score_load('current_length')} / denominator\n"
        "                        ).to(cutlass.BFloat16).to(cutlass.Float32)\n"
        + lines(
            "value_{j} = value_{j} + probability * tensor_bindings[" + str(new_v_index) + "][b, 0, kv_head, {d}].to(cutlass.Float32)",
            "                        ",
        )
        + "\n"
        + lines("scratch[" + str(warps) + " + warp * " + str(head_dim) + " + {d}] = value_{j}", "                ")
        + "\n                cute.arch.sync_threads()\n"
    )
    score_pass = f"""                query = tensor_bindings[{q_index}][(b, 0, h, None)].load().to(cutlass.Float32)
                # Scores: one key per thread, whole rows per load.
                maximum = cutlass.Float32(-1.0e30)
                for key in cutlass.range(start + thread, cached_stop, {threads}):
                    row = tensor_bindings[{k_index}][(b, key, kv_head, None)].load().to(cutlass.Float32)
                    score = cutlass.Float32((query * row).reduce(cute.ReductionOp.ADD, cutlass.Float32(0.0), 0))
                    score = score.to(cutlass.BFloat16).to(cutlass.Float32)
                    score = score * cutlass.Float32({scale!r})
                    {score_store("key", "score")}
                    maximum = cute.arch.fmax(maximum, score, nan=True)
                if stop == keys:
                    if thread == 0:
                        row = tensor_bindings[{new_k_index}][(b, 0, kv_head, None)].load().to(cutlass.Float32)
                        score = cutlass.Float32((query * row).reduce(cute.ReductionOp.ADD, cutlass.Float32(0.0), 0))
                        score = score.to(cutlass.BFloat16).to(cutlass.Float32)
                        score = score * cutlass.Float32({scale!r})
                        {score_store("current_length", "score")}
                        maximum = cute.arch.fmax(maximum, score, nan=True)
                maximum = cute.arch.warp_reduction_max(maximum)
"""

    def wait(counter: str) -> str:
        return (
            "                    cute.arch.sync_threads()\n"
            "                    if thread == 0:\n"
            "                        seen = cutlass.Int32(0)\n"
            "                        while seen < in_use:\n"
            "                            seen = cute.arch.atomic_cas(\n"
            f"                                counters.iterator + counter_base + bh * 3 + {counter},\n"
            "                                cmp=in_use,\n"
            "                                val=in_use,\n"
            '                                sem="acquire",\n'
            '                                scope="gpu",\n'
            "                            )\n"
            "                    cute.arch.sync_threads()\n"
            "                    cute.arch.fence_acq_rel_gpu()\n"
        )

    def publish(counter: str) -> str:
        return (
            "                    if thread == 0:\n"
            "                        cute.arch.atomic_add(\n"
            f"                            counters.iterator + counter_base + bh * 3 + {counter},\n"
            "                            cutlass.Int32(1),\n"
            '                            sem="release",\n'
            '                            scope="gpu",\n'
            "                        )\n"
        )

    source = f'''import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm


@cute.kernel
def run(
    work_iterator: cute.Tensor,
    tensor_bindings: tuple[{binding_type}],
    scratch: cute.Tensor,
    epilogue,
):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    workers, _, _ = cute.arch.grid_dim()
    floats, counters, float_base, counter_base = epilogue
    current_length = tensor_bindings[{length_index}][0].to(cutlass.Int32)
    if current_length < 0:
        llvm.intr_trap()
    if current_length + cutlass.Int32(1) > cutlass.Int32({kv_length}):
        llvm.intr_trap()
    keys = current_length + cutlass.Int32(1)
    in_use = (keys + cutlass.Int32({DECODE_MIN_CHUNK_KEYS - 1})) // cutlass.Int32({DECODE_MIN_CHUNK_KEYS})
    if in_use > cutlass.Int32({chunks}):
        in_use = cutlass.Int32({chunks})
    chunk_keys = (keys + in_use - cutlass.Int32(1)) // in_use
    warp = thread // 32
    lane = thread % 32
    # Only the chunks in use are items, row-major so a row's items are
    # consecutive (see the deadlock note above).
    for item in cutlass.range(block, cutlass.Int32({row_count}) * in_use, workers):
        bh = item // in_use
        chunk = item % in_use
        if True:
            b = bh // {query_heads}
            h = bh % {query_heads}
            kv_head = h // {group_size}
            start = chunk * chunk_keys
            stop = start + chunk_keys
            if stop > keys:
                stop = keys
            scores = float_base + bh * {kv_length}
            stats = float_base + {stats_base} + bh * {chunks * 2}
            # Cached keys come from the cache; the new key at current_length
            # (in the last chunk) from new_k and new_v.
            cached_stop = stop
            if cached_stop > current_length:
                cached_stop = current_length
            # Every chunk in use runs the whole exchange, even with no keys.
            if True:
{score_pass}                if lane == 0:
                    scratch[warp] = maximum
                cute.arch.sync_threads()
                if thread == 0:
                    chunk_maximum = scratch[0]
                    for other in cutlass.range_constexpr(1, {warps}):
                        chunk_maximum = cute.arch.fmax(chunk_maximum, scratch[other], nan=True)
                    if in_use > cutlass.Int32(1):
                        cute.arch.store(floats.iterator + stats + chunk * 2, chunk_maximum, cop="cg")
                    scratch[0] = chunk_maximum
                if in_use > cutlass.Int32(1):
                    cute.arch.fence_acq_rel_gpu()
{publish("0")}{wait("0")}                    if thread == 0:
                        row_maximum = cute.arch.load(floats.iterator + stats, cutlass.Float32, cop="cg")
                        for other in cutlass.range(1, in_use):
                            row_maximum = cute.arch.fmax(
                                row_maximum,
                                cute.arch.load(floats.iterator + stats + other * 2, cutlass.Float32, cop="cg"),
                                nan=True,
                            )
                        scratch[0] = row_maximum
                cute.arch.sync_threads()
                row_maximum = scratch[0]
                # Exponentials of this chunk's scores and their sum.
                total = cutlass.Float32(0.0)
                for key in cutlass.range(start + thread, stop, {threads}):
                    probability = cute.exp({score_load("key")} - row_maximum)
                    {score_store("key", "probability")}
                    total = total + probability
                total = cute.arch.warp_reduction_sum(total)
                cute.arch.sync_threads()
                if lane == 0:
                    scratch[warp] = total
                cute.arch.sync_threads()
                if thread == 0:
                    chunk_total = scratch[0]
                    for other in cutlass.range_constexpr(1, {warps}):
                        chunk_total = chunk_total + scratch[other]
                    if in_use > cutlass.Int32(1):
                        cute.arch.store(floats.iterator + stats + chunk * 2 + 1, chunk_total, cop="cg")
                    scratch[0] = chunk_total
                if in_use > cutlass.Int32(1):
                    cute.arch.fence_acq_rel_gpu()
{publish("1")}{wait("1")}                    if thread == 0:
                        row_total = cutlass.Float32(0.0)
                        for other in cutlass.range(0, in_use):
                            row_total = row_total + cute.arch.load(
                                floats.iterator + stats + other * 2 + 1, cutlass.Float32, cop="cg"
                            )
                        scratch[0] = row_total
                cute.arch.sync_threads()
                denominator = scratch[0]
                if denominator <= cutlass.Float32(0.0):
                    denominator = cutlass.Float32(1.0)
                cute.arch.sync_threads()
{value_pass}                if in_use == cutlass.Int32(1):
                    for d in cutlass.range(thread, {head_dim}, {threads}):
                        value = scratch[{warps} + d]
                        for other in cutlass.range_constexpr(1, {warps}):
                            value = value + scratch[{warps} + other * {head_dim} + d]
                        tensor_bindings[{output_index}][bh, 0, d] = value.to(cutlass.BFloat16)
                else:
                    partials = float_base + {partial_base} + bh * {chunks * head_dim}
                    for d in cutlass.range(thread, {head_dim}, {threads}):
                        value = scratch[{warps} + d]
                        for other in cutlass.range_constexpr(1, {warps}):
                            value = value + scratch[{warps} + other * {head_dim} + d]
                        cute.arch.store(
                            floats.iterator + partials + chunk * {head_dim} + d, value, cop="cg"
                        )
                    cute.arch.fence_acq_rel_gpu()
                    cute.arch.sync_threads()
                    if thread == 0:
                        arrived = cute.arch.atomic_add(
                            counters.iterator + counter_base + bh * 3 + 2,
                            cutlass.Int32(1),
                            sem="acq_rel",
                            scope="gpu",
                        )
                        scratch[0] = cutlass.Float32(0.0)
                        if arrived == in_use - cutlass.Int32(1):
                            scratch[0] = cutlass.Float32(1.0)
                    cute.arch.sync_threads()
                    if scratch[0] != cutlass.Float32(0.0):
                        cute.arch.fence_acq_rel_gpu()
                        for d in cutlass.range(thread, {head_dim}, {threads}):
                            value = cutlass.Float32(0.0)
                            for other in cutlass.range(0, in_use):
                                value = value + cute.arch.load(
                                    floats.iterator + partials + other * {head_dim} + d,
                                    cutlass.Float32,
                                    cop="cg",
                                )
                            tensor_bindings[{output_index}][bh, 0, d] = value.to(cutlass.BFloat16)
                        if thread == 0:
                            # Every chunk has passed both waits: reset for the next phase.
                            for counter in cutlass.range_constexpr(3):
                                cute.arch.atomic_exch(
                                    counters.iterator + counter_base + bh * 3 + counter,
                                    cutlass.Int32(0),
                                    sem="relaxed",
                                    scope="gpu",
                                )
                cute.arch.sync_threads()
            # Keys before current_length come from the cache and the new key
            # from new_k, so appending here cannot race with readers.
            if chunk == 0:
                if h % {group_size} == 0:
                    if warp == 0:
                        for d in cutlass.range(lane, {head_dim}, 32):
                            tensor_bindings[{k_index}][b, current_length, kv_head, d] = tensor_bindings[{new_k_index}][b, 0, kv_head, d]
                            tensor_bindings[{v_index}][b, current_length, kv_head, d] = tensor_bindings[{new_v_index}][b, 0, kv_head, d]
            cute.arch.sync_threads()
'''
    return GeneratedAttentionBody(
        source=source,
        threads_per_cta=threads,
        tile_count=row_count * chunks,
        work_iterator_encoding="one (query row, key chunk) item per CTA tile",
        phase_floats=floats,
        phase_counters=row_count * 3,
    )
