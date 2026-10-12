from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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


class GenericBodyProvider:
    provider_name = "generic"

    def get_candidates(
        self, region: Region, target: TargetFacts, policy: SelectionPolicy
    ) -> CandidateSet:
        if not policy.allow_generic:
            return CandidateSet(
                (), (ProviderRejection(self.provider_name, "generic providers disabled"),)
            )
        if not region.tensor_programs:
            return CandidateSet(
                (),
                (ProviderRejection(self.provider_name, "region has no tensor program"),),
            )
        if len(region.tensor_programs) != 1:
            return CandidateSet(
                (),
                (
                    ProviderRejection(
                        self.provider_name, "multi-program regions are not supported yet"
                    ),
                ),
            )

        program = region.tensor_programs[0]
        family = program.family
        if family == "view":
            return CandidateSet(
                (),
                (
                    ProviderRejection(
                        self.provider_name, "aliasing view needs no device body"
                    ),
                ),
            )
        if family not in {"elementwise", "reduction", "contraction"}:
            return CandidateSet(
                (),
                (
                    ProviderRejection(
                        self.provider_name, f"unsupported program family {family}"
                    ),
                ),
            )

        return CandidateSet((_generic_config(region, program),), ())


class G0RangeBodyProvider:
    provider_name = "g0_range"

    def get_candidates(
        self, region: Region, target: TargetFacts, policy: SelectionPolicy
    ) -> CandidateSet:
        if not policy.allow_tuned:
            return CandidateSet(
                (), (ProviderRejection(self.provider_name, "tuned providers disabled"),)
            )
        if not region.tensor_programs or len(region.tensor_programs) != 1:
            return CandidateSet(
                (),
                (
                    ProviderRejection(
                        self.provider_name, "region has no single tensor program"
                    ),
                ),
            )
        program = region.tensor_programs[0]
        if program.family != "contraction":
            return CandidateSet(
                (),
                (
                    ProviderRejection(
                        self.provider_name, "provider supports contraction only"
                    ),
                ),
            )
        if len(program.loads) != 2:
            return CandidateSet(
                (),
                (
                    ProviderRejection(
                        self.provider_name, "bias epilogues are not supported"
                    ),
                ),
            )
        if (
            any(load.dtype != "torch.bfloat16" for load in program.loads)
            or program.store.dtype != "torch.bfloat16"
        ):
            return CandidateSet(
                (),
                (ProviderRejection(self.provider_name, "provider supports BF16 only"),),
            )

        rows, columns = program.store.shape
        reduction_size = next(
            domain.size
            for domain in program.iteration_domains
            if domain.axis in program.reduction_axes
        )
        if columns % 64 or reduction_size % 64:
            return CandidateSet(
                (),
                (
                    ProviderRejection(
                        self.provider_name, "N and K must be multiples of 64"
                    ),
                ),
            )
        if target.architecture != "sm90a":
            return CandidateSet(
                (),
                (ProviderRejection(self.provider_name, "provider targets sm90a only"),),
            )
        if not target.supports_tma or not target.supports_wgmma:
            return CandidateSet(
                (),
                (
                    ProviderRejection(
                        self.provider_name, "target lacks TMA or WGMMA"
                    ),
                ),
            )

        small_m = rows <= 64
        tile = (64, 64) if small_m else (64, 128)
        stages = 13 if small_m else 8
        shared_bytes = 230_400 if small_m else 214_016
        return CandidateSet(
            (_g0_config(region, program, tile, stages, shared_bytes),), ()
        )


PROVIDERS: tuple[GenericBodyProvider | G0RangeBodyProvider, ...] = (
    GenericBodyProvider(),
    G0RangeBodyProvider(),
)


def get_candidates(
    region: Region, target: TargetFacts, policy: SelectionPolicy
) -> CandidateSet:
    configurations: list[BodyConfig] = []
    rejections: list[ProviderRejection] = []
    for provider in PROVIDERS:
        result = provider.get_candidates(region, target, policy)
        for config in result.configurations:
            try:
                validate_body_config(config, target, policy)
            except InvalidBodyConfig as error:
                rejections.append(ProviderRejection(config.provider, str(error)))
            else:
                configurations.append(config)
        rejections.extend(result.rejections)
    return CandidateSet(tuple(configurations), tuple(rejections))


def _generic_config(region: Region, program: Any) -> BodyConfig:
    output_axes = tuple(
        domain.axis
        for domain in program.iteration_domains
        if domain.axis not in program.reduction_axes
    )
    is_reduction = program.family == "reduction"
    is_contraction = program.family == "contraction"
    threads = 128
    scratch_bytes = 512 if is_reduction else 0
    barriers = ("cta_reduction",) if is_reduction else ()
    cta_shape = (128,) + (1,) * max(len(output_axes) - 1, 0)
    if is_contraction:
        cta_shape = (32, 32)
    completion = CompletionContract(
        outputs_ready="assigned output elements are stored when run returns",
        output_tiles_ready="each assigned element is ready after its predicated store",
        async_work_complete="run performs no asynchronous work",
        scratch_reusable="scratch is reusable after the CTA reduction barrier"
        if is_reduction
        else "scratch is reusable when run returns",
        register_fragments="none",
        warp_roles_on_return="unchanged",
    )
    return BodyConfig(
        provider=GenericBodyProvider.provider_name,
        name=f"generic_{program.family}",
        semantic=SemanticContract(
            operation_family=program.family,
            source_op_ids=region.source_op_ids,
            numerical_policy="source_semantics",
            expression_hooks=(),
            expression_hook_restrictions=(
                "none; expressions are generated from the tensor program",
            ),
            access_effects=_access_effects(program),
            access_restrictions=("loads and stores must match the tensor program",),
        ),
        layouts=_layouts(program),
        tiles=TileContract(
            cta_shape=cta_shape,
            cluster_shape=(1,) * len(cta_shape),
            work_partition="one output element per thread"
            if not is_contraction
            else "disjoint output tiles per CTA",
            mma_family=None,
            reduction_tile=(32,) if program.reduction_axes else (),
        ),
        participants=ParticipantContract(
            participating_threads=threads,
            waiting_threads=threads,
            warp_roles=("worker",) * (threads // 32),
            registers_per_thread=32,
        ),
        pipeline=PipelineContract(
            stage_count=1,
            barriers=barriers,
            async_operations=(),
            shared_memory_layouts=(
                {"reduction_partials": "one scalar per participating thread"}
                if is_reduction
                else {}
            ),
            drain_rule="no asynchronous pipeline to drain",
        ),
        scratch=ScratchContract(
            owner="cta" if is_reduction else "none",
            size_bytes=scratch_bytes,
            alignment_bytes=16,
            initial_state="uninitialized reduction partials"
            if is_reduction
            else "empty",
            reuse_rule=completion.scratch_reusable,
        ),
        completion=completion,
        evidence=ResourceEvidence(
            measured=False,
            source="generated baseline contract",
            revision="task-05",
            resources={"threads": threads, "shared_bytes": scratch_bytes},
            measurements={},
        ),
    )


def _g0_config(
    region: Region,
    program: Any,
    tile: tuple[int, int],
    stages: int,
    shared_bytes: int,
) -> BodyConfig:
    return BodyConfig(
        provider=G0RangeBodyProvider.provider_name,
        name=f"g0_range_{tile[0]}x{tile[1]}",
        semantic=SemanticContract(
            operation_family="view_plus_gemm"
            if region.semantic_facts.get("family") == "view_plus_gemm"
            else "contraction",
            source_op_ids=region.source_op_ids,
            numerical_policy="source_semantics",
            expression_hooks=(),
            expression_hook_restrictions=(
                "none; the provider implements the complete source contraction",
            ),
            access_effects=_access_effects(program),
            access_restrictions=(
                "BF16 2-D operands",
                "N and K are multiples of 64; partial M tiles are predicated",
            ),
        ),
        layouts=_layouts(program, alignment_bytes=16),
        tiles=TileContract(
            cta_shape=tile,
            cluster_shape=(1, 1),
            work_partition="persistent range of output tiles",
            mma_family="wgmma",
            reduction_tile=(64,),
        ),
        participants=ParticipantContract(
            participating_threads=256,
            waiting_threads=256,
            warp_roles=(
                "tma_load",
                "tma_load",
                "tma_load",
                "tma_load",
                "wgmma_epilogue",
                "wgmma_epilogue",
                "wgmma_epilogue",
                "wgmma_epilogue",
            ),
            registers_per_thread=232,
        ),
        pipeline=PipelineContract(
            stage_count=stages,
            barriers=("producer_tail", "wgmma_wait_group_0", "tma_store_producer_tail"),
            async_operations=("tma_load", "wgmma", "tma_store"),
            shared_memory_layouts={
                "A": "staged TMA input tile",
                "B": "staged TMA input tile",
                "C": "staged epilogue tile",
            },
            drain_rule=(
                "run returns only after input producer tail, WGMMA wait, "
                "and TMA store producer tail"
            ),
        ),
        scratch=ScratchContract(
            owner="cta",
            size_bytes=shared_bytes,
            alignment_bytes=128,
            initial_state="CTA-local staged buffers and pipeline barriers",
            reuse_rule="scratch is reusable only after all pipeline tails complete",
        ),
        completion=CompletionContract(
            outputs_ready=(
                "all assigned range outputs are globally stored when run returns; "
                "another CTA may read them after the enclosing kernel grid-synchronizes"
            ),
            output_tiles_ready=(
                "tile stores are submitted in the persistent range; cross-CTA "
                "visibility requires the enclosing grid synchronization"
            ),
            async_work_complete="all TMA and WGMMA work is drained before return",
            scratch_reusable="scratch is reusable when run returns",
            register_fragments=(
                "WGMMA accumulators feed the body epilogue without conversion"
            ),
            warp_roles_on_return="drained and unchanged for the next body",
        ),
        evidence=ResourceEvidence(
            measured=True,
            source="experiments/g0_2026_10_09/resident_range_probe.py",
            revision="0b55a2f691d69981583568fd9eb69687b1f0de8a",
            resources={
                "threads": 256,
                "registers_per_thread": 232,
                "shared_storage_bytes": shared_bytes,
                "active_blocks_per_sm": 1,
            },
            measurements={
                "selected_tile": tile,
                "pipeline_stages": stages,
                "max_admitted_workers": 60,
                "resident_range_gpu_us_by_m": {4: 4.057, 256: 4.339}
                if tile == (64, 64)
                else {256: 5.602, 1024: 9.458},
            },
        ),
    )


def _layouts(program: Any, alignment_bytes: int | None = None) -> LayoutContract:
    domain_sizes = {
        domain.axis: domain.size for domain in program.iteration_domains
    }
    inputs = {
        load.value_id: TensorLayout(
            dtype=load.dtype,
            axes=tuple(load.axes),
            strides=tuple(load.strides),
            alignment_bytes=alignment_bytes or _DTYPE_BYTES[load.dtype],
            shape=tuple(
                1 if isinstance(axis, int) else domain_sizes[axis] for axis in load.axes
            ),
        )
        for load in program.loads
    }
    store = program.store
    output_axes = tuple(
        domain.axis
        for domain in program.iteration_domains
        if domain.axis not in program.reduction_axes
    )
    outputs = {
        store.value_id: TensorLayout(
            dtype=store.dtype,
            axes=output_axes,
            strides=tuple(store.strides),
            alignment_bytes=alignment_bytes or _DTYPE_BYTES[store.dtype],
            shape=tuple(store.shape),
        )
    }
    return LayoutContract(inputs, outputs)


def _access_effects(program: Any) -> Mapping[str, Any]:
    return {
        "reads": tuple(load.value_id for load in program.loads),
        "writes": (program.store.value_id,),
        "owns_storage": program.store.owns_storage,
    }
