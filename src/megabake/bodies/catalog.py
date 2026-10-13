from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..graph import Region
from .gemm import G0RangeBodyProvider
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
        if family not in {"elementwise", "copy", "reduction", "contraction"}:
            return CandidateSet(
                (),
                (
                    ProviderRejection(
                        self.provider_name, f"unsupported program family {family}"
                    ),
                ),
            )

        return CandidateSet((_generic_config(region, program),), ())


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
    is_fused_chain = region.semantic_facts.get("fusion") == "vector_chain"
    threads = 128
    warp_count = threads // 32
    scratch_bytes = (
        warp_count * _DTYPE_BYTES[program.accumulator_dtype]
        if is_reduction
        else 0
    )
    barriers = ("cta_reduction",) if is_reduction else ()
    cta_shape = (128,) + (1,) * max(len(output_axes) - 1, 0)
    completion = CompletionContract(
        outputs_ready="assigned output elements are stored when run returns",
        output_tiles_ready=(
            "each assigned element is ready after its predicated store"
            if not is_contraction
            else "each assigned element is ready after its reduction and store"
        ),
        async_work_complete="run performs no asynchronous work",
        scratch_reusable="scratch is reusable after the CTA reduction barrier"
        if is_reduction
        else "scratch is reusable when run returns",
        register_fragments="none",
        warp_roles_on_return="unchanged",
    )
    evidence = (
        ResourceEvidence(
            measured=False,
            source="generated CuTe reduction; composition informed by Mirage primitive-body and Hopper norm tasks",
            revision="mirage@f9eb70c254acefc9f3667b2a973d0dcf25471fce; task-07",
            resources={"threads": threads, "shared_bytes": scratch_bytes},
            measurements={},
        )
        if is_reduction
        else ResourceEvidence(
            measured=False,
            source=(
                "generated CuTe indexed contraction; accumulation and epilogue "
                "structure informed by Mirage forloop-accum and threadblock epilogues"
            ),
            revision="mirage@f9eb70c254acefc9f3667b2a973d0dcf25471fce; task-08",
            resources={
                "threads": threads,
                "shared_bytes": scratch_bytes,
                "reduction_chunk": 32,
            },
            measurements={},
        )
        if is_contraction
        else ResourceEvidence(
            measured=False,
            source="generated baseline contract",
            revision="task-05",
            resources={"threads": threads, "shared_bytes": scratch_bytes},
            measurements=(
                {"fusion_saved_bytes": region.semantic_facts["fusion_saved_bytes"]}
                if is_fused_chain
                else {}
            ),
        )
    )
    return BodyConfig(
        provider=GenericBodyProvider.provider_name,
        name=f"generic_{program.family}_fused" if is_fused_chain else f"generic_{program.family}",
        semantic=SemanticContract(
            operation_family=program.family,
            source_op_ids=region.source_op_ids,
            numerical_policy="source_semantics",
            expression_hooks=("vector_chain",) if is_fused_chain else (),
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
            work_partition=(
                "one output row per CTA"
                if is_reduction
                else "one output element per thread"
            ),
            mma_family=None,
            reduction_tile=(threads,) if is_reduction else (32,)
            if is_contraction
            else (),
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
                {
                    "reduction_partials": (
                        "one accumulator scalar per warp in the CTA scratch row"
                    )
                }
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
        evidence=evidence,
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
