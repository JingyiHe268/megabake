"""Composable Blackwell GEMM using synchronous warp tensor-core MMA.

SM12x does not implement Hopper WGMMA. This provider admits SM10x and SM12x
without requiring the tcgen05/TMEM programming model. Every
warp keeps its fragments in registers; no asynchronous pipeline survives run().
"""
from __future__ import annotations

from dataclasses import replace

from ..graph import Region
from ..lowering import Accumulator, Binary, Cast, Constant, Load, TensorProgram
from .generic import GeneratedContractionBody
from .interface import (
    BodyConfig, CandidateSet, ParticipantContract, ProviderRejection, ResourceEvidence,
    SelectionPolicy, TargetFacts, TileContract, validate_body_config,
)

BLACKWELL_ARCHITECTURES = frozenset({"sm100", "sm103", "sm120", "sm121"})


class BlackwellGemmProvider:
    provider_name = "blackwell_mma"

    def get_candidates(
        self, region: Region, target: TargetFacts, policy: SelectionPolicy
    ) -> CandidateSet:
        def reject(reason):
            return CandidateSet((), (ProviderRejection(self.provider_name, reason),))

        if target.architecture not in BLACKWELL_ARCHITECTURES:
            return reject("provider requires a Blackwell target")
        if not policy.allow_tuned:
            return reject("tuned providers disabled")
        if len(region.tensor_programs) != 1:
            return reject("provider requires a single tensor program")
        program = region.tensor_programs[0]
        if (program.family != "contraction" or len(program.loads) != 2
                or len(program.store.shape) != 2 or program.reduction_axes != ("r0",)):
            return reject("provider supports unbiased 2-D GEMM only")
        if any(domain.size < 1 for domain in program.iteration_domains):
            return reject("provider requires nonempty GEMM dimensions")
        dtype = program.store.dtype
        if dtype not in {"torch.float16", "torch.bfloat16"} or any(
            load.dtype != dtype for load in program.loads
        ):
            return reject("provider requires matching FP16 or BF16 operands and output")
        left, right = program.loads
        if left.axes != ("o0", "r0") or right.axes not in {("r0", "o1"), ("o1", "r0")}:
            return reject("provider requires plain GEMM indexing")
        if any(stride < 0 for load in program.loads for stride in load.strides):
            return reject("negative operand strides are unsupported")
        # Reject scaled contractions and fused epilogues rather than silently
        # substituting plain GEMM for a different tensor program.
        acc = Accumulator("torch.float32")
        product = Binary("mul", Cast(Load(left.id, dtype), "torch.float32", "round_nearest_even"),
                         Cast(Load(right.id, dtype), "torch.float32", "round_nearest_even"), "torch.float32")
        if (program.accumulator_dtype != "torch.float32"
                or program.init != Constant(0.0, "torch.float32")
                or program.combine != Binary("add", acc, product, "torch.float32")
                or program.post != Cast(acc, dtype, "round_nearest_even")):
            return reject("provider requires an unscaled FP32 accumulation and result cast")
        from .catalog import _generic_config
        base = _generic_config(region, program)
        return CandidateSet((replace(
            base,
            provider=self.provider_name,
            name="blackwell_mma_16x32x16",
            tiles=TileContract(
                cta_shape=(16, 32), cluster_shape=(1, 1),
                work_partition="persistent range of output tiles",
                mma_family="mma.sync", reduction_tile=(16,),
            ),
            participants=ParticipantContract(
                participating_threads=128, waiting_threads=128,
                warp_roles=("mma",) * 4, registers_per_thread=64,
            ),
            completion=replace(
                base.completion,
                register_fragments="warp MMA FP32 accumulators are cast and stored before return",
                output_tiles_ready="all assigned 16x32 tiles are stored before return",
            ),
            evidence=ResourceEvidence(
                measured=False, source="megabake Blackwell warp MMA baseline",
                revision="blackwell-mma-v1", resources={"threads": 128, "shared_bytes": 0},
                measurements={},
            ),
        ),), ())


def emit_blackwell_gemm(
    program: TensorProgram, config: BodyConfig
) -> GeneratedContractionBody:
    """Emit a synchronous 16x32 CTA tile with predicated M/N/K tails."""
    validate_body_config(config)
    if config.provider != "blackwell_mma" or config.tiles.cta_shape != (16, 32):
        raise ValueError("Blackwell GEMM requires the 16x32 warp MMA body contract")
    rows, columns = program.store.shape
    reduction = next(d.size for d in program.iteration_domains if d.axis == "r0")
    dtype = "cutlass.BFloat16" if program.store.dtype == "torch.bfloat16" else "cutlass.Float16"
    right_index = "(k, n)" if program.loads[1].axes == ("r0", "o1") else "(n, k)"
    ntiles = (columns + 31) // 32
    source = f'''import cutlass
import cutlass.cute as cute

@cute.kernel
def run(
    work_iterator: cute.Tensor,
    tensor_bindings: tuple[cute.Tensor, cute.Tensor, cute.Tensor],
    scratch,
    epilogue,
):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    workers, _, _ = cute.arch.grid_dim()
    mma = cute.make_tiled_mma(cute.nvgpu.warp.MmaF16BF16Op({dtype}, cutlass.Float32, (16, 8, 16)), (1, 4, 1))
    thr_mma = mma.get_slice(thread)
    ca = thr_mma.partition_A(cute.make_identity_tensor((16, 16)))
    cb = thr_mma.partition_B(cute.make_identity_tensor((32, 16)))
    cc = thr_mma.partition_C(cute.make_identity_tensor((16, 32)))
    fa = cute.make_rmem_tensor(ca.shape, {dtype})
    fb = cute.make_rmem_tensor(cb.shape, {dtype})
    fc = cute.make_rmem_tensor(cc.shape, cutlass.Float32)
    for tile in cutlass.range(block, cute.size(work_iterator), workers):
        tile_id = cutlass.Int32(work_iterator[tile])
        row_base = (tile_id // {ntiles}) * 16
        col_base = (tile_id % {ntiles}) * 32
        fc.fill(0.0)
        for kbase in cutlass.range(0, {reduction}, 16):
            for i in cutlass.range_constexpr(cute.size(fa)):
                m = row_base + ca[i][0]
                k = kbase + ca[i][1]
                fa[i] = {dtype}(0.0)
                if m < {rows} and k < {reduction}:
                    fa[i] = tensor_bindings[0][m, k]
            for i in cutlass.range_constexpr(cute.size(fb)):
                n = col_base + cb[i][0]
                k = kbase + cb[i][1]
                fb[i] = {dtype}(0.0)
                if n < {columns} and k < {reduction}:
                    fb[i] = tensor_bindings[1][{right_index}]
            cute.gemm(mma, fc, fa, fb, fc)
        for i in cutlass.range_constexpr(cute.size(fc)):
            m = row_base + cc[i][0]
            n = col_base + cc[i][1]
            if m < {rows} and n < {columns}:
                tensor_bindings[2][m, n] = {dtype}(fc[i])
'''
    return GeneratedContractionBody(
        source=source, threads_per_cta=128, output_tile_elements=512,
        reduction_chunk=16, tile_count=((rows + 15) // 16) * ntiles,
        work_iterator_encoding="one row-major 16x32 output tile ordinal per CTA",
        phases=("output_tile", "warp_mma", "epilogue"),
        phase_descriptions={
            "output_tile": "four warps partition a predicated 16x32 output tile",
            "warp_mma": "synchronous 16x8x16 warp MMA accumulates into FP32 registers",
            "epilogue": "cast and store the owned output fragment with M/N predicates",
        },
    )
