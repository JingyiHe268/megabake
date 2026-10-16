"""Extract a range-preserving CuTe Hopper GEMM body and measure it."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "design_2026_10_08"))
from probe import REV, HASHES, samples, source


UPSTREAM_FILE = "dense_gemm_persistent.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _adapt(source_path):
    text = source_path.read_text()
    marker = "    @cute.kernel\n    def kernel("
    assert text.count(marker) == 1
    text = text.replace(marker, "    @cute.jit\n    def kernel(", 1)
    signature = "        tile_sched_params: utils.PersistentTileSchedulerParams,\n    ):"
    assert text.count(signature) == 1
    text = text.replace(
        signature,
        "        tile_sched_params: utils.PersistentTileSchedulerParams,\n        storage,\n        schedule_grid_dim,\n    ):",
        1,
    )
    assert text.count("cute.arch.grid_dim()") == 2
    text = text.replace("cute.arch.grid_dim()", "schedule_grid_dim")
    allocation = (
        "        smem = cutlass.utils.SmemAllocator()\n"
        "        storage = smem.allocate(self.shared_storage)\n"
    )
    assert text.count(allocation) == 1
    text = text.replace(allocation, "", 1)
    insert_at = text.index("    @cute.jit\n    def kernel(")
    run = '''    @cute.jit
    def run_range(self, work_iterator, tensor_bindings, scratch, epilogue, schedule_grid_dim):
        (
            tma_atom_a,
            mA_mkl,
            tma_atom_b,
            mB_nkl,
            tiled_mma,
            cta_layout_mnk,
            a_smem_layout_staged,
            b_smem_layout_staged,
        ) = tensor_bindings
        tma_atom_c, mC_mnl, epi_smem_layout_staged = epilogue
        self.kernel(
            tma_atom_a,
            mA_mkl,
            tma_atom_b,
            mB_nkl,
            tma_atom_c,
            mC_mnl,
            tiled_mma,
            cta_layout_mnk,
            a_smem_layout_staged,
            b_smem_layout_staged,
            epi_smem_layout_staged,
            work_iterator,
            scratch,
            schedule_grid_dim,
        )

'''
    text = text[:insert_at] + run + text[insert_at:]
    target = source_path.with_name("dense_gemm_persistent_range.py")
    target.write_text(text)
    return target


def _device_resources(compiled, threads, shared_bytes):
    rows = []
    for symbol in compiled.kernel_info:
        status, kernel = cuda.cuLibraryGetKernel(
            cuda.CUlibrary(int(compiled.library)), symbol.encode()
        )
        assert int(status) == 0, status
        values = {"name": symbol}
        for label, attribute in (
            ("registers_per_thread", cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS),
            ("static_shared_bytes", cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES),
            ("local_bytes_per_thread", cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),
        ):
            status, value = cuda.cuKernelGetAttribute(attribute, kernel, 0)
            assert int(status) == 0, status
            values[label] = value
        status, function = cuda.cuKernelGetFunction(kernel)
        assert int(status) == 0, status
        status, = cuda.cuFuncSetAttribute(
            function,
            cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            shared_bytes,
        )
        assert int(status) == 0, status
        status, active = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            function, threads, shared_bytes
        )
        assert int(status) == 0, status
        values["shared_storage_bytes"] = shared_bytes
        values["active_blocks_per_sm"] = active
        rows.append(values)
    assert len(rows) == 1, rows
    return rows[0]


def _range_class(base):
    class ResidentRange(base):
        @cute.kernel
        def range_kernel(
            self,
            tma_atom_a,
            mA_mkl,
            tma_atom_b,
            mB_nkl,
            tma_atom_c,
            mC_mnl,
            tile_sched_params,
            tiled_mma: cute.TiledMma,
            cta_layout_mnk: cute.Layout,
            a_smem_layout_staged: cute.ComposedLayout,
            b_smem_layout_staged: cute.ComposedLayout,
            epi_smem_layout_staged: cute.ComposedLayout,
        ):
            storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage)
            self.run_range(
                tile_sched_params,
                (
                    tma_atom_a,
                    mA_mkl,
                    tma_atom_b,
                    mB_nkl,
                    tiled_mma,
                    cta_layout_mnk,
                    a_smem_layout_staged,
                    b_smem_layout_staged,
                ),
                storage,
                (tma_atom_c, mC_mnl, epi_smem_layout_staged),
                cute.arch.grid_dim(),
            )

        @cute.jit
        def __call__(
            self,
            a: cute.Tensor,
            b: cute.Tensor,
            c: cute.Tensor,
            max_active_clusters: cutlass.Constexpr,
            workers: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            self.a_dtype = a.element_type
            self.b_dtype = b.element_type
            self.c_dtype = c.element_type
            self.a_layout = cutlass.utils.LayoutEnum.from_tensor(a)
            self.b_layout = cutlass.utils.LayoutEnum.from_tensor(b)
            self.c_layout = cutlass.utils.LayoutEnum.from_tensor(c)
            self._setup_attributes()

            tma_atom_a, tensor_a = self._make_tma_atoms_and_tensors(
                a,
                self.a_smem_layout_staged,
                (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
                self.cluster_shape_mn[1],
            )
            tma_atom_b, tensor_b = self._make_tma_atoms_and_tensors(
                b,
                self.b_smem_layout_staged,
                (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
                self.cluster_shape_mn[0],
            )
            tma_atom_c, tensor_c = self._make_tma_store_atoms_and_tensors(
                c,
                self.epi_smem_layout_staged,
                self.epi_tile,
            )
            work_iterator, _ = self._compute_grid(
                tensor_c,
                self.tile_shape_mnk,
                self.cluster_shape_mn,
                self.swizzle_size,
                self.raster_along_m,
                max_active_clusters,
            )

            @cute.struct
            class SharedStorage:
                mainloop_pipeline_array_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.ab_stage * 2
                ]
                sA: cute.struct.Align[
                    cute.struct.MemRange[
                        self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                    ],
                    self.buffer_align_bytes,
                ]
                sB: cute.struct.Align[
                    cute.struct.MemRange[
                        self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                    ],
                    self.buffer_align_bytes,
                ]
                sC: cute.struct.Align[
                    cute.struct.MemRange[
                        self.c_dtype, cute.cosize(self.epi_smem_layout_staged)
                    ],
                    self.buffer_align_bytes,
                ]

            self.shared_storage = SharedStorage
            self.range_kernel(
                tma_atom_a,
                tensor_a,
                tma_atom_b,
                tensor_b,
                tma_atom_c,
                tensor_c,
                work_iterator,
                self.tiled_mma,
                self.cta_layout_mnk,
                self.a_smem_layout_staged,
                self.b_smem_layout_staged,
                self.epi_smem_layout_staged,
            ).launch(
                # CuTe's persistent scheduler takes the grid index from z.
                grid=[1, 1, workers],
                block=[self.threads_per_cta, 1, 1],
                cluster=(*self.cluster_shape_mn, 1),
                min_blocks_per_mp=1,
                cooperative=True,
                stream=stream,
            )

    return ResidentRange


def _matrix(tensor):
    return from_dlpack(tensor.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )


def _run_case(name, tile, measured_m, followup_m, modules, hardware, stream):
    original_module, range_module, dense_module = modules
    tile_count = ((measured_m + tile[0] - 1) // tile[0]) * (
        (1536 + tile[1] - 1) // tile[1]
    )
    range_workers = max(1, tile_count // 2)
    b = torch.randn((1536, 576), device="cuda", dtype=torch.bfloat16)
    a = torch.randn((measured_m, 576), device="cuda", dtype=torch.bfloat16)
    c = torch.empty((measured_m, 1536), device="cuda", dtype=torch.bfloat16)
    ca, cb, cc = map(_matrix, (a, b, c))
    max_clusters = hardware.get_max_active_clusters(1)

    standalone = original_module.HopperWgmmaGemmPersistentKernel(
        cutlass.Float32, tile, (1, 1), 1, True
    )
    standalone_compiled = cute.compile(standalone, ca, cb, cc, max_clusters, stream)

    range_type = _range_class(range_module.HopperWgmmaGemmPersistentKernel)
    ranged = range_type(cutlass.Float32, tile, (1, 1), 1, True)
    range_compiled = cute.compile(
        ranged,
        ca,
        cb,
        cc,
        max_clusters,
        cutlass.Int32(range_workers),
        stream,
    )
    resources = _device_resources(
        range_compiled, ranged.threads_per_cta, ranged.shared_storage.__sizeof__()
    )
    max_workers = torch.cuda.get_device_properties(0).multi_processor_count * resources[
        "active_blocks_per_sm"
    ]
    assert range_workers <= max_workers

    from restart_body import adapt
    restart_path = adapt(source(Path("/tmp/megabake-design-research"), "dense_gemm.py"))
    restart_module = _load(restart_path, f"dense_gemm_restart_{name}")
    restarted = restart_module.HopperWgmmaGemmKernel(cutlass.Float32, tile, (1, 1))
    restarted.workers = torch.cuda.get_device_properties(0).multi_processor_count
    restarted_compiled = cute.compile(restarted, ca, cb, cc, stream)

    rows = []
    for m in (measured_m, followup_m):
        a_call = torch.randn((m, 576), device="cuda", dtype=torch.bfloat16)
        b_call = torch.randn((1536, 576), device="cuda", dtype=torch.bfloat16)
        c_call = torch.empty((m, 1536), device="cuda", dtype=torch.bfloat16)
        views = tuple(map(_matrix, (a_call, b_call, c_call)))
        ref = (a_call.float() @ b_call.float().T).bfloat16()
        if m == measured_m:
            workers = range_workers
        else:
            tile_count_call = ((m + tile[0] - 1) // tile[0]) * (
                (1536 + tile[1] - 1) // tile[1]
            )
            workers = min(max_workers, max(1, tile_count_call // 2))
        # The same compiled range entry sees new pointers, contents, and tile
        # counts. It keeps its selected tile shape and rebinds TMA descriptors.
        range_compiled(*views, cutlass.Int32(workers), stream)
        torch.cuda.synchronize()
        err = (c_call.float() - ref.float()).norm() / ref.float().norm()
        assert err.item() < 0.005, {
            "regime": name,
            "m": m,
            "workers": workers,
            "relative_l2_error": err.item(),
            "max_abs_error": (c_call.float() - ref.float()).abs().max().item(),
            "actual": c_call[0, :8].float().tolist(),
            "expected": ref[0, :8].float().tolist(),
        }
        assert torch.isfinite(c_call).all().item()
        rows.append({
            "m": m,
            "n": 1536,
            "k": 576,
            "tile": tile,
            "workers": workers,
            "max_admitted_workers": max_workers,
            "max_abs_error": (c_call.float() - ref.float()).abs().max().item(),
            "relative_l2_error": err.item(),
            "resident_range_gpu_us": samples(
                lambda: range_compiled(*views, cutlass.Int32(workers), stream),
                32,
                output=c_call,
            )["median_us"],
        })
        if m == measured_m:
            # Compare the original launch and the restarting adapter at the
            # same shape. These are references, not production providers.
            standalone_compiled(ca, cb, cc, stream)
            restarted_compiled(ca, cb, cc, stream)
            torch.cuda.synchronize()
            rows[-1]["standalone_persistent_gpu_us"] = samples(
                lambda: standalone_compiled(ca, cb, cc, stream),
                32,
                output=c,
            )["median_us"]
            rows[-1]["restarting_dense_body_gpu_us"] = samples(
                lambda: restarted_compiled(ca, cb, cc, stream),
                32,
                output=c,
            )["median_us"]

    return {
        "regime": name,
        "selected_tile": tile,
        "range_resources": resources,
        "records": rows,
        "launch_count_each": 1,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    root = Path("/tmp/megabake-design-research")
    root.mkdir(exist_ok=True)
    upstream = source(root, UPSTREAM_FILE)
    persistent_hash = hashlib.sha256(upstream.read_bytes()).hexdigest()
    assert persistent_hash == HASHES[UPSTREAM_FILE]
    original_module = _load(upstream, "megabake_original_persistent")
    dense_path = source(root, "dense_gemm.py")
    dense_module = _load(dense_path, "megabake_dense")
    generated = _adapt(upstream)
    range_module = _load(generated, "megabake_persistent_range")
    range_type = _range_class(range_module.HopperWgmmaGemmPersistentKernel)
    contract_config = range_type(cutlass.Float32, (64, 64), (1, 1), 1, True)
    hardware = cutlass.utils.HardwareInfo()
    torch.cuda.set_stream(torch.cuda.Stream())
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    report = {
        "gpu": str(torch.cuda.get_device_properties(0)),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cute": cutlass.__version__,
        "upstream_revision": REV,
        "upstream_file": UPSTREAM_FILE,
        "upstream_sha256": persistent_hash,
        "source_adaptation": "persistent kernel becomes a device-callable range body; enclosing worker allocates and passes shared storage; tile scheduler and TMA/WGMMA pipeline remain in the source body",
        "body_contract": {
            "entry": "persistent scheduler parameters, runtime scheduler grid, current TMA descriptors/tensors, CTA-local scratch, and epilogue binding",
            "cta_threads": contract_config.threads_per_cta,
            "warp_roles": {
                "dma_warp_group": "warps 0-3; warp 0 issues TMA input loads",
                "mma_warp_group": "warps 4-7 consume staged tiles and issue WGMMA",
                "epilogue_store_warp": 4
            },
            "pipeline": "source TMA/WGMMA stage ring remains active across the assigned persistent tile range; producer_tail drains input work, WGMMA wait_group(0) completes MMA, and TMA store producer_tail drains output writes",
            "scratch_release": "CTA-local shared storage is reusable only after the range body returns from both pipeline tails",
            "exit": "all output tiles assigned to the synthetic scheduler grid are globally stored; the enclosing kernel must grid-synchronize before another CTA reads them",
            "binding": "host rebuilds descriptors for each call; scheduler grid can be narrowed while idle outer CTAs bypass the range body"
        },
        "records": [],
    }
    for name, tile, measured_m, followup_m in (
        ("decode_small_m", (64, 64), 4, 256),
        ("prefill_larger_m", (64, 128), 256, 1024),
    ):
        report["records"].append(
            _run_case(
                name,
                tile,
                measured_m,
                followup_m,
                (original_module, range_module, dense_module),
                hardware,
                stream,
            )
        )
    output = Path(__file__).with_name("resident_range.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
