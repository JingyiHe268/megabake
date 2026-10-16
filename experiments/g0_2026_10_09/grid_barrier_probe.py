"""Check a cooperative CuTe grid barrier using invocation-owned ready slots.

This is a small G0 protocol probe. It is not a reusable scheduler.
"""

import json
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda


@cute.kernel
def _work_then_barrier(
    x: cute.Tensor,
    y: cute.Tensor,
    sums: cute.Tensor,
    ready: cute.Tensor,
    workers: cutlass.Int32,
    epoch: cutlass.Int32,
    delay_cta: cutlass.Int32,
):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()

    # Make one output per CTA. A selected CTA reads extra input before
    # publishing to exercise a late producer without changing the protocol.
    if thread == 0:
        value = x[block]
        if block == delay_cta:
            delayed_sum = cutlass.Float32(0.0)
            for i in cutlass.range(100_000):
                delayed_sum += x[i % workers]
            value += delayed_sum
        y[block] = value * cutlass.Float32(2.0)
    cute.arch.sync_threads()

    if thread == 0:
        cute.arch.atomic_exch(
            ready.iterator + block,
            epoch,
            sem="release",
            scope="gpu",
        )

    # Each invocation has a new epoch. Every CTA writes its own slot, so no
    # pre-launch clear is needed. Cooperative admission keeps all participants
    # resident while they wait.
    if thread == 0:
        all_ready = False
        while not all_ready:
            all_ready = True
            for i in cutlass.range(workers):
                seen = cute.arch.atomic_cas(
                    ready.iterator + i,
                    cmp=epoch,
                    val=epoch,
                    sem="acquire",
                    scope="gpu",
                )
                if seen != epoch:
                    all_ready = False
    cute.arch.sync_threads()

    total = cutlass.Float32(0.0)
    for i in cutlass.range(workers):
        total += y[i]
    sums[block] = total


@cute.jit
def _launch(
    x: cute.Tensor,
    y: cute.Tensor,
    sums: cute.Tensor,
    ready: cute.Tensor,
    workers: cutlass.Int32,
    epoch: cutlass.Int32,
    delay_cta: cutlass.Int32,
    stream: cuda.CUstream,
):
    _work_then_barrier(x, y, sums, ready, workers, epoch, delay_cta).launch(
        grid=[workers, 1, 1],
        block=[128, 1, 1],
        cooperative=True,
        stream=stream,
    )


def _view(tensor):
    return from_dlpack(tensor, assumed_align=16)


def _resources(compiled, threads):
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
        status, active_blocks_per_sm = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            function, threads, 0
        )
        assert int(status) == 0, status
        values["active_blocks_per_sm"] = active_blocks_per_sm
        rows.append(values)
    assert len(rows) == 1, rows
    return rows[0]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20261009)
    compile_workers = 2
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def bindings(values):
        x = torch.tensor(values, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        sums = torch.empty_like(x)
        ready = torch.empty(len(values), device="cuda", dtype=torch.int32)
        return tuple(map(_view, (x, y, sums, ready))), (x, y, sums, ready)

    first_views, first_tensors = bindings([3.0, 5.0, 7.0, 9.0])
    compiled = cute.compile(
        _launch,
        *first_views,
        cutlass.Int32(compile_workers),
        cutlass.Int32(1),
        cutlass.Int32(0),
        stream,
    )
    resources = _resources(compiled, 128)
    max_workers = (
        torch.cuda.get_device_properties(0).multi_processor_count
        * resources["active_blocks_per_sm"]
    )
    assert max_workers > 0

    reports = []
    calls = (
        (first_views, first_tensors, 2, 1, 0, [3.0, 5.0]),
        (*bindings([-2.0, 7.0, 4.0, 8.0]), 3, 2, 2, [-2.0, 7.0, 4.0]),
        (first_views, first_tensors, 4, 3, 1, [11.0, 13.0, 17.0, 19.0]),
    )
    for views, tensors, workers, epoch, delay_cta, values in calls:
        # The last call reuses the first binding and ready storage after a
        # larger grid, with a new epoch and new input contents.
        tensors[0].copy_(torch.tensor(values + [0.0] * (4 - len(values)), device="cuda"))
        compiled(
            *views,
            cutlass.Int32(workers),
            cutlass.Int32(epoch),
            cutlass.Int32(delay_cta),
            stream,
        )
        torch.cuda.synchronize()
        x, y, sums, ready = tensors
        delayed_sum = sum(values[i % workers] for i in range(100_000))
        expected_y = x[:workers] * 2
        expected_y[delay_cta] += 2 * delayed_sum
        expected_sums = expected_y.sum().expand_as(expected_y)
        torch.testing.assert_close(y[:workers], expected_y)
        torch.testing.assert_close(sums[:workers], expected_sums)
        assert torch.equal(ready[:workers], torch.full_like(ready[:workers], epoch))
        reports.append({
            "epoch": epoch,
            "workers": workers,
            "delay_cta": delay_cta,
            "output": y[:workers].cpu().tolist(),
            "sum": sums[:workers].cpu().tolist(),
        })

    # Run the largest grid admitted by the complete compiled kernel. A late
    # final CTA checks that every participant arrives at the barrier.
    large_values = [1.0] * max_workers
    large_views, large_tensors = bindings(large_values)
    largest = cute.compile(
        _launch,
        *large_views,
        cutlass.Int32(max_workers),
        cutlass.Int32(4),
        cutlass.Int32(max_workers - 1),
        stream,
    )
    largest(
        *large_views,
        cutlass.Int32(max_workers),
        cutlass.Int32(4),
        cutlass.Int32(max_workers - 1),
        stream,
    )
    torch.cuda.synchronize()
    _, y, sums, ready = large_tensors
    expected = torch.full_like(y, 2.0)
    expected[-1] += 200_000.0
    torch.testing.assert_close(y, expected)
    torch.testing.assert_close(sums, torch.full_like(sums, expected.sum().item()))
    assert torch.equal(ready, torch.full_like(ready, 4))

    report = {
        "gpu": str(torch.cuda.get_device_properties(0)),
        "cooperative_launch": True,
        "active_blocks_per_sm": resources["active_blocks_per_sm"],
        "largest_grid_workers": max_workers,
        "kernel_resources": resources,
        "block_threads": 128,
        "protocol": "CTA release exchange; CTA acquire CAS polling; CTA barriers on each side",
        "state": "one invocation-owned ready slot per CTA; host advances epoch; no memset launch",
        "repeated_bindings": reports,
        "largest_grid_passed": True,
        "status": "passed",
    }
    output = Path(__file__).with_name("grid_barrier.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
