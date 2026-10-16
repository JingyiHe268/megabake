import cutlass
import cutlass.cute as cute

@cute.jit
def _vector_affine(
    source: cute.Tensor,
    target: cute.Tensor,
    workers: cutlass.Int32,
    threads: cutlass.Constexpr,
    scale: cutlass.Float32,
    offset: cutlass.Float32,
):
    _, _, block = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    count = cute.size(source)
    first = block * threads + thread
    step = workers * threads
    for i in cutlass.range(first, count, step):
        value = (((source[i]).to(cutlass.Float32) * scale) + offset)
        target[i] = value.to(cutlass.BFloat16)

@cute.jit
def _vector_row_bias(
    source: cute.Tensor,
    rows: cute.Tensor,
    target: cute.Tensor,
    columns: cutlass.Int32,
    workers: cutlass.Int32,
    threads: cutlass.Constexpr,
    scale: cutlass.Float32,
    offset: cutlass.Float32,
):
    _, _, block = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    count = cute.size(source)
    first = block * threads + thread
    step = workers * threads
    for i in cutlass.range(first, count, step):
        value = (((source[i]).to(cutlass.Float32) + ((rows[i // columns]).to(cutlass.Float32) * scale)) + offset)
        target[i] = value.to(cutlass.BFloat16)

@cute.jit
def _row_reduce(
    source: cute.Tensor,
    target: cute.Tensor,
    rows: cutlass.Int32,
    columns: cutlass.Int32,
    workers: cutlass.Int32,
):
    _, _, block = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    if thread == 0:
        for row in cutlass.range(block, rows, workers):
            total = cutlass.Float32(0.0)
            for column in cutlass.range(columns):
                total = (total + (source[row * columns + column]).to(cutlass.Float32))
            target[row] = total.to(cutlass.BFloat16)
