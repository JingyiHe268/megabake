"""Adapt the pinned dense example to one device body per output tile.

Preserve its license in the generated file. Limit the experiment to batch=1,
cluster=(1,1). Keep its tile swizzle. Drain the store and synchronize before
reusing scratch. This is a measurement adapter, not production lowering.
"""
from pathlib import Path


def adapt(path: Path) -> Path:
    text = path.read_text()
    signature_start = text.index('    @cute.kernel\n    def kernel(')
    body_start = text.index('        """', signature_start)
    signature = text[signature_start:body_start]
    assert signature.endswith('    ):\n')
    args = ['tma_atom_a', 'mA_mkl', 'tma_atom_b', 'mB_nkl', 'tma_atom_c',
            'mC_mnl', 'tiled_mma', 'cta_layout_mnk', 'a_smem_layout_staged',
            'b_smem_layout_staged', 'epi_smem_layout_staged']
    wrapper = signature + '''        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        worker, _, _ = cute.arch.block_idx()
        workers, _, _ = cute.arch.grid_dim()
        gm = cute.ceil_div(cute.size(mC_mnl, mode=[0]), self.tile_shape_mnk[0])
        gn = cute.ceil_div(cute.size(mC_mnl, mode=[1]), self.tile_shape_mnk[1])
        for tile_id in cutlass.range(worker, gm * gn, workers):
            self.body(''' + ', '.join(args) + ''', tile_id, storage)
            cute.arch.sync_threads()

'''
    new_signature = signature.replace('@cute.kernel', '@cute.jit').replace('def kernel(', 'def body(').replace('    ):\n', '        tile_id: cutlass.Int32,\n        storage,\n    ):\n')
    text = text[:signature_start] + wrapper + new_signature + text[body_start:]
    replacements = {
        '            grid=grid,': '            grid=(min(cute.size(grid), self.workers), 1, 1),',
        '        bidx, bidy, bidz = cute.arch.block_idx()': '        bidx, bidy, bidz = 0, 0, 0',
        '        cidx, cidy, _ = cute.arch.cluster_idx()\n        cdimx, cdimy, _ = cute.arch.cluster_dim()': '''        cdimx = cutlass.Int32(cute.ceil_div(cute.size(mC_mnl, mode=[0]), self.tile_shape_mnk[0]))
        cdimy = cutlass.Int32(cute.ceil_div(cute.size(mC_mnl, mode=[1]), self.tile_shape_mnk[1]))
        cidx, cidy = tile_id % cdimx, tile_id // cdimx''',
        '        bidx_in_cluster = cute.arch.block_in_cluster_idx()': '        bidx_in_cluster = (0, 0, 0)',
        '            cute.arch.block_idx_in_cluster()': '            0',
    }
    for old, new in replacements.items():
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    allocation = '        smem = cutlass.utils.SmemAllocator()\n        storage = smem.allocate(self.shared_storage)'
    # Keep the wrapper allocation, remove the allocation inside the device body.
    assert text.count(allocation) == 2
    pos = text.rindex(allocation)
    text = text[:pos] + text[pos:].replace(allocation, '', 1)
    target = path.with_name('dense_gemm_restart.py')
    target.write_text(text)
    return target
