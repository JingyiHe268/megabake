import unittest
from profile_common import summarize_trace,category


class ProfileTests(unittest.TestCase):
    def test_gpu_union_does_not_double_count_overlapping_kernels(self):
        events=[dict(ph='X',cat='kernel',name='gemm',ts=10,dur=10),
                dict(ph='X',cat='kernel',name='other',ts=15,dur=10),
                dict(ph='X',cat='gpu_memcpy',name='memcpy',ts=30,dur=5),
                dict(ph='X',cat='cuda_runtime',name='cudaLaunchKernel',ts=0,dur=100)]
        row=summarize_trace({'traceEvents':events},5)
        self.assertEqual(row['gpu_busy_us_per_step'],4)
        self.assertEqual(row['gpu_gap_us_per_step'],1)
        self.assertEqual(row['kernel_count_per_step'],.4)
        self.assertEqual(row['cuda_apis'][0]['us_per_step'],20)

    def test_missing_gpu_activities_is_not_a_valid_profile(self):
        with self.assertRaises(RuntimeError):summarize_trace({'traceEvents':[]},16)

    def test_specialized_attention_and_norm_are_not_mislabeled_as_gemm(self):
        self.assertEqual(category('cutlass::device_kernel<flash::FlashAttnFwdSm90>'),'attention / softmax')
        self.assertEqual(category('cutlass_kernel_flashinfernormkernelsFusedAddRMSNorm'),'norm / position / activation')
        self.assertEqual(category('nvjet_tst_64x8_64x16'),'linear algebra')
        self.assertEqual(category('pytorch_flash::flash_fwd_splitkv_kernel'),'attention / softmax')
        self.assertEqual(category('fmha_cutlassF_bf16_aligned_64x64_rf_sm80'),'attention / softmax')
