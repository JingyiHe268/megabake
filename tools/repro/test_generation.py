"""Tests for continuous-generation counting and publication contracts."""
import json
from pathlib import Path
import tempfile
import unittest
from generation_common import record
from generation_publication import validated


class GenerationTests(unittest.TestCase):
    def evidence(self, folder):
        (folder/'status.json').write_text('{"state":"complete","execution_commit":"abc"}')
        for engine,token in [('sglang',1),('megabake',2)]:
            data=dict(state='complete',engine=engine,model_files={'model.safetensors':'same'},lengths=[128,256],
                      rounds=1,execution_commit='abc',prefill_tokens=128,batch_size=1,
                      validated_graph_replay_count=255,workload_kernel_count=1,native_reference_validated=True,
                      native_reference_logits_relative_l2={str(k):.01 for k in (1,127,255)},
                      native_reference_kv_relative_l2={str(k):.01 for k in (1,127,255)},
                      validation={'output_token_ids':[token]*256},
                      measurements=[record(n,1,(n-1)*.001,.01,[token]*n) for n in (128,256)])
            (folder/f'{engine}-generation.json').write_text(json.dumps(data))

    def test_output_length_counts_first_token_from_prefill(self):
        row=record(128,1,.127,.02,[1]*128)
        self.assertEqual(row['decode_steps'],127)
        self.assertEqual(row['final_cached_tokens'],255)
        self.assertAlmostEqual(row['decode_mean_ms_per_token'],1.)

    def test_generation_requires_native_reference_and_common_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);self.evidence(folder)
            validated(folder)  # Different greedy outputs are explicitly permitted.
            p=folder/'megabake-generation.json';data=json.loads(p.read_text())
            data['model_files']={'model.safetensors':'other'};p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):validated(folder)
            self.evidence(folder);data=json.loads(p.read_text());data['native_reference_validated']=False
            p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):validated(folder)

    def test_generation_rejects_off_by_one_decode_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);self.evidence(folder)
            p=folder/'sglang-generation.json';data=json.loads(p.read_text())
            data['measurements'][0]['decode_steps']=128;p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):validated(folder)


if __name__=='__main__':unittest.main()
