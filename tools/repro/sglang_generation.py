#!/usr/bin/env python3
"""Continuous greedy decode using the existing SGLang ModelRunner benchmark API."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
from generation_common import LENGTHS, PROMPT, CHECKS, fingerprint, record


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lengths',type=int,nargs='+',default=LENGTHS);p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--profile',action='store_true')
    args=p.parse_args()
    print('Importing SGLang benchmark and CUDA runtime',flush=True)
    import torch
    import sglang.benchmark.one_batch as b
    print('SGLang benchmark imported',flush=True)
    from safetensors.torch import save_file
    parser=argparse.ArgumentParser();b.ServerArgs.add_cli_args(parser)
    raw=parser.parse_args(['--model-path',args.model,'--tokenizer-path',args.model,'--dtype','bfloat16',
                          '--tp-size','1','--mem-fraction-static','0.5','--context-length','8192',
                          '--max-total-tokens','8192','--random-seed','20261026'])
    server_args=b.ServerArgs.from_cli_args(raw)
    bench_args=b.BenchArgs(batch_size=(1,),input_len=(128,),output_len=(max(args.lengths),))
    def execute(server_args,port_args,bench_args,gpu_id,tp_rank):
        b.publish(server_args,role='scheduler');b.initialize_moe_config();b.initialize_fp8_gemm_config();b.initialize_fp4_gemm_config()
        runner,_=b.load_model(server_args,port_args,gpu_id,tp_rank)
        def run(length,trial,verify=False):
            runner.clear()
            reqs=b.prepare_synthetic_inputs_for_latency_test(1,128,custom_inputs=[PROMPT])
            runner.synchronize();t=time.perf_counter();current,logits,batch=runner.extend(reqs);runner.synchronize()
            prefill=time.perf_counter()-t
            output=torch.empty(length,device='cuda',dtype=torch.int64);output[0].copy_(current[0])
            checks={'0':logits.detach().clone()} if verify else {}
            runner.synchronize();start=time.perf_counter()
            for index in range(1,length):
                current,logits=runner.decode(current,batch)
                output[index].copy_(current[0])
                if verify and index in CHECKS: checks[str(index)]=logits.detach().clone()
            runner.synchronize();elapsed=time.perf_counter()-start
            tokens=output.cpu().tolist();runner.cleanup(batch)
            return record(length,trial,elapsed,prefill,tokens),checks
        run(32,0)
        graph_runner=getattr(runner.torch_runner,'decode_cuda_graph_runner',None)
        if graph_runner is None: raise RuntimeError('SGLang CUDA Graph runner was not initialized')
        original_replay=graph_runner.execute
        replay_count=[0]
        def counted_replay(*a,**kw):
            replay_count[0]+=1
            return original_replay(*a,**kw)
        graph_runner.execute=counted_replay
        validation,checks=run(max(args.lengths),0,True)
        graph_runner.execute=original_replay
        if replay_count[0] != max(args.lengths)-1:
            raise RuntimeError(f'SGLang graph replay count {replay_count[0]} does not match decode steps')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        save_file(checks,str(args.output.parent/'sglang-validation.safetensors'))
        measurements=[]
        for trial in range(1,args.rounds+1):
            order=args.lengths if trial%2 else list(reversed(args.lengths))
            for length in order:
                row,_=run(length,trial)
                if row['output_token_ids'] != validation['output_token_ids'][:length]:
                    raise RuntimeError('SGLang generation was not deterministic across trials')
                measurements.append(row)
                print(f"SG output={length}, trial={trial}, decode={row['decode_total_ms']:.3f} ms",flush=True)
        profiles=[]
        if args.profile:
            from profile_common import LEADS,profile_window
            for lead in LEADS:
                runner.clear()
                reqs=b.prepare_synthetic_inputs_for_latency_test(1,128,custom_inputs=[PROMPT])
                current,_,batch=runner.extend(reqs)
                output=torch.empty(4096,device='cuda',dtype=torch.int64)
                def step(index):
                    nonlocal current
                    current,_=runner.decode(current,batch)
                    output[index+1].copy_(current[0])
                for index in range(lead):step(index)
                profiles.append(profile_window(args.output.parent,'sglang',lead,step))
                runner.cleanup(batch)
        cfg=b.resolving_view(server_args)
        result=dict(state='complete',engine='sglang',prefill_tokens=128,batch_size=1,lengths=args.lengths,
                    rounds=args.rounds,model_files=fingerprint(args.model),measurements=measurements,profiles=profiles,
                    validation=validation,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
                    sglang=importlib.metadata.version('sglang'),
                    cuda_graph_disabled=bool(getattr(cfg,'disable_cuda_graph',False)),
                    validated_graph_replay_count=replay_count[0],
                    sglang_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(b.__file__).resolve().parents[3],text=True).strip(),
                    sglang_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=Path(b.__file__).resolve().parents[3],text=True).strip()),
                    execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip())
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    b.latency_test=execute
    b.main(server_args,bench_args)


if __name__=='__main__': main()
