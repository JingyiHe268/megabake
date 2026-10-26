#!/usr/bin/env python3
"""Continuous greedy MegaBake decode from real 128-token prefill KV state."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
from generation_common import LENGTHS,PROMPT,CHECKS,fingerprint,record


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sglang-result',type=Path,required=True);p.add_argument('--lengths',type=int,nargs='+',default=LENGTHS)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--profile',action='store_true');p.add_argument('--phase-profile',action='store_true')
    args=p.parse_args()
    if args.phase_profile:os.environ['MEGABAKE_PHASE_TIMER']='1'
    import torch
    from transformers import AutoModelForCausalLM
    from megabake import make_backend
    from tests.fixtures import StatefulSmolLM
    from safetensors.torch import load_file
    torch.set_grad_enabled(False)
    torch.manual_seed(20261026)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).cuda().eval()
    stateful=StatefulSmolLM(model).eval()
    layers=model.config.num_hidden_layers;capacity=128+max(args.lengths)+128
    shape=(1,capacity,model.config.num_key_value_heads,model.config.hidden_size//model.config.num_attention_heads)
    state=[torch.zeros(shape,device='cuda',dtype=torch.bfloat16) for _ in range(2*layers)]
    state += [torch.zeros((1,),device='cuda',dtype=torch.int32) for _ in range(layers)]
    backend=make_backend(mode='decode',compile_kernel=True,cache=False,scheduler='phase',dump_dir=args.output.parent/'generated')
    compiled=torch.compile(stateful,backend=backend,fullgraph=True,dynamic=False)
    dummy=torch.tensor([[504]],device='cuda');zero=torch.tensor([[0]],device='cuda')
    start=time.perf_counter();compiled(dummy,zero,*state);torch.cuda.synchronize();compile_s=time.perf_counter()-start
    if backend.megabake_contexts[-1].requirements.get('execution') != 'megakernel':raise RuntimeError('No megakernel execution')
    for _ in range(5):
        for length in state[2*layers:]: length.zero_()
        compiled(dummy,zero,*state)
    torch.cuda.synchronize()
    prompt=torch.tensor([PROMPT],device='cuda')
    positions=[torch.tensor([[128+i]],device='cuda') for i in range(max(args.lengths)+32)]
    sg=json.loads(args.sglang_result.read_text())
    if sg['model_files'] != fingerprint(args.model):raise RuntimeError('Checkpoint differs between engines')
    expected=load_file(str(args.sglang_result.parent/'sglang-validation.safetensors'),device='cuda')
    errors={};native_errors={};native_cache_errors={}
    def prepare():
        result=model(prompt,use_cache=True,return_dict=True)
        for i,cache in enumerate(result.past_key_values.layers):
            state[i].zero_();state[layers+i].zero_()
            state[i][:,:128].copy_(cache.keys.transpose(1,2))
            state[layers+i][:,:128].copy_(cache.values.transpose(1,2))
            state[2*layers+i].fill_(128)
        return result.logits[:,-1,:].argmax(dim=-1),result.logits[:,-1,:]
    def run(length,trial,verify=False):
        torch.cuda.synchronize();start=time.perf_counter();current,logits=prepare();torch.cuda.synchronize()
        prefill=time.perf_counter()-start
        output=torch.empty(length,device='cuda',dtype=torch.int64);output[0].copy_(current[0])
        checks={'0':logits.detach().clone()} if verify else {}
        torch.cuda.synchronize();start=time.perf_counter()
        for index in range(1,length):
            if verify and index in CHECKS:
                reference_state=[v.clone() for v in state]
                reference_logits=stateful(current.view(1,1),positions[index-1],*reference_state)[:,-1,:]
            logits=compiled(current.view(1,1),positions[index-1],*state)[:,-1,:]
            if verify and index in CHECKS:
                error=float((logits.float()-reference_logits.float()).norm()/reference_logits.float().norm().clamp_min(1e-12))
                if not bool(torch.isfinite(logits).all()) or error>=.02:raise RuntimeError(f'Native-reference logits L2 {error} at {index}')
                native_errors[str(index)]=error
                at=128+index-1;cache_max=0.
                for actual_cache,ref_cache in zip(state[:2*layers],reference_state[:2*layers]):
                    if not torch.equal(actual_cache[:,:at],ref_cache[:,:at]):raise RuntimeError('Existing KV state changed')
                    actual_row=actual_cache[:,at:at+1].float();ref_row=ref_cache[:,at:at+1].float()
                    cache_max=max(cache_max,float((actual_row-ref_row).norm()/ref_row.norm().clamp_min(1e-12)))
                if cache_max>=.02:raise RuntimeError(f'Native-reference KV L2 {cache_max} at {index}')
                if any(int(a.item())!=int(b.item()) for a,b in zip(state[2*layers:],reference_state[2*layers:])):raise RuntimeError('KV length mismatch')
                native_cache_errors[str(index)]=cache_max
            current=logits.argmax(dim=-1)
            output[index].copy_(current[0])
            if verify and index in CHECKS:checks[str(index)]=logits.detach().clone()
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        if any(int(v.item())!=128+length-1 for v in state[2*layers:]):raise RuntimeError('KV lengths did not advance correctly')
        return record(length,trial,elapsed,prefill,output.cpu().tolist()),checks
    validation,checks=run(max(args.lengths),0,True)
    debug=dict(megabake_tokens=validation['output_token_ids'],sglang_tokens=sg['validation']['output_token_ids'],
               first_mismatch=next((i for i,(a,b) in enumerate(zip(validation['output_token_ids'],sg['validation']['output_token_ids'])) if a!=b),None),
               logits_relative_l2={k:float((v.float()-expected[k].float()).norm()/expected[k].float().norm().clamp_min(1e-12)) for k,v in checks.items()})
    (args.output.parent/'generation-validation-debug.json').write_text(json.dumps(debug,indent=2)+'\n')
    first_mismatch=debug['first_mismatch']
    for index,logits in checks.items():
        if first_mismatch is not None and int(index)>first_mismatch:continue
        target=expected[index]
        error=float((logits.float()-target.float()).norm()/target.float().norm().clamp_min(1e-12))
        errors[index]=error
    # Workload trace excludes the separate greedy argmax and output-buffer copy.
    current,_=prepare();torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
        compiled(current.view(1,1),positions[0],*state);torch.cuda.synchronize()
    kernels=[e for e in profiler.events() if e.device_type==torch.autograd.DeviceType.CUDA]
    if len(kernels)!=1:raise RuntimeError(f'Workload kernels={len(kernels)}')
    measurements=[]
    for trial in range(1,args.rounds+1):
        order=args.lengths if trial%2 else list(reversed(args.lengths))
        for length in order:
            row,_=run(length,trial)
            if row['output_token_ids'] != validation['output_token_ids'][:length]:raise RuntimeError('MegaBake output not deterministic')
            measurements.append(row)
            print(f"MB output={length}, trial={trial}, decode={row['decode_total_ms']:.3f} ms",flush=True)
    profiles=[];phase_profiles=[]
    if args.profile or args.phase_profile:
        from profile_common import LEADS,profile_window
        for lead in LEADS:
            current,_=prepare();output=torch.empty(8192,device='cuda',dtype=torch.int64)
            def step(index):
                nonlocal current
                logits=compiled(current.view(1,1),positions[index],*state)[:,-1,:]
                current=logits.argmax(dim=-1);output[index+1].copy_(current[0])
            for index in range(lead):step(index)
            if args.profile:profiles.append(profile_window(args.output.parent,'megabake',lead,step))
            if args.phase_profile:
                from tools.phase_profile import profile
                cursor=[lead]
                def one():
                    step(cursor[0]);cursor[0]+=1
                phase_profiles.append(dict(lead_decode_steps=lead,first_cached_tokens=128+lead,
                    **profile(backend.megabake_last_callable,backend.megabake_last_compiled,one,9)))
    result=dict(state='complete',engine='megabake',prefill_tokens=128,batch_size=1,lengths=args.lengths,
                profiles=profiles,phase_profiles=phase_profiles,phase_instrumented=args.phase_profile,
                rounds=args.rounds,model_files=fingerprint(args.model),measurements=measurements,validation=validation,
                cross_engine_same_prefix_logits_relative_l2=errors,greedy_tokens_match=first_mismatch is None,
                first_cross_engine_token_mismatch=first_mismatch,
                native_reference_logits_relative_l2=native_errors,native_reference_kv_relative_l2=native_cache_errors,
                native_reference_validated=True,workload_kernel_count=1,
                cache_capacity=capacity,compile_seconds=compile_s,torch=torch.__version__,
                cute=importlib.metadata.version('nvidia-cutlass-dsl'),gpu=torch.cuda.get_device_name(),
                execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip())
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
