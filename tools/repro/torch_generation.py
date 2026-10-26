#!/usr/bin/env python3
"""Shared-weight GPU-only attention ablation: eager graph, Inductor, Inductor graph."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import types
from generation_common import PROMPT, LENGTHS, fingerprint, record
from profile_common import LEADS, profile_window


def sdpa_kv(source,key,value,key_cache,value_cache,valid_length):
    import torch
    index=valid_length.to(torch.int64)
    key_cache.index_copy_(1,index,key);value_cache.index_copy_(1,index,value);valid_length.add_(1)
    mask=torch.arange(key_cache.shape[1],device=source.device)<valid_length[0]
    out=torch.nn.functional.scaled_dot_product_attention(
        source.permute(0,2,1,3),key_cache.permute(0,2,1,3),value_cache.permute(0,2,1,3),
        attn_mask=mask,enable_gqa=True)
    return out.permute(0,2,1,3).reshape(source.shape[0]*source.shape[2],1,source.shape[3])


def flash_kv(source,key,value,key_cache,value_cache,valid_length):
    """Packed FlashAttention with GPU-resident seqused_k; reads logical KV length."""
    import torch
    index=valid_length.to(torch.int64)
    key_cache.index_copy_(1,index,key);value_cache.index_copy_(1,index,value);valid_length.add_(1)
    capacity=key_cache.shape[1]
    cuq=torch.tensor([0,1],device=source.device,dtype=torch.int32)
    cuk=torch.tensor([0,capacity],device=source.device,dtype=torch.int32)
    output=torch.ops.aten._flash_attention_forward.default(
        source.reshape(1,source.shape[2],source.shape[3]),
        key_cache.reshape(capacity,key_cache.shape[2],key_cache.shape[3]),
        value_cache.reshape(capacity,value_cache.shape[2],value_cache.shape[3]),
        cuq,cuk,1,capacity,0.,False,False,seqused_k=valid_length)[0]
    return output.reshape(source.shape[0]*source.shape[2],1,source.shape[3])


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lengths',type=int,nargs='+',default=LENGTHS);p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--variants',nargs='+');a=p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM
    from tests.fixtures import StatefulSmolLM
    from mock_decode import tensor_kv
    torch.set_grad_enabled(False);torch.manual_seed(20261026)
    model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).cuda().eval()
    layers=model.config.num_hidden_layers;capacity=128+max(a.lengths)+128
    shape=(1,capacity,model.config.num_key_value_heads,model.config.hidden_size//model.config.num_attention_heads)
    prompt=torch.tensor([PROMPT],device='cuda');token=torch.tensor([[504]],device='cuda');position=torch.zeros((1,1),device='cuda',dtype=torch.int64)
    results=[]
    variants=[('tensor_eager_graph',tensor_kv,False,True),('tensor_inductor',tensor_kv,True,False),
              ('tensor_inductor_graph',tensor_kv,True,True),('sdpa_inductor_graph',sdpa_kv,True,True)]
    if a.variants:
        variants.append(('flash_inductor_graph',flash_kv,True,True))
        variants=[v for v in variants if v[0] in a.variants]
        if len(variants)!=len(a.variants):raise ValueError('Unknown/duplicate variant')
    compiled_cache={}
    for name,attention,use_inductor,use_graph in variants:
        print('BUILD '+name,flush=True)
        namespace=dict(StatefulSmolLM.forward.__globals__,kv_attention=attention)
        forward=types.FunctionType(StatefulSmolLM.forward.__code__,namespace)
        reference=type('GPUReference',(StatefulSmolLM,),{'forward':forward})(model).eval()
        if attention.__name__ in compiled_cache:
            function=compiled_cache[attention.__name__]
        else:
            function=torch.compile(reference,fullgraph=True,dynamic=False,options={'emulate_precision_casts':True}) if use_inductor else reference
            if use_inductor:compiled_cache[attention.__name__]=function
        state=[torch.zeros(shape,device='cuda',dtype=torch.bfloat16) for _ in range(2*layers)]
        state += [torch.zeros((1,),device='cuda',dtype=torch.int32) for _ in range(layers)]
        if attention is flash_kv:
            reference(token,position,*[v.clone() for v in state]);torch.cuda.synchronize()
            print('Native packed FlashAttention preflight passed',flush=True)
        start=time.perf_counter();function(token,position,*state);torch.cuda.synchronize();compile_s=time.perf_counter()-start
        def prepare():
            result=model(prompt,use_cache=True,return_dict=True)
            for i,cache in enumerate(result.past_key_values.layers):
                state[i].zero_();state[layers+i].zero_()
                state[i][:,:128].copy_(cache.keys.transpose(1,2));state[layers+i][:,:128].copy_(cache.values.transpose(1,2))
                state[2*layers+i].fill_(128)
            token.copy_(result.logits[:,-1,:].argmax(dim=-1).view(1,1));position.fill_(128)
            return int(token.item())
        def one():
            logits=function(token,position,*state)[:,-1,:]
            next_token=logits.argmax(dim=-1)
            token.copy_(next_token.view(1,1));position.add_(1)
            return logits,next_token
        for _ in range(3):
            prepare();one()
        torch.cuda.synchronize()
        graph=None
        if use_graph:
            prepare();torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):graph_logits,graph_token=one()
            def invoke():
                graph.replay();return graph_logits,graph_token
        else:invoke=one
        errors={};cache_errors={}
        def run(length,trial,verify=False):
            torch.cuda.synchronize();t=time.perf_counter();first=prepare();torch.cuda.synchronize();prefill=time.perf_counter()-t
            output=torch.empty(length,device='cuda',dtype=torch.int64);output[0]=first
            torch.cuda.synchronize();t=time.perf_counter()
            for index in range(1,length):
                check=verify and index in (1,127,1023,4095)
                if check:
                    refs=[s.clone() for s in state];expected=reference(token.clone(),position.clone(),*refs)[:,-1,:]
                logits,next_token=invoke();output[index].copy_(next_token[0])
                if check:
                    error=float((logits.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-12))
                    if not bool(torch.isfinite(logits).all()) or error>=.02:raise RuntimeError(f'{name}: L2 {error} at {index}')
                    errors[str(index)]=error;max_cache=0.;at=128+index-1
                    for actual,ref in zip(state[:2*layers],refs[:2*layers]):
                        if not torch.equal(actual[:,:at],ref[:,:at]):raise RuntimeError('Existing KV modified')
                        x=actual[:,at:at+1].float();y=ref[:,at:at+1].float()
                        max_cache=max(max_cache,float((x-y).norm()/y.norm().clamp_min(1e-12)))
                    if max_cache>=.02:raise RuntimeError('KV validation failed')
                    cache_errors[str(index)]=max_cache
            torch.cuda.synchronize();elapsed=time.perf_counter()-t
            if int(position.item())!=128+length-1 or any(int(v.item())!=128+length-1 for v in state[2*layers:]):raise RuntimeError('Invalid logical length')
            return record(length,trial,elapsed,prefill,output.cpu().tolist())
        validation=run(max(a.lengths),0,True);measurements=[]
        for trial in range(1,a.rounds+1):
            for length in a.lengths if trial%2 else list(reversed(a.lengths)):
                row=run(length,trial)
                if row['output_token_ids']!=validation['output_token_ids'][:length]:raise RuntimeError('Nondeterministic output')
                measurements.append(row);print(f'{name} output={length} trial={trial} TPOT={row["decode_mean_ms_per_token"]:.6f}',flush=True)
        profiles=[]
        for lead in LEADS:
            prepare();output=torch.empty(8192,device='cuda',dtype=torch.int64)
            def step(index):
                _,next_token=invoke();output[index+1].copy_(next_token[0])
            for index in range(lead):step(index)
            profiles.append(profile_window(a.output.parent,name,lead,step))
        results.append(dict(engine=name,state='complete',attention=attention.__name__,cuda_graph=use_graph,
                            inductor=use_inductor,inductor_options={'emulate_precision_casts':True} if use_inductor else {},
                            compile_seconds=compile_s,measurements=measurements,profiles=profiles,
                            native_reference_logits_relative_l2=errors,native_reference_kv_relative_l2=cache_errors,
                            validation=validation))
        # Incremental private checkpoint survives a later optional baseline failure.
        a.output.write_text(json.dumps(dict(state='running',variants=results),indent=2)+'\n')
        del graph,invoke,function,state
        torch.cuda.empty_cache()
    a.output.write_text(json.dumps(dict(state='complete',variants=results,torch=torch.__version__,
        prefill_tokens=128,batch_size=1,cache_capacity=capacity,lengths=a.lengths,rounds=a.rounds,
        model_files=fingerprint(a.model),execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2)+'\n')


if __name__=='__main__':main()
