#!/usr/bin/env python3
"""Counter-only follow-up; CUDA profiler markers exclude compilation and prefill."""
import argparse
import json
from pathlib import Path
from generation_common import PROMPT,fingerprint


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM
    from tests.fixtures import StatefulSmolLM
    from megabake import make_backend
    torch.set_grad_enabled(False)
    model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).cuda().eval()
    layers=model.config.num_hidden_layers;capacity=4352
    shape=(1,capacity,model.config.num_key_value_heads,model.config.hidden_size//model.config.num_attention_heads)
    state=[torch.zeros(shape,device='cuda',dtype=torch.bfloat16) for _ in range(2*layers)]
    state += [torch.zeros((1,),device='cuda',dtype=torch.int32) for _ in range(layers)]
    backend=make_backend(mode='decode',compile_kernel=True,cache=False,scheduler='phase')
    function=torch.compile(StatefulSmolLM(model).eval(),backend=backend,fullgraph=True,dynamic=False)
    dummy=torch.tensor([[504]],device='cuda');zero=torch.tensor([[0]],device='cuda')
    function(dummy,zero,*state);torch.cuda.synchronize()
    for _ in range(5):
        for v in state[2*layers:]:v.zero_()
        function(dummy,zero,*state)
    prompt=torch.tensor([PROMPT],device='cuda')
    positions=[torch.tensor([[128+i]],device='cuda') for i in range(4096)]
    for lead in (0,4079):
        prefill=model(prompt,use_cache=True,return_dict=True)
        for i,cache in enumerate(prefill.past_key_values.layers):
            state[i].zero_();state[layers+i].zero_()
            state[i][:,:128].copy_(cache.keys.transpose(1,2));state[layers+i][:,:128].copy_(cache.values.transpose(1,2))
            state[2*layers+i].fill_(128)
        token=prefill.logits[:,-1,:].argmax(dim=-1).view(1,1)
        for index in range(lead):token=function(token,positions[index],*state)[:,-1,:].argmax(dim=-1).view(1,1)
        torch.cuda.synchronize()
        print(f'NCU_CONTEXT={128+lead}',flush=True)
        torch.cuda.cudart().cudaProfilerStart()
        output=function(token,positions[lead],*state)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        if not bool(torch.isfinite(output).all()):raise RuntimeError('Counter run nonfinite')
    a.output.write_text(json.dumps(dict(state='complete',contexts=[128,4207],model_files=fingerprint(a.model)))+'\n')


if __name__=='__main__':main()
