#!/usr/bin/env python3
"""Run logical-length FlashAttention and hardware counters sequentially."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from run_node import now,write,run_stage,tool_environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();c=json.loads(a.config.read_text());root=Path(__file__).resolve().parents[2]
    if os.environ.get('SLURM_JOB_ID')!=str(c['job_id']) or not os.environ.get('SLURM_STEP_ID'):raise RuntimeError('Slurm step required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():raise RuntimeError('Dirty source')
    a.output.mkdir(exist_ok=False)
    env=tool_environment(os.environ,c['megabake_python'],c['cuda_home'],root/'src')
    env.update(PYTHONPATH=str(root/'src')+':'+str(root)+':'+str(root/'tools/repro'),CUDA_ROOT=c['cuda_home'],
               LD_LIBRARY_PATH=c['cuda_home']+'/lib64',TRITON_PTXAS_PATH=c['cuda_home']+'/bin/ptxas',
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
    stages=[('flash',[c['megabake_python'],str(root/'tools/repro/torch_generation.py'),'--model',c['model_path'],
                     '--output',str(a.output/'torch-flash-generation.json'),'--lengths','128','1024','4096','--rounds','3',
                     '--variants','flash_inductor_graph']),
            ('counters',[c['megabake_python'],str(root/'tools/repro/run_counter_node.py'),'--config',str(a.config),
                         '--output',str(a.output/'counters')])]
    result=dict(state='running',execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),stages=[])
    write(a.output/'status.json',result)
    for name,command in stages:
        stage=run_stage(name,command,a.output,env,3600,root);result['stages'].append(stage)
        write(a.output/'status.json',result)
        # Counters are independent of a failed optional attention implementation.
    result.update(state='complete' if all(s['state']=='complete' for s in result['stages']) else 'failed',finished_at=now())
    write(a.output/'status.json',result)


if __name__=='__main__':main()
