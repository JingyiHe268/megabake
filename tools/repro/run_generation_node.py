#!/usr/bin/env python3
"""Sequential shared-checkpoint SGLang/MegaBake continuous decode benchmark."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from run_node import now,write,run_stage,tool_environment
from workflow import validate_id
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--run-id',type=validate_id,required=True)
    p.add_argument('--lengths',type=int,nargs='+',default=[128,256,512,1024,2048,4096]);p.add_argument('--rounds',type=int,default=3)
    a=p.parse_args();c=json.loads(a.config.read_text())
    if c.get('profile')!='cuda12-generation' or min(a.lengths)<2 or a.rounds<1:raise RuntimeError('Invalid generation contract')
    if os.environ.get('SLURM_JOB_ID')!=str(c['job_id']) or not os.environ.get('SLURM_STEP_ID'):raise RuntimeError('Use configured Slurm step')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise RuntimeError('Dirty source')
    folder=ROOT/'results'/a.run_id;folder.mkdir(exist_ok=True)
    if (folder/'status.json').exists():raise RuntimeError('Run ID already used')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    status=dict(state='running',run_id=a.run_id,started_at=now(),execution_commit=commit,stages=[])
    write(folder/'status.json',status);write(folder/'config.json',c)
    mb_env=tool_environment(os.environ,c['megabake_python'],c['cuda_home'],ROOT/'src')
    mb_env.update(PYTHONPATH=str(ROOT/'src')+':'+str(ROOT)+':'+str(ROOT/'tools/repro'),
                  CUDA_ROOT=c['cuda_home'],LD_LIBRARY_PATH=c['cuda_home']+'/lib64',
                  TRITON_PTXAS_PATH=c['cuda_home']+'/bin/ptxas',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
    sg_env=tool_environment(os.environ,c['sglang_python'],c['cuda_home'],Path(c['sglang_repo'])/'python')
    sg_env.update(PYTHONPATH=c['sglang_repo']+'/python:'+str(ROOT/'tools/repro'),HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',LD_LIBRARY_PATH=c['cuda_home']+'/lib64')
    common=['--model',c['model_path'],'--lengths',*map(str,a.lengths),'--rounds',str(a.rounds)]
    stages=[('prepare-checkpoint',[c['megabake_python'],str(ROOT/'tools/repro/prepare_shared_mock.py'),'--model',c['model_path']],mb_env),
            ('sglang-generation',[c['sglang_python'],str(ROOT/'tools/repro/sglang_generation.py'),*common,'--output',str(folder/'sglang-generation.json')],sg_env),
            ('megabake-generation',[c['megabake_python'],str(ROOT/'tools/repro/megabake_generation.py'),*common,'--sglang-result',str(folder/'sglang-generation.json'),'--output',str(folder/'megabake-generation.json')],mb_env)]
    if c.get('profiling'):
        stages[1][1].append('--profile');stages[2][1].append('--profile')
        stages.append(('torch-comparison',[c['megabake_python'],str(ROOT/'tools/repro/torch_generation.py'),*common,
                       '--output',str(folder/'torch-generation.json')],mb_env))
        stages.append(('megabake-phases',[c['megabake_python'],str(ROOT/'tools/repro/megabake_generation.py'),
                       '--model',c['model_path'],'--lengths','4096','--rounds','1','--phase-profile',
                       '--sglang-result',str(folder/'sglang-generation.json'),'--output',str(folder/'megabake-phases.json')],mb_env))
    for name,command,env in stages:
        stage=run_stage(name,command,folder,env,c['timeout_seconds'],ROOT);status['stages'].append(stage)
        write(folder/'status.json',status)
        if stage['state']!='complete':break
    status.update(state='complete' if len(status['stages'])==len(stages) and all(s['state']=='complete' for s in status['stages']) else 'failed',finished_at=now())
    write(folder/'status.json',status)


if __name__=='__main__':main()
