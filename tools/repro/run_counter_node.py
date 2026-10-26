#!/usr/bin/env python3
"""Sequential Nsight Compute follow-up in the already allocated GPU step."""
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
    metrics=['gpu__time_duration.sum','dram__bytes.sum','dram__throughput.avg.pct_of_peak_sustained_elapsed',
             'sm__throughput.avg.pct_of_peak_sustained_elapsed','sm__warps_active.avg.pct_of_peak_sustained_active']
    command=['/APP/u22/ai_x86/CUDA/12.4/nsight-compute-2024.1.0/ncu','--target-processes','all',
             '--profile-from-start','off','--kernel-name','regex:kernel_cutlass_megabake','--launch-count','2',
             '--section','LaunchStats','--metrics',','.join(metrics),'--csv','--log-file',str(a.output/'ncu.csv'),
             c['megabake_python'],str(root/'tools/repro/ncu_decode.py'),'--model',c['model_path'],
             '--output',str(a.output/'counter-validation.json')]
    stage=run_stage('ncu',command,a.output,env,3600,root)
    write(a.output/'status.json',dict(state=stage['state'],finished_at=now(),stage=stage,
        execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
    if stage['state']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
