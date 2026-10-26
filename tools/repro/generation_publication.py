"""Export only validated, common-checkpoint continuous-generation comparisons."""
import csv
import io
import json
import math
import statistics
from generation_common import token_digest, CHECKS


def validated(directory):
    status=json.loads((directory/'status.json').read_text())
    sg=json.loads((directory/'sglang-generation.json').read_text())
    mb=json.loads((directory/'megabake-generation.json').read_text())
    if status.get('state')!='complete' or sg.get('state')!='complete' or mb.get('state')!='complete':
        raise ValueError('Continuous generation has not completed')
    if (sg['model_files']!=mb['model_files'] or sg['lengths']!=mb['lengths'] or sg['rounds']!=mb['rounds']
            or sg['execution_commit']!=mb['execution_commit'] or mb['execution_commit']!=status['execution_commit']
            or any(x['prefill_tokens']!=128 or x['batch_size']!=1 for x in (sg,mb))):
        raise ValueError('Cross-engine workload contract mismatch')
    if (not mb.get('native_reference_validated') or mb.get('workload_kernel_count')!=1
            or sg.get('validated_graph_replay_count')!=max(sg['lengths'])-1):
        raise ValueError('Token, megakernel or SGLang graph replay check missing')
    errors=mb.get('native_reference_logits_relative_l2',{})
    cache_errors=mb.get('native_reference_kv_relative_l2',{})
    expected_checks={str(i) for i in CHECKS if 0<i<max(mb['lengths'])}
    if (not errors or set(errors)!=expected_checks or set(errors)!=set(cache_errors)
            or not all(isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<.02
                       for v in [*errors.values(),*cache_errors.values()])):
        raise ValueError('Native-reference logits or KV validation failed')
    groups={}
    for data in (sg,mb):
        expected=data['validation']['output_token_ids']
        seen=set()
        for row in data['measurements']:
            length=row['output_tokens'];trial=row['trial'];key=(length,trial)
            if length not in data['lengths'] or not 1<=trial<=data['rounds'] or key in seen:
                raise ValueError('Unexpected or duplicate generation trial')
            seen.add(key)
            if (row['decode_steps']!=length-1 or row['prefill_tokens']!=128 or row['batch_size']!=1
                    or row['final_cached_tokens']!=128+length-1
                    or row['output_token_ids']!=expected[:length]
                    or row['output_sha256']!=token_digest(expected[:length])):
                raise ValueError('Wrong generation length or output sequence')
            for k in ('decode_total_ms','decode_mean_ms_per_token','decode_tokens_per_second','prefill_ms'):
                if not isinstance(row[k],(float,int)) or not math.isfinite(row[k]) or row[k]<=0:
                    raise ValueError('Nonfinite or invalid timing')
            if not math.isclose(row['decode_total_ms']/(length-1),row['decode_mean_ms_per_token'],rel_tol=1e-10):
                raise ValueError('TPOT does not match total decode time')
            if not math.isclose(1000/row['decode_mean_ms_per_token'],row['decode_tokens_per_second'],rel_tol=1e-10):
                raise ValueError('Throughput does not match decode time')
            groups.setdefault((data['engine'],length),[]).append(row)
        if len(seen)!=len(data['lengths'])*data['rounds']:
            raise ValueError('Incomplete generation sweep')
    return sg,mb,groups


def write_csv(path,rows):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows);path.write_text(stream.getvalue())


def export_generation(directory):
    sg,mb,groups=validated(directory)
    target=directory/'published';target.mkdir(exist_ok=True)
    trials=[];summary=[]
    for data in (sg,mb):
        for row in data['measurements']:
            trials.append(dict(engine=data['engine'],**{k:v for k,v in row.items() if k!='output_token_ids'},
                               torch=data['torch'],execution_commit=data['execution_commit']))
    text=('# Continuous generation: MegaBake vs SGLang\n\n'
          f'Shared random checkpoint, BF16, batch=1, prefill=128, {sg["rounds"]} runs per length. '
          'Each output length D includes one token obtained from prefill logits; the timed decode loop has D-1 steps. '
          'KV state grows naturally; no reset between decode steps. EOS stopping is disabled by the fixed-length benchmark.\n\n'
          'Timing starts after prefill and ends after the continuous decode loop and final GPU synchronization. '
          'Includes backend/model-runner calls, KV updates, greedy sampling and output-buffer writes. '
          'No per-step global synchronization. Excludes loading, compilation, warmup, prefill, HTTP and tokenization.\n\n'
          f'SGLang {sg["sglang"]}, torch {sg["torch"]}; MegaBake torch {mb["torch"]}, CuTe {mb["cute"]}. '
          'The stacks use different PyTorch/CUDA builds, so this is a backend-stack comparison, not an isolated compiler ablation.\n\n'
          f'SGLang source `{sg.get("sglang_commit", "unavailable")}`; existing dirty worktree: {sg.get("sglang_dirty", "unavailable")}. '
          'The existing SGLang environment and modifications were preserved.\n\n'
          f'MegaBake checked against native PyTorch using identical inputs and KV state at selected steps: '
          f'max logits L2 {max(mb["native_reference_logits_relative_l2"].values()):.6g}, '
          f'max new-KV L2 {max(mb["native_reference_kv_relative_l2"].values()):.6g}. '
          f'Cross-stack greedy sequences identical: {mb["greedy_tokens_match"]}; '
          f'first differing output index (zero-based): {mb["first_cross_engine_token_mismatch"]}. '
          'Each stack performs its own greedy generation; differing floating-point implementations may yield different token histories. '
          f'SGLang CUDA Graph replay verified ({sg["validated_graph_replay_count"]} validation decode calls); '
          'MegaBake workload verified as one kernel; sampling and output stores are separate operations.\n\n'
          f'Weight SHA256: `{sg["model_files"]["model.safetensors"]}`. '
          f'Execution commit: `{mb["execution_commit"]}`.\n\n'
          '| Output tokens | Decode steps | SGLang mean ms/token | MegaBake mean ms/token | MegaBake speedup |\n'
          '| ---: | ---: | ---: | ---: | ---: |\n')
    for length in sorted(sg['lengths']):
        med={engine:statistics.median(r['decode_mean_ms_per_token'] for r in groups[(engine,length)]) for engine in ('sglang','megabake')}
        for engine in med:
            rows=groups[(engine,length)];values=[r['decode_mean_ms_per_token'] for r in rows]
            summary.append(dict(engine=engine,output_tokens=length,decode_steps=length-1,prefill_tokens=128,batch_size=1,
                                median_run_mean_ms_per_token=med[engine],run_min_mean_ms_per_token=min(values),
                                run_max_mean_ms_per_token=max(values),median_decode_total_ms=statistics.median(r['decode_total_ms'] for r in rows),
                                median_decode_tokens_per_second=statistics.median(r['decode_tokens_per_second'] for r in rows),
                                runs=len(rows),megabake_speedup=med['sglang']/med['megabake']))
        text+=f'| {length} | {length-1} | {med["sglang"]:.6f} | {med["megabake"]:.6f} | {med["sglang"]/med["megabake"]:.3f}× |\n'
    text+='\nTable values are medians of per-run mean TPOTs; graph bands are observed run ranges, not confidence intervals. '
    text+='Random weights with the SmolLM-135M structural configuration and an explicit 8192-position synthetic context extension; no pretrained quality claim.\n'
    files=[target/'performance.csv',target/'trials.csv',target/'performance.md']
    write_csv(files[0],summary);write_csv(files[1],trials);files[2].write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'megabake-generation'})
    lengths=sorted(sg['lengths'])
    for metric,ylabel,name in [('decode_mean_ms_per_token','Decode mean latency (ms/token)','decode_tpot.svg'),
                              ('decode_total_ms','Decode loop time (ms)','decode_total.svg')]:
        fig,ax=plt.subplots(figsize=(9,5.4),layout='constrained')
        for engine,color in [('sglang','#337ac1'),('megabake','#14866a')]:
            values=[[r[metric] for r in groups[(engine,n)]] for n in lengths]
            ax.plot(lengths,[statistics.median(v) for v in values],'o-',label=engine,color=color)
            ax.fill_between(lengths,[min(v) for v in values],[max(v) for v in values],alpha=.15,color=color)
        ax.set_xscale('log',base=2);ax.set_xticks(lengths,labels=lengths);ax.set_ylim(bottom=0)
        ax.set_xlabel('Output tokens D (D-1 timed decode steps)');ax.set_ylabel(ylabel);ax.grid(alpha=.2);ax.legend()
        ax.set_title('Continuous greedy decode · batch=1 · prefill=128\nShared mock SmolLM-135M checkpoint · H100 · BF16')
        fig.supxlabel('Prefill excluded; no per-step synchronization; real growing KV; different software stacks.\n'
                      'Bands: observed run ranges; synthetic weights and extended context.',fontsize=8)
        path=target/name;fig.savefig(path,metadata={'Date':None});plt.close(fig);files.append(path)
    return files
