"""Only successful performance aggregates and charts; never raw profiler traces."""
import json
import math
import statistics
from generation_publication import validated,write_csv,export_generation
from profile_common import LEADS,WINDOW_STEPS,category


def export_profile(folder):
    sg,mb,_=validated(folder)
    baseline=json.loads((folder/'torch-generation.json').read_text())
    phase=json.loads((folder/'megabake-phases.json').read_text())
    if (baseline['state']!='complete' or phase['state']!='complete' or not phase.get('phase_instrumented')
            or mb.get('phase_instrumented') or baseline['model_files']!=sg['model_files']
            or phase['model_files']!=sg['model_files'] or baseline['execution_commit']!=sg['execution_commit']
            or phase['execution_commit']!=sg['execution_commit'] or baseline['prefill_tokens']!=128
            or baseline['batch_size']!=1 or baseline['lengths']!=sg['lengths'] or baseline['rounds']!=sg['rounds']):
        raise ValueError('Incomplete or mismatched profiling experiment')
    supplement=None
    if (folder/'torch-flash-generation.json').exists():
        supplement=json.loads((folder/'torch-flash-generation.json').read_text())
        if (supplement['state']!='complete' or supplement['model_files']!=sg['model_files'] or supplement['torch']!=baseline['torch']
                or supplement['lengths']!=sg['lengths'] or supplement['rounds']!=sg['rounds']
                or supplement['prefill_tokens']!=128 or supplement['batch_size']!=1
                or [v['engine'] for v in supplement['variants']]!=['flash_inductor_graph']):
            raise ValueError('Mismatched FlashAttention follow-up')
    variants=baseline['variants']+(supplement['variants'] if supplement else [])
    engines=[sg,mb,*variants]
    if {x['engine'] for x in baseline['variants']}!={'tensor_eager_graph','tensor_inductor','tensor_inductor_graph','sdpa_inductor_graph'}:
        raise ValueError('Missing compiler/graph ablation')
    for v in variants:
        if v['state']!='complete' or any(not math.isfinite(e) or e<0 or e>=.02 for kind in
                ('native_reference_logits_relative_l2','native_reference_kv_relative_l2') for e in v[kind].values()):
            raise ValueError('Baseline numerical validation failed')
        if set(v['native_reference_logits_relative_l2'])!={'1','127','1023','4095'}:
            raise ValueError('Missing growing-cache baseline checks')
        seen=set()
        for r in v['measurements']:
            pair=(r['output_tokens'],r['trial']);seen.add(pair)
            if (r['decode_steps']!=r['output_tokens']-1 or r['output_token_ids']!=v['validation']['output_token_ids'][:r['output_tokens']]
                    or not math.isfinite(r['decode_mean_ms_per_token']) or r['decode_mean_ms_per_token']<=0):
                raise ValueError('Invalid baseline generation')
        if len(seen)!=len(sg['lengths'])*sg['rounds']:raise ValueError('Incomplete baseline sweep')
    files=export_generation(folder);target=folder/'published'
    windows=[];kernels=[];apis=[];categories=[];timings=[];trials=[]
    for engine in engines:
        name=engine['engine']
        if [p['lead_decode_steps'] for p in engine['profiles']]!=LEADS:raise ValueError('Mismatched profiling contexts')
        for p in engine['profiles']:
            if (p['steps']!=WINDOW_STEPS or p['first_cached_tokens']!=128+p['lead_decode_steps']
                    or not math.isfinite(p['gpu_busy_us_per_step']) or p['gpu_busy_us_per_step']<=0):
                raise ValueError('Invalid GPU profiling window')
            base=dict(engine=name,lead_decode_steps=p['lead_decode_steps'],first_cached_tokens=p['first_cached_tokens'],steps=p['steps'])
            windows.append(dict(**base,**{k:p[k] for k in ('gpu_busy_us_per_step','gpu_span_us_per_step','gpu_gap_us_per_step',
                              'gpu_activities_per_step','kernel_count_per_step','profiled_wall_us_per_step')}))
            kernels.extend(dict(**base,**k) for k in p['kernels'])
            apis.extend(dict(**base,**k) for k in p['cuda_apis'])
            by_category={}
            for k in p['kernels']:
                c=category(k['name']);by_category[c]=by_category.get(c,0.)+k['us_per_step']
            categories.extend(dict(**base,category=k,us_per_step=v) for k,v in by_category.items())
        for length in sg['lengths']:
            rows=[r for r in engine['measurements'] if r['output_tokens']==length]
            values=[r['decode_mean_ms_per_token'] for r in rows]
            timings.append(dict(engine=name,output_tokens=length,decode_steps=length-1,
                                median_mean_tpot_ms=statistics.median(values),min_mean_tpot_ms=min(values),max_mean_tpot_ms=max(values)))
            trials.extend(dict(engine=name,**{k:r[k] for k in ('output_tokens','decode_steps','trial','decode_total_ms','decode_mean_ms_per_token','output_sha256')}) for r in rows)
    phases=[];phase_summary=[]
    for p in phase['phase_profiles']:
        phase_summary.append({k:p[k] for k in ('lead_decode_steps','first_cached_tokens','workers','phases','barrier_intervals','timed_total_us','trailing_phases_untimed')})
        phases.extend(dict(lead_decode_steps=p['lead_decode_steps'],category=c['category'],phases=c['phases'],approximate_us=c['us']) for c in p['categories'])
    if [p['lead_decode_steps'] for p in phase['phase_profiles']]!=LEADS:raise ValueError('Missing internal phase windows')
    for name,rows in [('comparison.csv',timings),('comparison_trials.csv',trials),('gpu_windows.csv',windows),
                      ('gpu_kernels.csv',kernels),('cuda_apis.csv',apis),('gpu_categories.csv',categories),
                      ('megabake_phase_summary.csv',phase_summary),('megabake_phase_categories.csv',phases)]:
        path=target/name;write_csv(path,rows);files.append(path)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'decode-profile-v1','font.size':10})
    labels={'sglang':'SGLang (CUDA Graph)','megabake':'MegaBake',
            'tensor_eager_graph':'Eager + CUDA Graph','tensor_inductor':'Inductor',
            'tensor_inductor_graph':'Inductor + CUDA Graph','sdpa_inductor_graph':'Inductor + masked SDPA + CUDA Graph',
            'flash_inductor_graph':'Inductor + FlashAttention + CUDA Graph'}
    fig,ax=plt.subplots(figsize=(9,5))
    for e in engines:
        rows=[r for r in timings if r['engine']==e['engine']]
        ax.plot([r['output_tokens'] for r in rows],[r['median_mean_tpot_ms'] for r in rows],marker='o',label=labels[e['engine']])
    ax.set(xlabel='Output length (includes first token from prefill)',ylabel='Mean decode ms/token',title='H100, shared mock SmolLM-135M, BF16, batch 1, prefill 128')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
    path=target/'compiler_graph_comparison.svg';fig.savefig(path,metadata={'Date':None});plt.close(fig);files.append(path)
    fig,axes=plt.subplots(1,3,figsize=(14,5),sharey=True)
    names=[e['engine'] for e in engines]
    for ax,lead in zip(axes,LEADS):
        rows=[next(r for r in windows if r['engine']==name and r['lead_decode_steps']==lead) for name in names]
        busy=[r['gpu_busy_us_per_step']/1000 for r in rows]
        ax.bar(range(len(names)),busy,label='GPU busy (union)')
        ax.set_title(f'Cached tokens: {128+lead}–{128+lead+15}')
        ax.set_xticks(range(len(names)),[labels[n] for n in names],rotation=65,ha='right',fontsize=8)
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Profiled GPU busy ms/step');axes[0].legend(fontsize=8)
    fig.suptitle('GPU activity time in separate 16-step CUPTI windows (host gaps excluded)')
    fig.tight_layout();path=target/'gpu_timeline.svg';fig.savefig(path,metadata={'Date':None});plt.close(fig);files.append(path)
    text=('# Matched decode profiling and compiler/graph ablation\n\n'
          'H100 80GB, shared nonzero random checkpoint, BF16, batch=1, prefill=128. '
          'Output length D includes the first token from prefill; timed decode steps = D−1. '
          'Three continuous, unprofiled runs per length; compilation, warmup and prefill excluded. '
          'All engines grow their real prefill KV state. Cross-stack natural greedy histories can differ.\n\n'
          f'Execution commit: `{sg["execution_commit"]}`. Weight SHA256: `{sg["model_files"]["model.safetensors"]}`.\n\n'
          'SGLang uses its existing PyTorch 2.13/cu129 environment; all other variants use PyTorch 2.14/cu126. '
          'Inductor options: `emulate_precision_casts=True`; same upstream StatefulSmolLM computation and weights, '
          'with its CPU-scalar-extracting declared attention replaced by GPU-only tensor operations. '
          f'The tensor and SDPA references attend a masked, padded static capacity of {baseline["cache_capacity"]} tokens; '
          'SGLang and MegaBake can use logical lengths, so this is an explicit attention/layout limitation of the compiler baseline. '
          'Masked SDPA is a separate attention implementation ablation, not a guaranteed fast attention dispatch. '
          'The optional packed FlashAttention variant uses device `seqused_k` to read the logical KV length, '
          'avoiding padded attention arithmetic. Each compiled/graph variant passes same-input/native-reference logits '
          'and new-KV relative L2 <0.02 at steps 1,127,1023,4095; old KV remains unchanged.\n\n'
          'CUDA Graph variants capture model, greedy argmax, fixed input-buffer update and position increment; '
          'output-buffer writes remain in the common outer loop. The current MegaBake table runtime increments a device call counter '
          'and expects a matching host-stamped address-ring slot for every launch. Replaying an unchanged captured launch '
          'does not maintain this protocol; therefore MegaBake+CUDA Graph is not presented as a valid baseline.\n\n'
          '| Engine | Output tokens | Median mean TPOT (ms) | Run range (ms) |\n| --- | ---: | ---: | ---: |\n')
    for r in timings:text+=f'| {labels[r["engine"]]} | {r["output_tokens"]} | {r["median_mean_tpot_ms"]:.6f} | {r["min_mean_tpot_ms"]:.6f}–{r["max_mean_tpot_ms"]:.6f} |\n'
    if supplement:text+=f'\nFlashAttention follow-up execution commit: `{supplement["execution_commit"]}`; unchanged model/core compiler, same software environment and workload.\n'
    text+=('\nSeparate CUPTI profiles record 16 consecutive decode steps after 0,2032,4079 prior decode steps. '
           'These profiles are not the primary latency measurements. GPU busy is the union of GPU intervals; kernel-category sums may overlap. '
           'CPU CUDA API duration overlaps GPU work and must not be added to GPU time. Profiling can enlarge host/GPU gaps.\n\n'
           '| Engine | First cached tokens | GPU busy µs/step | GPU gaps µs/step | Kernels/step |\n| --- | ---: | ---: | ---: | ---: |\n')
    for r in windows:text+=f'| {labels[r["engine"]]} | {r["first_cached_tokens"]} | {r["gpu_busy_us_per_step"]:.2f} | {r["gpu_gap_us_per_step"]:.2f} | {r["kernel_count_per_step"]:.2f} |\n'
    text+=('\nInternal MegaBake phase stamps come from a **separate instrumented build**. '
           'Barrier-to-barrier interval totals include computation and synchronization. Category attribution splits shared intervals '
           'in proportion to phase work-item counts and is approximate; it is not isolated operator time or pure barrier overhead. '
           'The final trailing phases may not be timed. No raw trace, log or failure screenshot is published.\n\n'
           '| First cached tokens | Workers | Phases | Barrier intervals | Timed total µs | Untimed trailing phases |\n| ---: | ---: | ---: | ---: | ---: | ---: |\n')
    for p in phase_summary:text+=f'| {p["first_cached_tokens"]} | {p["workers"]} | {p["phases"]} | {p["barrier_intervals"]} | {p["timed_total_us"]:.2f} | {p["trailing_phases_untimed"]} |\n'
    path=target/'profiling.md';path.write_text(text);files.append(path)
    counter_path=folder/'counter-aggregate.json'
    if counter_path.exists():
        counters=json.loads(counter_path.read_text())
        if counters['state']!='complete' or counters['model_files']!=sg['model_files']:raise ValueError('Invalid counter follow-up')
        path=target/'gpu_counters.csv';write_csv(path,counters['measurements']);files.append(path)
        text+='\nSeparate Nsight Compute counter run (kernel replay can perturb cache/timing): '
        text+=f'`{counters["execution_commit"]}`. Counters describe the complete MegaBake kernel, not individual phases.\n\n'
        text+='| Cached tokens | Metric | Value | Unit |\n| ---: | --- | ---: | --- |\n'
        for r in counters['measurements']:text+=f'| {r["first_cached_tokens"]} | {r["metric"]} | {r["value"]:.6g} | {r["unit"]} |\n'
        (target/'profiling.md').write_text(text)
    return files
