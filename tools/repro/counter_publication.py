"""Clean successful NCU hardware measurements into a performance-only table."""
import csv
import io
import json
import math


def extract(folder):
    status=json.loads((folder/'status.json').read_text())
    validation=json.loads((folder/'counter-validation.json').read_text())
    if status['state']!='complete' or validation['state']!='complete':raise ValueError('Counter run failed')
    text=(folder/'ncu.csv').read_text();lines=text.splitlines()
    start=next(i for i,s in enumerate(lines) if s.startswith('"ID","Process ID"'))
    rows=[]
    aliases={'Block Size':'launch__block_size','Grid Size':'launch__grid_size',
             'Registers Per Thread':'launch__registers_per_thread',
             'Dynamic Shared Memory Per Block':'launch__shared_mem_per_block_dynamic',
             'Static Shared Memory Per Block':'launch__shared_mem_per_block_static'}
    for row in csv.DictReader(io.StringIO('\n'.join(lines[start:]))):
        if row.get('Metric Value') is None:continue
        try:index=int(row['ID']);value=float(row['Metric Value'].replace(',',''))
        except (ValueError,TypeError):continue
        if index not in (0,1) or not math.isfinite(value):continue
        metric=aliases.get(row['Metric Name'],row['Metric Name'])
        if metric not in ('gpu__time_duration.sum','dram__bytes.sum','dram__throughput.avg.pct_of_peak_sustained_elapsed',
                          'sm__throughput.avg.pct_of_peak_sustained_elapsed','sm__warps_active.avg.pct_of_peak_sustained_active',
                          'launch__registers_per_thread','launch__shared_mem_per_block_static','launch__shared_mem_per_block_dynamic',
                          'launch__block_size','launch__grid_size'):continue
        rows.append(dict(first_cached_tokens=validation['contexts'][index],metric=metric,unit=row['Metric Unit'],value=value))
    required={'gpu__time_duration.sum','dram__bytes.sum','dram__throughput.avg.pct_of_peak_sustained_elapsed',
              'sm__throughput.avg.pct_of_peak_sustained_elapsed','sm__warps_active.avg.pct_of_peak_sustained_active'}
    if any(not required.issubset({r['metric'] for r in rows if r['first_cached_tokens']==context}) for context in (128,4207)):
        raise ValueError('Missing hardware counters')
    return dict(state='complete',execution_commit=status['execution_commit'],model_files=validation['model_files'],measurements=rows)
