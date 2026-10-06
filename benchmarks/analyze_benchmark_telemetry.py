#!/usr/bin/env python3
"""Summarize telemetry within Allen's throughput timer for each measured run."""
import argparse
import json
from pathlib import Path
import re
import statistics as stats


def cpu_nodes():
    nodes = {}
    for path in Path('/sys/devices/system/node').glob('node[0-9]*/cpulist'):
        node = int(path.parent.name[4:])
        for part in path.read_text().strip().split(','):
            lo, _, hi = part.partition('-')
            for cpu in range(int(lo), int(hi or lo) + 1):
                nodes[cpu] = node
    return nodes


def summary(values):
    return {'mean': stats.mean(values), 'min': min(values), 'max': max(values)} if values else None


def numeric(samples, field):
    values = []
    for sample in samples:
        try:
            values.append(float(sample['gpu'][field]))
        except (KeyError, ValueError):
            pass
    return summary(values)


def summarize_run(samples, node_for_cpu):
    out = {'timer_samples': len(samples)}
    for field, name in [('clocks.current.sm', 'sm_clock_MHz'),
                        ('clocks.current.memory', 'memory_clock_MHz'),
                        ('temperature.gpu', 'temperature_C'), ('power.draw', 'power_W'),
                        ('utilization.gpu', 'gpu_utilization_pct'),
                        ('utilization.memory', 'memory_utilization_pct')]:
        out[name] = numeric(samples, field)
    out['flag_active_fraction'] = {
        flag: stats.mean([s['gpu'].get('clocks_event_reasons.' + flag) == 'Active' for s in samples])
        for flag in ('sw_power_cap', 'hw_thermal_slowdown', 'hw_power_brake_slowdown')
    } if samples else {}
    out['max_gpu_compute_processes'] = max([len(s['gpu_processes']) for s in samples], default=0)
    load = [float(s['loadavg'].split()[0]) for s in samples if s['loadavg']]
    out['loadavg_1min'] = summary(load)
    nodes = sorted(set(node_for_cpu.values()))
    out['host_busy_pct_by_node'] = {}
    out['allen_thread_cpu_fraction_by_node'] = {}
    out['allen_memory_page_fraction_by_node'] = {}
    processes = [p for s in samples for p in s['processes'] if p['comm'] == 'Allen']
    placements = [cpu for p in processes for cpu in p['last_cpu_per_thread']]
    for node in nodes:
        busy = [value for s in samples for cpu, value in s['cpu_busy_pct'].items()
                if node_for_cpu.get(int(cpu)) == node]
        out['host_busy_pct_by_node'][str(node)] = summary(busy)
        if placements:
            out['allen_thread_cpu_fraction_by_node'][str(node)] = sum(node_for_cpu.get(c) == node for c in placements) / len(placements)
        fractions = [p['numa_pages'].get(str(node), 0) / sum(p['numa_pages'].values())
                     for p in processes if p['numa_pages']]
        if fractions:
            out['allen_memory_page_fraction_by_node'][str(node)] = stats.mean(fractions)
    per_pid = {}
    for p in processes:
        per_pid.setdefault(p['pid'], []).append(p)
    runtime = wait = 0
    for observed in per_pid.values():
        if len(observed) > 1:
            runtime += max(0, observed[-1]['runtime_ns'] - observed[0]['runtime_ns']) / 1e9
            wait += max(0, observed[-1]['runqueue_wait_ns'] - observed[0]['runqueue_wait_ns']) / 1e9
    out['observed_aggregate_thread_runtime_s'] = runtime
    out['observed_aggregate_thread_runqueue_wait_s'] = wait
    out['runqueue_wait_over_runtime'] = wait / runtime if runtime else None
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch_dir', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    groups = {}
    for line in (args.batch_dir / 'telemetry.jsonl').read_text().splitlines():
        try:
            sample = json.loads(line)
        except ValueError:
            continue  # permits inspecting a live batch with an incomplete last line
        phase = sample['phase']
        if (phase.get('phase') == 'start' and sample['timer_started']
                and not sample['processing_complete']):
            groups.setdefault((phase['repeat'], phase['role']), []).append(sample)
    nodes = cpu_nodes()
    runs = []
    for directory in sorted(args.batch_dir.glob('run_*')):
        rates = {}
        for line in (directory / 'rates.tsv').read_text().splitlines():
            key, value = line.split('\t')
            rates[key] = float(value)
        repeat = int(directory.name[4:])
        order = (directory / 'sequence_order.txt').read_text().splitlines()
        row = {'repeat': repeat, 'sequence_order': order, 'events_per_s': rates,
               'telemetry': {role: summarize_run(groups.get((repeat, role), []), nodes) for role in rates}}
        if rates.get('baseline') and 'pvs' in rates:
            row['loss_pct'] = 100 * (1 - rates['pvs'] / rates['baseline'])
        for role in rates:
            log = (directory / ('bench_' + role + '.log')).read_text()
            match = re.search(r'Ran test for ([\d.]+)', log)
            row['telemetry'][role]['allen_timer_s'] = float(match[1]) if match else None
        runs.append(row)
    result = {'batch_dir': str(args.batch_dir.resolve()), 'runs': runs,
              'notes': ['Only samples inside Allen throughput timer are included.',
                        'NUMA page fractions include all resident process mappings.',
                        'Aggregated scheduling deltas are approximate when threads exit.']}
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    for r in runs:
        print('repeat', r['repeat'], 'rates', r['events_per_s'], 'loss', r.get('loss_pct'))
        for role, t in r['telemetry'].items():
            print(' ', role, 'samples', t['timer_samples'], 'SM MHz', t['sm_clock_MHz'],
                  'CPU nodes', t['allen_thread_cpu_fraction_by_node'],
                  'memory nodes', t['allen_memory_page_fraction_by_node'])


if __name__ == '__main__':
    main()
