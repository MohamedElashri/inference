#!/usr/bin/env python3
"""Reproduce comparison.json and the variance figure from saved measurements."""
import json
import math
import os
from pathlib import Path
import statistics as stats


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def summarize(rows):
    rates = {role: [r['events_per_s'][role] for r in rows] for role in ('baseline', 'pvs')}
    losses = [100 * (1 - r['events_per_s']['pvs'] / r['events_per_s']['baseline']) for r in rows]
    n = len(losses)
    result = {'pairs': n, 'losses_pct': losses, 'mean_loss_pct': stats.mean(losses),
              'median_loss_pct': stats.median(losses),
              'loss_sample_sd_pp': stats.stdev(losses) if n > 1 else None,
              'loss_range_pct': [min(losses), max(losses)],
              'median_events_per_s': {k: stats.median(v) for k, v in rates.items()},
              'baseline_spread_pct': 100 * (max(rates['baseline']) - min(rates['baseline'])) / stats.median(rates['baseline']),
              'baseline_sample_cv_pct': 100 * stats.stdev(rates['baseline']) / stats.mean(rates['baseline']) if n > 1 else None}
    # Two-sided Student-t critical values for the three batch sizes in this experiment.
    critical = {2: 12.7062047364, 4: 3.1824463053, 5: 2.7764451052}.get(n)
    if critical:
        half_width = critical * stats.stdev(losses) / math.sqrt(n)
        result['approx_mean_loss_95pct_t_interval_pct'] = [stats.mean(losses) - half_width, stats.mean(losses) + half_width]
    return result


def main():
    original_path = ROOT / 'results/runs/20261006_101221_rtx3090_unet16-lc4-scnone-asym5-best-bf16_rta-2025-run321834.json'
    original = json.loads(original_path.read_text())
    unbound = json.loads((HERE / 'unbound.json').read_text())
    local = json.loads((HERE / 'local_cpu.json').read_text())
    batches = [('Original', original['results']['repeats']),
               ('Instrumented, unbound', unbound['runs']),
               ('GPU-local CPUs', local['runs'])]
    result = {'original_record': str(original_path.relative_to(ROOT)),
              'batches': {label: summarize(rows) for label, rows in batches},
              'notes': ['All batches use the same GPU, input, geometry, inference settings and built binary.',
                        'Later batches include telemetry and alternate sequence order.',
                        'CPU-local control changes affinity, which can also change first-touch memory placement.',
                        'Student-t intervals are descriptive; few repeats and shared-host drift limit inference.',
                        'The original batch had no timed telemetry; its specific variance cause cannot be reconstructed.']}
    (HERE / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print('| Batch | Pairs | Baseline median (kHz) | PVFinder median (kHz) | Baseline spread | Mean loss | Loss SD (pp) |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for label, _ in batches:
        s = result['batches'][label]
        print(f"| {label} | {s['pairs']} | {s['median_events_per_s']['baseline']/1000:.2f} | {s['median_events_per_s']['pvs']/1000:.2f} | {s['baseline_spread_pct']:.2f}% | {s['mean_loss_pct']:.2f}% | {s['loss_sample_sd_pp']:.2f} |")

    os.environ.setdefault('MPLCONFIGDIR', '/tmp/pvfinder-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    x = 0
    for group, (label, rows) in enumerate(batches):
        positions = list(range(x + 1, x + 1 + len(rows)))
        for role, marker, color, title in [('baseline', 'o', '#2060a0', 'HLT1 baseline'),
                                           ('pvs', 's', '#c56020', 'HLT1 + PVFinder shadow chain')]:
            axes[0].plot(positions, [r['events_per_s'][role] / 1000 for r in rows], marker=marker, color=color,
                         label=title if group == 0 else None)
        axes[1].plot(positions, result['batches'][label]['losses_pct'], 'o-', color='#6845a0')
        axes[0].text(sum(positions) / len(positions), 1.02, label, ha='center', transform=axes[0].get_xaxis_transform())
        if group:
            for ax in axes:
                ax.axvline(x + .5, color='#888888', linestyle=':', linewidth=1)
        x += len(rows)
    axes[0].set_ylabel('Throughput (k events/s)')
    axes[0].legend(loc='lower right', fontsize=9)
    axes[1].axhline(5, color='#a03030', linestyle='--', linewidth=1, label='5% target')
    axes[1].set_ylabel('Paired throughput loss (%)')
    axes[1].set_xlabel('Pair in chronological order (gaps between batches omitted)')
    axes[1].set_xticks(range(1, x + 1))
    axes[1].legend(loc='best', fontsize=9)
    for ax in axes:
        ax.grid(axis='y', alpha=.25)
    fig.savefig(HERE / 'variance.png', dpi=180)
    fig.savefig(HERE / 'variance.pdf')


if __name__ == '__main__':
    main()
