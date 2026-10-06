# Investigating RTA-sample timing variance

The initial [five-pair 2025 measurement](../rta_2025_321834_20261006/README.md)
had 15.31% baseline spread and paired losses from 0.26% to 8.17%.
This investigation repeats the same data, geometry, model and operating point
with GPU/host telemetry and alternating sequence order.

Initial probe: GPU 2 was idle, with no compute process. Host load was about
79 on 96 logical CPUs. GPU 2 is attached to NUMA node 1, whose CPUs are
`24-47,72-95`; the benchmark normally allows CPUs `0-95`.
These observations suggest host scheduling and placement as hypotheses,
alongside GPU boost/power/thermal behavior. They do not establish a cause.

The shared benchmark runner now supports:

- `--alternate-order`: reverse sequence order on even repeats.
- `--telemetry`: sample GPU state and host scheduling every two seconds;
  line-buffer Allen output to distinguish startup from its throughput timer.
- `--cpu-affinity LIST`: bind Allen's CPU threads for a placement control.

The first diagnostic batch uses four pairs, alternating baseline→PVFinder
and PVFinder→baseline, with `-n 500 -m 500 -r 1000 -t 16`. No GPU setting or
inference implementation is changed. A CPU-affinity control will follow if
the telemetry warrants it. Numerical correctness and the existing out-of-window
POCA diagnostic are unchanged.

Summarize a diagnostic batch (including a batch still running) with:

```bash
python3 benchmarks/analyze_benchmark_telemetry.py BATCH_DIR --report REPORT.json
```

The report includes only samples inside Allen's throughput timer, excluding
startup and shutdown. CPU placement describes the last CPU reported for all
Allen threads, including idle threads; resident-page fractions cover all process
mappings. These are placement indicators, not measurements of memory traffic.
Scheduling deltas are approximate when threads exit. GPU telemetry is sampled
every two seconds and can miss brief interference.

## Repeated unbound measurement

Four valid pairs use the same executable, library and model as the original
five-pair measurement; [SHA-256 checks](binary_model_identity.json) confirm
their identity. Input, geometry, stream count, repetitions, precision and grid
settings are unchanged. The sequence orders are baseline→PVFinder,
PVFinder→baseline, baseline→PVFinder, PVFinder→baseline.

| Pair | Baseline events/s | PVFinder shadow events/s | Loss |
|---|---:|---:|---:|
| 1 | 78,160.7 | 74,220.8 | 5.04% |
| 2 | 77,641.3 | 74,159.5 | 4.48% |
| 3 | 77,667.8 | 73,469.8 | 5.41% |
| 4 | 77,939.4 | 74,477.8 | 4.44% |

Baseline spread is **0.67%**, compared with **15.31%** originally. Mean paired
loss is **4.84%**, median **4.76%**, and sample standard deviation **0.46
percentage points**, compared with **3.01** originally. A descriptive
Student-t 95% interval for mean loss is **4.11–5.58%**; shared-host drift and
only four repeats limit its interpretation. This does **not** establish a
dependable margin below 5%.

The [first diagnostic batch](../../runs/20261006_105549_rtx3090_unet16-lc4-scnone-asym5-best-bf16_variance-unbound.json)
stopped after the fourth PVFinder run printed the integer rate `74187 events/s`:
the pre-existing parser required a decimal point. Allen completed successfully,
but no fourth baseline ran. The parser now accepts integer, decimal and
scientific notation and reports a clear extraction error if none is found.
The failed record is preserved. A [fresh PVFinder-first pair](../../runs/20261006_111411_rtx3090_unet16-lc4-scnone-asym5-best-bf16_variance-unbound-completion.json)
provides pair 4; the unpaired 74,187 events/s is not used in loss statistics.
[Combined data](unbound.json) retain each pair's source batch and repeat number.

Across the four measured pairs, Allen's sampled thread placement is overwhelmingly
on NUMA node 0; GPU 2 is on node 1. Most resident memory pages are also on node 0.
There is one observed GPU 2 compute process, Allen, throughout the timed samples.
Software power limiting is active almost continuously, with no hardware thermal
or power-brake slowdown flag. Per-run average SM clocks are approximately
1,767–1,778 MHz. Scheduling wait is small relative to Allen thread runtime.
These observations identify a placement issue and normal power-limited GPU boost,
but do not establish the cause of the original batch's large variance.

Baseline-first pairs average **5.22%** loss and PVFinder-first pairs **4.46%**.
Only two pairs have each order, so this is an order-associated difference,
not proof of a reproducible order effect. Keep both orders in timing comparisons.

Raw telemetry is preserved as gzip archives alongside compact reports; the
[manifest](telemetry_manifest.json) records original paths and SHA-256 hashes.
Raw logs and effective configurations remain in the recorded batch directories.

## GPU-local CPU control

Two more pairs bind Allen to `24-47,72-95`, the CPU socket attached to GPU 2,
and alternate sequence order. All sampled Allen threads are on node 1, and
98.2–99.9% of resident pages are there. All other benchmark settings are unchanged.

| Pair / order | Baseline events/s | PVFinder shadow events/s | Loss |
|---|---:|---:|---:|
| 1 / baseline first | 78,123.5 | 74,355.1 | 4.82% |
| 2 / PVFinder first | 76,540.7 | 74,303.5 | 2.92% |

PVFinder throughput changes by only **0.07%** between these pairs, while the
baseline falls by **2.03%**. The lower apparent loss in pair 2 is therefore
mainly a reference slowdown, not a PVFinder speedup. The overall pinned mean
loss of **3.87%** must not be presented as an optimization gain or evidence of
a dependable margin below 5%.

The slower baseline has **6.35%** aggregate CPU runqueue wait relative to thread
runtime, compared with **1.97%** in the first pinned baseline. Both runs have
about **91%** logical-CPU utilization on node 1. The second baseline's average
GPU clock is higher, **1,778 MHz** versus **1,770 MHz**, with no competing GPU 2
process or hardware thermal/power-brake flag observed in either timer window.
This makes host scheduling/reference drift a plausible source of the changed
loss. The scheduling correlation is not proof of causation, and cannot establish
what happened during the original uninstrumented batch.

Binding to GPU-local CPUs changes placement but gives no clear absolute
throughput gain in this comparison. Keep CPU affinity an explicit diagnostic
option. Further measurements should retain balanced ordering and telemetry;
reserved host CPU resources as well as a quiet GPU would help separate host
contention from implementation cost.

The [control record](../../runs/20261006_111926_rtx3090_unet16-lc4-scnone-asym5-best-bf16_variance-local-cpu.json)
and [timer-window report](local_cpu.json) preserve the outcome.
Effective configurations differ only in ordering of dependency lists generated
from Python sets. Normalizing those lists gives identical configurations, including
all algorithm properties and execution order, across the original and repeated
runs ([checks](config_identity.json)). Allen's scheduler uses dependency membership,
not that list order.

## Comparison and reproduction

| Batch | Pairs | Baseline median (k events/s) | PVFinder median (k events/s) | Baseline spread | Mean loss | Loss SD (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Original | 5 | 69.74 | 67.93 | 15.31% | 4.01% | 3.01 |
| Instrumented, unbound | 4 | 77.80 | 74.19 | 0.67% | 4.84% | 0.46 |
| GPU-local CPUs | 2 | 77.33 | 74.33 | 2.05% | 3.87% | 1.34 |

[Comparison plot](variance.png) ([PDF](variance.pdf)) shows absolute rates alongside
paired loss; [statistics](comparison.json) include descriptive intervals.
Reproduce these artifacts with:

```bash
MPLCONFIGDIR=/tmp/pvfinder-matplotlib .venv/bin/python3 \
    results/experiments/variance_20261006/make_report.py
```

The large original spread did not recur in the unbound repeat. The CPU-placement
control shows how a shifting baseline can still change apparent loss substantially.
Host scheduling is the strongest observed lead in this study; the exact original
cause remains unresolved. There is no evidence in these timed samples of a competing
GPU 2 compute job or hardware thermal throttling. Software power limiting is active.
The inference implementation, numerical tolerances and production operating point
are unchanged.
