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
