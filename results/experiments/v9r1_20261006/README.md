# Allen v9r1 optimization results

Final shadow loss: **4.648 ± 0.570%** (mean ± sample standard deviation, five paired repeats).
95% confidence interval for the mean across repeat variation: 3.940–5.355%.
Mean below 5%: True; upper confidence bound below 5%: False.

## Long confirmation

Same 500-event slice, 1,000 repetitions, 16 streams, RTX 3090, BF16 checkpoint, CUDA 12.8/LCG 108c. Candidate order alternated across repeats. No slice splitting was observed. GPU availability was checked at the start; process/power telemetry was recorded from the third repeat onward.
Loss = 100 × (1 − candidate throughput / paired HLT1 throughput). The intervals describe repeat variation on this fixed workload, not uncertainty across samples or GPU types.

| Configuration | Mean events/s | Paired loss, mean ± SD |
|---|---:|---:|
| HLT1 baseline | 86588 | 0.000 ± 0.000% |
| Original port, selected grid | 80769 | 6.718 ± 0.497% |
| Rebuilt host work list, selected grid | 80971 | 6.486 ± 0.404% |
| GPU work list, selected grid | 82562 | 4.648 ± 0.570% |

GPU throughput gain over Original port, selected grid: **2.219 ± 0.126%**, 95% CI 2.062–2.375%.
GPU throughput gain over Rebuilt host work list, selected grid: **1.965 ± 0.265%**, 95% CI 1.637–2.294%.

The earlier v9r1 result with the original FC 1/8 and UNet 1/4 limits was 7.639 ± 0.140%. That is a separate five-repeat campaign; the matched controls above isolate the code change in the current campaign.

## Ordered experiments and retained changes

- Fitter block_dim_y = 2 or 8: no repeatable gain; keep 4.
- UNet 2/3/4 warps and maximum shared-memory preference: no repeatable gain; revert the experimental templates/properties.
- GPU work list: accepted by the paired throughput checks against both controls.
- Selected grid limits: FC 0.0625, UNet 0.125. The short grid screen was independent of the long confirmation.

The GPU path builds compact rows and size-bucketed FC work items after the CSR kernel without a host round trip. Split-slot partials still sum in canonical chunk order. The host-built path remains available through gpu_work_list=false.

## Correctness

- Existing GPU unit tests: 12 cases, 299 assertions PASS.
- All five numerical validators PASS on 500 events/122,077 tracks.
- Coverage includes 14,550 empty intervals and 566 split intervals.
- Compute Sanitizer memcheck: zero errors, two streams, two repetitions.
- Full 10,000-event physics counts unchanged at thresholds 0.07 and 0.1: True.
- Host/GPU fitted vertices, KDE dump and seeds byte-identical at both working points; counts also match the initial v9r1 port.

Sample: 2024 minimum-bias MC, MagDown, nu7.6, 6.8 TeV beam, benchmark MDF/geometry. Runtime benchmarking uses its first 500 events; physics validation uses all 10,000.

## Tracked evidence and reproduction

The accepted source from isolated commit `b427d449b0c393734e361279e3922aa8aebea57c`
has been transferred to this repository's `Allen/` tree. Its upstream base is
Allen v9r1 (`9352e3650eed34142288e3fbc6ef9154973135da`). `origin.json` records
the import provenance; the historical run records retain their original paths
and git states rather than being relabelled as runs of master.

From the repository root:

```bash
bash benchmarks/build_v9r1.sh
bash benchmarks/benchmark_v9r1_optimized.sh
```

The first command builds `Allen/buildv9r1gpu12/Allen` and the GPU unit tests.
The second generates a fresh configuration with this checkout's model paths
and repeats the accepted BF16 full-PV shadow operating point, including the
GPU work list and the FC/UNet grid limits above. Models, input data, cuDNN and
the dependency sources are local prerequisites described in the root README.

- `measurements/`: measured rates, exact commands and configuration deltas for
  every ordered experiment and the long confirmation.
- `configurations/`: two complete canonical configurations. Each
  `*_config.delta.json` reconstructs its original configuration as a JSON
  object, without duplicating the full HLT1 graph for every experiment. For
  example:

  ```bash
  python3 results/experiments/v9r1_20261006/materialize_config.py \
    results/experiments/v9r1_20261006/measurements/confirmation/gpu_final_config.delta.json \
    /tmp/v9r1_measured.json
  ```

  These snapshots retain the original isolated paths. `original_sha256`
  identifies the original raw JSON file; reconstructed formatting may differ.
- `source_variants/`: accepted GPU work-list patch and rejected UNet variants,
  with manifests identifying their original source base.
- `validation/` and `physics/`: numerical reports, unit-test and memcheck output,
  physics comparisons, and host/GPU byte-comparison results.
- `confirmation_checks.json`, `decisions.json`, `selection.json` and the audit
  files: selection criteria, paired statistics, GPU telemetry and diagnostics.
- `accepted_manifest.json` and `restore_checks.json`: the source/model hashes
  and successful original archive restoration check (1,805 Allen files and
  four model/checkpoint files). That check concerned archive contents, not a
  rebuilt restore.

The larger original source/binary/model archives and raw logs/dumps remain in
the ignored `local_scripts/rebase_v9r1/optimization/` directory, named in
`origin.json`. Compact historical run records are also tracked under
`results/runs/`. Fresh verification of the main-repository rebuild is recorded
separately in `master_verification.json`.

The settings have been measured on the RTX 3090 only. This is the full PV shadow configuration, with baseline PV reconstruction still present. No claim is made here for other GPUs, samples, or replacement configurations.

The earlier tensor-instruction audit found no other tensor-core algorithm in this benchmarked v9r1 HLT1 build. The experiments do not provide a complete causal attribution of the v7r9-to-v9r1 regression.
