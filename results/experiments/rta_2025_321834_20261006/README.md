# RTA-recommended 2025 throughput input

On 6 October 2026, RTA supplied `hlt1_input_data_2025_run_321834_mdf`, and the
user confirmed that 2025 data is acceptable. This resolves the input selection
left open by the [initial 2026 search](../input_2026_20261006/README.md).

The exact file is mirrored on CVMFS; authenticated EOS access is unnecessary:

```text
/cvmfs/lhcbdev.cern.ch/testfiledb-mirror/rta/samples/data/321834-LHCb-MEP/MEP_2025_pp_pD2_bu_321834_LHCb_ECEB01_BU_0.mdf
```

The [official TestFileDB entry](https://gitlab.cern.ch/lhcb-datapkg/PRConfig/-/blob/59e4a28c074f03398f138bdfc1b39c616d73cf6a/python/PRConfig/TestFileDB.py#L6992)
describes unbiased real data from run 321834, full machine with average mu
5.26 plus Deuterium, converted from MEP to MDF with a passthrough sequence.
It specifies `run3/2025-v00.01` geometry and `master` conditions.

The current vendored Allen `scripts/ci/test_config.yaml` pairs this same
dataset with `geometry_run3_2025-v00.01` for `hlt1_pp_default` throughput:

```text
/cvmfs/lhcb.cern.ch/lib/lhcb/ALLEN/ALLEN_v9r2/input/allen_geometries/geometry_run3_2025-v00.01
```

All 15 local geometry files match this CVMFS pair byte for byte
([manifest](geometry_manifest.json)). This is the documented upstream CI
pair; no new conditions dump was generated. The first 500 MDF records and
their ODIN banks all identify run 321834 ([metadata](input_metadata.json)).
The file contains 2,008,416,688 bytes.

## Execution and numerical checks

The shared builder completed successfully. The current GPU build then ran
the full `hlt1_pp_pvs_pvfinder_unet_benchmark` chain on 500 distinct events,
with 118,002 VELO tracks: HLT1 plus FC, UNet, peak finding and PV fitting.
As usual, the numerical dump repeats the slice twice.

FC, UNet, full-model BF16 agreement and peak finding passed their existing
checks. The feature validator **failed** its absolute POCA threshold for one
track: 0.0010324 mm difference, compared with the 0.001 mm limit. That track
has z approximately −15,078 mm, far outside the network input range; the
float32 spacing there is 0.0009766 mm. The maximum difference for tracks
within −100 to 300 mm is 0.00006345 mm.
The outlier is absent from the dumped FC work list and does not enter the network.

[The outlier diagnostic](feature_outliers.json) and all five numerical
reports preserve this outcome. The tolerance and GPU implementation are
unchanged; this is not an all-checks-PASS validation result. The ellipsoid
matrix and training-rule checks passed.

## Workflow selection

Throughput and numerical dumps now default to this real-data sample and
the matching CVMFS geometry. Existing `MDF_FILE` and `GEOMETRY_DIR` overrides
remain available. This changes the input, not the HLT1 sequence, model,
precision or accepted work-list/grid settings.

Truth-based physics checks retain the 2024 minimum-bias MC reference because
real data has no MC truth. `MC_MDF_FILE` and `MC_GEOMETRY_DIR` select an
alternative MC pair; general input overrides remain supported too. An
upstream import verification retains the fixed MC regression fixture for
numerical checks and its MC reference counts for physics checks.

## Throughput measurement

Five paired repeats completed at the existing production settings:
GPU 2 (RTX 3090), 500 events, 500 MB per stream, 1,000 repetitions and
16 streams. The comparison retains the baseline PV finder and adds the
full PVFinder shadow chain. Historical 2024 throughput numbers remain tied
to that sample and must not be relabeled as 2025 results.

| Repeat | Baseline events/s | Full PVFinder shadow events/s | Loss |
|---|---:|---:|---:|
| 1 | 69,737.6 | 64,039.7 | 8.17% |
| 2 | 66,950.7 | 66,776.1 | 0.26% |
| 3 | 69,397.9 | 67,925.3 | 2.12% |
| 4 | 77,627.9 | 74,159.9 | 4.47% |
| 5 | 77,263.3 | 73,363.1 | 5.05% |

The median paired loss is **4.47%**. The mean is **4.01%**, with sample
standard deviation **3.01 percentage points**. A Student-t 95% interval
across these five paired repeat losses is **0.28–7.75%**. This describes
repeat variability on the same 500-event slice, not uncertainty across all
collision conditions.

Baseline throughput spread is **15.31%**, so the run recorder marks the batch
`contended`. No competing GPU 2 compute process was observed at the start or
in the saved [process probe](gpu_process_probe.csv); that spread-based label
does not establish its cause. A [clock/power probe](gpu_probe.csv) is also
retained. There were no slice splits. **This batch does not establish a
dependable margin below 5%.** A more controlled timing study is needed before
using that claim.

The [run record](../../runs/20261006_101221_rtx3090_unet16-lc4-scnone-asym5-best-bf16_rta-2025-run321834.json)
captures the new dataset, geometry, effective configuration and clean source
commit `babae82cb1`. [Paired statistics](throughput.json),
[the batch summary](benchmark_summary.md), and
[the command](benchmark_command.cmd) preserve the measurement. The existing
mtime heuristic flags `CMakeLists.txt` and `PVFinderModel.h` relative to the
older host library link time; the shared builder completed successfully
before the smoke test and measurement, and no Allen source changed during
this migration. [Compatibility provenance](compatibility.json) records the
executable and model hashes.
