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

Five paired repeats at the existing production settings are the next check:
GPU 2 (RTX 3090), 500 events, 500 MB per stream, 1,000 repetitions and
16 streams. The comparison retains the baseline PV finder and adds the
full PVFinder shadow chain. Historical 2024 throughput numbers remain tied
to that sample and must not be relabeled as 2025 results.
