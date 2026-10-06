# 2026 input and geometry availability

Checked on 6 October 2026 following RTA feedback that the 2024 minimum-bias
input should be replaced. **2026 data and geometry are available on CVMFS,
but an equivalent unselected, 6.8 TeV beam-energy Run 3 minimum-bias sample
was not found in the sources checked.** The benchmark default is unchanged
until a representative replacement is identified.

The checks covered the CVMFS TestFileDB mirror, released PRConfig v1r116,
and [PRConfig master at 59e4a28c](https://gitlab.cern.ch/lhcb-datapkg/PRConfig/-/blob/59e4a28c074f03398f138bdfc1b39c616d73cf6a/python/PRConfig/TestFileDB.py).
[Catalog entries](catalog_candidates.json) retain the relevant metadata and
file locations. This search does not establish that a suitable sample is
absent from private or authenticated EOS storage.

| Input | Availability | Consequence |
|---|---|---|
| `2026-hlt1-mdf` | Four real-data MDF files on CVMFS, about 7.8 GiB; runs 341725–341737 | Already HLT1-selected; useful for compatibility, but representativeness for HLT1 throughput needs RTA agreement. No MC truth. |
| `expected_2026_pp1200GeV_NOSMOG_mdf_10kevents` | Unselected minimum-bias MC, 10,000 events, MagDown; MDF and matching standalone geometry available | 1.2 TeV beam energy, so it does not replace the historical 6.8 TeV sample for a like-for-like performance comparison. |
| `hlt2_input_data_2026_run_343424` | One mirrored real-data MDF, about 2 GiB | HLT1-filtered data with Neon; not an unselected pp minimum-bias replacement. |
| `upgrade2_minbias_*Jan2026` | Some DIGI files mirrored | Upgrade II / Run 5 geometry; not Run 3 input. |

## Matching geometry and compatibility checks

The low-energy MC geometry is the exact pair used by the upstream Allen CI
configuration, not just a directory with the same year in its name:

```text
/cvmfs/lhcbdev.cern.ch/testfiledb-mirror/lhcb/swtest/2026-converted_MC-pp1200GeV-NOSMOG/Converted_MC_pp1200GeV_NOSMOG_10kevents_00372315_00000001_1.mdf
/cvmfs/lhcb.cern.ch/lib/lhcb/ALLEN/ALLEN_v9r2/input/allen_geometries/geometry_2026-v00.00_sim10-2026.W17-v00.00-md100
```

The existing GPU build ran `pvfinder_pv_validation` on 500 distinct events,
containing 22,357 VELO tracks. The full-model BF16 numerical validator and
peak validator passed. MC PV banks were readable, with 807 reconstructible
MC PVs. The dump runner repeats the slice twice, so its PV files contain
1,000 records; these are not 1,000 independent events.

For real data, CVMFS has the DD4hep geometry source in
`DETECTOR_v3r18/compact/run3/2026-v00.00` and the conditions database.
The released Allen/v9r2 CPU stack successfully dumped standalone geometry
using `run3/2026-v00.00`, conditions `master`, and the first event of
`2026-hlt1-mdf/data-0000.mdf`. The run-specific dump is installed locally at:

```text
Allen/input/allen_geometries/geometry_2026-v00.00_run341725_master20261006
```

The first 500 MDF records and their ODIN banks all identify run 341725
([metadata check](real_first500_metadata.json)). Using this dump, the
existing GPU build ran the FC and UNet chain on 500 real events with
180,247 VELO tracks. The full-model BF16 numerical validator passed.
This test did not exercise the full HLT1 sequence or PV fitting. Upstream's
real-2026 CI test uses a sequence without UT; our production benchmark
sequence currently includes UT and downstream reconstruction.

[Availability and geometry hashes](availability.json), numerical reports,
smoke logs and build snapshots preserve the checks. The snapshots flag
two source timestamps newer than `libAllenLib.so`; no tracked Allen source
was changed in this investigation. These results establish compatibility,
not a new certified throughput or physics result.

The real-data geometry dump recipe is preserved in
[the Python configuration](pvfinder_2026_dump_geometry.py) and
[its options](pvfinder_2026_dump_geometry.yaml). On this host, regenerate it
from the repository root with:

```bash
PYTHONPATH="$PWD/results/experiments/input_2026_20261006:${PYTHONPATH:-}" \
    /cvmfs/lhcb.cern.ch/lib/var/lib/LbEnv/3985/stable/linux-64/bin/lb-run \
    -c x86_64_v3-el9-gcc15-opt+g Allen/v9r2 \
    lbexec pvfinder_2026_dump_geometry:main \
    results/experiments/input_2026_20261006/pvfinder_2026_dump_geometry.yaml
```

This generates the dump under `benchmark_results/input_2026_20261006/real_geometry`.
The real-data dump is specific to run 341725 and the recorded conditions
revision; it should not be applied indiscriminately to other runs.

## Selecting input for the existing workflows

Throughput, physics comparisons and weights dumps now read `MDF_FILE` and
`GEOMETRY_DIR` from the shared defaults. Always set both to a matched pair.
For a real-data FC/UNet compatibility check:

```bash
MDF_FILE=/cvmfs/lhcbdev.cern.ch/testfiledb-mirror/lhcb/swtest/2026-hlt1-mdf/data-0000.mdf \
GEOMETRY_DIR="$PWD/Allen/input/allen_geometries/geometry_2026-v00.00_run341725_master20261006" \
    bash weights/scripts/allen_dump.sh \
    --model-file "$PWD/weights/out/unet16_lc4_scnone_asym5_best_bf16/pvfinder_model.json" \
    --sequence pvfinder_unet --events 500 --device 2 \
    --dump-dir "$PWD/benchmark_results/real_2026_check" \
    --set pvfinder_fc_aggregation.precision=bfloat16 \
    --set pvfinder_unet.precision=bfloat16 \
    --set pvfinder_fc_aggregation.gpu_work_list=true \
    --set pvfinder_fc_aggregation.fused_grid_fraction=0.0625 \
    --set pvfinder_unet.fused_grid_fraction=0.125
```

Real data cannot provide the MC efficiency and false-positive measurements
in `pv_comparison.sh`. The upstream import validator also retains its fixed
2024 MC reference counts, which must be updated when its reference sample
changes.

## Remaining input requirement

The next step is to obtain RTA's recommended unselected, full-energy 2026
sample: a TestFileDB key or MDF path plus matching geometry/conditions.
If real data is requested for throughput, an appropriate unselected sample
is needed, while MC remains necessary for truth-based physics validation.

Listing `/eos/lhcb/wg/rta/WP6/Allen/geometries` failed with server error
3010, unauthorized identity ([access result](eos_geometry_access.txt)).
No usable CERN Kerberos ticket was present. Authenticated EOS access may
reveal additional geometry dumps or samples; it was not available for this
check. No performance claim or slide result has been updated.
