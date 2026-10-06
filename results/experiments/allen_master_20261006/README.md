# Rebase onto Allen master, 6 October 2026

Upstream: `b567e104f23ebfd0670d6272e47d9b1ce37d905c`, fetched from CERN on
6 October 2026 (423 commits beyond v9r1). The `allen/pvfinder` branch has
61 integration files across two commits on top of this upstream revision.
`benchmarks/allen_upstream.json` records both revisions and the vendored tree.

The initial rebase required three conflict resolutions: ignore rules, standalone
model initialization and the Gaudi wrapper's initialization order. Network
kernels and the accepted GPU work-list implementation are unchanged. The
standalone loader still reads models after configuring properties and before
initializing algorithms; upstream output-manager and prefetch changes remain.
The Gaudi wrapper keeps upstream's algorithm-name initialization.

Current master removed the standalone MC checker framework. Our PV dump now
reads MC-PV raw-bank payloads directly and preserves the existing vertex/MC
binary format. Offline `compare_pvs.py` applies the same matching rules. The
reader supports the benchmark's MDF layout; its unit tests cover empty payloads,
coordinates, track counts and invalid/truncated payloads.

## Completed isolated checks

- CUDA 12.8/LCG 108c/cuDNN build: PASS.
- GPU unit tests: 14 cases, 314 assertions PASS.
- All five numerical validators: PASS for BF16 and FP32, on 500 events and
  122,077 tracks.
- Both 10,000-event physics working points: vertex counts and reconstructed-PV/MC
  event payloads match v9r1 exactly, including the beamline reference. One mean
  differs by 1.11e-16 micrometres due to floating-point summation order.
- The reusable upstream-validation command passed end to end.
- Update-tool tests exercise successive rebases, local edits, a post-rebase API
  adaptation, manual conflict resolution, validation rejection, existing source
  collisions, ancestry preservation and preservation of local input data.

`preview_verification.json`, `preview_validation/`, `preview_fp32_validation/`
and `preview_physics/` retain the first checks. `automated_validation/` records
the reusable validator's run. Raw logs/dumps and the isolated checkout remain
in `benchmark_results/allen_master_rebase20261006/` and
`/tmp/pvfinder-current-master20261006/`. The committed source and upstream
ancestry are retained by the import merge and the `allen/pvfinder` branch.
The Gaudi stack build was not tested.

## Future updates

Follow the root README's `allen_upstream.py prepare`, build, validation and
import commands. Preparation captures committed changes from `Allen/` before
rebasing. Import preserves upstream ancestry as a merge parent and excludes
`input/`, keeping the existing local sample and geometry. Conflicts and API
changes still require adaptation and validation; upstream updates no longer
require reconstructing our integration from a copied source tree.

## Main checkout and unified tools

The main checkout rebuilt successfully using `benchmarks/build_allen.sh`.
All 1,643 tracked vendored source files match the validated isolated checkout.
The rebuilt GPU tests passed 14 cases and 314 assertions. The documented
`make -C weights dump validate` command passed all five numerical validators.
The default physics command passed both 10,000-event working points; all
physics summaries match the isolated checks to 1e-12 floating-point tolerance,
with integer counts identical. `main_validation/`, `main_physics/` and
`main_unit_tests.txt` retain the checks.

Build, benchmark, physics and weights tools use `Allen/build/` and shared
settings in `benchmarks/defaults.sh`. The version-specific build and benchmark
helpers have been removed; `ballen` delegates to the one build script. The
model catalog records architecture and checkpoint provenance without separate
build names. Active docs describe only this workflow; historical experiment
records retain their original commands.

On this existing checkout, the ignored `Allen/build` alias uses the completed
build at `Allen/buildupstreamgpu12`; the builder resolves the path before
configuring CMake. `.cache/allen-dependencies` references the existing dependency
source cache. A fresh checkout creates a real `Allen/build/` and needs its
dependency cache configured as documented. Other old build folders are not
used by the default tools.
