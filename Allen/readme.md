# Allen — LHCb GPU HLT1 Trigger

**Depends on:** Rec (→ Lbcom → LHCb → Gaudi, Detector)

[![pipeline status](https://gitlab.cern.ch/lhcb/Allen/badges/master/pipeline.svg)](https://gitlab.cern.ch/lhcb/Allen/-/commits/master)

GPU-accelerated first-stage trigger (HLT1) for LHCb Run 3. Allen processes ~30 MHz of proton-proton collisions and filters to ~1 MHz for HLT2 (Moore). It supports CUDA (Nvidia), HIP (AMD), and CPU backends, selected at build time via `TARGET_DEVICE`, and is also used standalone for detector commissioning and studies.

Full documentation: https://allen-doc.docs.cern.ch/index.html

---

## Subpackages

| Directory              | Role |
|------------------------|------|
| `device/velo/`         | Velo hit decoding (RetinaCluster, sparse CCL), search-by-triplet track finding, simplified Kalman filter |
| `device/UT/`           | UT hit decoding and CompassUT track reconstruction |
| `device/SciFi/`        | SciFi hit preprocessing, hybrid seeding, and looking-forward track reconstruction |
| `device/track_matching/` | Velo–SciFi matching for Long track assembly |
| `device/downstream/`   | Downstream track reconstruction, consolidation, and vertexing |
| `device/ttrack/`       | T-station-only track reconstruction |
| `device/kalman/`       | Parametric Kalman filter (`ParKalman`) for momentum and state estimation on Long tracks |
| `device/PV/`           | Primary vertex finding from Velo tracks |
| `device/vertex_fit/`   | Two-track MVA vertex fitter |
| `device/muon/`         | Muon hit decoding, Velo/upstream muon matching, muon-ID neural network |
| `device/calo/`         | Calorimeter hit decoding, cluster finding, and electron-ID neural network |
| `device/rich/`         | RICH raw bank decoding and Cherenkov PID |
| `device/plume/`        | Plume luminosity-detector reconstruction |
| `device/selections/`   | HLT1 trigger line definitions (muon, hadron, electron, photon, charm, downstream, heavy-ion, …) |
| `device/lumi/`         | Luminosity monitoring algorithms |
| `device/associate/`    | Track–cluster and track–PV associations |
| `device/combiners/`    | Multi-track candidate combiners for two- and three-body selections |
| `device/jets/`         | Jet reconstruction |
| `device/event_model/`  | GPU event model: packed track, hit, and cluster containers |
| `device/utils/`        | Shared GPU utility functions |
| `device/validators/`   | Per-algorithm output validation (debug builds) |
| `host/`                | CPU-side infrastructure: data providers, global event cut, routing bits, TAE handling, validators |
| `backend/`             | CUDA/HIP/CPU dispatch abstraction — macros, warp/block helpers, memory utilities |
| `stream/`              | Execution framework: `Scheduler` (algorithm ordering), `Store` (GPU memory management), `Sequence` |
| `configuration/`       | Python sequence generation — `AllenConf/` (algorithm builders), `AllenSequences/` (named sequences → JSON) |
| `main/`                | Application entry point and Gaudi integration layer |
| `checker/`             | Physics performance checkers (tracking efficiency, PV resolution, selection yields, perfscans) |
| `mdf/`                 | MDF raw-event file I/O |
| `zmq/`                 | ZMQ-based event distribution for online use |
| `Dumpers/`             | Raw-bank dumpers for standalone data collection |

---

## Building

From the top-level LHCb stack:

```bash
make Allen              # build Allen and all dependencies
make fast/Allen         # build Allen only (skip dependency check)
make Allen/test         # run all tests
make fast/Allen/test    # run tests without rebuilding dependencies
make clean/Allen        # clean build artefacts
```

Build artefacts appear in `build.x86_64_v3-el9-gcc15-opt/` and `InstallArea/` inside this directory.

By default Allen builds with a CPU backend. To target a GPU:

```bash
make Allen TARGET_DEVICE=CUDA    # Nvidia GPU (requires CUDA toolkit)
make Allen TARGET_DEVICE=HIP     # AMD GPU (requires ROCm)
```

---

## Running

```bash
cd Allen && ./run gaudirun.py options/myoptions.py
cd Allen && ./run python myscript.py
```

The `run` symlink sets up the full runtime environment (LCG externals, paths, etc.). For standalone Allen operation (without the full LHCb stack) see the [setup documentation](https://allen-doc.docs.cern.ch/setup/index.html).

---

## Testing

Tests cover unit tests, integration tests with reference output, and physics performance checks.

```bash
make Allen/test                    # full test suite via CTest
ctest --test-dir build.*/          # run CTest directly
pytest test/unit_tests/            # run unit tests only
```

Physics performance (tracking efficiency, PV resolution, selection yields) is validated by the checkers in `checker/` against reference output in `test/reference/`. Throughput benchmarking is available via the performance scans in `checker/perfscans/`.

---

## Package Structure

Allen uses its own GPU algorithm pattern rather than the Gaudi Functional framework. Each algorithm declares its I/O parameters and properties in a `.cuh` header and implements kernels in a `.cu` source:

```cpp
// device/mydet/include/my_algorithm.cuh
struct my_algorithm_t : public DeviceAlgorithm, Parameters {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned);
    DEVICE_INPUT(dev_hits_t, HitType);
    DEVICE_OUTPUT(dev_tracks_t, TrackType);
    PROPERTY(block_dim_t, "BlockDimensions", DeviceDimensions, {256, 1, 1});
  };
  void set_arguments_size(ArgumentReferences<Parameters>, ...) const;
  void operator()(const ArgumentReferences<Parameters>&, ...) const;
};

// device/mydet/src/my_algorithm.cu
__global__ void my_kernel(my_algorithm::Parameters params) {
  const unsigned event = blockIdx.x;  // one block per event
  // process hits for this event
}
```

The `Store` manages GPU memory allocation per algorithm to minimise peak usage. Sequences are generated from Python (`AllenConf/`) into JSON at build time and parsed by the `Scheduler` at runtime.

Top-level layout:

```text
Allen/
├── backend/          # CUDA/HIP/CPU dispatch macros and primitives
├── checker/          # physics performance checkers and perfscans
├── configuration/    # Python sequence builders (AllenConf/) and named sequences (AllenSequences/)
├── device/           # GPU algorithms organised by sub-detector and function
├── host/             # CPU algorithms (data providers, GEC, validators)
├── main/             # application entry point and Gaudi bridge
├── mdf/              # MDF file I/O
├── stream/           # Scheduler, Store, Sequence execution framework
├── test/             # unit tests, reference files, algorithm contracts
└── zmq/              # ZMQ event distribution
```

---

## Resources

- **Documentation**: https://allen-doc.docs.cern.ch/index.html
- **Throughput evolution**: https://lbgrafana.cern.ch/d/Qvm54N3Mz/allen-performance?orgId=1
- **Physics performance dashboard**: https://lblhcbpr.cern.ch/dashboards/allen
- **Allen developers** (Mattermost): https://mattermost.web.cern.ch/lhcb/channels/allen-developers
- **Allen core** (Mattermost): https://mattermost.web.cern.ch/lhcb/channels/allen-core
- **AllenPR throughput** (Mattermost): https://mattermost.web.cern.ch/lhcb/channels/allenpr-throughput

---

## Contributing

**Active branches:**

| Branch          | Purpose                                              |
|-----------------|------------------------------------------------------|
| `master`        | Run 3 long-term development                          |
| `2026-patches`  | Fixes and additions for 2026 data-taking             |
| `2025-patches`  | Fixes and additions for 2025 data-taking             |
| `2024-patches`  | Fixes and additions for 2024 data-taking             |

All protected branches require a merge request. The CI runs the full physics-performance checker suite and throughput benchmarks on each MR.

GitLab remote: `ssh://git@gitlab.cern.ch:7999/lhcb/Allen.git`

---

## Dependencies

```text
LCG (external: ROOT, Boost, …)  [via CVMFS]
└── Gaudi          ← application framework
    ├── Detector   ← DD4hep geometry and conditions
    └── LHCb       ← core event model, detector interfaces
        └── Lbcom  ← detector-specific algorithms and MC linkers
            └── Rec      ← CPU reconstruction algorithms
                └── Allen  ← this package
```
