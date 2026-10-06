# v9r1 throughput investigation

The measured v9r1 shadow loss is **7.639 ± 0.140%**, compared with
**4.567 ± 0.250%** on v7r9. These are means and sample standard deviations of
five paired baseline/shadow measurements, with the same 500-event slice,
1,000 repetitions, 16 streams, BF16 checkpoint and RTX 3090.

## Tensor-core hypothesis

No other HLT1 algorithm in the benchmarked v9r1 build uses tensor cores.
`cuobjdump --dump-sass Allen/buildgpu12/Allen` inspected 512 compiled CUDA
functions. Only `pvfinder_fused_fc_tc_kernel` and `fused_unet_bf16_kernel`
contain tensor matrix instructions. An upstream v9r1 source search also finds
no WMMA/MMA, cuBLAS, cuDNN or CUTLASS implementation. Half-precision storage
in UT/SciFi code does not by itself imply tensor-core execution.
The dynamically linked cuDNN library belongs to our FP32 implementation;
the BF16 path uses our direct CUDA kernels. This finding is about this build
and HLT1 configuration, not every possible external Allen extension.

Evidence: `tensor_instruction_audit.json`.

## What changed

| Measurement | v7r9 | v9r1 |
|---|---:|---:|
| Baseline mean throughput, events/s | 87,085 | 86,774 |
| Network shadow loss | 3.565% | 5.940% |
| Full PV shadow loss | 4.567% | 7.639% |
| Effective added time, network shadow | 0.425 us/event | 0.728 us/event |
| Effective added time, full shadow | 0.550 us/event | 0.953 us/event |

The baseline changes by only about 0.36%, so a smaller baseline denominator
does not explain the regression. Roughly three quarters of the increase in
effective added time occurs already in the network shadow measurement.
The network-to-full-shadow difference is 1.00 percentage point on v7r9 and
1.70 points on v9r1. These are differences between configurations, not isolated
stage costs: overlap and scheduling can change when a stage is added.

The upstream PV fitter and histogram now accumulate in 64-bit fixed point
to ensure reproducible results. In the 16-stream baseline Nsight Systems
profiles, mean fitter duration rises from 383 to 538 us per launch and the
histogram from 58 to 99 us. In the single-stream profiles the fitter rises
from 114 to 327 us. Full shadow runs a second fitter on our seeds. This is
a measured additional cost, but does not explain the whole network regression.
Reverting upstream deterministic arithmetic would change the reconstruction's
numerical guarantees and is not the proposed optimization.

The existing single-stream full-grid profiles show essentially unchanged
network durations: FC 105 versus 104 us, UNet 213 versus 209 us, CSR 50 us
in both versions. Multistream network durations are also similar. Therefore
there is no evidence here for a large isolated FC/UNet kernel slowdown.
Scheduling, shared-resource overlap and host work remain the leading
explanations for the network-only regression; their individual contributions
have not yet been isolated by controlled experiments.

The port's PVFinder device sources differ only in library registration and
qualification of the existing cross-product helper. The toolchain nevertheless
changes from CUDA 12.6.85/LCG_106c to CUDA 12.8.93/LCG_108c; gcc remains 13.1.
A controlled compiler/dependency comparison would help separate environment
effects from upstream reconstruction changes. The profiles do not establish
a complete causal attribution of the extra network cost.

## Concrete resource limits

CUDA block residency is limited by registers and shared memory, even when
different kernels use different arithmetic units. See NVIDIA's
[hardware multithreading documentation](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#hardware-multithreading).

Nsight Compute reports FC at 128 registers/thread, 256 threads/block and
43,136 allocated shared bytes/block. UNet uses 141 registers/thread
(144 allocated), 128 threads/block and 61,056 allocated shared bytes/block.
Their combined shared requirement is **104,192 bytes**, exceeding the
3090's **102,400 bytes/SM**. An FC block and a UNet block cannot reside
together on one SM with these layouts. UNet has only one resident block/SM
and about 8.3% achieved occupancy when profiled in isolation.

The isolated UNet launch also selects a 65,536-byte shared-memory carveout,
versus 102,400 for FC. A preferred carveout should be tested together with
a smaller block footprint, balancing available shared space against L1 cache.
Grid fractions cap block counts; they do not reserve particular SMs.

FC copies CSR offsets to the host, synchronizes its stream, builds row maps
and a sorted/chunked work list on the CPU, then copies those back. In the
16-stream v9r1 trace the associated sync waits average 964 us/slice, including
828 us of queue wait before the CSR kernel and about 100 us in the CSR kernel.
These waits overlap other streams and must not be added to event throughput
as if they were serial costs. They identify a synchronization point worth
removing, not a measured 964-us saving.

Evidence: saved `v9r1_ncu_{full,caps}/profile.csv`,
`host_roundtrip_audit.json`, and profile stages in `performance_diagnosis.json`.

## Completed short grid scan

Each setting uses three paired repeats, 500 events, 100 repetitions and
16 streams. Uncertainties below are sample standard deviations; these are
screening results, not replacements for the longer headline benchmark.

| FC grid fraction | UNet grid fraction | Full shadow loss |
|---|---|---:|
| 1/16 | 1/16 | 6.748 ± 0.526% |
| 1/32 | 1/32 | 13.782 ± 0.278% |
| 1/16 | 1/32 | 12.277 ± 0.122% |
| 1/32 | 1/16 | 7.413 ± 0.158% |

All batches completed successfully; baseline spreads are below 0.8% and
the records report no other GPU compute processes at startup. Short timing
measurements still have startup and run-to-run effects. The 1/16 result
reproduces the longer sweep; reducing UNet further is particularly harmful.
The scan identifies an insufficient-parallelism limit, not a sub-5% setting.

## Optimization order

1. Retune FC and UNet grid fractions independently. The five-repeat overnight
   sweep gives 9.119% loss at full grids, 7.024% at 1/8 each and
   **6.704 ± 0.443% at 1/16 each**. This is useful progress but remains above
   the target. The completed independent 1/16–1/32 screening scan finds no
   improvement; any further candidate needs 1,000 repetitions and five repeats.
   Also screen our fitter's `block_dim_y` at 2, 4 and 8. It currently uses four
   warps, one seed per warp, so this property changes the number of seeds fitted
   concurrently without removing the fixed-point accumulation. Change only
   `pv_beamline_multi_fitter_pvfinder`, retaining the baseline fitter's settings,
   and validate the resulting vertex output.
2. Reduce UNet shared memory. Test 2- or 3-warp blocks and compare at similar
   total warp counts, with grid and preferred shared-memory carveout tuned
   together. Three warps reduce the dynamic footprint by 9,088 bytes without
   changing an interval's arithmetic. This crosses the FC/UNet co-residency
   threshold, but may lose standalone parallelism; measure the full HLT1 result.
   Two warps also bring two UNet blocks below the shared-memory limit. Another
   option is staging only the active layer's weights instead of the entire
   network; compare its extra loads against improved residency.
3. Build the compact row map and chunked FC work list on the GPU. Keep counts
   device-side and let the BF16 kernels consume them, so no CSR download or
   per-slice CPU work-list construction is needed. Preserve empty-interval
   responses, split-slot reduction and physics validation.
4. Reprofile the added vertex fit after the network changes. Optimize our
   seed-fit integration and launch properties while retaining upstream
   reproducibility. Replacement/hybrid measurements are a different physics
   configuration and cannot be substituted for the shadow target.

At the measured v9r1 baseline rate, a 5% loss permits about 0.607 us/event
of effective added time. The present 0.953 us/event must fall by about 36%.
None of the available 16-stream measurements yet demonstrates a reliable
sub-5% result. Every kernel change needs the existing numerical validators,
full-sample vertex checks, and a repeated paired throughput confirmation.

## Reproduction

Run `python3 local_scripts/rebase_v9r1/analyze_performance.py` to regenerate
the throughput/profile comparison from structured records. It reads the actual
`pvs` role: the existing batch Markdown summary assumes an `unet` role and
prints spurious 100% overhead for baseline-plus-PVs-only batches.
`diagnostic_grid_scan.sh` runs the short screening scan in the isolated port.
All source builds and benchmark experiments remain separate from v7r9 and
the talk deck.
