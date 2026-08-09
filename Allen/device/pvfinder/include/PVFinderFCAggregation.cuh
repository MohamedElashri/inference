#pragma once

#include "VeloConsolidated.cuh"
#include "AlgorithmTypes.cuh"

namespace pvfinder_fc_aggregation {

struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_reconstructed_velo_tracks_t, unsigned) host_number_of_reconstructed_velo_tracks;
    
    DEVICE_INPUT(dev_velo_tracks_view_t, Allen::Views::Velo::Consolidated::Tracks) dev_velo_tracks_view;
    DEVICE_INPUT(dev_pvfinder_track_features_t, float) dev_pvfinder_track_features;      // Track size x 9
    
    // Output array: [events x 40 intervals x 100 bins] -> 4000 floats per event mapping the KDE layout targets
    DEVICE_OUTPUT(dev_pvfinder_output_histogram_t, float) dev_pvfinder_output_histogram;
    // 8-channel interval features: [events x 40 intervals x 8 channels x 100 bins] = 32000 floats per event
    // Preserved un-collapsed for UNet NCW input: channel c of interval i = latent dim c summed over tracks in i
    DEVICE_OUTPUT(dev_pvfinder_interval_features_t, float) dev_pvfinder_interval_features;
    // CSR index structure for interval-sorted track access:
    //   interval_start[n_events * 42]: start offset into track_idx for each interval + 1 sentinel
    //   track_idx[total_tracks * 2]:   track indices sorted by interval (boundary tracks appear twice)
    DEVICE_OUTPUT(dev_pvfinder_interval_start_t, int) dev_pvfinder_interval_start;
    DEVICE_OUTPUT(dev_pvfinder_track_idx_t,      int) dev_pvfinder_track_idx;
    // cuBLAS L6A GEMM intermediate buffers — sized per chunk, reused across chunks.
    //   dev_pvfinder_l5_output: L1-L5 hidden states, shape [T_chunk_max × 20] row-major.
    //   dev_pvfinder_l6a_output: raw L6A GEMM output, shape [800 × T_chunk_max] col-major (cuBLAS layout).
    //   Both are allocated only when ALLEN_WITH_CUBLAS is defined; zero-sized otherwise.
    DEVICE_OUTPUT(dev_pvfinder_l5_output_t,  float) dev_pvfinder_l5_output;
    DEVICE_OUTPUT(dev_pvfinder_l6a_output_t, float) dev_pvfinder_l6a_output;
    // Phase 15 (optimization_plan.md): single-element atomic work counter for
    // pvfinder_reduce_l6a_kernel's grid-stride work-stealing mode -- reset to
    // 0 before each chunk's launch, claimed via atomicAdd by one thread per
    // block. See m_use_grid_stride_reduce's doc comment.
    DEVICE_OUTPUT(dev_pvfinder_reduce_work_counter_t, unsigned) dev_pvfinder_reduce_work_counter;
};

struct pvfinder_fc_aggregation_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;

    void operator()(
        const ArgumentReferences<Parameters>& arguments,
        const RuntimeOptions& runtime_options,
        const Constants& constants,
        const Allen::Context& context) const;

private:
    Allen::Property<dim3> m_block_dim {this, "block_dim", {256, 1, 1}, "block dimensions"};

    // Throughput-only override of how many of L6A's 800 neurons actually get
    // computed and reduced. Default 800 = physics-valid (8 channels x 100 bins,
    // matches w6A/b6A and every downstream buffer, which all stay sized for 800
    // regardless of this value). A smaller value shrinks the cuBLAS GEMM's M,
    // the bias/ReLU kernel's work, AND the per-track accumulation loop in the
    // reduction kernel (the actual dominant cost in this block, ~6x the GEMM's
    // own share -- an earlier GEMM-only version of this property, isolating just
    // the GEMM to test cuBLAS tile-alignment effects, undersold any real width
    // reduction because it left the reduction kernel doing full-800 work
    // regardless). Neurons >= this value simply never get a nonzero
    // contribution; downstream buffers stay the same 800/100 shape (just
    // partially zero), so nothing else needs to change to test this. Any value
    // other than 800 is NOT physics-valid.
    Allen::Property<unsigned> m_l6a_m {
        this, "l6a_m", 800u,
        "Override how many of L6A's 800 neurons are computed (GEMM + bias/ReLU "
        "+ reduction, all three) for width/throughput testing "
        "(800 = physics-valid default; any other value is throughput-only)"};

    // Profiling (2026-07-30, optimization_plan.md) found pvfinder_reduce_l6a_kernel
    // costs ~9x the L6A GEMM it reduces -- by far the single most expensive kernel
    // in the FC stage. The per-track accumulation loop writes each shared-memory
    // slot s_feat[n] via atomicAdd, but the thread<->n mapping (n = thread_id,
    // thread_id+blockDim.x, ...) is identical on every iteration of the enclosing
    // track loop, so each slot is written by exactly one thread for the block's
    // entire lifetime -- no two threads ever touch the same slot. The atomic
    // appears to be unnecessary; this flag swaps it for a plain += to test that.
    // Physics-identical if the no-race analysis is correct (same arithmetic, same
    // result) -- default false (keep atomicAdd) until benchmarked and verified.
    Allen::Property<bool> m_use_nonatomic_l6a_reduce {
        this, "use_nonatomic_l6a_reduce", false,
        "Replace atomicAdd with a plain += in pvfinder_reduce_l6a_kernel's "
        "per-track accumulation loop (see comment above) -- candidate fix for "
        "the reduction being ~9x the cost of the GEMM it reduces"};

    // Phase 6 (optimization_plan.md, 2026-08-04): hardware profiling with ncu
    // found pvfinder_reduce_l6a_kernel is occupancy-bound (1.7% SM throughput,
    // 10.9% warp occupancy), not compute- or memory-bound. Idea 1 (compact
    // non-empty (event, interval) blocks out of the grid using a host-built
    // map) was implemented and benchmarked here, but regressed real
    // throughput at every thread count (worst at -t4/-t16) -- the host-side
    // cost of building the map and an extra per-chunk cudaMemcpy outweighed
    // the grid-shrinking benefit, especially under multi-stream contention.
    // Rejected and removed; see optimization_plan.md Phase 6 for the full
    // benchmark table and root-cause analysis. Idea 2 below is what actually
    // worked.
    //
    // Idea 2: even among non-empty intervals, the per-track
    // accumulation loop is fully serial -- all threads in a block jointly
    // process one track before moving to the next, so an interval with many
    // tracks takes proportionally longer with no way for the rest of the
    // block's warps to help. This flag splits tracks round-robin across the
    // block's warps (mirroring the pattern pvfinder_fused_fc_aggregation_kernel
    // already uses), each warp accumulating its assigned tracks into private
    // registers, then combining the (small, fixed-size = 4-way) per-warp
    // partial sums into shared memory via a bounded atomicAdd (not the
    // O(n_local) atomics use_nonatomic_l6a_reduce targets -- this is a
    // different, much smaller combine step that happens once per warp, not
    // once per track).
    Allen::Property<bool> m_use_warp_parallel_reduce {
        this, "use_warp_parallel_reduce", false,
        "Split pvfinder_reduce_l6a_kernel's per-track accumulation across the "
        "block's warps (round-robin over tracks) instead of processing tracks "
        "serially with the whole block -- candidate fix for track-heavy "
        "intervals dominating kernel wall-clock time"};

    // Idea 3: independent of idea 2, the FC pipeline's chunk size (events
    // processed per L1-L5/GEMM/bias-relu/reduce launch) caps
    // pvfinder_reduce_l6a_kernel's grid at fc_chunk_size*40 blocks of 128
    // threads (4 warps) each -- e.g. 800 blocks/3,200 warps at the default of
    // 20, below the RTX 3090's ~5,248-warp concurrent capacity. Raising this
    // widens the grid (and proportionally the
    // dev_pvfinder_l5_output/l6a_output intermediate buffers, sized off this
    // value in set_arguments_size). Modest/mixed on its own, but combines
    // cleanly with idea 2 (use_warp_parallel_reduce) -- see
    // optimization_plan.md Phase 6 for the combined benchmark results.
    Allen::Property<unsigned> m_fc_chunk_size {
        this, "fc_chunk_size", 20u,
        "Number of events processed per L1-L5/GEMM/bias-relu/reduce chunk "
        "(default 20 = current baseline); raising this widens "
        "pvfinder_reduce_l6a_kernel's grid at the cost of larger intermediate "
        "buffers"};

    // Phase 7 (optimization_plan.md, 2026-08-04): post-Phase-6 re-profiling
    // found pvfinder_reduce_l6a_kernel and pvfinder_l6a_bias_relu_kernel now
    // tied for FC's single biggest cost (~28% each). Nothing else reads
    // dev_pvfinder_l6a_output between the bias/ReLU kernel's in-place write
    // and the reduce kernel's read -- they're two full passes (one
    // read-modify-write, one read) over the same buffer that can be fused
    // into the reduce kernel's own read, applying bias+LeakyReLU inline on
    // the raw GEMM output instead of reading a value a separate kernel
    // already wrote back. Same math, one fewer kernel launch, one fewer full
    // DRAM read-modify-write pass over an ~24 MB buffer.
    Allen::Property<bool> m_use_fused_bias_relu_reduce {
        this, "use_fused_bias_relu_reduce", false,
        "Apply L6A bias+LeakyReLU inline inside pvfinder_reduce_l6a_kernel's "
        "read of the raw GEMM output instead of running "
        "pvfinder_l6a_bias_relu_kernel as a separate pass -- candidate fix "
        "for FC's now-largest kernel cost post-Phase-6"};

    // Phase 7 also tried replacing pvfinder_l1_to_l5_kernel's per-thread
    // linear CSR walk with a shared-memory prefix sum + binary search --
    // implemented, benchmarked, found to be a net wash-to-slight-regression
    // (the binary search kernel's own per-instance cost was ~2% *higher*
    // than the linear scan; the shared-memory setup/sync overhead roughly
    // cancels the reduced iteration count at the chunk sizes actually in
    // use). Removed. See optimization_plan.md Phase 7 for the numbers.

    // Phase 7 also tried forcing CUBLAS_TF32_TENSOR_OP_MATH on the L6A
    // GEMM's cuBLAS handle (default is the library's own choice, which ncu
    // showed already runs the classic CUDA-core kernel at 70% SM / 46% DRAM
    // throughput -- already well utilized, not idling like every other
    // kernel touched in Phases 6-7). Benchmarked: a wash at low thread
    // counts (+0.6pp) and a small net *negative* at -t8/-t16 (-0.4 to
    // -0.75pp), the production-relevant regime. Correctness was fine (diff
    // landed within the pipeline's existing noise floor, surprisingly, given
    // TF32's real precision trade-off -- plausibly because this GEMM's K=20
    // is too shallow a reduction depth for the coarser mantissa to compound
    // into anything larger) but the performance case didn't hold up.
    // Removed. See optimization_plan.md Phase 7 for the numbers.

    // Phase 8 (optimization_plan.md, 2026-08-05): profiling at real -t16
    // contention (not the -t1 clean-attribution methodology used through
    // Phase 7) found cudaMemsetAsync at 14.4% of total CUDA API time --
    // much more prominent than the quieter -t1 measurements suggested.
    // pvfinder_reduce_l6a_kernel's block still gets launched for every
    // (event, interval) slot including empty ones (block compaction was
    // tried and rejected in Phase 6), so an early-return-and-rely-on-a-
    // separate-whole-buffer-memset design pays for a memset that's mostly
    // redundant with work the kernel is already positioned to do itself.
    // pvfinder_reduce_l6a_kernel now unconditionally writes explicit zeros
    // for empty slots instead of early-returning (always-on, not gated by
    // this flag -- provably correctness-preserving on its own, since it
    // writes literal 0.0f wherever the memset already would have). This
    // flag controls whether operator() still also runs the now-redundant
    // memsets: dev_pvfinder_output_histogram never needs one in this mode
    // (exactly n_events-sized, fully covered by the kernel);
    // dev_pvfinder_interval_features still needs a small memset for its
    // padding tail (padded_events > n_events, for UNet's batch alignment --
    // never written by any FC kernel), just not the full buffer.
    Allen::Property<bool> m_skip_redundant_memset {
        this, "skip_redundant_memset", false,
        "Skip pvfinder_output_histogram's full memset and shrink "
        "pvfinder_interval_features's memset to just its padding tail, "
        "relying on pvfinder_reduce_l6a_kernel's own explicit zero-writes "
        "for empty (event, interval) slots instead -- candidate fix for "
        "cudaMemsetAsync's real cost under -t16 contention"};

    // Phase 10 also tried launching pvfinder_l1_to_l5_kernel once across the
    // whole batch instead of once per L6A chunk, targeting its low measured
    // occupancy (18.7% warps active vs. 30.8%/63.9% for reduce/GEMM) by
    // decoupling its grid width from L6A's memory-constrained chunk size.
    // Implemented, correctness-verified, benchmarked: +0.09 percentage
    // points -- statistically indistinguishable from noise, no real win.
    // Removed. See optimization_plan.md Phase 10 for the numbers. (Phase 10
    // also found and fixed a genuine, pre-existing correctness bug in
    // pvfinder_reduce_l6a_kernel while investigating this -- see that
    // kernel's own doc comment in the .cu file.)

    // Phase 11 (optimization_plan.md, 2026-08-06) tried storing the L6A GEMM's
    // output (dev_pvfinder_l6a_output) as FP16 instead of FP32, via
    // cublasGemmEx with CUDA_R_16F as the C type and CUBLAS_COMPUTE_32F
    // (compute itself stayed FP32) -- targeting that buffer's ~24 MB
    // footprint as the fixed memory ceiling that has capped fc_chunk_size at
    // every point in this project (Phase 6's 20->40, Phase 8's 40->100,
    // Phase 9's crash/fix all trace back to this buffer's size). Implemented
    // (a parallel __half buffer, a 4th template bool on
    // pvfinder_reduce_l6a_kernel converting back to float via __half2float
    // at the point of read, a use_fp16_l6a_storage property), and
    // correctness-checked two ways: self-consistency across chunk sizes
    // (fp16 chunk=100 vs chunk=200, same 200 events) matched to within
    // 2.4e-7 -- ruling out any indexing/chunking bug in the new code -- but
    // fp16 chunk=100 vs the FP32 reference (same events, same chunk size)
    // showed massive, widespread divergence: 63.5% of dev_pvfinder_interval_
    // features elements changed measurably, 23% by more than 100 (against a
    // mean reference magnitude of only 423), 9% by more than the mean
    // magnitude itself, max_abs_diff=140175 -- essentially as large as the
    // reference's own most extreme value. Only 146 of the ~589k
    // large-divergence elements actually involved a reference value beyond
    // FP16's max representable magnitude (65504), ruling out simple overflow
    // as the dominant cause -- the real driver is that s_feat[n]'s
    // accumulation sums up to hundreds of per-track FP16-quantized terms
    // (each with only 10 mantissa bits, ~0.05% relative precision) per
    // interval, and that quantization noise compounds across the sum far
    // more than a "just shrink the storage format" framing suggested. This
    // is a genuine, physics-breaking precision loss, not a rare edge case --
    // rejected without proceeding to the planned chunk-size sweep (a
    // throughput win isn't worth shipping if it corrupts the majority of
    // output elements). Removed. See optimization_plan.md Phase 11 for the
    // full percentile breakdown and methodology.

    // Phase 12 (optimization_plan.md, 2026-08-06) tried capturing the
    // per-chunk kernel-launch sequence (L1-L5, cuBLAS GEMM, optional
    // bias/ReLU, reduce) for all of a call's non-empty chunks as one CUDA
    // graph, replayed via a single cudaGraphLaunch instead of up to ~3 host
    // API calls per chunk -- targeting cudaLaunchKernel overhead under real
    // -t16 contention. Implemented (capture-once/replay-verbatim, verified
    // safe since Allen's argument buffer addresses and each chunk's T_chunk
    // are bit-identical across repeated calls with the same n_events -- no
    // per-replay parameter patching needed, unlike PVFinderUNet's own CUDA
    // graph), correctness-verified (matched the eager path within float
    // accumulation noise, both on first capture+launch and after 7 cached
    // replays), and benchmarked at two chunk sizes: at fc_chunk_size=100
    // (5 chunks/call, the production default) it was a wash -- 73.34%
    // eager vs. 73.72% graph FC-alone retention, well within the 0.07-0.38%
    // baseline run-to-run spread. At fc_chunk_size=20 (25 chunks/call, more
    // host calls bundled per graph launch) it was a real *regression* --
    // 68.81% eager vs. 66.37% graph, five times larger than that
    // configuration's own 0.10-0.43% baseline spread. Rejected: no benefit
    // at the config that matters, and bundling more chunks into a bigger
    // graph made things worse, not better -- the opposite of what
    // "eliminating N host API calls via 1 graph launch" would predict if
    // host-side dispatch overhead were the bottleneck it was hypothesized
    // to be. Removed. See optimization_plan.md Phase 12 for the full
    // implementation notes and both benchmark tables.

    // Phase 13 (optimization_plan.md, 2026-08-06) tried shrinking
    // pvfinder_reduce_l6a_kernel's grid to only non-empty (event, interval)
    // slots via a compacted work-list -- built on the host from CSR data
    // already resident there (the existing DtoH readback used for T_chunk
    // already copies each event's full 42-word interval_start array, not
    // just the sentinel) and uploaded via a single H2D copy for the whole
    // batch before the chunk loop, specifically to avoid the *extra
    // per-chunk* cudaMemcpy that Phase 6's own (host-map-based) attempt at
    // this was attributed to regressing on. Correctness-verified (matched
    // the eager path within float-accumulation noise). But because every
    // launched block became guaranteed non-empty, this mode couldn't rely
    // on Phase 8's self-zeroing trick for empty slots and had to force a
    // full memset back on regardless of skip_redundant_memset -- and that
    // reintroduced cost, plus whatever the compacted grid's more uniform
    // block sizes did to GPU-side scheduling, outweighed the grid-shrinking
    // benefit: FC-alone retention dropped from 73.07% (eager) to 69.37%
    // (compacted), a real -3.70pp regression, confirmed clean after an
    // initial contended run (baseline spread 10.64%, another user's process
    // on the shared GPU) was discarded and retried (0.23% spread). So this
    // DID avoid Phase 6's specific extra-memcpy cost driver, but still
    // regressed for a different reason -- the reintroduced memset (or a
    // GPU-scheduling effect from the new grid shape) dominates instead.
    // Rejected. Removed. See optimization_plan.md Phase 13 for the full
    // numbers.

    // Phase 14 (optimization_plan.md, 2026-08-06) tried replacing
    // pvfinder_l1_to_l5_kernel's per-thread linear CSR scan with a per-thread
    // binary search over a precomputed, once-per-batch-uploaded per-event
    // cumulative CSR offset array (dev_pvfinder_event_csr_offset) -- a
    // different mechanism than Phase 7's rejected *block-collaborative*
    // shared-memory binary search (no per-block setup/sync cost here, just
    // a plain per-thread search over a small, heavily-L1/L2-cached global
    // array). csr_offset, the kernel's existing but previously-dead
    // parameter, was resurrected into real use to convert a thread's
    // chunk-relative t into the batch-wide position the search needed.
    // Correctness-verified (matched the eager path within float-accumulation
    // noise). But benchmarked at a real, reproducible -3.73pp retention
    // regression (73.20% eager vs. 69.47% lookup, both runs clean at
    // 0.08-0.55% baseline spread) -- remarkably close in magnitude to Phase
    // 13's own -3.70pp regression from a different, unrelated idea. Rejected.
    // Plausible reading (not confirmed via further profiling): the linear
    // scan has strong warp-coherence properties a binary search doesn't
    // preserve -- neighboring threads share consecutive CSR slot t values,
    // which given typical event sizes (~196 entries/event) usually land in
    // the same or an adjacent event, so a warp's threads scan nearly
    // identical iteration counts via cheap, predictable, cached sequential
    // reads. A binary search's data-dependent branching can diverge more at
    // event-boundary threads even while doing fewer total iterations
    // (log2(100)~7 vs. up to 100 worst-case for the scan), and its access
    // pattern is less prefetch-friendly than the scan's simple stride.
    // Removed. See optimization_plan.md Phase 14 for the full numbers.

    // Phase 15 (optimization_plan.md, 2026-08-06): pvfinder_reduce_l6a_kernel
    // launches exactly one block per (event, interval) slot in the current
    // chunk (n_chunk_events*40 blocks), which Phase 6's own profiling found
    // only 30.8% warp-occupied. Ideas 1 (Phase 13, shrink the grid via
    // compaction) and 2 (Phase 14, cut L1-L5's scan cost) both targeted
    // occupancy/iteration-count via restructuring how work is distributed
    // across blocks/threads, and both regressed by almost exactly the same
    // amount (-3.70pp, -3.73pp) -- a pattern worth being aware of, since this
    // idea makes a similar kind of bet. This one: instead of a grid sized to
    // the chunk's slot count, launch a FIXED number of blocks sized to the
    // GPU's actual occupancy ceiling for this kernel (SM count *
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor, queried once per thread
    // and cached -- portable, no hardcoded RTX 3090 SM count), each looping
    // via an atomicAdd-claimed work counter over the SAME dense
    // (event, interval) slot space (including empty slots, which still
    // self-zero exactly as today -- Phase 13's forced-memset regression risk
    // doesn't apply here since nothing about empty-slot handling changes).
    // Only combined with the winning warp_parallel+fused_bias_relu
    // combination (matching Phase 11/13's scoping precedent).
    Allen::Property<bool> m_use_grid_stride_reduce {
        this, "use_grid_stride_reduce", false,
        "Launch pvfinder_reduce_l6a_kernel as a fixed, occupancy-sized grid "
        "that work-steals over all (event, interval) slots via an atomic "
        "counter, instead of one block per slot -- candidate fix for "
        "reduce's still-partial (30.8%) warp occupancy via a different "
        "mechanism than Phase 13's rejected static compaction"};

    // Phase 17 (optimization_plan.md, 2026-08-07) tried routing L1-L5's 5
    // dense layers through cuBLAS (a gather kernel materializing the
    // scattered per-track input into a dense buffer, then 5 cublasSgemm
    // calls mirroring L6A's own GEMM convention, each followed by a shared
    // bias+LeakyReLU epilogue kernel, ping-ponging between two scratch
    // buffers) instead of the hand-written per-thread register kernel --
    // motivated by L1-L5's low 18.7% occupancy vs. L6A's 70% SM utilization
    // via cuBLAS, and by Phase 16's confirmation that cuBLAS decisively
    // beats a hand-written equivalent when the whole FC pipeline is
    // compared (38% vs. 74%). Correctness-verified (matched the eager path
    // within float-accumulation noise, max_abs_diff=0.039 -- larger than
    // single-GEMM-only diffs elsewhere in this file but still negligible,
    // consistent with 5 sequential GEMMs compounding rounding differently
    // than the hand-written version's in-register chain). But benchmarked
    // at a real, tightly-reproducible -2.04pp regression (74.00% eager+
    // grid-stride-reduce vs. 71.96% cublas_l1_l5, both clean at 0.19-0.71%
    // baseline spread). Plausible reading (not confirmed via further
    // profiling): unlike L6A's single GEMM (20->800, wide enough for
    // cuBLAS's tiling advantage to amortize against), L1-L5's 5 layers are
    // narrow (out_f=20 throughout) and, critically, the hand-written
    // kernel does all 5 layers back-to-back in registers with only ONE
    // final global-memory write -- while the cuBLAS-routed version pays
    // for 4 extra full round-trips through DRAM to materialize each
    // layer's intermediate output, a real memory-traffic cost the thin
    // per-layer GEMMs don't have enough compute to amortize away. Removed.
    // See optimization_plan.md Phase 17 for the full numbers.
};

} // namespace pvfinder_fc_aggregation
