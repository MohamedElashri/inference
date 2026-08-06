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
};

} // namespace pvfinder_fc_aggregation
