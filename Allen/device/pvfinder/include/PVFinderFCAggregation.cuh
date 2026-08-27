#pragma once

#include "VeloConsolidated.cuh"
#include "AlgorithmTypes.cuh"

namespace pvfinder_fc_aggregation {

// N_LATENT_CHANNELS shares PVFinderUNet.cuh's PVFINDER_UNET_N_BATCH_CHANNELS
// build macro (wired via ballen's --unet-batch-channels flag) rather than
// getting its own -- both represent the exact same physical quantity (the
// UNet's bottleneck/output channel count, which is also the number of
// channels in the per-interval features FC aggregation reduces into), and
// a build where the two disagreed would silently pair a UNet sized for one
// architecture with FC aggregation sized for another. One flag keeps them
// consistent by construction.
#ifdef PVFINDER_UNET_N_BATCH_CHANNELS
constexpr unsigned N_LATENT_CHANNELS = PVFINDER_UNET_N_BATCH_CHANNELS;
#else
constexpr unsigned N_LATENT_CHANNELS = 8u;
#endif
constexpr unsigned N_BINS_PER_CHANNEL = 100u;
constexpr unsigned N_INTERVALS = 40u;
// L6A's physical width: one neuron per (channel, bin) pair. 800 by default.
constexpr unsigned L6A_WIDTH = N_LATENT_CHANNELS * N_BINS_PER_CHANNEL;
// Layer6A's weight matrix is [L6A_WIDTH x 20]; 16000 floats by default.
constexpr unsigned L6A_WEIGHT_FLOATS = L6A_WIDTH * 20u;
// dev_pvfinder_interval_features's per-event stride (40 intervals x
// L6A_WIDTH); 32000 floats by default.
constexpr unsigned INTERVAL_FEATURES_STRIDE = N_INTERVALS * L6A_WIDTH;

struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_reconstructed_velo_tracks_t, unsigned) host_number_of_reconstructed_velo_tracks;
    
    DEVICE_INPUT(dev_velo_tracks_view_t, Allen::Views::Velo::Consolidated::Tracks) dev_velo_tracks_view;
    DEVICE_INPUT(dev_pvfinder_track_features_t, float) dev_pvfinder_track_features;      // Track size x 9
    
    // Output array: [events x 40 intervals x 100 bins] -> 4000 floats per event mapping the KDE layout targets
    DEVICE_OUTPUT(dev_pvfinder_output_histogram_t, float) dev_pvfinder_output_histogram;
    // N_LATENT_CHANNELS-channel interval features: [events x 40 intervals x
    // N_LATENT_CHANNELS channels x 100 bins] = INTERVAL_FEATURES_STRIDE
    // floats per event (32000 by default, i.e. 8 channels).
    // Preserved un-collapsed for UNet NCW input: channel c of interval i = latent dim c summed over tracks in i
    DEVICE_OUTPUT(dev_pvfinder_interval_features_t, float) dev_pvfinder_interval_features;
    // CSR index structure for interval-sorted track access:
    //   interval_start[n_events * 42]: start offset into track_idx for each interval + 1 sentinel
    //   track_idx[total_tracks * 2]:   track indices sorted by interval (boundary tracks appear twice)
    DEVICE_OUTPUT(dev_pvfinder_interval_start_t, int) dev_pvfinder_interval_start;
    DEVICE_OUTPUT(dev_pvfinder_track_idx_t,      int) dev_pvfinder_track_idx;
    // cuBLAS L6A GEMM intermediate buffers — sized per chunk, reused across chunks.
    //   dev_pvfinder_l5_output: L1-L5 hidden states, shape [T_chunk_max × 20] row-major.
    //   dev_pvfinder_l6a_output: raw L6A GEMM output, shape [L6A_WIDTH × T_chunk_max] col-major (cuBLAS layout, L6A_WIDTH=800 by default).
    //   Both are allocated only when ALLEN_WITH_CUBLAS is defined; zero-sized otherwise.
    DEVICE_OUTPUT(dev_pvfinder_l5_output_t,  float) dev_pvfinder_l5_output;
    DEVICE_OUTPUT(dev_pvfinder_l6a_output_t, float) dev_pvfinder_l6a_output;
    // Single-element atomic work counter for pvfinder_reduce_l6a_kernel's
    // grid-stride work-stealing mode -- reset to 0 before each chunk's
    // launch, claimed via atomicAdd by one thread per block. See
    // m_use_grid_stride_reduce's doc comment.
    DEVICE_OUTPUT(dev_pvfinder_reduce_work_counter_t, unsigned) dev_pvfinder_reduce_work_counter;
    // Per-chunk cumulative CSR column offsets,
    // one per event in the chunk plus a leading 0 (size B_CHUNK_max+1),
    // precomputed on the host from the same host_csr readback that already
    // computes T_chunk and uploaded once per chunk. Only used when
    // m_use_precomputed_csr_offset is true -- see that property's doc
    // comment. Zero-sized otherwise.
    DEVICE_OUTPUT(dev_pvfinder_event_col_offset_t, unsigned) dev_pvfinder_event_col_offset;
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

    // Path to the FC-stage weight file, mirroring PVFinderUNet.cuh's
    // m_weight_file / --cnn-weights on the CNN side.
    Allen::Property<std::string> m_weight_file {
        this, "weight_file", "/data/home/melashri/iris/inference/fc_weights.bin",
        "path to fc_weights.bin produced by convert_weights.py"};

    // Throughput-only override of how many of L6A's L6A_WIDTH neurons
    // actually get computed and reduced. Default L6A_WIDTH (800 for the
    // default N_LATENT_CHANNELS=8; scales with --unet-batch-channels) =
    // physics-valid, matching w6A/b6A and every downstream buffer, which all
    // stay sized for L6A_WIDTH regardless of this value. A smaller value
    // shrinks the cuBLAS GEMM's M, the bias/ReLU kernel's work, AND the
    // per-track accumulation loop in the reduction kernel (the actual
    // dominant cost in this block, ~6x the GEMM's own share -- an earlier
    // GEMM-only version of this property, isolating just the GEMM to test
    // cuBLAS tile-alignment effects, undersold any real width reduction
    // because it left the reduction kernel doing full-width work
    // regardless). Neurons >= this value simply never get a nonzero
    // contribution; downstream buffers stay the same L6A_WIDTH/100 shape
    // (just partially zero), so nothing else needs to change to test this.
    // Any value other than L6A_WIDTH is NOT physics-valid -- this is a
    // sub-block-of-a-wider-real-buffer throughput probe, not a way to
    // actually run a smaller latentChannels architecture (that requires a
    // build with a matching --unet-batch-channels, see N_LATENT_CHANNELS
    // above).
    Allen::Property<unsigned> m_l6a_m {
        this, "l6a_m", L6A_WIDTH,
        "Override how many of L6A's L6A_WIDTH neurons are computed (GEMM + "
        "bias/ReLU + reduction, all three) for width/throughput testing "
        "(L6A_WIDTH, 800 by default, is the physics-valid default for this "
        "build; any other value is throughput-only)"};

    // pvfinder_reduce_l6a_kernel is by far the single most expensive kernel
    // in the FC stage. Its per-track accumulation loop writes each shared-memory
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

    // The per-track accumulation loop above is also fully serial within a
    // block -- all threads jointly process one track before moving to the
    // next, so an interval with many tracks takes proportionally longer
    // with no way for the rest of the block's warps to help. This flag
    // splits tracks round-robin across the block's warps instead (mirroring
    // the pattern pvfinder_fused_fc_aggregation_kernel already uses), each
    // warp accumulating its assigned tracks into private registers, then
    // combining the small, fixed-size per-warp partial sums into shared
    // memory via a bounded atomicAdd (unlike use_nonatomic_l6a_reduce's
    // O(n_local) atomics, this combine step happens once per warp, not once
    // per track).
    Allen::Property<bool> m_use_warp_parallel_reduce {
        this, "use_warp_parallel_reduce", false,
        "Split pvfinder_reduce_l6a_kernel's per-track accumulation across the "
        "block's warps (round-robin over tracks) instead of processing tracks "
        "serially with the whole block -- candidate fix for track-heavy "
        "intervals dominating kernel wall-clock time"};

    // The FC pipeline's chunk size (events processed per
    // L1-L5/GEMM/bias-relu/reduce launch) caps pvfinder_reduce_l6a_kernel's
    // grid width. Raising it widens the grid (and proportionally the
    // dev_pvfinder_l5_output/l6a_output intermediate buffers, sized off
    // this value in set_arguments_size); combines well with
    // use_warp_parallel_reduce above.
    Allen::Property<unsigned> m_fc_chunk_size {
        this, "fc_chunk_size", 20u,
        "Number of events processed per L1-L5/GEMM/bias-relu/reduce chunk "
        "(default 20 = current baseline); raising this widens "
        "pvfinder_reduce_l6a_kernel's grid at the cost of larger intermediate "
        "buffers"};

    // Nothing else reads dev_pvfinder_l6a_output between
    // pvfinder_l6a_bias_relu_kernel's in-place write and
    // pvfinder_reduce_l6a_kernel's read -- they're two full passes (one
    // read-modify-write, one read) over the same buffer that can be fused
    // into the reduce kernel's own read, applying bias+LeakyReLU inline on
    // the raw GEMM output instead of reading a value a separate kernel
    // already wrote back. Same math, one fewer kernel launch, one fewer
    // full DRAM read-modify-write pass over the L6A output buffer.
    Allen::Property<bool> m_use_fused_bias_relu_reduce {
        this, "use_fused_bias_relu_reduce", false,
        "Apply L6A bias+LeakyReLU inline inside pvfinder_reduce_l6a_kernel's "
        "read of the raw GEMM output instead of running "
        "pvfinder_l6a_bias_relu_kernel as a separate pass -- candidate fix "
        "for FC's largest kernel cost"};

    // Throughput-ceiling probe: if L1-L5 were architecturally 1 hidden layer
    // instead of 5, how much of FC's runtime would that buy back? Not a
    // real architecture change: reuses layer1's real trained weights
    // (w1/b1, 9->20) and skips layers 2-5 entirely, writing layer1's raw
    // output straight to dev_pvfinder_l5_output in place of layer5's. Every
    // downstream buffer is unchanged -- layer1's output is already the same
    // 20-wide shape layer5's would have been. Output is not physically
    // meaningful (layer6A's weights expect layer5's transformation, not
    // layer1's) -- this flag answers a timing question only, never a
    // correctness one.
    Allen::Property<bool> m_fc_single_hidden_layer {
        this, "fc_single_hidden_layer", false,
        "Throughput-ceiling probe: skip pvfinder_l1_to_l5_kernel's layers "
        "2-5, feeding layer1's raw 20-wide output straight to L6A (default "
        "false = physics-valid all-5-layers; true is NOT physics-valid, "
        "timing only)"};

    // Replaces pvfinder_reduce_l6a_kernel's per-slot ev_col_offset
    // computation (a serial walk over this chunk's per-event CSR sentinels)
    // with an O(1) lookup into a per-chunk offset array, precomputed on the
    // host (piggybacking on the existing T_chunk host walk) and uploaded
    // once per chunk. A measured, real win under production-scale
    // multi-thread contention, though invisible in single-thread profiling.
    Allen::Property<bool> m_use_precomputed_csr_offset {
        this, "use_precomputed_csr_offset", false,
        "Replace pvfinder_reduce_l6a_kernel's O(events-in-chunk) ev_col_offset "
        "walk with an O(1) lookup into a per-chunk offset array, precomputed "
        "on the host (piggybacking on the existing T_chunk host walk) and "
        "uploaded once per chunk"};

    // Per-event CSR-entry safety margin used to size T_chunk_max = this *
    // fc_chunk_size. This bound is empirical (calibrated against a large
    // sample of real events), not provably safe: setting it too low risks a
    // real illegal-memory-access crash if some dataset exceeds what it was
    // calibrated against. Smaller values reclaim memory to allow a larger
    // fc_chunk_size, at that risk. Exposed as a runtime property (rather
    // than a compile-time constant) so a candidate value can be tested and
    // dialed back without recompiling.
    Allen::Property<unsigned> m_safe_avg_entries_per_event {
        this, "safe_avg_entries_per_event", 600u,
        "Per-event CSR-entry safety margin used to size T_chunk_max = "
        "this * fc_chunk_size (default 600; smaller values reclaim memory "
        "for a larger fc_chunk_size at real crash risk if set too low)"};

    // Throughput-ceiling probe: use only the first N of L1-L5's 20 real
    // hidden neurons per layer, and correspondingly shrink L6A's GEMM K
    // dimension (a valid cuBLAS sub-block read of the same wider-strided
    // real weight buffer, lda/ldb held at the real stride 20 -- the same
    // trick m_l6a_m uses for the GEMM's M dimension), so the simulated
    // throughput reflects a narrower L1-L5 output feeding a correspondingly
    // narrower L6A input, not just L1-L5's own kernel in isolation.
    Allen::Property<unsigned> m_l1_l5_hidden_width {
        this, "l1_l5_hidden_width", 20u,
        "Throughput-ceiling probe: use only the first N of L1-L5's 20 real "
        "hidden neurons per layer, and correspondingly shrink L6A's GEMM K "
        "dimension (default 20 = physics-valid; smaller is NOT "
        "physics-valid, timing only)"};

    // Complements m_l6a_m: that property bounds how many of L6A_WIDTH's
    // flat neurons the GEMM/accumulation step touches, but pvfinder_reduce_
    // l6a_kernel's shared-memory zero-init, its softplus reduction's channel
    // loop, and its output write-back are all hardcoded to the buffer's
    // full shape (L6A_WIDTH neurons / N_LATENT_CHANNELS channels) regardless
    // of l6a_m. This property bounds exactly those three operations, in
    // channel units, as a throughput-only sub-block probe *within this
    // build's real N_LATENT_CHANNELS* -- unlike N_LATENT_CHANNELS itself (a
    // build-time constant sized to match the loaded weight file, see its
    // definition above), this never actually shrinks the buffer, so it
    // cannot be used to run a genuinely different architecture the way
    // rebuilding with a different --unet-batch-channels can. Intended usage:
    // set this to l6a_m/100 so both probes represent the SAME hypothetical
    // (narrower-than-this-build) architecture consistently -- NOT
    // physics-valid when less than N_LATENT_CHANNELS, timing only.
    Allen::Property<unsigned> m_l6a_active_channels {
        this, "l6a_active_channels", N_LATENT_CHANNELS,
        "Throughput-ceiling probe: bound pvfinder_reduce_l6a_kernel's "
        "shared-memory zero-init, channel-reduction loop, and output "
        "write-back to this many of this build's real N_LATENT_CHANNELS "
        "channels (default N_LATENT_CHANNELS = physics-valid; set to "
        "l6a_m/100 for a consistent narrower-L6A simulation with m_l6a_m)"};

    // pvfinder_reduce_l6a_kernel's block still gets launched for every
    // (event, interval) slot including empty ones, so an
    // early-return-and-rely-on-a-separate-whole-buffer-memset design pays
    // for a memset that's mostly redundant with work the kernel is already
    // positioned to do itself under real multi-thread contention.
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

    // pvfinder_reduce_l6a_kernel launches exactly one block per
    // (event, interval) slot in the current chunk (n_chunk_events*40
    // blocks), which typically leaves it partially warp-occupied. Instead
    // of a grid sized to the chunk's slot count, this launches a FIXED
    // number of blocks sized to the GPU's actual occupancy ceiling for this
    // kernel (SM count * cudaOccupancyMaxActiveBlocksPerMultiprocessor,
    // queried once per thread and cached -- portable across devices), each
    // looping via an atomicAdd-claimed work counter over the same dense
    // (event, interval) slot space (including empty slots, which still
    // self-zero exactly as in the static-grid path). Only combined with
    // warp_parallel_reduce + fused_bias_relu_reduce above.
    Allen::Property<bool> m_use_grid_stride_reduce {
        this, "use_grid_stride_reduce", false,
        "Launch pvfinder_reduce_l6a_kernel as a fixed, occupancy-sized grid "
        "that work-steals over all (event, interval) slots via an atomic "
        "counter, instead of one block per slot -- candidate fix for "
        "reduce's partial warp occupancy"};
};

} // namespace pvfinder_fc_aggregation
