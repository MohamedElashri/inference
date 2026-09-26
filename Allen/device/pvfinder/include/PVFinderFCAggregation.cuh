#pragma once

#include "VeloConsolidated.cuh"
#include "AlgorithmTypes.cuh"
#include "ParticleTypes.cuh"

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
    // Events that passed the node's prefilters (the HLT1 physics prefilters in
    // the PVFinder sequences). Only these get tracks; every other event's
    // intervals are empty.
    MASK_INPUT(dev_event_list_t) dev_event_list;
    
    DEVICE_INPUT(dev_velo_tracks_view_t, Allen::Views::Velo::Consolidated::Tracks) dev_velo_tracks_view;
    DEVICE_INPUT(dev_velo_states_view_t, Allen::Views::Physics::KalmanStates) dev_velo_states_view;
    // Per-track input features [tracks x 9] (PVFinderTrackFeatures.cuh),
    // computed by the CSR build's first pass (which reads them there anyway).
    DEVICE_OUTPUT(dev_pvfinder_track_features_t, float) dev_pvfinder_track_features;
    
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
    // Row layout of dev_pvfinder_interval_features for the UNet, see
    // m_skip_empty_intervals:
    //   host_pvfinder_unet_rows[0]: 0 = dense (row = event * 40 + interval,
    //       padded to a multiple of unet_batch_events events), 1 = compact
    //   host_pvfinder_unet_rows[1]: compact only, number of rows in use
    //   host_pvfinder_unet_rows[2]: storage type of the features, 0 = float32,
    //       1 = bfloat16 (see m_unet_input_dtype)
    //   host_pvfinder_unet_rows[3]: 1 when each row is stored channels last,
    //       [bin][channel] (see m_unet_input_layout)
    //   dev_pvfinder_slot_row[event * 40 + interval]: compact only, the
    //       interval's row, or -1 when the UNet skips it
    HOST_OUTPUT(host_pvfinder_unet_rows_t, unsigned) host_pvfinder_unet_rows;
    DEVICE_OUTPUT(dev_pvfinder_slot_row_t, int) dev_pvfinder_slot_row;
    // Its inverse, compact only: dev_pvfinder_row_slot[row] = event * 40 +
    // interval, written by the FC kernels with the row's features (the fused
    // UNet writes each row's KDE straight to its slot with it).
    DEVICE_OUTPUT(dev_pvfinder_row_slot_t, int) dev_pvfinder_row_slot;
    // Work list of the per-warp fused FC kernels (see m_fc_largest_first):
    // uint4 {slot, first CSR entry, entries, feature row} by decreasing
    // entries, 4 words per slot.
    DEVICE_OUTPUT(dev_pvfinder_slot_order_t, unsigned) dev_pvfinder_slot_order;
    // Split slots of the tensor-core fused FC: partial sums per chunk
    // [rows][L6A_WIDTH] and, per split slot, the number of chunks done.
    DEVICE_OUTPUT(dev_pvfinder_fc_partial_t, float) dev_pvfinder_fc_partial;
    DEVICE_OUTPUT(dev_pvfinder_fc_arrive_t, unsigned) dev_pvfinder_fc_arrive;
};

struct pvfinder_fc_aggregation_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;

    void operator()(
        const ArgumentReferences<Parameters>& arguments,
        const RuntimeOptions& runtime_options,
        const Constants& constants,
        const Allen::Context& context) const;

    // Loads the beamline into dev_beamline (the track features are in its frame).
    void update(const Constants& constants) const;

private:
    // Block of the CSR build (one per event). 512 is fastest on the RTX 3090:
    // the busiest events' canonical ranking is the kernel's tail, and more
    // threads shorten it (256: 6.0 ms, 512: 5.0, 1024: 7.3 per 100 slices).
    Allen::Property<dim3> m_block_dim {this, "block_dim", {512, 1, 1}, "block dimensions"};

    // Required, no default, like pvfinder_unet's weight_file: the
    // repository's weights/ pipeline produces fc_weights.bin
    // (make -C weights convert MODEL=<name>), and AllenConf fills this in from
    // PVFINDER_WEIGHTS_DIR when the sequence configuration is generated.
    Allen::Property<std::string> m_weight_file {
        this, "weight_file", "",
        "path to fc_weights.bin (required; produced by the weights/ pipeline, "
        "set by AllenConf from PVFINDER_WEIGHTS_DIR)"};

    // Throughput-only override of how many of L6A's L6A_WIDTH neurons
    // actually get computed and reduced. Default L6A_WIDTH (800 for the
    // default N_LATENT_CHANNELS=8; scales with --unet-batch-channels) =
    // physics-valid, matching w6A/b6A and every downstream buffer, which all
    // stay sized for L6A_WIDTH regardless of this value. A smaller value
    // shrinks the cuBLAS GEMM's M, the bias/ReLU kernel's work, AND the
    // per-track accumulation loop in the reduction kernel. The reduction is
    // the dominant cost in this block, so all three stages use the same bound.
    // Neurons >= this value never get a nonzero
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
    // can therefore use a plain += without a cross-thread race. The default
    // retains atomicAdd; the flag selects the non-atomic implementation.
    Allen::Property<bool> m_use_nonatomic_l6a_reduce {
        this, "use_nonatomic_l6a_reduce", false,
        "Replace atomicAdd with a plain += in pvfinder_reduce_l6a_kernel's "
        "per-track accumulation loop (see comment above)"};

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
        this, "use_warp_parallel_reduce", true,
        "Split pvfinder_reduce_l6a_kernel's per-track accumulation across the "
        "block's warps (round-robin over tracks) instead of processing tracks "
        "serially with the whole block"};

    // The FC pipeline's chunk size (events processed per
    // L1-L5/GEMM/bias-relu/reduce launch) caps pvfinder_reduce_l6a_kernel's
    // grid width. Raising it widens the grid (and proportionally the
    // dev_pvfinder_l5_output/l6a_output intermediate buffers, sized off
    // this value in set_arguments_size); combines well with
    // use_warp_parallel_reduce above.
    Allen::Property<unsigned> m_fc_chunk_size {
        this, "fc_chunk_size", 130u,
        "Number of events processed per L1-L5/GEMM/bias-relu/reduce chunk "
        "(default 130); raising this widens "
        "pvfinder_reduce_l6a_kernel's grid at the cost of larger intermediate "
        "buffers"};

    // dev_pvfinder_interval_features is padded to a multiple of the UNet's
    // cuDNN batch size so the UNet can always read whole batches. Must match
    // pvfinder_unet.unet_batch_events.
    Allen::Property<unsigned> m_unet_batch_events {
        this, "unet_batch_events", 20u,
        "pad interval features to a multiple of this many events; must match "
        "pvfinder_unet.unet_batch_events"};

    // Hand the UNet only the intervals that can contribute. An interval with
    // no tracks has all-zero features, so the UNet's output there is its
    // response to a zero input, the same for every such interval (exactly 0
    // for the current models). With this on, the reduce kernel writes the
    // features of the intervals with at least min_interval_tracks tracks to
    // consecutive rows, the UNet runs on those rows only and writes its
    // zero-input response to every other interval. The row map is built on
    // the host from the CSR offsets operator() already copies back to size
    // its chunks, so it adds no device synchronisation. The pvfinder_unet
    // downstream follows host_pvfinder_unet_rows, so only this algorithm
    // needs configuring. cuBLAS builds only (ignored otherwise). On by
    // default: validated exact on every UNet path, see
    // docs/pvfinder/skip_empty_intervals.md.
    Allen::Property<bool> m_skip_empty_intervals {
        this, "skip_empty_intervals", true,
        "give the UNet only intervals with >= min_interval_tracks tracks; it writes its "
        "zero-input response to all others (exact for min_interval_tracks = 1)"};

    // Put each interval's tracks in a canonical order (by their features) when
    // building the CSR. The atomic scatter and the VELO reconstruction both
    // change the order from run to run, and with it the rounding of every sum
    // over an interval's tracks; with this on (and fc_fused, whose sums run in
    // a fixed order) the whole PVFinder output is bit-for-bit reproducible.
    Allen::Property<bool> m_canonical_track_order {
        this, "canonical_track_order", true,
        "sort each interval's tracks by their features when building the CSR, so the "
        "sums over tracks (and, with fc_fused, the whole output) are reproducible"};

    // One fused kernel for the whole FC stage (pvfinder_fused_fc_kernel):
    // L1-L5, L6A, bias + LeakyReLU and the sum over each interval's tracks,
    // without writing the per-entry L6A output to memory, and without the
    // cuBLAS GEMM, the chunking and its buffers. FP32, deterministic sums.
    // The throughput probes (l6a_m, l1_l5_hidden_width, l6a_active_channels,
    // fc_single_hidden_layer) and the chunk/reduce options apply to the
    // unfused path only. cuBLAS builds only (ignored otherwise).
    Allen::Property<bool> m_fc_fused {
        this, "fc_fused", true,
        "compute the FC stage in one fused kernel (no L6A round trip through memory, "
        "no cuBLAS GEMM, deterministic sums)"};

    // Precision of layers 2-5 in the fused FC's tensor-core kernel (fc_fused,
    // fc_fused_per_warp, bfloat16 L6A): "bfloat16" runs them as BF16 tensor
    // core products with FP32 accumulation, activations rounded to BF16
    // between layers; layer 1 (raw track features) stays FP32. "auto" (the
    // default) = bfloat16 whenever L6A is on tensor cores.
    Allen::Property<std::string> m_fc_hidden_dtype {
        this, "fc_hidden_dtype", "auto",
        "float32, bfloat16 or auto: precision of FC layers 2-5 in the tensor-core fused FC "
        "(auto = bfloat16 when L6A is bfloat16)"};

    // Fraction of the full-occupancy grid the tensor-core fused FC kernel is
    // launched with. Its blocks take most of an SM's shared memory; below 1,
    // other streams' kernels keep SMs while it runs. At 16 streams on the RTX
    // 3090 (FC + UNet, both set alike): 1: 6.3% loss, 1/2: 4.9%, 1/4: 4.3%,
    // 1/8: 4.0%, 1/16: 3.9% (docs/pvfinder/pvfinder_16_streams.md).
    Allen::Property<float> m_fused_grid_fraction {
        this, "fused_grid_fraction", 0.125f,
        "fraction of the full-occupancy grid for the tensor-core fused FC kernel"};

    // dev_pvfinder_output_histogram, the FC's own KDE estimate, is not read
    // by the UNet (which takes the interval features) nor by anything else in
    // the sequence. The tensor-core fused FC writes it only when this is set,
    // or when dumping for validation (dump_dir); the other FC paths always do.
    Allen::Property<bool> m_write_histogram {
        this, "write_histogram", false,
        "write dev_pvfinder_output_histogram in the tensor-core fused FC (always when dump_dir is set)"};

    // With fc_fused_per_warp: warps take the slots from a work list built on
    // the host from the CSR readback, largest first (the biggest intervals do
    // not start last and run alone at the end of the kernel), each entry
    // carrying the slot's CSR range and row (one load instead of a chain of
    // dependent ones); slots that need no output (empty, with compact rows
    // and no histogram) are left out. Same results: each slot is still
    // computed by one warp, in fixed order.
    Allen::Property<bool> m_fc_largest_first {
        this, "fc_largest_first", true,
        "with fc_fused_per_warp: process the slots in decreasing track count"};

    // With fc_fused: one warp per (event, interval) slot instead of one
    // block (no block-wide barriers; tracks one per lane through L1-L5).
    // Same results: FP32 bit-identical, tensor-core L6A the same sums.
    Allen::Property<bool> m_fc_fused_per_warp {
        this, "fc_fused_per_warp", true,
        "with fc_fused: one warp per slot instead of one block"};

    // Storage type of dev_pvfinder_interval_features. "bfloat16" rounds the
    // Precision of L6A. With "bfloat16", L6A's inputs (the layer-5 outputs)
    // and W6A are rounded to bfloat16 and multiplied on tensor cores with
    // float32 accumulation; bias, LeakyReLU and the sum over tracks stay
    // float32. With fc_fused this is the fused kernel's tensor-core L6A
    // (compute capability 8.0 or newer); on the unfused path the cuBLAS GEMM
    // also stores its output as bfloat16, halving what the reduce kernel
    // reads back (needs use_fused_bias_relu_reduce). Meant for the bfloat16
    // UNet path; float32 is exact. The default "auto" is bfloat16 for the
    // fused FC when it writes bfloat16 features (unet_input_dtype) on a
    // device with bfloat16 tensor cores, float32 otherwise.
    Allen::Property<std::string> m_l6a_dtype {
        this, "l6a_dtype", "auto",
        "float32, bfloat16 or auto: precision of the L6A matrix product (bfloat16 = tensor cores; "
        "auto = bfloat16 in the fused FC when the UNet input is bfloat16)"};
    // features once, in the reduce kernel, so the UNet's BF16 path
    // (pvfinder_unet.use_bf16) reads them without a conversion pass; only
    // that path accepts it. The FC arithmetic and histogram stay float32.
    // cuBLAS builds only (ignored otherwise).
    Allen::Property<std::string> m_unet_input_dtype {
        this, "unet_input_dtype", "float32",
        "storage type of the interval features handed to the UNet: float32 (default) or "
        "bfloat16 (for pvfinder_unet.use_bf16 = true, avoids its input conversion)"};

    // Layout of each bfloat16 row: "ncw" ([channel][bin], default) or "nwc"
    // ([bin][channel]), the layout of the UNet's channels-last BF16 path
    // (pvfinder_unet.bf16_layout = nwc), which then reads the features as they
    // are. Only valid with unet_input_dtype = bfloat16.
    Allen::Property<std::string> m_unet_input_layout {
        this, "unet_input_layout", "ncw",
        "layout of each bfloat16 interval-feature row: ncw (default) or nwc (for "
        "pvfinder_unet.bf16_layout = nwc)"};

    // On top of skip_empty_intervals: also skip non-empty intervals with fewer
    // tracks. The UNet writes its zero-input response there, which is NOT
    // what it would compute from their (non-zero) features. 2 skips
    // single-track intervals, about 23% of the UNet's rows, for about 1% of
    // throughput; on the validation sample they hold 0.28% of the intervals
    // with a KDE bin above 1e-3, at most 0.056
    // (docs/pvfinder/pvfinder_16_streams.md). Not worth it: default 1.
    Allen::Property<unsigned> m_min_interval_tracks {
        this, "min_interval_tracks", 1u,
        "with skip_empty_intervals: tracks an interval needs to go through the UNet "
        "(default 1 = every non-empty interval, exact; larger is NOT physics-exact)"};

    // Validation dump: when non-empty, the first operator() call writes the
    // raw FC inputs and outputs (track-to-interval CSR, per-event track
    // offsets, track features, interval features, histogram) to this
    // directory, for weights/scripts/validate_fc.py to recompute from the checkpoint.
    Allen::Property<std::string> m_dump_dir {
        this, "dump_validation", "",
        "if non-empty, dump FC inputs/outputs of the first slice to this "
        "directory (read by weights/scripts/validate_fc.py)"};
    mutable bool m_dump_done = false;

    // Nothing else reads dev_pvfinder_l6a_output between
    // pvfinder_l6a_bias_relu_kernel's in-place write and
    // pvfinder_reduce_l6a_kernel's read -- they're two full passes (one
    // read-modify-write, one read) over the same buffer that can be fused
    // into the reduce kernel's own read, applying bias+LeakyReLU inline on
    // the raw GEMM output instead of reading a value a separate kernel
    // already wrote back. Same math, one fewer kernel launch, one fewer
    // full DRAM read-modify-write pass over the L6A output buffer.
    Allen::Property<bool> m_use_fused_bias_relu_reduce {
        this, "use_fused_bias_relu_reduce", true,
        "Apply L6A bias+LeakyReLU inline inside pvfinder_reduce_l6a_kernel's "
        "read of the raw GEMM output instead of running "
        "pvfinder_l6a_bias_relu_kernel as a separate pass"};

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
    // host during the T_chunk walk and uploaded once per chunk. This removes
    // the per-slot O(events-in-chunk) scan under concurrent execution.
    Allen::Property<bool> m_use_precomputed_csr_offset {
        this, "use_precomputed_csr_offset", true,
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
    // than a compile-time constant) so datasets can choose an appropriate
    // safety bound without recompiling.
    Allen::Property<unsigned> m_safe_avg_entries_per_event {
        this, "safe_avg_entries_per_event", 450u,
        "Per-event CSR-entry safety margin used to size T_chunk_max = "
        "this * fc_chunk_size (default 450; smaller values reclaim memory "
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
    // pvfinder_reduce_l6a_kernel unconditionally writes explicit zeros
    // for empty slots instead of early-returning (always-on, not gated by
    // this flag -- provably correctness-preserving on its own, since it
    // writes literal 0.0f wherever the memset already would have). This
    // flag controls whether operator() also runs the redundant
    // memsets: dev_pvfinder_output_histogram never needs one in this mode
    // (exactly n_events-sized, fully covered by the kernel);
    // dev_pvfinder_interval_features still needs a small memset for its
    // padding tail (padded_events > n_events, for UNet's batch alignment --
    // never written by any FC kernel), just not the full buffer.
    Allen::Property<bool> m_skip_redundant_memset {
        this, "skip_redundant_memset", true,
        "Skip pvfinder_output_histogram's full memset and shrink "
        "pvfinder_interval_features's memset to just its padding tail, "
        "relying on pvfinder_reduce_l6a_kernel's own explicit zero-writes "
        "for empty (event, interval) slots instead"};

    // The static reduction dispatch uses one block per (event, interval) slot
    // and may not fill the device. The grid-stride dispatch launches a fixed
    // number of blocks sized to the GPU's actual occupancy ceiling for this
    // kernel (SM count * cudaOccupancyMaxActiveBlocksPerMultiprocessor,
    // queried once per thread and cached -- portable across devices), each
    // looping via an atomicAdd-claimed work counter over the same dense
    // (event, interval) slot space (including empty slots, which still
    // self-zero exactly as in the static-grid path). Only combined with
    // warp_parallel_reduce + fused_bias_relu_reduce above.
    Allen::Property<bool> m_use_grid_stride_reduce {
        this, "use_grid_stride_reduce", true,
        "Launch pvfinder_reduce_l6a_kernel as a fixed, occupancy-sized grid "
        "that work-steals over all (event, interval) slots via an atomic "
        "counter, instead of one block per slot"};
};

} // namespace pvfinder_fc_aggregation
