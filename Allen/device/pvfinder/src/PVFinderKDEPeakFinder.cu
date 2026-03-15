#include "PVFinderKDEPeakFinder.cuh"

#include <cstdio>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

INSTANTIATE_ALGORITHM(pvfinder_kde_peak_finder::pvfinder_kde_peak_finder_t)

namespace pvfinder_kde_peak_finder {

// Convert flat bin index b in [0, KDE_N_BINS) to z bin-centre in mm.
// Derivation: interval i = b/100, bin-in-interval j = b%100
//   z_centre = -100 + 10*i + 0.1*j + 0.05
//            = -100 + 0.1*(100*i + j) + 0.05
//            = KDE_ZMIN + KDE_DZ*b + 0.5*KDE_DZ
__device__ __forceinline__ float bin_to_z(int b)
{
    return KDE_ZMIN + KDE_DZ * static_cast<float>(b) + 0.5f * KDE_DZ;
}

// ---------------------------------------------------------------------------
// set_arguments_size
// ---------------------------------------------------------------------------
void pvfinder_kde_peak_finder_t::set_arguments_size(
    ArgumentReferences<Parameters> arguments,
    const RuntimeOptions&,
    const Constants&) const
{
    const unsigned n_events = first<host_number_of_events_t>(arguments);
    set_size<dev_nn_zpeaks_t>(arguments, n_events * PV::max_number_vertices);
    set_size<dev_nn_number_of_zpeaks_t>(arguments, n_events);
}

// ---------------------------------------------------------------------------
// operator()
// ---------------------------------------------------------------------------
void pvfinder_kde_peak_finder_t::operator()(
    const ArgumentReferences<Parameters>& arguments,
    const RuntimeOptions&,
    const Constants&,
    const Allen::Context& context) const
{
    // One block per event in the event list, one warp (32 threads) per block.
    global_function(pvfinder_kde_peak_finder)(
        dim3(size<dev_event_list_t>(arguments)),
        m_block_dim,
        context)(
        arguments,
        static_cast<float>(m_kde_peak_threshold),
        static_cast<float>(m_min_integral_tracks));

    // Validation dump -- fires once when dump_dir property is non-empty.
    const std::string& dump_dir = m_dump_dir.value();
    if (!dump_dir.empty() && !m_dump_done) {
        cudaStreamSynchronize(context.stream());

        const unsigned n_events = first<host_number_of_events_t>(arguments);
        std::vector<float>    h_zpeaks(n_events * PV::max_number_vertices);
        std::vector<unsigned> h_npeaks(n_events);

        cudaMemcpy(
            h_zpeaks.data(),
            data<dev_nn_zpeaks_t>(arguments),
            n_events * PV::max_number_vertices * sizeof(float),
            cudaMemcpyDeviceToHost);
        cudaMemcpy(
            h_npeaks.data(),
            data<dev_nn_number_of_zpeaks_t>(arguments),
            n_events * sizeof(unsigned),
            cudaMemcpyDeviceToHost);

        // Binary layout:
        //   uint32  magic = 0xAB2F
        //   uint32  n_events
        //   for each event e:
        //       uint32  n_peaks
        //       float32[n_peaks]  z_seeds  (sorted by cluster order, ascending z)
        const std::string path = dump_dir + "/allen_nn_zpeaks.bin";
        std::ofstream f(path, std::ios::binary);
        if (f) {
            const uint32_t magic = 0xAB2Fu;
            const uint32_t ne    = n_events;
            f.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
            f.write(reinterpret_cast<const char*>(&ne),    sizeof(ne));
            for (unsigned e = 0; e < n_events; ++e) {
                const uint32_t np = h_npeaks[e];
                f.write(reinterpret_cast<const char*>(&np), sizeof(np));
                f.write(
                    reinterpret_cast<const char*>(h_zpeaks.data() + e * PV::max_number_vertices),
                    np * sizeof(float));
            }
            printf("[pvfinder_kde_peak_finder] Validation dump written to %s (%u events)\n",
                   dump_dir.c_str(), n_events);
        } else {
            printf("[pvfinder_kde_peak_finder] WARNING: could not open %s for writing\n",
                   path.c_str());
        }
        m_dump_done = true;
    }
}

// ---------------------------------------------------------------------------
// CUDA kernel -- one block per event, exactly one warp (blockDim.x = 32).
//
// Three phases:
//   1. Find proto-cluster edges (contiguous above-threshold regions) using
//      warp-ballot intrinsics -- same technique as pv_beamline_peak.
//      KDE_N_BINS = 4000 = 32 * 125, so every iteration uses a full warp.
//   2. Filter proto-clusters by integrated KDE weight >= min_integral_tracks.
//   3. For each accepted cluster compute the KDE-weighted centroid z-seed.
// ---------------------------------------------------------------------------
__global__ void pvfinder_kde_peak_finder(
    pvfinder_kde_peak_finder::Parameters parameters,
    const float kde_peak_threshold,
    const float min_integral_tracks)
{
    // seq_idx: sequential position matching UNet's output ordering (0..n_events-1).
    // event_number: actual event index used to write into global output buffers.
    const unsigned seq_idx      = blockIdx.x;
    const unsigned event_number = parameters.dev_event_list[seq_idx];

    // Pointers to this event's KDE input and z-peaks output.
    const float* __restrict__ kde    = parameters.dev_pvfinder_kde_output + seq_idx * KDE_N_BINS;
    float*                    zpeaks = parameters.dev_nn_zpeaks + event_number * PV::max_number_vertices;

    // Shared memory: cluster edge index pairs -- (ibegin, iend) per proto-cluster.
    using BinIndex = unsigned short;  // sufficient for indices up to 4000
    __shared__ BinIndex clusteredges[PV::max_number_clusteredges];  // 200 entries
    unsigned number_of_clusteredges = 0u;

    // =========================================================================
    // Phase 1 -- proto-cluster edge detection via warp ballot.
    //
    // All 32 threads process KDE_N_BINS / 32 = 125 full-warp iterations.
    // prev_mask initialised to ~0u ("all empty") so the very first non-empty
    // bin produces an opening edge.
    // =========================================================================
#if defined(TARGET_DEVICE_CUDA)
    {
        uint32_t prev_mask  = ~0u;
        const uint32_t lane_eq = 1u << threadIdx.x;
        const uint32_t lane_lt = lane_eq - 1u;

        for (int i = threadIdx.x; i < KDE_N_BINS; i += blockDim.x) {
            const bool empty = (kde[i] < kde_peak_threshold);

            // cur_mask: bit j = 1 if thread j's bin is empty.
            const uint32_t cur_mask = __ballot_sync(~0u, empty);

            // edge_mask: bit j = 1 where there is a transition between bins i-1 and i.
            // __funnelshift_l(hi, lo, 1) = (lo << 1) | (hi >> 31), i.e. shifts {hi:lo}
            // left by 1 and returns the upper 32 bits -- equivalent to "prev bit in cur_mask".
            const uint32_t edge_mask = cur_mask ^ __funnelshift_l(prev_mask, cur_mask, 1u);
            prev_mask = cur_mask;

            // Compute this thread's slot in clusteredges[] and write if it has an edge.
            const int idx = static_cast<int>(number_of_clusteredges)
                          + static_cast<int>(__popc(edge_mask & lane_lt));
            if ((edge_mask & lane_eq) && idx < static_cast<int>(PV::max_number_clusteredges)) {
                clusteredges[idx] = static_cast<BinIndex>(i);
            }
            number_of_clusteredges += __popc(edge_mask);
        }
    }
#else
    // Non-CUDA fallback: single-threaded serial scan.
    {
        bool prev_empty = true;
        for (int i = 0; i < KDE_N_BINS; ++i) {
            const bool empty = (kde[i] < kde_peak_threshold);
            if (empty != prev_empty && number_of_clusteredges < PV::max_number_clusteredges) {
                clusteredges[number_of_clusteredges++] = static_cast<BinIndex>(i);
            }
            prev_empty = empty;
        }
        // Close any cluster still open at the array boundary.
        if (!prev_empty && number_of_clusteredges < PV::max_number_clusteredges) {
            clusteredges[number_of_clusteredges++] = static_cast<BinIndex>(KDE_N_BINS);
        }
    }
#endif

    // Guard: if KDE ends above threshold the last cluster has no closing edge.
    // Round down to nearest even count to discard the orphan opening edge.
    number_of_clusteredges &= ~1u;

    __syncthreads();  // make clusteredges visible to all threads before Phase 2

    // =========================================================================
    // Phase 2 -- filter proto-clusters by integrated KDE weight.
    //
    // Each cluster [ibegin, iend) is integrated with a parallel warp reduction.
    // Accepted clusters are compacted back into clusteredges[] (outIdx <= c).
    // The condition is identical in all threads (warp-reduced integral is
    // broadcast to all lanes), so outIdx stays consistent across the warp.
    // =========================================================================
    int outIdx = 0;
    for (int c = 0; c < static_cast<int>(number_of_clusteredges); c += 2) {
        const BinIndex ibegin = clusteredges[c];
        const BinIndex iend   = clusteredges[c + 1];

        float integral = 0.f;
        for (int j = static_cast<int>(ibegin) + threadIdx.x;
             j < static_cast<int>(iend);
             j += blockDim.x) {
            integral += kde[j];
        }
#if defined(TARGET_DEVICE_CUDA)
        for (int offset = 16; offset > 0; offset >>= 1) {
            integral += __shfl_xor_sync(~0u, integral, offset);
        }
        __syncwarp();
#endif

        if (integral > min_integral_tracks) {
            if (threadIdx.x == 0) {
                clusteredges[outIdx]     = ibegin;
                clusteredges[outIdx + 1] = iend;
            }
            outIdx += 2;
        }
        __syncthreads();  // before next read of clusteredges[c+2..] or write at outIdx
    }
    number_of_clusteredges = static_cast<unsigned>(outIdx);
    __syncthreads();

    // =========================================================================
    // Phase 3 -- KDE-weighted centroid z-seed per accepted cluster.
    //
    // z_seed = sum(z_centre(b) * kde[b]) / sum(kde[b])  for b in [ibegin, iend)
    //
    // n_seeds is broadcast from thread 0 after each write so all threads see
    // the same value and the early-exit check is consistent across the warp.
    // =========================================================================
    const unsigned n_clusters = number_of_clusteredges / 2u;
    unsigned n_seeds = 0u;

    for (unsigned c = 0u; c < n_clusters; ++c) {
        if (n_seeds >= PV::max_number_vertices) break;

        const BinIndex ibegin = clusteredges[c * 2u];
        const BinIndex iend   = clusteredges[c * 2u + 1u];

        float wz = 0.f;
        float w  = 0.f;
        for (int j = static_cast<int>(ibegin) + threadIdx.x;
             j < static_cast<int>(iend);
             j += blockDim.x) {
            const float kval = kde[j];
            wz += kval * bin_to_z(j);
            w  += kval;
        }
#if defined(TARGET_DEVICE_CUDA)
        for (int offset = 16; offset > 0; offset >>= 1) {
            wz += __shfl_xor_sync(~0u, wz, offset);
            w  += __shfl_xor_sync(~0u, w,  offset);
        }
        __syncwarp();
#endif

        if (threadIdx.x == 0) {
            // Fall back to bin midpoint if cluster integral is pathologically zero.
            const float z_seed = (w > 0.f)
                ? wz / w
                : bin_to_z((static_cast<int>(ibegin) + static_cast<int>(iend)) / 2);
            zpeaks[n_seeds] = z_seed;
            ++n_seeds;
        }

        // Broadcast updated n_seeds to all threads for consistent loop control.
#if defined(TARGET_DEVICE_CUDA)
        n_seeds = __shfl_sync(~0u, n_seeds, 0u);
#endif
    }

    if (threadIdx.x == 0) {
        parameters.dev_nn_number_of_zpeaks[event_number] = n_seeds;
    }
}

} // namespace pvfinder_kde_peak_finder
