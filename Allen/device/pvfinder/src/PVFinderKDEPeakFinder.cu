#include "PVFinderKDEPeakFinder.cuh"

#include <cstdio>
#include <cstdint>
#include <filesystem>
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
        static_cast<float>(m_min_integral_tracks),
        static_cast<unsigned>(m_min_width));

    // Validation dump -- fires once when dump_dir property is non-empty.
    const std::string& dump_dir = m_dump_dir.value();
    if (!dump_dir.empty() && !m_dump_done) {
        cudaStreamSynchronize(context.stream());
        std::error_code ec;
        std::filesystem::create_directories(dump_dir, ec);

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
// CUDA kernel -- one block per event.
//
// The logic below intentionally mirrors pvfinder_pytorch/utils.py:
//   * thresholded "active" regions
//   * minimum region width in bins
//   * minimum integral
//   * KDE-weighted centroid
//   * special merged-peak split/end heuristic
//
// A single thread performs the scan so the behavior matches the Python
// reference as closely as possible during validation/debugging.
// ---------------------------------------------------------------------------
__global__ void pvfinder_kde_peak_finder(
    pvfinder_kde_peak_finder::Parameters parameters,
    const float kde_peak_threshold,
    const float min_integral_tracks,
    const unsigned min_width)
{
    const unsigned seq_idx      = blockIdx.x;
    const unsigned event_number = parameters.dev_event_list[seq_idx];

    const float* __restrict__ kde    = parameters.dev_pvfinder_kde_output + seq_idx * KDE_N_BINS;
    float*                    zpeaks = parameters.dev_nn_zpeaks + event_number * PV::max_number_vertices;
    if (threadIdx.x == 0) {
        unsigned state = 0u;
        float integral = 0.f;
        float weighted_z = 0.f;
        unsigned n_seeds = 0u;
        bool peak_passed = false;

        for (int i = 0; i < KDE_N_BINS; ++i) {
            const float v = kde[i];
            if (v >= kde_peak_threshold) {
                ++state;
                integral += v;
                weighted_z += v * bin_to_z(i);

                if (i > 0 && kde[i - 1] > v + 0.05f && kde[i - 1] > 1.1f * v) {
                    peak_passed = true;
                }
            }

            const bool end_region =
                (v < kde_peak_threshold) ||
                (i == KDE_N_BINS - 1) ||
                ((i > 0) && (kde[i - 1] < v) && peak_passed);

            if (end_region && state > 0u) {
                if (state >= min_width && integral >= min_integral_tracks && n_seeds < PV::max_number_vertices) {
                    zpeaks[n_seeds] = weighted_z / integral;
                    ++n_seeds;
                }

                state = 0u;
                integral = 0.f;
                weighted_z = 0.f;
                peak_passed = false;
            }
        }

        parameters.dev_nn_number_of_zpeaks[event_number] = n_seeds;
    }
}

} // namespace pvfinder_kde_peak_finder
