#include "PVFinderNCWLayout.cuh"

INSTANTIATE_ALGORITHM(pvfinder_ncw_layout::pvfinder_ncw_layout_t)

namespace pvfinder_ncw_layout {

// ---------------------------------------------------------------------------
// Kernel: element-wise copy from interval-features buffer to NCW tensor.
//
// Both buffers have identical byte layout:
//   [event_idx][interval_idx][channel_idx][bin_idx]
// The output is logically viewed as:
//   [N = event_idx*N_INTERVALS + interval_idx][channel_idx][bin_idx]
// Because the strides are identical this is a flat memcpy in parallel.
// A dedicated kernel is used so Allen's pool tracks the output independently
// and future phases (e.g. pre-processing normalization) can be inserted here.
// ---------------------------------------------------------------------------
__global__ void pvfinder_ncw_layout_kernel(
    pvfinder_ncw_layout_t::Parameters parameters,
    unsigned total_slices)  // = n_events * N_INTERVALS
{
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread handles one float element across the full flat tensor.
    const unsigned total_elems = total_slices * ELEMS_PER_SLICE;
    if (tid >= total_elems) return;

    parameters.dev_pvfinder_ncw_tensor[tid] =
        parameters.dev_pvfinder_interval_features[tid];
}

void pvfinder_ncw_layout_t::set_arguments_size(
    ArgumentReferences<Parameters> arguments,
    const RuntimeOptions&,
    const Constants&) const
{
    const unsigned n_events = first<host_number_of_events_t>(arguments);
    // Output: N = n_events * N_INTERVALS slices of [C=8, W=100]
    set_size<dev_pvfinder_ncw_tensor_t>(arguments, n_events * ELEMS_PER_EVENT_IN);
}

void pvfinder_ncw_layout_t::operator()(
    const ArgumentReferences<Parameters>& arguments,
    const RuntimeOptions&,
    const Constants&,
    const Allen::Context& context) const
{
    const unsigned n_events     = first<host_number_of_events_t>(arguments);
    const unsigned total_slices = n_events * N_INTERVALS;
    const unsigned total_elems  = total_slices * ELEMS_PER_SLICE;

    const dim3 block = m_block_dim;
    const dim3 grid((total_elems + block.x - 1) / block.x);

    global_function(pvfinder_ncw_layout_kernel)(grid, block, context)(
        arguments, total_slices);
}

} // namespace pvfinder_ncw_layout
