/*****************************************************************************\
* (c) Copyright 2018-2026 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "RichPix2Tracks.cuh"
#include <BinarySearch.cuh>
#include <PrefixSum.cuh>

INSTANTIATE_ALGORITHM(rich_pix2track::rich_pix2track_t);

__global__ void rich_pix2track_count_k(
  const unsigned number_of_photons,
  const Allen::Rich::PhotonReco::Photon* photons,
  unsigned* pix2track_counts)
{
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;

  for (unsigned ph = threadId; ph < number_of_photons; ph += stride) {
    const unsigned pix = photons[ph].pixelIdx;
    atomicAdd(&pix2track_counts[pix], 1u);
  }
}

__global__ void rich_pix2track_fill_k(
  const unsigned number_of_tracks,
  const unsigned number_of_photons,
  const unsigned* photons_offsets,
  const Allen::Rich::PhotonReco::Photon* photons,
  unsigned* pix2track_cursors,
  unsigned* pix2track,
  unsigned* pix2photon)
{
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;

  for (unsigned ph = threadId; ph < number_of_photons; ph += stride) {
    const unsigned track_id = binary_search_rightmost(photons_offsets, number_of_tracks + 1, ph);
    const unsigned pix = photons[ph].pixelIdx;
    // cursors array to avoid ruining offsets order
    const unsigned slot = atomicAdd(&pix2track_cursors[pix], 1u);
    pix2track[slot] = track_id;
    pix2photon[slot] = ph;
  }
}

__global__ void rich_pix2track_debug_k(
  const unsigned n_pixels,
  const unsigned n_photons,
  const unsigned* pix2track_offsets,
  const unsigned* pix2track)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // test counts: sentinel equals n_photons
    printf("pix2track sentinel = %u, expected n_photons = %u\n", pix2track_offsets[n_pixels], n_photons);

    // test if it makes  sense: is monotonic
    for (unsigned pix = 0; pix < min(20u, n_pixels); pix++) {
      if (pix2track_offsets[pix + 1] < pix2track_offsets[pix]) {
        printf(
          "MONOTONICITY VIOLATION at pix %u: offsets[%u]=%u > offsets[%u]=%u\n",
          pix,
          pix,
          pix2track_offsets[pix],
          pix + 1,
          pix2track_offsets[pix + 1]);
      }
    }

    // some track lists: for some pixels that have entries, print their track lists
    for (unsigned pix = 0; pix < n_pixels; pix++) {
      unsigned start = pix2track_offsets[pix];
      unsigned end = pix2track_offsets[pix + 1];
      if (end - start > 1) { // only print pixels shared by multiple tracks
        printf("pixel %u -> %u tracks: ", pix, end - start);
        for (unsigned k = start; k < end; k++) {
          printf("%u ", pix2track[k]);
        }
        printf("\n");
      }
    }
  }
}

void rich_pix2track::rich_pix2track_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  const unsigned n_pixels = first<host_number_of_pixels_t>(arguments);
  const unsigned n_photons = first<host_number_of_photons_t>(arguments);

  set_size<dev_pix2track_offsets_t>(arguments, n_pixels + 1);
  set_size<dev_pix2track_t>(arguments, n_photons);
  set_size<dev_pix2photon_t>(arguments, n_photons);
}

void rich_pix2track::rich_pix2track_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  const unsigned n_tracks = first<host_number_of_tracks_t>(arguments);
  const unsigned n_photons = first<host_number_of_photons_t>(arguments);
  const unsigned n_pixels = first<host_number_of_pixels_t>(arguments);

  Allen::memset_async<dev_pix2track_offsets_t>(arguments, 0, context);

  // count pass
  global_function(rich_pix2track_count_k)(dim3(32), dim3(m_block_dim), context)(
    n_photons, data<dev_rich_photons_t>(arguments), data<dev_pix2track_offsets_t>(arguments));

  PrefixSum::prefix_sum<dev_pix2track_offsets_t>(*this, arguments, context);

  // cursor array to consume pix-track pairs without ruining offsets order
  auto cursors = arguments.template make_buffer<Allen::Store::Scope::Device, unsigned>(n_pixels);
  Allen::memcpy_async(
    cursors.data(),
    data<dev_pix2track_offsets_t>(arguments),
    n_pixels * sizeof(unsigned),
    Allen::memcpyDeviceToDevice,
    context);

  // fill pass
  global_function(rich_pix2track_fill_k)(dim3(32), dim3(m_block_dim), context)(
    n_tracks,
    n_photons,
    data<dev_offsets_rich_photons_t>(arguments),
    data<dev_rich_photons_t>(arguments),
    cursors.data(),
    data<dev_pix2track_t>(arguments),
    data<dev_pix2photon_t>(arguments));
}
