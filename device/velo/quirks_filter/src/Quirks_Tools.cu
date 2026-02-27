/*****************************************************************************\
 * (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
 *                                                                             *
 * This software is distributed under the terms of the Apache License          *
 * version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
 *                                                                             *
 * In applying this licence, CERN does not waive the privileges and immunities *
 * granted to it by virtue of its status as an Intergovernmental Organization  *
 * or submit itself to any jurisdiction.                                       *
 \*****************************************************************************/
#include "Quirks_Tools.cuh"
#include <BackendCommon.h>
#include <MEPTools.h>

INSTANTIATE_ALGORITHM(quirks_tools::quirks_tools_t)

namespace quirks_tools {

  // Pair-list generator kernel: compute per-hit r/phi and original hit indices, then for each A hit
  // find the nearest hit ~180 degrees apart (binary search on module phi) and emit pairs.
  __global__ void generate_pair_list_kernel(
    quirks_tools::Parameters parameters,
    const unsigned max_opposite_considered,
    const unsigned hit_threshold,
    const unsigned max_pairs_per_module,
    const float maxPHI,
    const float maxR)
  {
    const unsigned number_of_VELO_layers = Velo::Constants::n_module_pairs;
    const unsigned block_index = blockIdx.x;
    const unsigned event_block = block_index / number_of_VELO_layers;
    const unsigned layer = block_index % number_of_VELO_layers;

    const unsigned event_number = parameters.dev_event_list[event_block];
    const unsigned off_idx = event_number * number_of_VELO_layers + layer;
    const unsigned count = parameters.dev_module_cluster_num[off_idx];

    // initialize pair count for this module (must be done for every module, even if we early-return)
    if (threadIdx.x == 0 && threadIdx.y == 0) parameters.dev_pair_count[off_idx] = 0u;
    __syncthreads();

    if (count == 0u || count >= hit_threshold) return;

    const unsigned layer_offset = parameters.dev_offsets_estimated_input_size[off_idx];
    const float maxR_sq = maxR * maxR;

    const auto& clusters = parameters.dev_velo_clusters[event_number];

    // convert maxPHI to i16 domain for tolerance around 180 deg
    const int16_t maxPHI_i16 = hit_phi_float_to_16(maxPHI * Allen::constants::pi_f_float / 180.0f);
    const int16_t window_start = hit_phi_float_to_16((180 - maxPHI) / 180 * Allen::constants::pi_f_float);

    // For each A-side precomp entry, find candidate opposite-side hit using binary search on clusters.phi
    for (unsigned ia = threadIdx.x; ia < count; ia += blockDim.x) {
      const unsigned hitA = layer_offset + ia;
      const int16_t phi_i = clusters.phi(hitA);
      // phii is defined in radians betewwn -pi and pi,
      if (phi_i > 0) break; // A side only
      // search target = phii + 180deg
      const int16_t target = phi_i + window_start;

      // binary search on the module phi array (sorted)
      // We want to get all candidates that are within maxPHI of 180 deg from phii
      const unsigned m = binary_search_leftmost<int16_t>(clusters.phi_begin() + layer_offset, count, target);

      for (unsigned j = 0; j < max_opposite_considered; ++j) {
        const unsigned cand = m + j;
        // normalize candidate into [0,count) then add layer_offset
        const unsigned hit_idx_in_layer = (cand < count) ? cand : (cand - count);
        const unsigned hitCand = layer_offset + hit_idx_in_layer;
        // phi distance from 180 deg
        const int16_t phi_j = clusters.phi(hitCand);
        // We have to calculate the difference in int16_t domain carefully to account for wrap-around
        // phii<0 and phij>0
        const int16_t dphi_i16 = phi_j - phi_i;
        const int16_t abs_phi_diff =
          dphi_i16 < 0 ? Velo::Tools::max_output_value_i16 + dphi_i16 : Velo::Tools::max_output_value_i16 - dphi_i16;
        if (abs_phi_diff > maxPHI_i16 or abs_phi_diff < 0) continue;

        // quick r test
        const float xi = clusters.x(hitA);
        const float yi = clusters.y(hitA);
        const float xj = clusters.x(hitCand);
        const float yj = clusters.y(hitCand);
        const float rA = sqrtf(xi * xi + yi * yi);
        const float rB = sqrtf(xj * xj + yj * yj);
        const float dr = rA - rB;
        if (dr * dr >= maxR_sq) continue;

        // append pair (atomic)
        const unsigned pos = atomicAdd(&parameters.dev_pair_count[off_idx], 1u);
        if (pos < (unsigned) max_pairs_per_module) {
          const unsigned module_pair_base = off_idx * (unsigned) max_pairs_per_module;
          // store the normalized candidate index
          parameters.dev_pair_list[module_pair_base + pos] =
            (uint32_t)((ia << 16) | (uint32_t)(hit_idx_in_layer & 0xFFFFu));
        }
        else {
          break;
        }
      }
    } // for ia
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0 && parameters.dev_pair_count[off_idx] > max_pairs_per_module) {
      parameters.dev_pair_count[off_idx] = 0u;
    }
  }

  // Selection kernel: iterate over pair-lists
  __global__ void quirks_selection_kernel(
    quirks_tools::Parameters parameters,
    const float maxPHIDF,
    const unsigned minStations,
    const unsigned max_pairs_per_module)
  {
    const int16_t maxPHIDF_i16 = hit_phi_float_to_16(maxPHIDF * Allen::constants::pi_f_float / 180.0f);

    const unsigned event_block_index = blockIdx.x;
    const unsigned event_number = parameters.dev_event_list[event_block_index];

    const unsigned number_of_VELO_layers = Velo::Constants::n_module_pairs;
    const unsigned effective_layers = number_of_VELO_layers - minStations;

    __shared__ int s_found;
    if (threadIdx.x == 0 && threadIdx.y == 0) s_found = 0;
    __syncthreads();

    const unsigned seed = threadIdx.x;
    const auto& clusters = parameters.dev_velo_clusters[event_number];

    // main search: each seed-thread processes multiple seed indices
    for (unsigned seed_layer = seed; seed_layer < effective_layers; seed_layer += blockDim.x) {
      if (seed_layer > effective_layers) continue;
      if (s_found) break;
      const unsigned off_idx_base = event_number * number_of_VELO_layers + seed_layer;
      // use module start as base for precomp per-side arrays
      const unsigned layer_offset = parameters.dev_offsets_estimated_input_size[off_idx_base];
      const unsigned pair_module_base = (unsigned) off_idx_base * (unsigned) max_pairs_per_module;
      const unsigned pair_count = parameters.dev_pair_count[off_idx_base];

      if (pair_count == (unsigned) 0) {
        continue;
      }

      for (unsigned p = threadIdx.y; p < pair_count; p += blockDim.y) {
        if (s_found) break;
        // Get indexes of the pair that makes a hit-pair.
        const uint32_t pair_enc = parameters.dev_pair_list[pair_module_base + p];
        const int i = (int) (pair_enc >> 16);
        const int j = (int) (pair_enc & 0xFFFF);

        // read precomputed coordinates using compact bases
        const float x1 = clusters.x(layer_offset + i);
        const float y1 = clusters.y(layer_offset + i);
        const float r1 = sqrtf(x1 * x1 + y1 * y1);
        const int16_t phi1_i16 = clusters.phi(layer_offset + i);

        const float x2 = clusters.x(layer_offset + j);
        const float y2 = clusters.y(layer_offset + j);
        const float r2 = sqrtf(x2 * x2 + y2 * y2);
        const int16_t phi2_i16 = clusters.phi(layer_offset + j);

        unsigned station_count = 1;
        float mean_r = 0.5f * (r1 + r2);
        bool sel_r = false;
        bool sel_r2 = false;
        bool skipped_layer = false;
        // extend across subsequent layers by scanning their pair-lists
        for (unsigned layer = seed_layer + 1; layer < number_of_VELO_layers; ++layer) {
          if (s_found) {
            break;
          }
          const unsigned off_idx_j = event_number * number_of_VELO_layers + layer;
          const unsigned layer_offset_j = parameters.dev_offsets_estimated_input_size[off_idx_j];
          const unsigned pair_module_base_j = off_idx_j * max_pairs_per_module;
          const unsigned pair_count_j = parameters.dev_pair_count[off_idx_j];

          bool found_in_layer = false;

          // scan pair-list of layer layer
          for (unsigned pj = 0; pj < pair_count_j && !found_in_layer; ++pj) {
            const uint32_t enc = parameters.dev_pair_list[pair_module_base_j + pj];
            const int a = (int) (enc >> 16);
            const int b = (int) (enc & 0xFFFF);

            const float xj1 = clusters.x(layer_offset_j + a);
            const float yj1 = clusters.y(layer_offset_j + a);
            const float rj1 = sqrtf(xj1 * xj1 + yj1 * yj1);
            const int16_t phij1_i16 = clusters.phi(layer_offset_j + a);

            int16_t dphi1_i16 = phij1_i16 - phi1_i16;
            if (dphi1_i16 < 0) {
              dphi1_i16 = -dphi1_i16;
            }
            int16_t dphi2_i16 = abs(phij1_i16 - phi2_i16);
            if (dphi2_i16 < 0) {
              dphi2_i16 = -dphi2_i16;
            }

            int16_t dphi_i16 = dphi1_i16 < dphi2_i16 ? dphi1_i16 : dphi2_i16;

            if (dphi_i16 > maxPHIDF_i16) {
              continue;
            }

            const float xj2 = clusters.x(layer_offset_j + b);
            const float yj2 = clusters.y(layer_offset_j + b);
            const float rj2 = sqrtf(xj2 * xj2 + yj2 * yj2);

            const float mean_r_j = 0.5f * (rj1 + rj2);
            if (mean_r_j <= mean_r) sel_r = true;
            if (mean_r_j >= mean_r) sel_r2 = true;
            mean_r = mean_r_j;

            ++station_count;
            found_in_layer = true;
            break;
          } // pj

          if (!found_in_layer) {
            if (skipped_layer) {
              break;
            }
            else {
              skipped_layer = true;
            }
          }

          if (station_count >= minStations && sel_r && sel_r2) {
            if (atomicCAS((int*) &s_found, 0, 1) == 0) {
            }
            break;
          }
        } // layer

      } // for p
    }   // seed_layer  stride loop

    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
      parameters.dev_quirks_pairs[event_number] = (unsigned) s_found;
    }
  }
  void quirks_tools_t::set_arguments_size(
    ArgumentReferences<Parameters> arguments,
    const RuntimeOptions&,
    const Constants&) const
  {
    set_size<dev_quirks_pairs_t>(arguments, first<host_number_of_events_t>(arguments));

    const unsigned num_events = first<host_number_of_events_t>(arguments);
    const unsigned number_of_VELO_layers = Velo::Constants::n_module_pairs;

    // allocate pair-list storage (uint32 pairs) and counts
    const unsigned max_pairs = m_max_pairs_per_module;
    set_size<dev_pair_list_t>(arguments, num_events * number_of_VELO_layers * max_pairs);
    set_size<dev_pair_count_t>(arguments, num_events * number_of_VELO_layers);
  }

  void quirks_tools_t::operator()(
    const ArgumentReferences<Parameters>& arguments,
    const RuntimeOptions&,
    const Constants&,
    const Allen::Context& context) const
  {
    const unsigned number_of_VELO_layers = Velo::Constants::n_module_pairs;
    const unsigned num_events = size<dev_event_list_t>(arguments);

    const unsigned precomp_blocks = num_events * number_of_VELO_layers;

    // Launch pair-list generator (one block per module)
    const dim3 pair_threads = dim3(128, 1, 1); // tweak as needed; using 128 threads per module
    global_function(generate_pair_list_kernel)(dim3(precomp_blocks), pair_threads, context)(
      arguments,
      static_cast<unsigned>(m_max_opposite_considered),
      static_cast<unsigned>(m_hit_threshold),
      static_cast<unsigned>(m_max_pairs_per_module),
      static_cast<float>(m_maxPHI),
      static_cast<float>(m_maxR));

    // Launch selection kernel
    const unsigned bx = m_block_dim_x;
    const unsigned by = m_block_dim_y;
    dim3 block_dim(bx, by);
    dim3 grid_dim(num_events);
    global_function(quirks_selection_kernel)(grid_dim, block_dim, context)(
      arguments,
      static_cast<float>(m_maxPHIDF),
      static_cast<float>(m_minStations),
      static_cast<unsigned>(m_max_pairs_per_module));
  }

} // namespace quirks_tools
