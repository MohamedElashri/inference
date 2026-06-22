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

#include "RichRayTraceCherenkovCones.cuh"
#include <cmath>
#include <fstream>
#include <BinarySearch.cuh>
#include <PrefixSum.cuh>

INSTANTIATE_ALGORITHM(rich_raytrace_cherenkov_cones::rich_raytrace_cherenkov_cones_t);

namespace Rich::Raytracing {
  Allen::Rich::DetectorArray<Allen::Rich::PanelArray<Allen::Rich::Detector::PDPanelLookup>> pd_panel_lut {};
} // namespace Rich::Raytracing

template<Allen::Rich::Detector::DetectorType richIdx>
void rich_raytrace_cherenkov_cones::rich_raytrace_cherenkov_cones_t::updateRich(
  const Allen::Rich::RichDetector<richIdx>* rich) const
{
  // Build photodetector panel lookup tables
  for (const auto side : Allen::Rich::Detector::sides()) {
    const auto& allenPanel = rich->m_panels[side];
    const auto& g2panel = allenPanel.globalToPDPanel();
    auto& lut = Rich::Raytracing::pd_panel_lut[richIdx][side];
    const unsigned npds = Allen::Rich::Detector::PDPanel<richIdx>::PDsPerPanel;
    // printf("Rich%d side %d\n", richIdx + 1, side);
    lut.init(richIdx, g2panel, allenPanel.pds(), npds);
  }
}

void rich_raytrace_cherenkov_cones::rich_raytrace_cherenkov_cones_t::update(const Constants& constants) const
{
  const auto rich = Allen::Rich::Detector::detectorTypeFromNumber(m_current_rich.value());
  if (rich == Allen::Rich::Detector::Rich1) {
    updateRich<Allen::Rich::Detector::Rich1>(constants.host_rich_1_geometry);
  }
  else {
    updateRich<Allen::Rich::Detector::Rich2>(constants.host_rich_2_geometry);
  }
}

__global__ void rich_raytrace_npoints_k(
  const unsigned total_number_of_tracks,
  const Allen::Rich::ParticleHypos* hypos_tracks,
  unsigned* photons_offset,
  const unsigned nPointsMin,
  const unsigned nPointsMax)
{
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;
  for (unsigned i = threadId; i < total_number_of_tracks; i += stride) {
    const auto hypos = hypos_tracks[i];
    float lightestCKtheta = 0.f;
    UNROLL(Allen::Rich::NRealParticleTypes)
    for (unsigned hypo_index = 0; hypo_index < Allen::Rich::NRealParticleTypes; ++hypo_index) {
      const auto hypo = static_cast<Allen::Rich::ParticleIDType>(hypo_index);
      const float ckTheta = hypos.ckTheta[hypo];
      if (std::isfinite(ckTheta) && ckTheta > 0.f) {
        lightestCKtheta = ckTheta;
        break;
      }
    }

    UNROLL(Allen::Rich::NRealParticleTypes)
    for (unsigned hypo_index = 0; hypo_index < Allen::Rich::NRealParticleTypes; ++hypo_index) {
      const auto hypo = static_cast<Allen::Rich::ParticleIDType>(hypo_index);
      const float ckTheta = hypos.ckTheta[hypo];
      unsigned count = 0; // if invalid hypo
      if (std::isfinite(ckTheta) && ckTheta > 0.f && lightestCKtheta > 0.f) {
        const float nPoints =
          fminf(static_cast<float>(nPointsMax), nPointsMin + (nPointsMax - nPointsMin) * ckTheta / lightestCKtheta);
        count = static_cast<unsigned>(nPoints);
      }
      photons_offset[i * Allen::Rich::NRealParticleTypes + hypo] = count;
    }
  }
}

template<Allen::Rich::Detector::DetectorType richIdx>
__global__ void rich_raytrace_ck_cones_k(
  const Allen::Rich::RichDetector<richIdx>* rich,
  const Allen::Rich::PanelArray<Allen::Rich::Detector::PDPanelLookup> pd_panel_lut,
  const unsigned number_of_events,
  const unsigned total_number_of_tracks,
  const unsigned* track_offsets,
  const unsigned total_number_of_photons,
  const unsigned* hypo_offsets,
  const float3* segs_best_point,
  const float3* segs_best_momentum,
  const Allen::Rich::ParticleHypos* hypos_tracks,
  int* pd_ids)
{
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;
  for (unsigned i = threadId; i < total_number_of_photons; i += stride) {
    const unsigned track_hypo_id =
      binary_search_rightmost(hypo_offsets, total_number_of_tracks * Allen::Rich::NRealParticleTypes + 1, i);
    const unsigned hypo_offset = hypo_offsets[track_hypo_id];
    const unsigned nPoints = hypo_offsets[track_hypo_id + 1] - hypo_offset;

    const unsigned track_id = track_hypo_id / Allen::Rich::NRealParticleTypes;
    const unsigned event_number = binary_search_rightmost(track_offsets, number_of_events + 1, track_id);
    const auto hypo = static_cast<Allen::Rich::ParticleIDType>(track_hypo_id % Allen::Rich::NRealParticleTypes);
    const unsigned photon_id = i - hypo_offset;

    const float3 emission_point = segs_best_point[track_id];
    const auto side = Allen::Rich::side<richIdx>(emission_point);
    const float ckTheta = hypos_tracks[track_id].ckTheta[hypo];
    const float ckPhi = ((float) (2 * M_PI)) * photon_id / nPoints;

    Allen::Rich::PhotonReco::RotationMatrix rotation {segs_best_momentum[track_id]};
    float3 dir = rotation.vectorAtThetaPhi(ckTheta, ckPhi);

    float2 lpos = Allen::Rich::pointAtPanel(rich, emission_point, dir, side);
    const unsigned panel_offset =
      (side * number_of_events + event_number) * Allen::Rich::Detector::PDPanel<richIdx>::PDsPerPanel;
    int pd_index = pd_panel_lut[side].find(lpos);
    if (pd_index != -1) pd_index += panel_offset;
    pd_ids[i] = pd_index;
    /*if (track_id == 0 && richIdx == 0 && hypo == Allen::Rich::Electron) {
      std::cout << "[Allen] id "<< photon_id << " / " << nPoints << " (" <<lpos.x<<", "<<lpos.y<<") side: "<<side<<" pd:
    "<<pd_index<<std::endl;
    }*/
  }
}

__global__ void rich_raytrace_count_distinct_k(
  const unsigned total_number_of_hypos,
  const unsigned* hypo_offsets,
  const int* pd_ids,
  unsigned* counts)
{
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;
  for (unsigned i = threadId; i < total_number_of_hypos; i += stride) {
    const auto start = hypo_offsets[i];
    const auto end = hypo_offsets[i + 1];

    unsigned count = 0;
    int last_pd = -1;
    for (unsigned j = start; j < end; j++) {
      const int pd_id = pd_ids[j];
      if (pd_id == -1 || pd_id == last_pd) continue;
      last_pd = pd_id;
      count++;
    }
    counts[i] = count;
  }
}

__global__ void rich_raytrace_fill_geomeffs_k(
  const unsigned total_number_of_hypos,
  const unsigned* hypo_offsets,
  const int* pd_ids,
  const unsigned* geomeffs_offsets,
  int* geomeffs_pd_ids,
  float* geomeffs_fractions,
  Allen::Rich::HypoData<float>* geomeffs_fractions_per_hypo)
{
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;
  for (unsigned i = threadId; i < total_number_of_hypos; i += stride) {
    const auto start = hypo_offsets[i];
    const auto end = hypo_offsets[i + 1];
    int* geomeffs_pd_ids_hypo = geomeffs_pd_ids + geomeffs_offsets[i];
    float* geomeffs_fractions_hypo = geomeffs_fractions + geomeffs_offsets[i];
    const float pdInc = 1.f / (end - start);

    unsigned idx = 0;
    int last_pd = -1;
    float pd_fraction = 0;
    float total_fraction = 0;
    for (unsigned j = start; j < end; j++) {
      int pd_id = pd_ids[j];
      if (pd_id == -1) continue;
      total_fraction += pdInc;
      if (last_pd == pd_id) {
        pd_fraction += pdInc;
      }
      else {
        if (last_pd != -1) {
          geomeffs_pd_ids_hypo[idx] = last_pd;
          geomeffs_fractions_hypo[idx] = pd_fraction;
          idx++;
        }
        pd_fraction = pdInc;
        last_pd = pd_id;
      }
    }
    if (last_pd != -1) {
      geomeffs_pd_ids_hypo[idx] = last_pd;
      geomeffs_fractions_hypo[idx] = pd_fraction;
    }

    const unsigned track_id = i / Allen::Rich::NRealParticleTypes;
    const auto hypo = static_cast<Allen::Rich::ParticleIDType>(i % Allen::Rich::NRealParticleTypes);
    geomeffs_fractions_per_hypo[track_id][hypo] = total_fraction;
  }
}

// This is the computation done in Rec:
__global__ void rich_sum_track_signal_rec_k(
  const unsigned number_of_tracks,
  const Allen::Rich::ParticleHypos* hypos,
  const Allen::Rich::HypoData<float>* geomEffs,
  Allen::Rich::HypoData<float>* track_total_signals)
{
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;
  for (unsigned i = threadId; i < number_of_tracks * Allen::Rich::NParticleTypes; i += stride) {
    const unsigned track_id = i / Allen::Rich::NParticleTypes;
    const auto hypo = static_cast<Allen::Rich::ParticleIDType>(i % Allen::Rich::NParticleTypes);

    track_total_signals[track_id][hypo] = hypos[track_id].yield[hypo] * geomEffs[track_id][hypo];

    /*if (track_id == 0) {
      std::cout << "[Allen] Track 0 Hypo " << hypo << " track_signals "
                << track_total_signals[track_id][hypo] << " geomEffs " << geomEffs[track_id][hypo] << std::endl;
    }*/
  }
}

void rich_raytrace_cherenkov_cones::rich_raytrace_cherenkov_cones_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  const unsigned number_of_tracks = first<host_number_of_tracks_t>(arguments);

  set_size<host_total_number_of_photons_t>(arguments, 1);
  set_size<dev_rich_photons_offsets_t>(arguments, number_of_tracks * Allen::Rich::NRealParticleTypes + 1);
  set_size<dev_rich_cone_pd_ids_t>(arguments, 1); // will be allocated when we know the count
  set_size<host_total_number_of_geomeffs_t>(arguments, 1);
  set_size<dev_rich_geomeff_offsets_t>(arguments, number_of_tracks * Allen::Rich::NRealParticleTypes + 1);
  set_size<dev_rich_geomeff_pd_ids_t>(arguments, 1);    // will be allocated when we know the count
  set_size<dev_rich_geomeff_fractions_t>(arguments, 1); // will be allocated when we know the count
  set_size<dev_rich_geomeff_fractions_per_hypo_t>(arguments, number_of_tracks);
  set_size<dev_track_total_signals_t>(arguments, number_of_tracks);
}

template<Allen::Rich::Detector::DetectorType richIdx>
void rich_raytrace_cherenkov_cones::rich_raytrace_cherenkov_cones_t::launchForRich(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context,
  const Allen::Rich::RichDetector<richIdx>* rich) const
{
  const unsigned number_of_events = first<host_number_of_events_t>(arguments);
  const unsigned number_of_tracks = first<host_number_of_tracks_t>(arguments);

  // Memset not necessary since we override in the count kernels:
  // Allen::memset_async<dev_rich_photons_offsets_t>(arguments, 0, context);
  // Allen::memset_async<dev_rich_geomeff_offsets_t>(arguments, 0, context);

  // Compute number of ring points for each track/hypo
  global_function(rich_raytrace_npoints_k)(dim3(32), m_block_dim, context)(
    number_of_tracks,
    data<dev_rich_hypos_t>(arguments),
    data<dev_rich_photons_offsets_t>(arguments),
    m_nPointsMin.value()[richIdx],
    m_nPointsMax.value()[richIdx]);

  // Compute offsets and allocate pd_id buffer:
  PrefixSum::prefix_sum<dev_rich_photons_offsets_t, host_total_number_of_photons_t>(*this, arguments, context);
  resize<dev_rich_cone_pd_ids_t>(arguments, first<host_total_number_of_photons_t>(arguments));

  // Raytrace through detector
  const auto& raytrace_kernel = rich_raytrace_ck_cones_k<richIdx>;
  global_function(raytrace_kernel)(dim3(32), m_block_dim, context)(
    rich,
    Rich::Raytracing::pd_panel_lut[richIdx],
    number_of_events,
    number_of_tracks,
    data<dev_offsets_rich_states_t>(arguments),
    first<host_total_number_of_photons_t>(arguments),
    data<dev_rich_photons_offsets_t>(arguments),
    data<dev_segs_best_point_t>(arguments),
    data<dev_segs_best_momentum_t>(arguments),
    data<dev_rich_hypos_t>(arguments),
    data<dev_rich_cone_pd_ids_t>(arguments));

  // Count distinct pd_id
  global_function(rich_raytrace_count_distinct_k)(dim3(32), m_block_dim, context)(
    size<dev_rich_photons_offsets_t>(arguments) - 1,
    data<dev_rich_photons_offsets_t>(arguments),
    data<dev_rich_cone_pd_ids_t>(arguments),
    data<dev_rich_geomeff_offsets_t>(arguments));

  // Compute offsets and allocate geomeffs buffer:
  PrefixSum::prefix_sum<dev_rich_geomeff_offsets_t, host_total_number_of_geomeffs_t>(*this, arguments, context);
  resize<dev_rich_geomeff_pd_ids_t>(arguments, first<host_total_number_of_geomeffs_t>(arguments));
  resize<dev_rich_geomeff_fractions_t>(arguments, first<host_total_number_of_geomeffs_t>(arguments));

  // Fill geomeffs ids and fractions
  global_function(rich_raytrace_fill_geomeffs_k)(dim3(32), m_block_dim, context)(
    size<dev_rich_photons_offsets_t>(arguments) - 1,
    data<dev_rich_photons_offsets_t>(arguments),
    data<dev_rich_cone_pd_ids_t>(arguments),
    data<dev_rich_geomeff_offsets_t>(arguments),
    data<dev_rich_geomeff_pd_ids_t>(arguments),
    data<dev_rich_geomeff_fractions_t>(arguments),
    data<dev_rich_geomeff_fractions_per_hypo_t>(arguments));

  // Compute track signals
  global_function(rich_sum_track_signal_rec_k)(dim3(32), m_block_dim, context)(
    number_of_tracks,
    data<dev_rich_hypos_t>(arguments),
    data<dev_rich_geomeff_fractions_per_hypo_t>(arguments),
    data<dev_track_total_signals_t>(arguments));
}

void rich_raytrace_cherenkov_cones::rich_raytrace_cherenkov_cones_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& options,
  const Constants& constants,
  const Allen::Context& context) const
{
  const auto rich = Allen::Rich::Detector::detectorTypeFromNumber(m_current_rich.value());
  if (rich == Allen::Rich::Detector::Rich1) {
    launchForRich<Allen::Rich::Detector::Rich1>(arguments, options, constants, context, constants.dev_rich_1_geometry);
  }
  else {
    launchForRich<Allen::Rich::Detector::Rich2>(arguments, options, constants, context, constants.dev_rich_2_geometry);
  }
}
