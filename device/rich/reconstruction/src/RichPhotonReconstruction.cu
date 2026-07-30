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
#include "RichPhotonReconstruction.cuh"
#include <ParabolicExtrapolator.cuh>
#include <RungeKuttaExtrapolator.cuh>
#include <BinarySearch.cuh>
#include <PrefixSum.cuh>

INSTANTIATE_ALGORITHM(rich_photon_reconstruction::rich_photon_reconstruction_t);

template<Allen::Rich::Detector::DetectorType richIdx>
__global__ void rich_prefilter_photon_count_k(
  const short2* pixels_lpos,
  const unsigned* rich_hit_offsets,
  const unsigned* offsets_states,
  const unsigned number_of_events,
  const float3* segs_best_point,
  const float2* segs_point_at_panel,
  const Allen::Rich::ParticleHypos* track_hypos,
  unsigned* photons_offsets,
  const float min_separation2,
  const float max_separation2,
  const float rad_scale,
  const float scalePreSel,
  const float nSigmaPreSel)
{
  unsigned event_number = blockIdx.x;
  unsigned track_offset = offsets_states[event_number];
  unsigned track_end = offsets_states[event_number + 1];
  for (unsigned i = track_offset + threadIdx.x; i < track_end; i += blockDim.x) {

    // Load segment data
    const auto side = Allen::Rich::side<richIdx>(segs_best_point[i]);
    const auto segPanelPnt = segs_point_at_panel[i];
    auto hypos = track_hypos[i];
    UNROLL(Allen::Rich::NRealParticleTypes)
    for (unsigned hypo_index = 0; hypo_index < Allen::Rich::NRealParticleTypes; ++hypo_index) {
      const auto hypo = static_cast<Allen::Rich::ParticleIDType>(hypo_index);
      hypos.ckRes[hypo] *= nSigmaPreSel;
    }

    // Count hits
    unsigned count_prefilter2 = 0; // Mass hypo prefilter

    const unsigned hits_start = rich_hit_offsets[event_number + side * number_of_events];
    const unsigned hits_end = rich_hit_offsets[event_number + side * number_of_events + 1];

    for (unsigned j = hits_start; j < hits_end; j++) {
      const auto lpos16 = pixels_lpos[j];
      const auto lpos = make_float2(lpos16.x * (750.f / (1 << 15)), lpos16.y * (750.f / (1 << 15)));
      const auto pixP = Allen::Rich::radLocalCorrection(lpos, rad_scale);

      const auto dx = pixP.x - segPanelPnt.x;
      const auto dy = pixP.y - segPanelPnt.y;
      const auto sep2 = dx * dx + dy * dy;

      if (min_separation2 < sep2 && sep2 < max_separation2) {
        // estimated CK theta
        const auto ckThetaEsti = sqrtf(sep2) * scalePreSel;

        // Is any hit close to any mass hypo in local coordinate space ?
        bool keep = false;
        UNROLL(Allen::Rich::NRealParticleTypes)
        for (unsigned hypo_index = 0; hypo_index < Allen::Rich::NRealParticleTypes; ++hypo_index) {
          const auto hypo = static_cast<Allen::Rich::ParticleIDType>(hypo_index);
          if (std::isnan(hypos.ckTheta[hypo])) continue;
          keep |= fabsf(hypos.ckTheta[hypo] - ckThetaEsti) < hypos.ckRes[hypo];
        }

        if (keep) count_prefilter2++;
      }
    }
    photons_offsets[i] = count_prefilter2;
  }
}

template<Allen::Rich::Detector::DetectorType richIdx>
__global__ void rich_prefilter_photon_fill_k(
  const short2* pixels_lpos,
  const unsigned* rich_hit_offsets,
  const unsigned* offsets_states,
  const unsigned number_of_events,
  const float3* segs_best_point,
  const float2* segs_point_at_panel,
  const Allen::Rich::ParticleHypos* track_hypos,
  const unsigned* photons_offsets,
  Allen::Rich::PhotonReco::Photon* output_photons,
  const float min_separation2,
  const float max_separation2,
  const float rad_scale,
  const float scalePreSel,
  const float nSigmaPreSel)
{
  unsigned event_number = blockIdx.x;
  unsigned track_offset = offsets_states[event_number];
  unsigned track_end = offsets_states[event_number + 1];
  for (unsigned i = track_offset + threadIdx.x; i < track_end; i += blockDim.x) {

    // Load segment data
    const auto side = Allen::Rich::side<richIdx>(segs_best_point[i]);
    const auto segPanelPnt = segs_point_at_panel[i];
    auto hypos = track_hypos[i];
    UNROLL(Allen::Rich::NRealParticleTypes)
    for (unsigned hypo_index = 0; hypo_index < Allen::Rich::NRealParticleTypes; ++hypo_index) {
      const auto hypo = static_cast<Allen::Rich::ParticleIDType>(hypo_index);
      hypos.ckRes[hypo] *= nSigmaPreSel;
    }

    // Iterate hits
    unsigned photon_count = 0;
    Allen::Rich::PhotonReco::Photon* track_photons_output = output_photons + photons_offsets[i];

    const unsigned hits_start = rich_hit_offsets[event_number + side * number_of_events];
    const unsigned hits_end = rich_hit_offsets[event_number + side * number_of_events + 1];

    for (unsigned j = hits_start; j < hits_end; j++) {
      const auto lpos16 = pixels_lpos[j];
      const auto lpos = make_float2(lpos16.x * (750.f / (1 << 15)), lpos16.y * (750.f / (1 << 15)));
      const auto pixP = Allen::Rich::radLocalCorrection(lpos, rad_scale);

      const auto dx = pixP.x - segPanelPnt.x;
      const auto dy = pixP.y - segPanelPnt.y;
      const auto sep2 = dx * dx + dy * dy;

      if (min_separation2 < sep2 && sep2 < max_separation2) {
        // estimated CK theta
        const auto ckThetaEsti = sqrtf(sep2) * scalePreSel;

        // Is any hit close to any mass hypo in local coordinate space ?
        bool keep = false;
        UNROLL(Allen::Rich::NRealParticleTypes)
        for (unsigned hypo_index = 0; hypo_index < Allen::Rich::NRealParticleTypes; ++hypo_index) {
          const auto hypo = static_cast<Allen::Rich::ParticleIDType>(hypo_index);
          if (std::isnan(hypos.ckTheta[hypo])) continue;
          keep |= fabsf(hypos.ckTheta[hypo] - ckThetaEsti) < hypos.ckRes[hypo];
        }

        if (keep) {
          // All prefilter passed
          track_photons_output[photon_count++].pixelIdx = j;
        }
      }
    }
  }
}

template<Allen::Rich::Detector::DetectorType richIdx>
__global__ void rich_photon_reco_k(
  const Allen::Rich::RichDetector<richIdx>* rich,
  const float3* pixels_gpos,
  const unsigned total_number_of_tracks,
  const float3* segs_best_point,
  const float3* segs_best_momentum,
  const Allen::Rich::ParticleHypos* track_hypos,
  const unsigned* photons_offsets_prefilter,
  Allen::Rich::PhotonReco::Photon* output_photons,
  unsigned* photons_offsets,
  const float ckThetaCorr,
  const float minCKtheta,
  const float maxCKtheta,
  const float nSigma)
{
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;
  const unsigned total_number_of_photons = photons_offsets_prefilter[total_number_of_tracks];
  for (unsigned i = threadId; i < total_number_of_photons; i += stride) {
    // Determine track and pixel ids
    const unsigned track_id = binary_search_rightmost(photons_offsets_prefilter, total_number_of_tracks + 1, i);

    const unsigned pixel_id = output_photons[i].pixelIdx;
    const float3 gpos = pixels_gpos[pixel_id];

    // Load segment data
    const float3 emissionPoint = segs_best_point[track_id];

    // Do quartic reco and store results:
    const float3 photonDirection = Allen::Rich::photonDirection(rich, emissionPoint, gpos);
    /*if (i == 0) {
      std::cout << "[Allen] gpos: (" << gpos.x << ", " << gpos.y << ", " << gpos.z << ")"
                << " emissionPoint: (" << emissionPoint.x << ", " << emissionPoint.y << ", " << emissionPoint.z << ")"
                << " dir: (" << photonDirection.x << ", " << photonDirection.y << ", " << photonDirection.z << ")"
                << " mom: (" << middle_states[track_id].px() << ", " << middle_states[track_id].py() << ", " <<
    middle_states[track_id].pz() << ")" << std::endl;
    }*/

    float thetaCherenkov = 0.0f, phiCherenkov = 0.0f;
    const Allen::Rich::PhotonReco::RotationMatrix rotation {segs_best_momentum[track_id]};
    rotation.angleToDirection(photonDirection, thetaCherenkov, phiCherenkov);

    // thetaCherenkov = Allen::Rich::Maths::truncate_relative<18>(thetaCherenkov);
    // phiCherenkov   = Allen::Rich::Maths::truncate_relative<18>(phiCherenkov);

    thetaCherenkov += ckThetaCorr;

    // Post filter
    auto hypos = track_hypos[track_id];

    bool keep = false;
    UNROLL(Allen::Rich::NRealParticleTypes)
    for (unsigned hypo_index = 0; hypo_index < Allen::Rich::NRealParticleTypes; ++hypo_index) {
      const auto hypo = static_cast<Allen::Rich::ParticleIDType>(hypo_index);
      if (std::isnan(hypos.ckTheta[hypo])) continue;
      keep |= fabsf(hypos.ckTheta[hypo] - thetaCherenkov) < (hypos.ckRes[hypo] * nSigma);
    }
    keep &= thetaCherenkov < maxCKtheta && thetaCherenkov > minCKtheta;

    if (!keep) {
      thetaCherenkov = NAN; // Flag the photon as invalid
    }
    else {
      atomicAdd(&photons_offsets[track_id], 1); // Count the numbers of kept photons
    }

    output_photons[i].ckTheta = thetaCherenkov;
    output_photons[i].ckPhi = phiCherenkov;
  }
}

__global__ void rich_photon_copy_postfilter_k(
  const unsigned* photons_offsets_prefilter,
  const unsigned* photons_offsets,
  const Allen::Rich::PhotonReco::Photon* photons_prefilter,
  Allen::Rich::PhotonReco::Photon* photons)
{
  unsigned track_id = blockIdx.x;
  unsigned in_start = photons_offsets_prefilter[track_id];
  unsigned in_size = photons_offsets_prefilter[track_id + 1] - in_start;
  unsigned out_start = photons_offsets[track_id];

  __shared__ unsigned out_idx[1];
  if (threadIdx.x == 0) *out_idx = 0;
  __syncthreads();

  for (unsigned i = threadIdx.x; i < in_size; i += blockDim.x) {
    auto photon = photons_prefilter[in_start + i];
    bool keep = !std::isnan(photon.ckTheta);
    if (keep) {
      unsigned j = atomicAdd(out_idx, 1); // TODO, warp level compaction
      photons[out_start + j] = photon;
    }
  }
}

void rich_photon_reconstruction::rich_photon_reconstruction_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  const unsigned number_of_tracks = first<host_number_of_tracks_t>(arguments);

  set_size<host_total_number_of_photons_t>(arguments, 1);
  set_size<dev_offsets_rich_photons_prefilter_t>(arguments, number_of_tracks + 1);
  set_size<dev_offsets_rich_photons_t>(arguments, number_of_tracks + 1);
  set_size<dev_rich_photons_prefilter_t>(arguments, 1); // will be allocated when we know the count
  set_size<dev_rich_photons_t>(arguments, 1);           // will be allocated when we know the count
}

template<Allen::Rich::Detector::DetectorType richIdx>
void rich_photon_reconstruction::rich_photon_reconstruction_t::launchForRich(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context,
  const Allen::Rich::RichDetector<richIdx>* rich) const
{

  const unsigned number_of_events = first<host_number_of_events_t>(arguments);
  const unsigned number_of_tracks = first<host_number_of_tracks_t>(arguments);

  const float minROIPreSel2 = m_minROIPreSel.value()[richIdx] * m_minROIPreSel.value()[richIdx];
  const float maxROIPreSel2 = m_maxROIPreSel.value()[richIdx] * m_maxROIPreSel.value()[richIdx];
  const float radScale = m_radScale.value()[richIdx];
  const float scalePreSel = m_ckThetaScale.value()[richIdx] / m_sepGScale.value()[richIdx];
  const float nSigmaPreSel = m_nSigmaPreSel.value()[richIdx];

  const float ckBiasCorr = m_ckBiasCorrs.value()[richIdx];

  const float minCKtheta = m_minCKtheta.value()[richIdx];
  const float maxCKtheta = m_maxCKtheta.value()[richIdx];
  const float nSigma = m_nSigma.value()[richIdx];

  Allen::memset_async<dev_offsets_rich_photons_t>(arguments, 0, context);

  // First count how many photons are prefiltered:
  const auto& count_kernel = rich_prefilter_photon_count_k<richIdx>;
  global_function(count_kernel)(dim3(number_of_events), m_block_dim, context)(
    data<dev_rich_pixels_lpos_t>(arguments),
    data<dev_rich_pd_offsets_t>(arguments),
    data<dev_offsets_rich_states_t>(arguments),
    number_of_events,
    data<dev_segs_best_point_t>(arguments),
    data<dev_segs_point_at_panel_t>(arguments),
    data<dev_rich_hypos_t>(arguments),
    data<dev_offsets_rich_photons_prefilter_t>(arguments),
    minROIPreSel2,
    maxROIPreSel2,
    radScale,
    scalePreSel,
    nSigmaPreSel);

  // Compute offsets and allocate photon buffer:
  PrefixSum::prefix_sum<dev_offsets_rich_photons_prefilter_t, host_total_number_of_photons_t>(
    *this, arguments, context);
  resize<dev_rich_photons_prefilter_t>(arguments, first<host_total_number_of_photons_t>(arguments));

  // Now do the quartic photon reconstruction and fill the buffer:
  const auto& fill_kernel = rich_prefilter_photon_fill_k<richIdx>;
  global_function(fill_kernel)(dim3(number_of_events), m_block_dim, context)(
    data<dev_rich_pixels_lpos_t>(arguments),
    data<dev_rich_pd_offsets_t>(arguments),
    data<dev_offsets_rich_states_t>(arguments),
    number_of_events,
    data<dev_segs_best_point_t>(arguments),
    data<dev_segs_point_at_panel_t>(arguments),
    data<dev_rich_hypos_t>(arguments),
    data<dev_offsets_rich_photons_prefilter_t>(arguments),
    data<dev_rich_photons_prefilter_t>(arguments),
    minROIPreSel2,
    maxROIPreSel2,
    radScale,
    scalePreSel,
    nSigmaPreSel);

  // Run quartic reco + postfilter
  const auto& quartic_kernel = rich_photon_reco_k<richIdx>;
  global_function(quartic_kernel)(dim3(32), m_block_dim, context)(
    rich,
    data<dev_rich_pixels_gpos_t>(arguments),
    number_of_tracks,
    data<dev_segs_best_point_t>(arguments),
    data<dev_segs_best_momentum_t>(arguments),
    data<dev_rich_hypos_t>(arguments),
    data<dev_offsets_rich_photons_prefilter_t>(arguments),
    data<dev_rich_photons_prefilter_t>(arguments),
    data<dev_offsets_rich_photons_t>(arguments),
    ckBiasCorr,
    minCKtheta,
    maxCKtheta,
    nSigma);

  // Copy the photons into a postfilter compact array:
  PrefixSum::prefix_sum<dev_offsets_rich_photons_t, host_total_number_of_photons_t>(*this, arguments, context);
  resize<dev_rich_photons_t>(arguments, first<host_total_number_of_photons_t>(arguments));

  global_function(rich_photon_copy_postfilter_k)(dim3(number_of_tracks), m_block_dim, context)(
    data<dev_offsets_rich_photons_prefilter_t>(arguments),
    data<dev_offsets_rich_photons_t>(arguments),
    data<dev_rich_photons_prefilter_t>(arguments),
    data<dev_rich_photons_t>(arguments));
}

void rich_photon_reconstruction::rich_photon_reconstruction_t::operator()(
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
