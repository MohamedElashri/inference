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

#include "RichQuarticSignals.cuh"
#include <cassert>
#include <cmath>
#include <fstream>
#include <ParabolicExtrapolator.cuh>
#include <RungeKuttaExtrapolator.cuh>
#include <BinarySearch.cuh>
#include <PrefixSum.cuh>

INSTANTIATE_ALGORITHM(rich_quartic_signals::rich_quartic_signals_t);

template<Allen::Rich::Detector::DetectorType richIdx>
void rich_quartic_signals::rich_quartic_signals_t::updateRich(const Allen::Rich::RichDetector<richIdx>* rich) const
{
  // Precompute pd infos
  std::vector<PDShortInfo> pd_infos {};
  std::vector<short2> pd_lpos16 {};
  std::vector<float3> pd_corners {};
  std::vector<float>
    pd_diag {}; // TODO: since this is all the same value for R1 and 2 possible values for R2, this could be optimized..
  Allen::Rich::PanelArray<unsigned> n_pds_per_side {};
  constexpr std::array<float2, 4> corners {{{-3.5f, -3.5f}, {-3.5f, +3.5f}, {+3.5f, -3.5f}, {+3.5f, +3.5f}}};
  float smallPixelArea = 0.f;
  float largePixelArea = 0.f;
  const float scalePreSel = m_ckThetaScale.value()[richIdx] / m_sepGScale.value()[richIdx];
  for (const auto side : Allen::Rich::Detector::sides()) {
    const unsigned n_pds = Allen::Rich::Detector::PDPanel<richIdx>::PDsPerPanel;
    const auto* pds = rich->pdPanels()[side].pds();
    const auto& g2panel = rich->pdPanels()[side].globalToPDPanel();
    for (unsigned i = 0; i < n_pds; i++) {
      const auto& pd = pds[i];
      if (pd.getIsNull()) continue; // skip

      float2 p = pd.centrePointPanel(g2panel);
      p = Allen::Rich::radLocalCorrection(p, m_radScale.value()[richIdx]);
      pd_lpos16.emplace_back(make_short2(p.x * ((1 << 15) / 750.f) + .5f, p.y * ((1 << 15) / 750.f) + .5f));

      n_pds_per_side[side]++;
      pd_infos.emplace_back(i, pd.isLarge());

      if (pd.isLarge())
        largePixelArea = pd.effectivePixelArea();
      else
        smallPixelArea = pd.effectivePixelArea();

      for (unsigned corner = 0; corner < 4; corner++) {
        pd_corners.emplace_back(pd.localToGlobal(corners[corner]));
      }

      pd_diag.emplace_back(static_cast<float>(std::sqrt(2.0) * 4.0) * pd.getEffectivePixelXSize() * scalePreSel);
    }
  }

  assert(
    n_pds_per_side[0] == n_pds_per_side[1] &&
    "RICH quartic signals require equal non-null photodetector counts per panel");
  m_n_pds = n_pds_per_side[0]; // per panel

  if (m_pd_infos != nullptr) Allen::free(m_pd_infos);
  Allen::malloc((void**) &m_pd_infos, pd_infos.size() * sizeof(PDShortInfo));
  Allen::memcpy(m_pd_infos, pd_infos.data(), pd_infos.size() * sizeof(PDShortInfo), Allen::memcpyHostToDevice);

  if (m_pd_lpos16 != nullptr) Allen::free(m_pd_lpos16);
  Allen::malloc((void**) &m_pd_lpos16, pd_lpos16.size() * sizeof(short2));
  Allen::memcpy(m_pd_lpos16, pd_lpos16.data(), pd_lpos16.size() * sizeof(short2), Allen::memcpyHostToDevice);

  if (m_pd_corners != nullptr) Allen::free(m_pd_corners);
  Allen::malloc((void**) &m_pd_corners, pd_corners.size() * sizeof(float3));
  Allen::memcpy(m_pd_corners, pd_corners.data(), pd_corners.size() * sizeof(float3), Allen::memcpyHostToDevice);

  // TODO: this shouldnt be an array:
  if (m_pd_diag != nullptr) Allen::free(m_pd_diag);
  Allen::malloc((void**) &m_pd_diag, pd_diag.size() * sizeof(float));
  Allen::memcpy(m_pd_diag, pd_diag.data(), pd_diag.size() * sizeof(float), Allen::memcpyHostToDevice);

  const double R = static_cast<double>(rich->sphMirrorRadius());
  m_factor[0] = static_cast<float>(4.0 / (R * R * std::pow(2.0 * M_PI, 1.5))) * smallPixelArea;
  m_factor[1] = static_cast<float>(4.0 / (R * R * std::pow(2.0 * M_PI, 1.5))) * largePixelArea;
}

void rich_quartic_signals::rich_quartic_signals_t::update(const Constants& constants) const
{
  const auto rich = Allen::Rich::Detector::detectorTypeFromNumber(m_current_rich.value());
  if (rich == Allen::Rich::Detector::Rich1) {
    updateRich<Allen::Rich::Detector::Rich1>(constants.host_rich_1_geometry);
  }
  else {
    updateRich<Allen::Rich::Detector::Rich2>(constants.host_rich_2_geometry);
  }
}

__device__ inline bool pdPassesPrefilter(
  const short2 lpos16,
  const float diag,
  const float2 segPanelPnt,
  const Allen::Rich::ParticleHypos& hypos,
  const float min_separation2,
  const float max_separation2,
  const float scalePreSel,
  const float nSigmaPreSel)
{
  const auto pixP = make_float2(lpos16.x * (750.f / (1 << 15)), lpos16.y * (750.f / (1 << 15)));
  const auto dx = pixP.x - segPanelPnt.x;
  const auto dy = pixP.y - segPanelPnt.y;
  const auto sep2 = dx * dx + dy * dy;
  if (!(min_separation2 < sep2 && sep2 < max_separation2)) return false;

  const auto ckThetaEsti = sqrtf(sep2) * scalePreSel;
  UNROLL(Allen::Rich::NRealParticleTypes)
  for (unsigned hypo_index = 0; hypo_index < Allen::Rich::NRealParticleTypes; ++hypo_index) {
    const auto hypo = static_cast<Allen::Rich::ParticleIDType>(hypo_index);
    if (std::isnan(hypos.ckTheta[hypo])) break; // break on first below threshold
    if (fabsf(hypos.ckTheta[hypo] - ckThetaEsti) < (hypos.ckRes[hypo] * nSigmaPreSel + diag)) return true;
  }
  return false;
}

template<Allen::Rich::Detector::DetectorType richIdx>
__global__ void rich_prefilter_pd_count_k(
  const unsigned n_pds,
  const short2* pd_lpos16,
  const float* pd_diag,
  const unsigned number_of_tracks,
  const float3* segs_best_point,
  const float2* segs_point_at_panel,
  const Allen::Rich::ParticleHypos* track_hypos,
  unsigned* pd_counts,
  const float min_separation2,
  const float max_separation2,
  const float scalePreSel,
  const float nSigmaPreSel)
{
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;
  for (unsigned i = threadId; i < number_of_tracks; i += stride) {

    // Load segment data
    const auto side = Allen::Rich::side<richIdx>(segs_best_point[i]);
    const auto segPanelPnt = segs_point_at_panel[i];
    const auto& hypos = track_hypos[i];

    // Count hits
    unsigned count_prefilter = 0; // Mass hypo prefilter

    for (unsigned j = 0; j < n_pds; j++) {
      const auto pd_index = j + side * n_pds;
      if (pdPassesPrefilter(
            pd_lpos16[pd_index],
            pd_diag[pd_index],
            segPanelPnt,
            hypos,
            min_separation2,
            max_separation2,
            scalePreSel,
            nSigmaPreSel)) {
        count_prefilter++;
      }
    }
    pd_counts[i] = count_prefilter;
  }
}

template<Allen::Rich::Detector::DetectorType richIdx>
__global__ void rich_prefilter_pd_fill_k(
  const rich_quartic_signals::PDShortInfo* pd_infos,
  const unsigned n_pds,
  const short2* pd_lpos16,
  const float* pd_diag,
  const unsigned number_of_events,
  const unsigned* track_offsets,
  const unsigned number_of_tracks,
  const float3* segs_best_point,
  const float2* segs_point_at_panel,
  const Allen::Rich::ParticleHypos* track_hypos,
  const uint64_t* pd_pixels,
  const unsigned* track_signal_offsets,
  int* pd_indices,
  unsigned* pix_counts,
  const float min_separation2,
  const float max_separation2,
  const float scalePreSel,
  const float nSigmaPreSel)
{
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;
  for (unsigned i = threadId; i < number_of_tracks; i += stride) {

    // Load segment data
    const auto side = Allen::Rich::side<richIdx>(segs_best_point[i]);
    const auto segPanelPnt = segs_point_at_panel[i];
    const auto& hypos = track_hypos[i];

    const auto event_number = binary_search_rightmost(track_offsets, number_of_events + 1, i);
    const auto pd_offset =
      (side * number_of_events + event_number) * Allen::Rich::Detector::PDPanel<richIdx>::PDsPerPanel;

    // Iterate hits
    unsigned pd_count = 0;
    int* track_pd_indices = pd_indices + track_signal_offsets[i];
    unsigned* track_pixel_counts = pix_counts + track_signal_offsets[i];

    for (unsigned j = 0; j < n_pds; j++) {
      const auto pd_index = j + side * n_pds;
      if (pdPassesPrefilter(
            pd_lpos16[pd_index],
            pd_diag[pd_index],
            segPanelPnt,
            hypos,
            min_separation2,
            max_separation2,
            scalePreSel,
            nSigmaPreSel)) {
        track_pixel_counts[pd_count] = __popcll(pd_pixels[pd_offset + pd_infos[pd_index].index()]);
        track_pd_indices[pd_count++] = pd_index;
      }
    }
  }
}

__device__ inline float interp(float r, float a, float b) { return a + (b - a) * r; }

__device__ inline float2 interp(float r, const float2& a, const float2& b)
{
  return {interp(r, a.x, b.x), interp(r, a.y, b.y)};
}

__device__ inline float3 interp(float r, const float3& a, const float3& b)
{
  return {interp(r, a.x, b.x), interp(r, a.y, b.y), interp(r, a.z, b.z)};
}

template<typename T>
__device__ inline T interp2D(float rx, float ry, const T& A, const T& B, const T& C, const T& D)
{
  const T y0 = interp(ry, A, B);
  const T y1 = interp(ry, C, D);
  return interp(rx, y0, y1);
}

template<typename T>
__device__ inline T interp2D(float rx, float ry, const std::array<T, 4>& values)
{
  const T y0 = interp(ry, values[0], values[1]);
  const T y1 = interp(ry, values[2], values[3]);
  return interp(rx, y0, y1);
}

template<Allen::Rich::Detector::DetectorType richIdx>
__global__ void rich_quartic_corners_k(
  const Allen::Rich::RichDetector<richIdx>* rich,
  const float3* pd_corners,
  const unsigned number_of_tracks,
  const float3* segs_best_point,
  const float3* segs_best_momentum,
  [[maybe_unused]] const uint64_t* pd_pixels,
  const unsigned* track_signal_offsets,
  const int* pd_indices,
  std::array<float2, 4>* dir_corners)
{
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;
  const unsigned total_number_of_pds = track_signal_offsets[number_of_tracks];

  for (unsigned i = threadId; i < total_number_of_pds; i += stride) {
    // Determine track and pd ids
    const unsigned track_id = binary_search_rightmost(track_signal_offsets, number_of_tracks + 1, i);

    auto pd_id = pd_indices[i];

    // Load segment data
    const float3 emissionPoint = segs_best_point[track_id];
    const Allen::Rich::PhotonReco::RotationMatrix rotation {segs_best_momentum[track_id]};

    // Reconstruct corners
    UNROLL(4)
    for (unsigned corner = 0; corner < 4; corner++) {
      // PD's localToGlobal matrix is 12 floats, so for the same memory cost
      // we have the 4 precomputed float3 corners and save the flops
      // They can also be loaded sequentially and discarded, thus saving registers
      const float3 gpos = pd_corners[pd_id * 4 + corner];
      const float3 photonDirection = Allen::Rich::photonDirection(rich, emissionPoint, gpos);

      // Normalize the dir so that z = 1 to save 4 registers and some flops in interp
      // this also simplifies theta computation since atan2(perp,1) = atan(perp);
      const float3 dir = rotation.globalToTrack(photonDirection);
      const float zInv = 1.f / dir.z;
      dir_corners[i][corner] = make_float2(dir.x * zInv, dir.y * zInv);
    }
    // TODO: filter and count pixels ?
  }
}

template<Allen::Rich::Detector::DetectorType richIdx>
__global__ void rich_interp_pixel_signals_k(
  // const Allen::Rich::RichDetector<richIdx>* rich,
  const rich_quartic_signals::PDShortInfo* pd_infos,
  const unsigned number_of_events,
  const unsigned* track_offsets,
  const unsigned number_of_tracks,
  const float3* segs_best_point,
  const Allen::Rich::ParticleHypos* track_hypos,
  const unsigned* pd_offsets,
  const uint64_t* pd_pixels,
  const unsigned* track_signal_offsets,
  const int* pd_indices,
  const std::array<float2, 4>* dir_corners,
  Allen::Rich::HypoData<float>* pd_fractions,
  const unsigned* photon_offsets,
  Allen::Rich::PhotonReco::Photon* photons,
  unsigned* photon_count,
  const float ckThetaCorr,
  const std::array<float, 2> factor)
{
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;
  const unsigned total_number_of_pds = track_signal_offsets[number_of_tracks];

  for (unsigned i = threadId; i < total_number_of_pds; i += stride) {
    // Determine track and pd ids
    const unsigned track_id = binary_search_rightmost(track_signal_offsets, number_of_tracks + 1, i);

    auto pd_id = pd_indices[i];

    // Load segment data
    const float3 emissionPoint = segs_best_point[track_id];
    const auto& hypos = track_hypos[track_id];

    const std::array<float2, 4> dirCorners = dir_corners[i];

    // TODO: check if all corners use the same mirrors
    // TODO: might also be useful to check corners theta and early exit for R2 as the estimated
    // theta used in the prefilter are less precise

    // Find the PD infos
    const auto side = Allen::Rich::side<richIdx>(emissionPoint);
    const unsigned event_number = binary_search_rightmost(track_offsets, number_of_events + 1, track_id);
    const unsigned pd_offset =
      (side * number_of_events + event_number) * Allen::Rich::Detector::PDPanel<richIdx>::PDsPerPanel;
    const uint64_t pixels_bitmask = pd_pixels[pd_offset + pd_infos[pd_id].index()];
    const unsigned pixel_id_base = pd_offsets[pd_offset + pd_infos[pd_id].index()];
    const unsigned out_base = photon_offsets[i];
    unsigned out_id = 0;
    unsigned postfilter_count = 0;

    // Precompute pixel independent values:
    Allen::Rich::HypoData<float> invRes {};
    UNROLL(Allen::Rich::NRealParticleTypes)
    for (unsigned hypo_index = 0; hypo_index < Allen::Rich::NRealParticleTypes; ++hypo_index) {
      const auto hypo = static_cast<Allen::Rich::ParticleIDType>(hypo_index);
      const float res = hypos.ckRes[hypo];
      invRes[hypo] = 1.f / res;
    }
    const float Afactor = factor[richIdx == Allen::Rich::Detector::Rich1 ? 0 : pd_infos[pd_id].isLarge()];

    // Integrate photon signals
    Allen::Rich::HypoData<float> fractions {};
    for (int y = 0; y < 8; y++) {
      const float ry = y / 7.f;
      const auto interp_0 = interp(ry, dirCorners[0], dirCorners[1]);
      const auto interp_1 = interp(ry, dirCorners[2], dirCorners[3]);
      for (int x = 0; x < 8; x++) {
        const float rx = x / 7.f;
        // Interpolate in cartesian space to avoid mathematical aberations
        const auto r = interp(rx, interp_0, interp_1);

        const float perp = sqrtf(r.x * r.x + r.y * r.y);
        float theta = atanf(perp); // atan2f(perp, r.z);
        theta += ckThetaCorr;
        // float phi = atan2f(r.y, r.x); // for debug
        // if (phi < 0) phi += (float) (2 * M_PI);

        const bool save_photon = (pixels_bitmask >> (y * 8 + x)) & 1;

        const bool validTheta = std::isfinite(theta) && theta > 0.f;
        const float hypo_indep = validTheta ? Afactor / theta : 0.f;

        // Compute signals for that pixel
        //[[maybe_unused]] Allen::Rich::HypoData<float> signals{};
        bool has_signal_over_threshold = false;
        UNROLL(Allen::Rich::NRealParticleTypes)
        for (unsigned hypo_index = 0; hypo_index < Allen::Rich::NRealParticleTypes; ++hypo_index) {
          const auto hypo = static_cast<Allen::Rich::ParticleIDType>(hypo_index);
          const float expTheta = hypos.ckTheta[hypo];
          if (std::isnan(expTheta)) break; // break on first below threshold
          if (!validTheta) break;
          if (!(expTheta > 0.f)) continue;

          // aij = yield * 1/((2pi)^(3/2)*sigma(theta)) * exp(-1/2 * sep^2) * 4A/(R^2 theta)
          const float sep = (theta - expTheta) * invRes[hypo];
          const float expArg = expf(-0.5f * sep * sep);

          float sig = hypo_indep * expArg * invRes[hypo];
          // if (sig <= minPhotonProb) sig = 0.f;

          if (save_photon) {
            const float pix_sig = sig * hypos.yield[hypo];
            if (pix_sig > 1e-5f) has_signal_over_threshold = true;
          }

          // signals[hypo] = sig * hypos.yield[hypo];
          fractions[hypo] += sig;
        }

        /*if (track_id == 0 && richIdx == 1) {
          // Compute "real" theta/phi to evaluate the approximation error due to interpolation
          //const auto& pd = rich->pdPanels()[side].pds()[pd_infos[pd_id].index()];
          //constexpr std::array<float2, 4> corners {{{-3.5f, -3.5f}, {-3.5f, +3.5f}, {+3.5f, -3.5f}, {+3.5f, +3.5f}}};
          const float3 gpos = interp2D(rx, ry, pd_corners[pd_id * 4], pd_corners[pd_id * 4 + 1],
                                               pd_corners[pd_id * 4 + 2], pd_corners[pd_id * 4 + 3]);
          const float3 photonDirection = Allen::Rich::photonDirection(rich, emissionPoint, gpos);
          float rTheta{}, rPhi{};
          rotation.angleToDirection(photonDirection, rTheta, rPhi);
          rTheta += ckThetaCorr;

          std::cout << "[Allen] PD " << pd_id << " side " << side
                    << " pixel [" << x << "," << y << "]"
                    << " theta/phi: [" <<theta<<","<< phi << "]"
                    << " real theta/phi: ["<<rTheta<<","<< rPhi << "]"
                    << " signals: " << signals
                    << " firing: " << ((pixels_bitmask >> (y * 8 + x)) & 1) << std::endl;
        }*/

        if (save_photon) {
          photons[out_base + out_id].ckTheta = has_signal_over_threshold ? theta : NAN;
          // photons[out_base + out_id].ckPhi = phi;
          photons[out_base + out_id].pixelIdx = pixel_id_base + out_id;

          if (has_signal_over_threshold) postfilter_count++;
          out_id++;
        }
      }
    }
    UNROLL(Allen::Rich::NRealParticleTypes)
    for (unsigned hypo_index = 0; hypo_index < Allen::Rich::NRealParticleTypes; ++hypo_index) {
      const auto hypo = static_cast<Allen::Rich::ParticleIDType>(hypo_index);
      fractions[hypo] *= hypos.yield[hypo];
    }
    pd_fractions[i] = fractions;
    atomicAdd(&photon_count[track_id], postfilter_count);
  }
}

__global__ void rich_photon_copy_postfilter_k(
  const unsigned* track_signal_offsets,
  const unsigned* photons_offsets_prefilter,
  const unsigned* photons_offsets,
  const Allen::Rich::PhotonReco::Photon* photons_prefilter,
  Allen::Rich::PhotonReco::Photon* photons)
{
  unsigned track_id = blockIdx.x;
  unsigned in_start = photons_offsets_prefilter[track_signal_offsets[track_id]];
  unsigned in_size = photons_offsets_prefilter[track_signal_offsets[track_id + 1]] - in_start;
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

__global__ void rich_sum_track_geomeffs_k(
  const unsigned number_of_tracks,
  const unsigned* track_signal_offsets,
  const Allen::Rich::HypoData<float>* pd_signals,
  Allen::Rich::HypoData<float>* track_total_signals)
{
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;
  for (unsigned i = threadId; i < number_of_tracks * Allen::Rich::NParticleTypes; i += stride) {
    const unsigned track_id = i / Allen::Rich::NParticleTypes;
    const auto hypo = static_cast<Allen::Rich::ParticleIDType>(i % Allen::Rich::NParticleTypes);

    const unsigned start = track_signal_offsets[track_id];
    const unsigned end = track_signal_offsets[track_id + 1];

    float sum = 0.f;
    for (unsigned j = start; j < end; j++) {
      sum += pd_signals[j][hypo];
    }
    track_total_signals[track_id][hypo] = sum;

    /*if (track_id == 0) {
      std::cout << "[Allen] hypo " << hypo_id << " signal: " << sum<<std::endl;
    }*/
  }
}

// Final pass: transform dev_rich_geomeff_pd_ids_t from compacted (per-panel, non-null-only)
// indices in-place into the same fully-packed global dense index convention
// rich_raytrace_ck_cones_k already uses for FromCones (panel_offset = (side*number_of_events +
// event_number) * PDsPerPanel, plus the raw per-panel index from pd_infos).
template<Allen::Rich::Detector::DetectorType richIdx>
__global__ void rich_geomeff_pd_ids_to_global_k(
  const rich_quartic_signals::PDShortInfo* pd_infos,
  const unsigned number_of_events,
  const unsigned* track_offsets,
  const unsigned number_of_tracks,
  const float3* segs_best_point,
  const unsigned* track_signal_offsets,
  int* pd_indices)
{
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;
  const unsigned total_number_of_pds = track_signal_offsets[number_of_tracks];

  for (unsigned i = threadId; i < total_number_of_pds; i += stride) {
    const unsigned track_id = binary_search_rightmost(track_signal_offsets, number_of_tracks + 1, i);

    const auto side = Allen::Rich::side<richIdx>(segs_best_point[track_id]);
    const unsigned event_number = binary_search_rightmost(track_offsets, number_of_events + 1, track_id);
    const unsigned panel_offset =
      (side * number_of_events + event_number) * Allen::Rich::Detector::PDPanel<richIdx>::PDsPerPanel;

    const int compacted_id = pd_indices[i];
    pd_indices[i] = static_cast<int>(panel_offset + pd_infos[compacted_id].index());
  }
}

void rich_quartic_signals::rich_quartic_signals_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  const unsigned number_of_tracks = first<host_number_of_tracks_t>(arguments);

  // Expected signal:
  set_size<host_total_number_of_geomeffs_t>(arguments, 1);
  set_size<dev_rich_geomeff_offsets_t>(arguments, number_of_tracks + 1);
  set_size<dev_rich_geomeff_pd_ids_t>(arguments, 1);       // will be allocated when we know the count
  set_size<dev_rich_geomeff_pd_fractions_t>(arguments, 1); // will be allocated when we know the count
  set_size<dev_rich_geomeff_fractions_t>(arguments, number_of_tracks);

  // Observed signal:
  set_size<host_total_number_of_photons_t>(arguments, 1);

  set_size<dev_offsets_rich_photons_prefilter_t>(arguments, 1); // will be allocated when we know the count
  set_size<dev_rich_photons_prefilter_t>(arguments, 1);         // will be allocated when we know the count

  set_size<dev_offsets_rich_photons_t>(arguments, number_of_tracks + 1);
  set_size<dev_rich_photons_t>(arguments, 1); // will be allocated when we know the count

  // Temporaries:
  set_size<dev_pd_photon_dir_corners_t>(arguments, 1); // will be allocated when we know the count
}

template<Allen::Rich::Detector::DetectorType richIdx>
void rich_quartic_signals::rich_quartic_signals_t::launchForRich(
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
  const float scalePreSel = m_ckThetaScale.value()[richIdx] / m_sepGScale.value()[richIdx];
  const float nSigmaPreSel = m_nSigmaPreSel.value()[richIdx];

  const float ckBiasCorr = m_ckBiasCorrs.value()[richIdx];

  Allen::memset_async<dev_rich_geomeff_offsets_t>(arguments, 0, context);
  Allen::memset_async<dev_offsets_rich_photons_t>(arguments, 0, context);

  // First count how many photodetectors pass the prefilter:
  const auto& count_kernel = rich_prefilter_pd_count_k<richIdx>;
  global_function(count_kernel)(dim3(number_of_events), m_block_dim, context)(
    m_n_pds,
    m_pd_lpos16,
    m_pd_diag,
    number_of_tracks,
    data<dev_segs_best_point_t>(arguments),
    data<dev_segs_point_at_panel_t>(arguments),
    data<dev_rich_hypos_t>(arguments),
    data<dev_rich_geomeff_offsets_t>(arguments),
    minROIPreSel2,
    maxROIPreSel2,
    scalePreSel,
    nSigmaPreSel);

  // Compute offsets and allocate per track x hypo x pd buffer:
  PrefixSum::prefix_sum<dev_rich_geomeff_offsets_t, host_total_number_of_geomeffs_t>(*this, arguments, context);
  resize<dev_rich_geomeff_pd_ids_t>(arguments, first<host_total_number_of_geomeffs_t>(arguments));
  resize<dev_rich_geomeff_pd_fractions_t>(arguments, first<host_total_number_of_geomeffs_t>(arguments));
  resize<dev_offsets_rich_photons_prefilter_t>(arguments, first<host_total_number_of_geomeffs_t>(arguments) + 1);

  // Fill PD indices
  const auto& fill_kernel = rich_prefilter_pd_fill_k<richIdx>;
  global_function(fill_kernel)(dim3(number_of_events), m_block_dim, context)(
    m_pd_infos,
    m_n_pds,
    m_pd_lpos16,
    m_pd_diag,
    number_of_events,
    data<dev_offsets_rich_states_t>(arguments),
    number_of_tracks,
    data<dev_segs_best_point_t>(arguments),
    data<dev_segs_point_at_panel_t>(arguments),
    data<dev_rich_hypos_t>(arguments),
    data<dev_pd_pixels_t>(arguments),
    data<dev_rich_geomeff_offsets_t>(arguments),
    data<dev_rich_geomeff_pd_ids_t>(arguments),
    data<dev_offsets_rich_photons_prefilter_t>(arguments),
    minROIPreSel2,
    maxROIPreSel2,
    scalePreSel,
    nSigmaPreSel);

  // Compute offsets for per PD photons output:
  PrefixSum::prefix_sum<dev_offsets_rich_photons_prefilter_t, host_total_number_of_photons_t>(
    *this, arguments, context);
  resize<dev_rich_photons_prefilter_t>(arguments, first<host_total_number_of_photons_t>(arguments));

  // Now do the quartic photon reconstruction and compute geom effs:
  resize<dev_pd_photon_dir_corners_t>(arguments, first<host_total_number_of_geomeffs_t>(arguments));

  const auto& corners_kernel = rich_quartic_corners_k<richIdx>;
  global_function(corners_kernel)(dim3(32), m_block_dim, context)(
    rich,
    m_pd_corners,
    number_of_tracks,
    data<dev_segs_best_point_t>(arguments),
    data<dev_segs_best_momentum_t>(arguments),
    data<dev_pd_pixels_t>(arguments),
    data<dev_rich_geomeff_offsets_t>(arguments),
    data<dev_rich_geomeff_pd_ids_t>(arguments),
    data<dev_pd_photon_dir_corners_t>(arguments));

  const auto& interp_kernel = rich_interp_pixel_signals_k<richIdx>;
  global_function(interp_kernel)(dim3(32), m_block_dim, context)(
    // rich,
    m_pd_infos,
    number_of_events,
    data<dev_offsets_rich_states_t>(arguments),
    number_of_tracks,
    data<dev_segs_best_point_t>(arguments),
    data<dev_rich_hypos_t>(arguments),
    data<dev_rich_pd_offsets_t>(arguments),
    data<dev_pd_pixels_t>(arguments),
    data<dev_rich_geomeff_offsets_t>(arguments),
    data<dev_rich_geomeff_pd_ids_t>(arguments),
    data<dev_pd_photon_dir_corners_t>(arguments),
    data<dev_rich_geomeff_pd_fractions_t>(arguments),
    data<dev_offsets_rich_photons_prefilter_t>(arguments),
    data<dev_rich_photons_prefilter_t>(arguments),
    data<dev_offsets_rich_photons_t>(arguments),
    ckBiasCorr,
    m_factor);

  // Copy the photons and signals into a post-filtered compact array:
  PrefixSum::prefix_sum<dev_offsets_rich_photons_t, host_total_number_of_photons_t>(*this, arguments, context);
  resize<dev_rich_photons_t>(arguments, first<host_total_number_of_photons_t>(arguments));

  global_function(rich_photon_copy_postfilter_k)(dim3(number_of_tracks), m_block_dim, context)(
    data<dev_rich_geomeff_offsets_t>(arguments),
    data<dev_offsets_rich_photons_prefilter_t>(arguments),
    data<dev_offsets_rich_photons_t>(arguments),
    data<dev_rich_photons_prefilter_t>(arguments),
    data<dev_rich_photons_t>(arguments));

  // Sum hypos for each tracks (compute per track expected signals from per track-pd expected signals)
  global_function(rich_sum_track_geomeffs_k)(dim3(32), m_block_dim, context)(
    number_of_tracks,
    data<dev_rich_geomeff_offsets_t>(arguments),
    data<dev_rich_geomeff_pd_fractions_t>(arguments),
    data<dev_rich_geomeff_fractions_t>(arguments)); // TODO: this is actually signal not geomeff, rename

  // Transform geomeff_pd_ids from compacted to the global packed index convention matching HLT2 and FromCones
  const auto& to_global_kernel = rich_geomeff_pd_ids_to_global_k<richIdx>;
  global_function(to_global_kernel)(dim3(32), m_block_dim, context)(
    m_pd_infos,
    number_of_events,
    data<dev_offsets_rich_states_t>(arguments),
    number_of_tracks,
    data<dev_segs_best_point_t>(arguments),
    data<dev_rich_geomeff_offsets_t>(arguments),
    data<dev_rich_geomeff_pd_ids_t>(arguments));
}

void rich_quartic_signals::rich_quartic_signals_t::operator()(
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
