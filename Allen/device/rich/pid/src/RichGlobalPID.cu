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

#include "RichGlobalPID.cuh"
#include <BinarySearch.cuh>
#include <array>

INSTANTIATE_ALGORITHM_WITH_ID(
  rich_global_pid::rich_global_pid_t<rich_global_pid::BackgroundEstimationMethod::FromReco>,
  "rich_global_pid_from_reco_t")
INSTANTIATE_ALGORITHM_WITH_ID(
  rich_global_pid::rich_global_pid_t<rich_global_pid::BackgroundEstimationMethod::FromCones>,
  "rich_global_pid_from_cones_t")

// Same fixed-point convention used for the DLL values themselves (as opposed
// to the pixel signals above): a delta log-likelihood of 1e-3 is the
// smallest step that's tracked.
constexpr float dll_fixed_point_scale = 1e3f;
constexpr float inv_dll_fixed_point_scale = 1.f / dll_fixed_point_scale;

constexpr float pix_signals_scale = 1e5f;
constexpr float inv_pix_signals_scale = 1.f / pix_signals_scale;

inline __device__ int signalToFixedPoint(const float signal) { return static_cast<int>(signal * pix_signals_scale); }

inline __device__ float signalFromFixedPoint(const int signal)
{
  return static_cast<float>(signal) * inv_pix_signals_scale;
}

template<rich_global_pid::BackgroundEstimationMethod bkg_method>
template<Allen::Rich::Detector::DetectorType richIdx>
void rich_global_pid::rich_global_pid_t<bkg_method>::updateRich(const Allen::Rich::RichDetector<richIdx>* rich) const
{
  const unsigned ec_per_panel = Allen::Rich::Detector::PDPanel<richIdx>::ECsPerPanel;
  std::vector<uint16_t> effNumPixs {};
  effNumPixs.reserve(Allen::Rich::NPDPanelsPerRICH * ec_per_panel);

  for (const auto panel : Allen::Rich::Detector::sides()) {
    for (unsigned ec = 0; ec < ec_per_panel; ec++) {
      uint16_t count = 0;
      for (unsigned i = 0; i < Allen::Rich::Decoding::SmartID::MaxPDsPerEC; i++) {
        const auto& pd = rich->pdPanels()[panel].pds()[ec * Allen::Rich::Decoding::SmartID::MaxPDsPerEC + i];
        if (!pd.getIsNull()) {
          count += pd.m_numPixels;
        }
      }
      effNumPixs.emplace_back(count);
    }
  }

  if (m_cached_effNumPixsEC[richIdx] != nullptr) Allen::free(m_cached_effNumPixsEC[richIdx]);
  Allen::malloc((void**) &m_cached_effNumPixsEC[richIdx], effNumPixs.size() * sizeof(uint16_t));
  Allen::memcpy(
    m_cached_effNumPixsEC[richIdx], effNumPixs.data(), effNumPixs.size() * sizeof(uint16_t), Allen::memcpyHostToDevice);
}

template<rich_global_pid::BackgroundEstimationMethod bkg_method>
void rich_global_pid::rich_global_pid_t<bkg_method>::update(const Constants& constants) const
{
  updateRich<Allen::Rich::Detector::Rich1>(constants.host_rich_1_geometry);
  updateRich<Allen::Rich::Detector::Rich2>(constants.host_rich_2_geometry);
}

// Sum the signals of every photon from the selected hypo into each pixel:
template<Allen::Rich::Detector::DetectorType richIdx>
__global__ void rich_acc_pixel_signal_k(
  const unsigned number_of_tracks,
  const unsigned number_of_photons,
  const Allen::Rich::ParticleIDType* pids,
  const unsigned* photons_offsets,
  const Allen::Rich::PhotonReco::Photon* photons,
  const Allen::Rich::HypoData<float>* photon_pix_signals,
  int* pixels_signals)
{
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;
  for (unsigned i = threadId; i < number_of_photons; i += stride) {
    const unsigned track_id = binary_search_rightmost(photons_offsets, number_of_tracks + 1, i);

    // TODO: variant where we know all pids are equal to avoid the track_id search ?
    const auto pid = pids[track_id];
    if (pid == Allen::Rich::ParticleIDType::Unknown) continue;
    const float sig = photon_pix_signals[i][pid];
    if (sig <= 0.f) continue;

    const unsigned pix_id = photons[i].pixelIdx;

    // We accumulate the signals as signed integers to avoid atomic float
    // non-determinism while preserving negative hypothesis-change deltas.
    atomicAdd(&pixels_signals[pix_id], signalToFixedPoint(sig));
  }
}

template<rich_global_pid::BackgroundEstimationMethod bkg_method>
template<Allen::Rich::Detector::DetectorType richIdx>
void rich_global_pid::rich_global_pid_t<bkg_method>::pixelSignalsForRich(
  const ArgumentReferences<Parameters<bkg_method>>& arguments,
  const Allen::Context& context,
  const Allen::Rich::ParticleIDType* pids_in) const
{

  using P = Parameters<bkg_method>;

  const unsigned number_of_tracks = first<typename P::host_number_of_tracks_t>(arguments);
  const unsigned number_of_photons = richIdx == Allen::Rich::Detector::Rich1 ?
                                       first<typename P::host_number_of_photons_r1_t>(arguments) :
                                       first<typename P::host_number_of_photons_r2_t>(arguments);

  if constexpr (richIdx == Allen::Rich::Detector::Rich1) {
    Allen::memset_async<typename P::dev_pixel_signals_r1_t>(arguments, 0, context);
  }
  else {
    Allen::memset_async<typename P::dev_pixel_signals_r2_t>(arguments, 0, context);
  }

  const auto& acc_kernel = rich_acc_pixel_signal_k<richIdx>;
  global_function(acc_kernel)(dim3(32), dim3(m_block_dim), context)(
    number_of_tracks,
    number_of_photons,
    pids_in,
    richIdx == Allen::Rich::Detector::Rich1 ? data<typename P::dev_offsets_rich_photons_r1_t>(arguments) :
                                              data<typename P::dev_offsets_rich_photons_r2_t>(arguments),
    richIdx == Allen::Rich::Detector::Rich1 ? data<typename P::dev_rich_photons_r1_t>(arguments) :
                                              data<typename P::dev_rich_photons_r2_t>(arguments),
    richIdx == Allen::Rich::Detector::Rich1 ? data<typename P::dev_photon_pix_signals_r1_t>(arguments) :
                                              data<typename P::dev_photon_pix_signals_r2_t>(arguments),
    richIdx == Allen::Rich::Detector::Rich1 ? data<typename P::dev_pixel_signals_r1_t>(arguments) :
                                              data<typename P::dev_pixel_signals_r2_t>(arguments));
}

template<Allen::Rich::Detector::DetectorType richIdx>
__global__ void rich_exp_signal_from_reco_k(
  const unsigned number_of_tracks,
  const Allen::Rich::ParticleIDType* pids,
  const unsigned* geomeff_offsets,
  const int* geomeff_pd_ids,
  const Allen::Rich::HypoData<float>* geomeff_pd_fractions,
  int* exp_signal_ec)
{
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;

  for (unsigned track_id = threadId; track_id < number_of_tracks; track_id += stride) {
    const auto cur_pid = pids[track_id];
    if (cur_pid == Allen::Rich::ParticleIDType::Unknown) continue;

    const unsigned start = geomeff_offsets[track_id];
    const unsigned end = geomeff_offsets[track_id + 1];

    for (unsigned j = start; j < end; j++) {
      const int pd_id = geomeff_pd_ids[j];
      if (pd_id < 0) continue;

      const float sig = geomeff_pd_fractions[j][cur_pid];
      if (sig <= 0.f) continue;

      const unsigned ec = static_cast<unsigned>(pd_id) / Allen::Rich::Decoding::SmartID::MaxPDsPerEC;
      atomicAdd(&exp_signal_ec[ec], signalToFixedPoint(sig));
    }
  }
}

// Background kernels in the FromCones path. Adapted from
// Rec/Rich/RichFutureRecPixelAlgorithms/src/RichSIMDPixelBackgroundsEstiAvHPD.cpp. Scatter yield[hypo] * per-EC
// geometrical efficiency from every track's current best-fit hypothesis into the EC it geometrically overlaps. Mirrors
// Rec's "fill expected signal" loop (SIMDPixelBackgroundsEstiAvHPD::operator(), step 3): for each track, for each PD
// its ring overlaps at the current hypothesis, accumulate detYield[hypo] * PD.eff.
template<Allen::Rich::Detector::DetectorType richIdx>
__global__ void rich_exp_signal_from_cones_k(
  const unsigned number_of_tracks,
  const Allen::Rich::ParticleIDType* pids,
  const Allen::Rich::ParticleHypos* hypos,
  const unsigned* geomeff_offsets,
  const int* geomeff_pd_ids,
  const float* geomeff_fractions,
  int* exp_signal_ec)
{
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;

  // loop over track data
  for (unsigned track_id = threadId; track_id < number_of_tracks; track_id += stride) {
    const auto cur_pid = pids[track_id];
    if (cur_pid == Allen::Rich::ParticleIDType::Unknown) continue;
    const unsigned hypo = static_cast<unsigned>(cur_pid);

    const float yield = hypos[track_id].yield[cur_pid];
    if (yield <= 0.f) continue;

    const unsigned idx = track_id * Allen::Rich::NRealParticleTypes + hypo;
    const unsigned start = geomeff_offsets[idx];
    const unsigned end = geomeff_offsets[idx + 1];

    // Loop over the per PD geom. effs. for this track hypo
    for (unsigned j = start; j < end; j++) {
      const int pd_id = geomeff_pd_ids[j];
      if (pd_id < 0) continue; // ray missed the panel for this ring sample point

      const unsigned ec = static_cast<unsigned>(pd_id) / Allen::Rich::Decoding::SmartID::MaxPDsPerEC;
      // fill expected signal for this PD
      atomicAdd(&exp_signal_ec[ec], signalToFixedPoint(yield * geomeff_fractions[j]));
    }
  }
}

// Per-EC background estimate, matching Rec's
// SIMDPixelBackgroundsEstiAvHPD steps 4-5 (normalize + broadcast to pixels), without the iterative nBelow/nAbove/rnorm
// redistribution bi = (observed - expected) / nActivePixels, no cross-EC refinement.
template<Allen::Rich::Detector::DetectorType richIdx>
__global__ void rich_bkg_from_expected_signal_k(
  const uint16_t* effNumPixsEC,
  const unsigned number_of_events,
  const unsigned* pixels_offsets,
  const int* exp_signal_ec,
  float* ec_bkg,
  int* pixel_signals,
  const float bkg_weight,
  const float bkg_threshold,
  const float bkg_min,
  const float bkg_max)
{
  const unsigned ec_per_panel = Allen::Rich::Detector::PDPanel<richIdx>::ECsPerPanel;
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;

  for (unsigned i = threadId; i < 2 * number_of_events * ec_per_panel; i += stride) {
    const unsigned side = i / (number_of_events * ec_per_panel);
    const unsigned ec_index = i % ec_per_panel;

    const unsigned effNumPixs = effNumPixsEC[ec_index + side * ec_per_panel];
    if (effNumPixs == 0) continue;

    const unsigned pixStart = pixels_offsets[i * Allen::Rich::Decoding::SmartID::MaxPDsPerEC];
    const unsigned pixEnd = pixels_offsets[(i + 1) * Allen::Rich::Decoding::SmartID::MaxPDsPerEC];
    if (pixEnd == pixStart) continue; // no observed hits in this EC

    const unsigned obsSignal = pixEnd - pixStart;

    // TODO: Add here iterative background refinement loop
    // code here...

    float expBackgrd = static_cast<float>(obsSignal) - signalFromFixedPoint(exp_signal_ec[i]);

    // normalize
    // divide by effective pixel count, only if positive
    expBackgrd = expBackgrd > 0.f ? expBackgrd / static_cast<float>(effNumPixs) : 0.f;
    // scale by the per-RICH weight
    expBackgrd *= bkg_weight;
    // threshold cut
    if (expBackgrd < bkg_threshold) expBackgrd = 0.f;
    // clamp to [min, max]
    expBackgrd = fminf(fmaxf(expBackgrd, bkg_min), bkg_max);

    ec_bkg[i] = expBackgrd;

    // broadcast to every pixel in this EC, added into the combined signal+bkg buffer
    for (unsigned j = pixStart; j < pixEnd; j++) {
      pixel_signals[j] += signalToFixedPoint(expBackgrd);
    }
  }
}

// Backgrounds FromReco and FromCones shared launcher
template<rich_global_pid::BackgroundEstimationMethod bkg_method>
template<Allen::Rich::Detector::DetectorType richIdx>
void rich_global_pid::rich_global_pid_t<bkg_method>::backgroundsForRich(
  const ArgumentReferences<Parameters<bkg_method>>& arguments,
  const Allen::Context& context,
  const Allen::Rich::ParticleIDType* pids,
  const unsigned it) const
{
  using P = Parameters<bkg_method>;
  const unsigned number_of_events = first<typename P::host_number_of_events_t>(arguments);
  const unsigned number_of_tracks = first<typename P::host_number_of_tracks_t>(arguments);

  if constexpr (richIdx == Allen::Rich::Detector::Rich1) {
    Allen::memset_async<typename P::dev_exp_signal_ec_r1_t>(arguments, 0, context);
  }
  else {
    Allen::memset_async<typename P::dev_exp_signal_ec_r2_t>(arguments, 0, context);
  }

  auto* exp_signal_ec = richIdx == Allen::Rich::Detector::Rich1 ? data<typename P::dev_exp_signal_ec_r1_t>(arguments) :
                                                                  data<typename P::dev_exp_signal_ec_r2_t>(arguments);

  if (!m_ignoreExpSignal.value()[it]) {
    if constexpr (bkg_method == BackgroundEstimationMethod::FromReco) {
      global_function(rich_exp_signal_from_reco_k<richIdx>)(dim3(32), dim3(m_block_dim), context)(
        number_of_tracks,
        pids,
        richIdx == Allen::Rich::Detector::Rich1 ? data<typename P::dev_rich_geomeff_offsets_r1_t>(arguments) :
                                                  data<typename P::dev_rich_geomeff_offsets_r2_t>(arguments),
        richIdx == Allen::Rich::Detector::Rich1 ? data<typename P::dev_rich_geomeff_pd_ids_r1_t>(arguments) :
                                                  data<typename P::dev_rich_geomeff_pd_ids_r2_t>(arguments),
        richIdx == Allen::Rich::Detector::Rich1 ? data<typename P::dev_rich_geomeff_pd_fractions_r1_t>(arguments) :
                                                  data<typename P::dev_rich_geomeff_pd_fractions_r2_t>(arguments),
        exp_signal_ec);
    }
    else { // BackgroundEstimationMethod::FromCones
      global_function(rich_exp_signal_from_cones_k<richIdx>)(dim3(32), dim3(m_block_dim), context)(
        number_of_tracks,
        pids,
        richIdx == Allen::Rich::Detector::Rich1 ? data<typename P::dev_rich_hypos_r1_t>(arguments) :
                                                  data<typename P::dev_rich_hypos_r2_t>(arguments),
        richIdx == Allen::Rich::Detector::Rich1 ? data<typename P::dev_rich_geomeff_offsets_r1_t>(arguments) :
                                                  data<typename P::dev_rich_geomeff_offsets_r2_t>(arguments),
        richIdx == Allen::Rich::Detector::Rich1 ? data<typename P::dev_rich_geomeff_pd_ids_r1_t>(arguments) :
                                                  data<typename P::dev_rich_geomeff_pd_ids_r2_t>(arguments),
        richIdx == Allen::Rich::Detector::Rich1 ? data<typename P::dev_rich_geomeff_fractions_r1_t>(arguments) :
                                                  data<typename P::dev_rich_geomeff_fractions_r2_t>(arguments),
        exp_signal_ec);
    }
  }

  // bkg property index
  const unsigned bkgIdx = it * 2 + static_cast<unsigned>(richIdx);
  global_function(rich_bkg_from_expected_signal_k<richIdx>)(dim3(32), dim3(m_block_dim), context)(
    m_cached_effNumPixsEC[richIdx],
    number_of_events,
    richIdx == Allen::Rich::Detector::Rich1 ? data<typename P::dev_rich_pd_offsets_r1_t>(arguments) :
                                              data<typename P::dev_rich_pd_offsets_r2_t>(arguments),
    exp_signal_ec,
    richIdx == Allen::Rich::Detector::Rich1 ? data<typename P::dev_pix_bkg_r1_t>(arguments) :
                                              data<typename P::dev_pix_bkg_r2_t>(arguments),
    richIdx == Allen::Rich::Detector::Rich1 ? data<typename P::dev_pixel_signals_r1_t>(arguments) :
                                              data<typename P::dev_pixel_signals_r2_t>(arguments),
    m_bkgWeight.value()[bkgIdx],
    m_bkgThreshold.value()[bkgIdx],
    m_bkgMin.value()[bkgIdx],
    m_bkgMax.value()[bkgIdx]);
}

inline __device__ float sigFunc(float sig)
{
  // Floor to the fixed-point signal granularity and use the small-x expansion
  // of log(exp(x) - 1) to avoid cancellation near zero.
  constexpr float min_sig = inv_pix_signals_scale;
  const float x = fmaxf(sig, min_sig);
  return x < 1e-2f ? logf(x) + x * (0.5f + x * (1.f / 24.f)) : logf(expm1f(x));
}

// Accumulate the pixel part of the delta log likelihood
__global__ void rich_acc_pixel_dll_k(
  const unsigned number_of_tracks,
  const unsigned number_of_photons,
  const Allen::Rich::ParticleIDType* pids,
  const unsigned* photons_offsets,
  const Allen::Rich::PhotonReco::Photon* photons,
  const Allen::Rich::HypoData<float>* photon_pix_signals,
  int* pixel_signals,
  Allen::Rich::HypoData<float>* dlls)
{
  Allen::Rich::HypoData<int>* dlls_int = reinterpret_cast<Allen::Rich::HypoData<int>*>(dlls);
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;
  for (unsigned i = threadId; i < number_of_photons; i += stride) {
    const unsigned track_id = binary_search_rightmost(photons_offsets, number_of_tracks + 1, i);

    const auto cur_pid = pids[track_id];
    const float cur_sig = (cur_pid != Allen::Rich::ParticleIDType::Unknown) ? photon_pix_signals[i][cur_pid] : 0.f;
    const unsigned pix_id = photons[i].pixelIdx;
    float pix_sig_bkg = signalFromFixedPoint(pixel_signals[pix_id]);
    const float deltaLLbase = sigFunc(pix_sig_bkg);
    pix_sig_bkg -= cur_sig;

    UNROLL(Allen::Rich::NParticleTypes)
    for (unsigned pid_index = 0; pid_index < Allen::Rich::NParticleTypes; ++pid_index) {
      const auto new_pid = static_cast<Allen::Rich::ParticleIDType>(pid_index);
      if (new_pid == cur_pid) continue;

      const float new_sig = photon_pix_signals[i][new_pid];
      const float deltaLL = deltaLLbase - sigFunc(pix_sig_bkg + new_sig);
      const auto deltaInt = static_cast<int>(deltaLL * dll_fixed_point_scale);

      if (deltaInt != 0) atomicAdd(&dlls_int[track_id][new_pid], deltaInt);
    }
  }
}

// Finish the delta log likelihood computation (convert back to float and add the track part),
// then find the hypothesis that maximize the DLL:
__global__ void rich_init_dll_best_hypo_k(
  const unsigned number_of_tracks,
  const Allen::Rich::ParticleIDType* pids_in,
  const Allen::Rich::HypoData<float>* track_signals_r1,
  const Allen::Rich::HypoData<float>* track_signals_r2,
  Allen::Rich::HypoData<float>* dlls,
  Allen::Rich::ParticleIDType* pids_out)
{
  Allen::Rich::HypoData<int>* dlls_int = reinterpret_cast<Allen::Rich::HypoData<int>*>(dlls);
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;
  for (unsigned track_id = threadId; track_id < number_of_tracks; track_id += stride) {

    const auto cur_pid = pids_in[track_id];
    const float cur_sig = (cur_pid != Allen::Rich::ParticleIDType::Unknown) ?
                            track_signals_r1[track_id][cur_pid] + track_signals_r2[track_id][cur_pid] :
                            0.f;

    Allen::Rich::ParticleIDType bestPID = cur_pid;
    float bestDLL = 0.f;
    for (unsigned pid_index = 0; pid_index < Allen::Rich::NParticleTypes; ++pid_index) {
      const auto new_pid = static_cast<Allen::Rich::ParticleIDType>(pid_index);
      if (new_pid == cur_pid) continue;

      const float new_sig = track_signals_r1[track_id][new_pid] + track_signals_r2[track_id][new_pid];

      const float cur_dll =
        static_cast<float>(dlls_int[track_id][new_pid]) * inv_dll_fixed_point_scale + (new_sig - cur_sig);
      if (cur_dll < bestDLL) {
        bestDLL = cur_dll;
        bestPID = new_pid;
      }
      // Store back in fixed-point: dlls stays int across every kernel in this
      // algorithm (only converted to float once, by rich_dll_finalise_k,
      // after all outer likelihood iterations are done) so it never needs
      // converting back and forth per iteration.
      dlls_int[track_id][new_pid] = static_cast<int>(cur_dll * dll_fixed_point_scale);
    }

    // std::cout << "[Allen] Track " << track_id << " pid " << cur_pid << " -> " << bestPID << " DLL " << dlls[track_id]
    // << std::endl;

    pids_out[track_id] = bestPID;
  }
}

// Finalise the whole DLL buffer: convert it out of the fixed-point integer
// representation used throughout every kernel above (to keep GPU atomicAdd
// accumulation associative) back to real floats, and normalise it to the
// pion-relative, inverted convention expected by the rest of the pipeline.
// Both are a single post-processing pass over the same per-track DLL array
// - done together in one kernel rather than two back-to-back launches - run
// once, after all of the outer likelihood iterations are done (not per
// iteration, since dlls is discarded (memset back to 0) at the start of
// every outer iteration anyway).
__global__ void rich_dll_finalise_k(const unsigned number_of_tracks, Allen::Rich::HypoData<float>* dlls)
{
  const Allen::Rich::HypoData<int>* dlls_int = reinterpret_cast<const Allen::Rich::HypoData<int>*>(dlls);
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;

  for (unsigned t = threadId; t < number_of_tracks; t += stride) {
    float dll[Allen::Rich::NParticleTypes];
    UNROLL(Allen::Rich::NParticleTypes)
    for (unsigned particle_index = 0; particle_index < Allen::Rich::NParticleTypes; ++particle_index) {
      const auto particle = static_cast<Allen::Rich::ParticleIDType>(particle_index);
      dll[particle_index] = static_cast<float>(dlls_int[t][particle]) * inv_dll_fixed_point_scale;
    }

    // Get dll relative to pion
    const float dll_pion = dll[Allen::Rich::ParticleIDType::Pion];

    UNROLL(Allen::Rich::NParticleTypes)
    for (unsigned particle_index = 0; particle_index < Allen::Rich::NParticleTypes; ++particle_index) {
      const auto particle = static_cast<Allen::Rich::ParticleIDType>(particle_index);
      // Internally, the Global PID normalises the DLL values to the best hypothesis
      // and also works in "-loglikelihood" space.
      // For final storage, renormalise the DLLS w.r.t. the pion hypothesis and
      // invert the values
      dlls[t][particle] = dll_pion - dll[particle_index];
    }

    // Ensure pion is exactly 0
    dlls[t][Allen::Rich::ParticleIDType::Pion] = 0.f;
  }
}

__global__ void rich_global_pid_init_update_signals_k(
  const unsigned number_of_tracks,
  const unsigned number_of_photons,
  const Allen::Rich::ParticleIDType* pids_old,
  const Allen::Rich::ParticleIDType* pids_new,
  const unsigned* photons_offsets,
  const Allen::Rich::PhotonReco::Photon* photons,
  const Allen::Rich::HypoData<float>* photon_pix_signals,
  int* pixel_signals)
{
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;

  // TODO: to iterate over tracks or over photons...?
  for (unsigned i = threadId; i < number_of_photons; i += stride) {
    const unsigned track_id = binary_search_rightmost(photons_offsets, number_of_tracks + 1, i);

    const auto old_pid = pids_old[track_id];
    const auto new_pid = pids_new[track_id];

    if (old_pid == new_pid) continue;

    const unsigned pix_id = photons[i].pixelIdx;
    const float delta = photon_pix_signals[i][new_pid] - photon_pix_signals[i][old_pid];

    atomicAdd(&pixel_signals[pix_id], signalToFixedPoint(delta));
  }
}

// Phase D helper for a single RICH detector.
// For every photon of the track whose hypothesis changed (gt):
//   * update the pixel signal in place (regular store, no race: each photon
//     of a track hits a distinct pixel, and each warp owns one event),
//   * accumulate the pixel term of gt's new DLLs into dll_reg,
//   * propagate the resulting DLL correction to every other track sharing
//     that pixel (direct lookup via pix2photon).
inline __device__ void rich_global_pid_update_and_propagate(
  const unsigned gt,
  const Allen::Rich::ParticleIDType old_pid,
  const Allen::Rich::ParticleIDType new_pid,
  const unsigned ph_start,
  const unsigned ph_end,
  const Allen::Rich::PhotonReco::Photon* photons,
  const Allen::Rich::HypoData<float>* photon_pix_signals,
  int* pixel_signals,
  const unsigned* pix2track_offsets,
  const unsigned* pix2track,
  const unsigned* pix2photon,
  const Allen::Rich::ParticleIDType* pids,
  Allen::Rich::HypoData<int>* dlls_int,
  std::array<float, Allen::Rich::NParticleTypes>& dll_reg)
{
  const unsigned lane_id = threadIdx.x % warp_size;

  for (unsigned p = ph_start + lane_id; p < ph_end; p += warp_size) {
    const int delta = signalToFixedPoint(photon_pix_signals[p][new_pid] - photon_pix_signals[p][old_pid]);
    const unsigned pix = photons[p].pixelIdx;
    const int S_old_i = pixel_signals[pix];
    const int S_new_i = S_old_i + delta;
    const float S_new = signalFromFixedPoint(S_new_i);

    const float sig_new = sigFunc(S_new);

    // gt's own pixel term: always accumulate, using the post-update signal.
    const float sig_cur_gt = photon_pix_signals[p][new_pid];
    UNROLL(Allen::Rich::NParticleTypes)
    for (unsigned pid_idx = 0; pid_idx < Allen::Rich::NParticleTypes; ++pid_idx) {
      if (pid_idx == static_cast<unsigned>(new_pid)) continue;
      const auto particle = static_cast<Allen::Rich::ParticleIDType>(pid_idx);
      const float sig_h = photon_pix_signals[p][particle];
      dll_reg[pid_idx] += sig_new - sigFunc(S_new - sig_cur_gt + sig_h);
    }

    if (delta == 0) continue; // no pixel change: skip write and propagation

    pixel_signals[pix] = S_new_i;
    const float S_old = signalFromFixedPoint(S_old_i);
    const float sig_old = sigFunc(S_old);

    // Propagate to the other tracks sharing this pixel.
    for (unsigned k = pix2track_offsets[pix]; k < pix2track_offsets[pix + 1]; k++) {
      const unsigned tp = pix2track[k];
      if (tp == gt) continue;

      const auto cur_pid_tp = pids[tp];
      if (cur_pid_tp == Allen::Rich::ParticleIDType::Unknown) continue;

      const unsigned p2 = pix2photon[k];
      const float sig_cur_tp = photon_pix_signals[p2][cur_pid_tp];

      UNROLL(Allen::Rich::NParticleTypes)
      for (unsigned particle_index = 0; particle_index < Allen::Rich::NParticleTypes; ++particle_index) {
        const auto particle = static_cast<Allen::Rich::ParticleIDType>(particle_index);
        if (particle == cur_pid_tp) continue;
        const float sig_h = photon_pix_signals[p2][particle];
        const float d_dll =
          (sig_new - sigFunc(S_new - sig_cur_tp + sig_h)) - (sig_old - sigFunc(S_old - sig_cur_tp + sig_h));
        atomicAdd(&dlls_int[tp][particle], static_cast<int>(d_dll * dll_fixed_point_scale));
      }
    }
  }
}

__global__ void rich_global_pid_iterations_k(
  const unsigned number_of_events,
  const unsigned* dev_offsets_tracks,
  const unsigned* dev_offsets_rich_photons_r1,
  const Allen::Rich::PhotonReco::Photon* dev_rich_photons_r1,
  const Allen::Rich::HypoData<float>* dev_photon_pix_signals_r1,
  const Allen::Rich::HypoData<float>* dev_track_total_signals_r1,
  int* dev_pixel_signals_r1,
  const unsigned* dev_offsets_rich_photons_r2,
  const Allen::Rich::PhotonReco::Photon* dev_rich_photons_r2,
  const Allen::Rich::HypoData<float>* dev_photon_pix_signals_r2,
  const Allen::Rich::HypoData<float>* dev_track_total_signals_r2,
  int* dev_pixel_signals_r2,
  Allen::Rich::ParticleIDType* pids,
  Allen::Rich::HypoData<float>* dlls,
  const unsigned* dev_pix2track_offsets_r1,
  const unsigned* dev_pix2track_r1,
  const unsigned* dev_pix2photon_r1,
  const unsigned* dev_pix2track_offsets_r2,
  const unsigned* dev_pix2track_r2,
  const unsigned* dev_pix2photon_r2,
  const float epsilon,
  const unsigned max_iterations)
{
  // Multiple warps per block, each warp handles one event.
  const unsigned warp_id = threadIdx.x / warp_size;
  const unsigned lane_id = threadIdx.x % warp_size;
  const unsigned warps_per_block = blockDim.x / warp_size;
  const unsigned event_id = blockIdx.x * warps_per_block + warp_id;
  if (event_id >= number_of_events) return;

  const unsigned track_start = dev_offsets_tracks[event_id];
  const unsigned track_end = dev_offsets_tracks[event_id + 1];
  const unsigned n_tracks = track_end - track_start;

  if (n_tracks == 0) return;

  // dlls is kept as a fixed-point integer representation throughout this
  // whole algorithm (every kernel, across every outer likelihood iteration -
  // only converted to float once, at the very end, by
  // rich_dll_finalise_k), precisely so that no kernel needs to convert
  // back and forth: the atomicAdd below can then accumulate over associative
  // integer addition instead of order-dependent float addition, without any
  // entry/exit conversion of its own.
  auto* dlls_int = reinterpret_cast<Allen::Rich::HypoData<int>*>(dlls);

  const int epsilon_int = static_cast<int>(epsilon * dll_fixed_point_scale);

  unsigned iteration = 0;
  while (true) {

    // Phase A: track scan to read existing DLL values
    // find best log likelihood
    // thread "local" variables
    int local_best_dll_int = 0; // 0 = no improvement found yet
    int local_best_track = -1;  // event-local index
    auto local_best_pid = Allen::Rich::ParticleIDType::Unknown;

    // t is event-local, so each warp must scan with its lane-local index.
    for (unsigned t = lane_id; t < n_tracks; t += warp_size) {
      const unsigned gt = track_start + t; // global track index
      const auto cur_pid = pids[gt];
      if (cur_pid == Allen::Rich::ParticleIDType::Unknown) continue;

      int track_best_dll_int = 0;
      auto track_best_pid = Allen::Rich::ParticleIDType::Unknown;

      for (unsigned pid_index = 0; pid_index < Allen::Rich::NParticleTypes; ++pid_index) {
        const auto new_pid = static_cast<Allen::Rich::ParticleIDType>(pid_index);
        if (new_pid == cur_pid) continue;
        const int dll_int = dlls_int[gt][new_pid];

        // update dll
        if (dll_int < track_best_dll_int) {
          track_best_dll_int = dll_int;
          track_best_pid = new_pid;
        }
      } // end hypo loop

      // update thread values
      if (track_best_dll_int < local_best_dll_int) {
        local_best_dll_int = track_best_dll_int;
        local_best_track = static_cast<int>(t);
        local_best_pid = track_best_pid;
      }
    } // end track loop

    // Phase B: Warp butterfly reduction
    int global_best_dll_int = local_best_dll_int;
    int global_best_trk = local_best_track;

    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
      const int neighbor_dll_int = __shfl_xor_sync(0xffffffff, global_best_dll_int, offset);
      const int neighbor_trk = __shfl_xor_sync(0xffffffff, global_best_trk, offset);
      if (
        neighbor_dll_int < global_best_dll_int ||
        (neighbor_dll_int == global_best_dll_int && neighbor_trk < global_best_trk)) {
        global_best_dll_int = neighbor_dll_int;
        global_best_trk = neighbor_trk;
      }
    }
    // All lanes now have the same global_best_dll_int and global_best_trk.
    // Find the winning lane and broadcast its local_best_pid.
    const unsigned winner_lane = __ffs(__ballot_sync(0xffffffff, local_best_track == global_best_trk)) - 1;
    const auto global_best_pid =
      static_cast<Allen::Rich::ParticleIDType>(__shfl_sync(0xffffffff, static_cast<int>(local_best_pid), winner_lane));

    // Phase C: convergence check
    if (global_best_dll_int >= epsilon_int || global_best_trk == -1) break;
    if (++iteration >= max_iterations) break;

    // Phase D: update values for next iter (hypothesis change)
    const unsigned gt = track_start + static_cast<unsigned>(global_best_trk);
    const auto old_pid = pids[gt];
    const auto new_pid = global_best_pid;

    // Phase D: merged single-pass update.
    // For every photon of gt: update the pixel signal in place, accumulate the
    // pixel term of gt's new DLLs in registers, and propagate the DLL delta to
    // every other track sharing that pixel.
    std::array<float, Allen::Rich::NParticleTypes> dll_reg {};

    rich_global_pid_update_and_propagate(
      gt,
      old_pid,
      new_pid,
      dev_offsets_rich_photons_r1[gt],
      dev_offsets_rich_photons_r1[gt + 1],
      dev_rich_photons_r1,
      dev_photon_pix_signals_r1,
      dev_pixel_signals_r1,
      dev_pix2track_offsets_r1,
      dev_pix2track_r1,
      dev_pix2photon_r1,
      pids,
      dlls_int,
      dll_reg);

    rich_global_pid_update_and_propagate(
      gt,
      old_pid,
      new_pid,
      dev_offsets_rich_photons_r2[gt],
      dev_offsets_rich_photons_r2[gt + 1],
      dev_rich_photons_r2,
      dev_photon_pix_signals_r2,
      dev_pixel_signals_r2,
      dev_pix2track_offsets_r2,
      dev_pix2track_r2,
      dev_pix2photon_r2,
      pids,
      dlls_int,
      dll_reg);

    // Warp-reduce gt's per-hypothesis pixel terms.
    UNROLL(Allen::Rich::NParticleTypes)
    for (unsigned pid_idx = 0; pid_idx < Allen::Rich::NParticleTypes; ++pid_idx) {
      if (pid_idx == static_cast<unsigned>(new_pid)) continue;
      for (int offset = warp_size / 2; offset > 0; offset /= 2)
        dll_reg[pid_idx] += __shfl_down_sync(0xffffffff, dll_reg[pid_idx], offset);
    }

    // Finalise gt's DLLs: add the track term and store.
    if (lane_id == 0) {
      pids[gt] = new_pid;
      dlls_int[gt][new_pid] = 0;
      UNROLL(Allen::Rich::NParticleTypes)
      for (unsigned pid_idx = 0; pid_idx < Allen::Rich::NParticleTypes; ++pid_idx) {
        if (pid_idx == static_cast<unsigned>(new_pid)) continue;
        const auto particle = static_cast<Allen::Rich::ParticleIDType>(pid_idx);
        dll_reg[pid_idx] += (dev_track_total_signals_r1[gt][particle] + dev_track_total_signals_r2[gt][particle]) -
                            (dev_track_total_signals_r1[gt][new_pid] + dev_track_total_signals_r2[gt][new_pid]);
        dlls_int[gt][particle] = static_cast<int>(dll_reg[pid_idx] * dll_fixed_point_scale);
      }
    }
    __syncwarp(); // ensure all updates visible before next iteration's Phase A
  }               // iterations while loop
}

template<rich_global_pid::BackgroundEstimationMethod bkg_method>
void rich_global_pid::rich_global_pid_t<bkg_method>::initDLLs(
  const ArgumentReferences<Parameters<bkg_method>>& arguments,
  const Allen::Context& context,
  const Allen::Rich::ParticleIDType* pids_in,
  Allen::Rich::ParticleIDType* pids_out) const
{

  using P = Parameters<bkg_method>;

  const unsigned number_of_tracks = first<typename P::host_number_of_tracks_t>(arguments);
  const unsigned number_of_photons_r1 = first<typename P::host_number_of_photons_r1_t>(arguments);
  const unsigned number_of_photons_r2 = first<typename P::host_number_of_photons_r2_t>(arguments);

  Allen::memset_async<typename P::dev_dll_out_t>(arguments, 0, context);

  global_function(rich_acc_pixel_dll_k)(dim3(32), dim3(m_block_dim), context)(
    number_of_tracks,
    number_of_photons_r1,
    pids_in,
    data<typename P::dev_offsets_rich_photons_r1_t>(arguments),
    data<typename P::dev_rich_photons_r1_t>(arguments),
    data<typename P::dev_photon_pix_signals_r1_t>(arguments),
    data<typename P::dev_pixel_signals_r1_t>(arguments),
    data<typename P::dev_dll_out_t>(arguments));

  global_function(rich_acc_pixel_dll_k)(dim3(32), dim3(m_block_dim), context)(
    number_of_tracks,
    number_of_photons_r2,
    pids_in,
    data<typename P::dev_offsets_rich_photons_r2_t>(arguments),
    data<typename P::dev_rich_photons_r2_t>(arguments),
    data<typename P::dev_photon_pix_signals_r2_t>(arguments),
    data<typename P::dev_pixel_signals_r2_t>(arguments),
    data<typename P::dev_dll_out_t>(arguments));

  global_function(rich_init_dll_best_hypo_k)(dim3(32), dim3(m_block_dim), context)(
    number_of_tracks,
    pids_in,
    data<typename P::dev_track_total_signals_r1_t>(arguments),
    data<typename P::dev_track_total_signals_r2_t>(arguments),
    data<typename P::dev_dll_out_t>(arguments),
    pids_out);

  // recompute the DLLs based on the best hypo that was selected
  global_function(rich_global_pid_init_update_signals_k)(dim3(32), dim3(m_block_dim), context)(
    number_of_tracks,
    number_of_photons_r1,
    pids_in,
    pids_out,
    data<typename P::dev_offsets_rich_photons_r1_t>(arguments),
    data<typename P::dev_rich_photons_r1_t>(arguments),
    data<typename P::dev_photon_pix_signals_r1_t>(arguments),
    data<typename P::dev_pixel_signals_r1_t>(arguments));

  global_function(rich_global_pid_init_update_signals_k)(dim3(32), dim3(m_block_dim), context)(
    number_of_tracks,
    number_of_photons_r2,
    pids_in,
    pids_out,
    data<typename P::dev_offsets_rich_photons_r2_t>(arguments),
    data<typename P::dev_rich_photons_r2_t>(arguments),
    data<typename P::dev_photon_pix_signals_r2_t>(arguments),
    data<typename P::dev_pixel_signals_r2_t>(arguments));
}

template<rich_global_pid::BackgroundEstimationMethod bkg_method>
void rich_global_pid::rich_global_pid_t<bkg_method>::doIterations(
  const ArgumentReferences<Parameters<bkg_method>>& arguments,
  const Allen::Context& context,
  Allen::Rich::ParticleIDType* pids,
  Allen::Rich::HypoData<float>* dlls,
  const unsigned* pix2track_offsets_r1,
  const unsigned* pix2track_r1,
  const unsigned* pix2photon_r1,
  const unsigned* pix2track_offsets_r2,
  const unsigned* pix2track_r2,
  const unsigned* pix2photon_r2) const
{

  using P = Parameters<bkg_method>;

  const unsigned number_of_events = first<typename P::host_number_of_events_t>(arguments);

  constexpr unsigned warps_per_block = 2;
  const unsigned n_blocks = (number_of_events + warps_per_block - 1) / warps_per_block;

  global_function(rich_global_pid_iterations_k)(dim3(n_blocks), dim3(warps_per_block * warp_size), context)(
    number_of_events,
    data<typename P::dev_offsets_tracks_t>(arguments),
    data<typename P::dev_offsets_rich_photons_r1_t>(arguments),
    data<typename P::dev_rich_photons_r1_t>(arguments),
    data<typename P::dev_photon_pix_signals_r1_t>(arguments),
    data<typename P::dev_track_total_signals_r1_t>(arguments),
    data<typename P::dev_pixel_signals_r1_t>(arguments),
    data<typename P::dev_offsets_rich_photons_r2_t>(arguments),
    data<typename P::dev_rich_photons_r2_t>(arguments),
    data<typename P::dev_photon_pix_signals_r2_t>(arguments),
    data<typename P::dev_track_total_signals_r2_t>(arguments),
    data<typename P::dev_pixel_signals_r2_t>(arguments),
    pids,
    dlls,
    pix2track_offsets_r1,
    pix2track_r1,
    pix2photon_r1,
    pix2track_offsets_r2,
    pix2track_r2,
    pix2photon_r2,
    m_epsilon.value(),
    m_maxEventIterations.value());
}

template<rich_global_pid::BackgroundEstimationMethod bkg_method>
void rich_global_pid::rich_global_pid_t<bkg_method>::set_arguments_size(
  ArgumentReferences<Parameters<bkg_method>> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  using P = Parameters<bkg_method>;

  const unsigned number_of_events = first<typename P::host_number_of_events_t>(arguments);
  const unsigned number_of_tracks = first<typename P::host_number_of_tracks_t>(arguments);
  const unsigned number_of_pixels_r1 = first<typename P::host_number_of_pixels_r1_t>(arguments);
  const unsigned number_of_pixels_r2 = first<typename P::host_number_of_pixels_r2_t>(arguments);

  set_size<typename P::dev_pixel_signals_r1_t>(arguments, number_of_pixels_r1);
  set_size<typename P::dev_pixel_signals_r2_t>(arguments, number_of_pixels_r2);

  set_size<typename P::dev_exp_signal_ec_r1_t>(
    arguments,
    Allen::Rich::NPDPanelsPerRICH * number_of_events *
      Allen::Rich::Detector::PDPanel<Allen::Rich::Detector::Rich1>::ECsPerPanel);
  set_size<typename P::dev_exp_signal_ec_r2_t>(
    arguments,
    Allen::Rich::NPDPanelsPerRICH * number_of_events *
      Allen::Rich::Detector::PDPanel<Allen::Rich::Detector::Rich2>::ECsPerPanel);

  set_size<typename P::dev_pix_bkg_r1_t>(
    arguments,
    Allen::Rich::NPDPanelsPerRICH * number_of_events *
      Allen::Rich::Detector::PDPanel<Allen::Rich::Detector::Rich1>::ECsPerPanel);
  set_size<typename P::dev_pix_bkg_r2_t>(
    arguments,
    Allen::Rich::NPDPanelsPerRICH * number_of_events *
      Allen::Rich::Detector::PDPanel<Allen::Rich::Detector::Rich2>::ECsPerPanel);

  set_size<typename P::dev_pid_out_t>(arguments, number_of_tracks);
  set_size<typename P::dev_dll_out_t>(arguments, number_of_tracks);
}

template<rich_global_pid::BackgroundEstimationMethod bkg_method>
void rich_global_pid::rich_global_pid_t<bkg_method>::operator()(
  const ArgumentReferences<Parameters<bkg_method>>& arguments,
  const RuntimeOptions&,
  const Constants& constants,
  const Allen::Context& context) const
{

  using P = Parameters<bkg_method>;

  [[maybe_unused]] const Allen::Rich::RichDetector<Allen::Rich::Detector::Rich1>* rich1 = constants.dev_rich_1_geometry;
  [[maybe_unused]] const Allen::Rich::RichDetector<Allen::Rich::Detector::Rich2>* rich2 = constants.dev_rich_2_geometry;

  // const unsigned number_of_events = first<host_number_of_events_t>(arguments);
  const unsigned number_of_tracks = first<typename P::host_number_of_tracks_t>(arguments);
  auto pids_tmp_buffer =
    arguments.template make_buffer<Allen::Store::Scope::Device, Allen::Rich::ParticleIDType>(number_of_tracks);

  Allen::Rich::ParticleIDType* pids_in = pids_tmp_buffer.data();
  Allen::Rich::ParticleIDType* pids_out = data<typename P::dev_pid_out_t>(arguments);

  if (m_nLikelihoodIterations.value() % 2 == 0) {
    // Make sure the last iteration writes to dev_pid_out_t
    std::swap(pids_in, pids_out);
  }

  Allen::memcpy_async(
    pids_in,
    data<typename P::dev_pid_in_t>(arguments),
    number_of_tracks * sizeof(Allen::Rich::ParticleIDType),
    Allen::memcpyDeviceToDevice,
    context);

  for (unsigned it = 0; it < m_nLikelihoodIterations.value(); it++) {

    // Init pixel signals:
    pixelSignalsForRich<Allen::Rich::Detector::Rich1>(arguments, context, pids_in);
    pixelSignalsForRich<Allen::Rich::Detector::Rich2>(arguments, context, pids_in);

    // Compute backgrounds:
    backgroundsForRich<Allen::Rich::Detector::Rich1>(arguments, context, pids_in, it);
    backgroundsForRich<Allen::Rich::Detector::Rich2>(arguments, context, pids_in, it);

    // Init DLLs and set to best hypothesis:
    initDLLs(arguments, context, pids_in, pids_out);

    // Do iterations until convergence:
    doIterations(
      arguments,
      context,
      pids_out,
      data<typename P::dev_dll_out_t>(arguments),
      data<typename P::dev_pix2track_offsets_r1_t>(arguments),
      data<typename P::dev_pix2track_r1_t>(arguments),
      data<typename P::dev_pix2photon_r1_t>(arguments),
      data<typename P::dev_pix2track_offsets_r2_t>(arguments),
      data<typename P::dev_pix2track_r2_t>(arguments),
      data<typename P::dev_pix2photon_r2_t>(arguments));

    std::swap(pids_in, pids_out);
  }

  // Convert the fixed-point DLL state (kept as int across every kernel above
  // to make atomicAdd accumulation associative) back to float, and normalise
  // to the pion convention expected by the converter - now that all outer
  // likelihood iterations are done.
  global_function(rich_dll_finalise_k)(dim3(32), dim3(m_block_dim), context)(
    number_of_tracks, data<typename P::dev_dll_out_t>(arguments));
}

template struct rich_global_pid::rich_global_pid_t<rich_global_pid::BackgroundEstimationMethod::FromReco>;
template struct rich_global_pid::rich_global_pid_t<rich_global_pid::BackgroundEstimationMethod::FromCones>;
