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

INSTANTIATE_ALGORITHM(rich_global_pid::rich_global_pid_t);

constexpr float pix_signals_scale = 1e5f;
constexpr float inv_pix_signals_scale = 1.f / pix_signals_scale;

inline __device__ int signalToFixedPoint(const float signal) { return static_cast<int>(signal * pix_signals_scale); }

inline __device__ float signalFromFixedPoint(const int signal)
{
  return static_cast<float>(signal) * inv_pix_signals_scale;
}

template<Allen::Rich::Detector::DetectorType richIdx>
void rich_global_pid::rich_global_pid_t::updateRich(const Allen::Rich::RichDetector<richIdx>* rich) const
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

void rich_global_pid::rich_global_pid_t::update(const Constants& constants) const
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

template<Allen::Rich::Detector::DetectorType richIdx>
void rich_global_pid::rich_global_pid_t::pixelSignalsForRich(
  const ArgumentReferences<Parameters>& arguments,
  const Allen::Context& context,
  const Allen::Rich::ParticleIDType* pids_in) const
{
  const unsigned number_of_tracks = first<host_number_of_tracks_t>(arguments);
  const unsigned number_of_photons = richIdx == Allen::Rich::Detector::Rich1 ?
                                       first<host_number_of_photons_r1_t>(arguments) :
                                       first<host_number_of_photons_r2_t>(arguments);

  if constexpr (richIdx == Allen::Rich::Detector::Rich1) {
    Allen::memset_async<dev_pixel_signals_r1_t>(arguments, 0, context);
  }
  else {
    Allen::memset_async<dev_pixel_signals_r2_t>(arguments, 0, context);
  }

  const auto& acc_kernel = rich_acc_pixel_signal_k<richIdx>;
  global_function(acc_kernel)(dim3(32), dim3(m_block_dim), context)(
    number_of_tracks,
    number_of_photons,
    pids_in,
    richIdx == Allen::Rich::Detector::Rich1 ? data<dev_offsets_rich_photons_r1_t>(arguments) :
                                              data<dev_offsets_rich_photons_r2_t>(arguments),
    richIdx == Allen::Rich::Detector::Rich1 ? data<dev_rich_photons_r1_t>(arguments) :
                                              data<dev_rich_photons_r2_t>(arguments),
    richIdx == Allen::Rich::Detector::Rich1 ? data<dev_photon_pix_signals_r1_t>(arguments) :
                                              data<dev_photon_pix_signals_r2_t>(arguments),
    richIdx == Allen::Rich::Detector::Rich1 ? data<dev_pixel_signals_r1_t>(arguments) :
                                              data<dev_pixel_signals_r2_t>(arguments));
}

// Compute the average background in an EC (group of 1-4 PDs), using the pixel signals
template<Allen::Rich::Detector::DetectorType richIdx, bool ignoreExpSignal>
__global__ void rich_avg_bkg_from_reco_k(
  const uint16_t* effNumPixsEC,
  const unsigned number_of_events,
  const unsigned* pixels_offsets,
  int* pixel_signals,
  float* ec_bkg)
{
  const unsigned ec_per_panel = Allen::Rich::Detector::PDPanel<richIdx>::ECsPerPanel;

  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;
  for (unsigned i = threadId; i < Allen::Rich::NPDPanelsPerRICH * number_of_events * ec_per_panel; i += stride) {
    const auto side = static_cast<Allen::Rich::Detector::Side>(i / (number_of_events * ec_per_panel));
    const unsigned ec_index = i % ec_per_panel;

    const unsigned effNumPixs = effNumPixsEC[ec_index + side * ec_per_panel];
    const unsigned pixStart = pixels_offsets[i * Allen::Rich::Decoding::SmartID::MaxPDsPerEC];
    const unsigned pixEnd = pixels_offsets[(i + 1) * Allen::Rich::Decoding::SmartID::MaxPDsPerEC];
    const unsigned obsSignal = pixEnd - pixStart;

    float expBackgrd = 0.f;
    if (effNumPixs > 0) {

      float expSignal = 0.f;
      if constexpr (!ignoreExpSignal) {
        for (unsigned j = pixStart; j < pixEnd; j++) {
          expSignal += signalFromFixedPoint(pixel_signals[j]);
        }

        if (obsSignal > 0) expSignal *= effNumPixs / obsSignal;
      }

      expBackgrd = max((obsSignal - expSignal) / effNumPixs, 0.f);

      /*printf(
        "[Allen] Rich %d id %d side %d obsSignal: %d expSignal: %f bkg: %f effNumPixs: %d (from reco)\n",
        richIdx + 1,
        ec_index,
        side,
        obsSignal,
        expSignal,
        expBackgrd,
        effNumPixs);*/

      // Broadcast back the expected background to the pixel_signals
      // because they are used together in the log(exp(signal + bkg) + 1) formula, and that way
      // we save some memory.
      // this will have to be subtracted at the end of the iteration (if its not the last one)
      for (unsigned j = pixStart; j < pixEnd; j++) {
        pixel_signals[j] += signalToFixedPoint(expBackgrd);
      }
    }

    ec_bkg[i] = expBackgrd; // save to subtract later
  }
}

template<Allen::Rich::Detector::DetectorType richIdx>
void rich_global_pid::rich_global_pid_t::backgroundsForRichFromReco(
  const ArgumentReferences<Parameters>& arguments,
  const Allen::Context& context,
  const unsigned it) const
{
  const unsigned number_of_events = first<host_number_of_events_t>(arguments);

  const auto& avg_kernel =
    m_ignoreExpSignal.value()[it] ? rich_avg_bkg_from_reco_k<richIdx, true> : rich_avg_bkg_from_reco_k<richIdx, false>;
  global_function(avg_kernel)(dim3(32), dim3(m_block_dim), context)(
    m_cached_effNumPixsEC[richIdx],
    number_of_events,
    richIdx == Allen::Rich::Detector::Rich1 ? data<dev_rich_pd_offsets_r1_t>(arguments) :
                                              data<dev_rich_pd_offsets_r2_t>(arguments),
    richIdx == Allen::Rich::Detector::Rich1 ? data<dev_pixel_signals_r1_t>(arguments) :
                                              data<dev_pixel_signals_r2_t>(arguments),
    richIdx == Allen::Rich::Detector::Rich1 ? data<dev_pix_bkg_r1_t>(arguments) : data<dev_pix_bkg_r2_t>(arguments));
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
      const auto deltaInt = static_cast<int>(deltaLL * 1e3f);

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

      const float cur_dll = static_cast<float>(dlls_int[track_id][new_pid]) * 1e-3f + (new_sig - cur_sig);
      if (cur_dll < bestDLL) {
        bestDLL = cur_dll;
        bestPID = new_pid;
      }
      dlls[track_id][new_pid] = cur_dll;
    }

    // std::cout << "[Allen] Track " << track_id << " pid " << cur_pid << " -> " << bestPID << " DLL " << dlls[track_id]
    // << std::endl;

    pids_out[track_id] = bestPID;
  }
}

__global__ void rich_normalise_dlls_k(const unsigned number_of_tracks, Allen::Rich::HypoData<float>* dlls)
{
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;

  for (unsigned t = threadId; t < number_of_tracks; t += stride) {
    // Get dll relative to pion
    const float dll_pion = dlls[t][Allen::Rich::ParticleIDType::Pion];

    for (unsigned particle_index = 0; particle_index < Allen::Rich::NParticleTypes; ++particle_index) {
      const auto particle = static_cast<Allen::Rich::ParticleIDType>(particle_index);
      // Internally, the Global PID normalises the DLL values to the best hypothesis
      // and also works in "-loglikelihood" space.
      // For final storage, renormalise the DLLS w.r.t. the pion hypothesis and
      // invert the values
      dlls[t][particle] = dll_pion - dlls[t][particle];
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
  float* dev_s_old_r1,
  float* dev_s_old_r2,
  const unsigned* dev_pix2track_offsets_r1,
  const unsigned* dev_pix2track_r1,
  const unsigned* dev_pix2track_offsets_r2,
  const unsigned* dev_pix2track_r2,
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

  unsigned iteration = 0;
  while (true) {

    // Phase A: track scan to read existing DLL values
    // find best log likelihood
    // thread "local" variables
    float local_best_dll = 0.f; // 0 = no improvement found yet
    int local_best_track = -1;  // event-local index
    auto local_best_pid = Allen::Rich::ParticleIDType::Unknown;

    // t is event-local, so each warp must scan with its lane-local index.
    for (unsigned t = lane_id; t < n_tracks; t += warp_size) {
      const unsigned gt = track_start + t; // global track index
      const auto cur_pid = pids[gt];
      if (cur_pid == Allen::Rich::ParticleIDType::Unknown) continue;

      float track_best_dll = 0.f;
      auto track_best_pid = Allen::Rich::ParticleIDType::Unknown;

      for (unsigned pid_index = 0; pid_index < Allen::Rich::NParticleTypes; ++pid_index) {
        const auto new_pid = static_cast<Allen::Rich::ParticleIDType>(pid_index);
        if (new_pid == cur_pid) continue;
        const float dll = dlls[gt][new_pid];

        // update dll
        if (dll < track_best_dll) {
          track_best_dll = dll;
          track_best_pid = new_pid;
        }
      } // end hypo loop

      // update thread values
      if (track_best_dll < local_best_dll) {
        local_best_dll = track_best_dll;
        local_best_track = static_cast<int>(t);
        local_best_pid = track_best_pid;
      }
    } // end track loop

    // Phase B: Warp butterly reduction
    float global_best_dll = local_best_dll;
    int global_best_trk = local_best_track;

    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
      const float neighbor_dll = __shfl_xor_sync(0xffffffff, global_best_dll, offset);
      const int neighbor_trk = __shfl_xor_sync(0xffffffff, global_best_trk, offset);
      if (neighbor_dll < global_best_dll || (neighbor_dll == global_best_dll && neighbor_trk < global_best_trk)) {
        global_best_dll = neighbor_dll;
        global_best_trk = neighbor_trk;
      }
    }
    // All lanes now have the same global_best_dll and global_best_trk.
    // Find the winning lane and broadcast its local_best_pid.
    const unsigned winner_lane = __ffs(__ballot_sync(0xffffffff, local_best_track == global_best_trk)) - 1;
    const auto global_best_pid =
      static_cast<Allen::Rich::ParticleIDType>(__shfl_sync(0xffffffff, static_cast<int>(local_best_pid), winner_lane));

    // Phase C: convergence check
    if (global_best_dll >= epsilon || global_best_trk == -1) break;
    if (++iteration >= max_iterations) break;

    // Phase D: update values for next iter (hypothesis change)
    const unsigned gt = track_start + static_cast<unsigned>(global_best_trk);
    const auto old_pid = pids[gt];
    const auto new_pid = global_best_pid;

    const unsigned ph_start_r1 = dev_offsets_rich_photons_r1[gt];
    const unsigned ph_end_r1 = dev_offsets_rich_photons_r1[gt + 1];
    const unsigned ph_start_r2 = dev_offsets_rich_photons_r2[gt];
    const unsigned ph_end_r2 = dev_offsets_rich_photons_r2[gt + 1];

    // Step 1: update pixel signals
    for (unsigned p = ph_start_r1 + lane_id; p < ph_end_r1; p += warp_size) {
      const unsigned pix = dev_rich_photons_r1[p].pixelIdx;
      dev_s_old_r1[p] = signalFromFixedPoint(dev_pixel_signals_r1[pix]);
      const float delta = dev_photon_pix_signals_r1[p][new_pid] - dev_photon_pix_signals_r1[p][old_pid];
      atomicAdd(&dev_pixel_signals_r1[dev_rich_photons_r1[p].pixelIdx], signalToFixedPoint(delta));
    }
    for (unsigned p = ph_start_r2 + lane_id; p < ph_end_r2; p += warp_size) {
      const unsigned pix = dev_rich_photons_r2[p].pixelIdx;
      dev_s_old_r2[p] = signalFromFixedPoint(dev_pixel_signals_r2[pix]);
      const float delta = dev_photon_pix_signals_r2[p][new_pid] - dev_photon_pix_signals_r2[p][old_pid];
      atomicAdd(&dev_pixel_signals_r2[dev_rich_photons_r2[p].pixelIdx], signalToFixedPoint(delta));
    }

    if (lane_id == 0) {
      pids[gt] = new_pid;
      dlls[gt][new_pid] = 0.f;
    }

    __syncwarp();

    // Step 2: recompute DLLs for t*
    // t* changed hypothesis so its old DLL values are wrong
    for (unsigned particle_index = 0; particle_index < Allen::Rich::NParticleTypes; ++particle_index) {
      const auto particle = static_cast<Allen::Rich::ParticleIDType>(particle_index);
      if (particle == new_pid) {
        if (lane_id == 0) dlls[gt][particle] = 0.f;
        continue;
      }

      // Each lane accumulates its subset of photons
      float dll = 0.f;

      for (unsigned p = ph_start_r1 + lane_id; p < ph_end_r1; p += warp_size) {
        const float S = signalFromFixedPoint(dev_pixel_signals_r1[dev_rich_photons_r1[p].pixelIdx]);
        const float sig_cur = dev_photon_pix_signals_r1[p][new_pid];
        const float sig_h = dev_photon_pix_signals_r1[p][particle];
        dll += sigFunc(S) - sigFunc(S - sig_cur + sig_h);
      }

      for (unsigned p = ph_start_r2 + lane_id; p < ph_end_r2; p += warp_size) {
        const float S = signalFromFixedPoint(dev_pixel_signals_r2[dev_rich_photons_r2[p].pixelIdx]);
        const float sig_cur = dev_photon_pix_signals_r2[p][new_pid];
        const float sig_h = dev_photon_pix_signals_r2[p][particle];
        dll += sigFunc(S) - sigFunc(S - sig_cur + sig_h);
      }

      // Warp reduce: sum dll across all 32 lanes
      for (int offset = warp_size / 2; offset > 0; offset /= 2)
        dll += __shfl_down_sync(0xffffffff, dll, offset);

      // Add track term
      if (lane_id == 0) {
        dll += (dev_track_total_signals_r1[gt][particle] + dev_track_total_signals_r2[gt][particle]) -
               (dev_track_total_signals_r1[gt][new_pid] + dev_track_total_signals_r2[gt][new_pid]);
        dlls[gt][particle] = dll;
      }
    } // end of step 2
    __syncwarp();

    // Step 3: propagate DLL corrections to tracks sharing pixels with t*
    // RICH 1
    for (unsigned p = ph_start_r1 + lane_id; p < ph_end_r1; p += warp_size) {
      const unsigned pix = dev_rich_photons_r1[p].pixelIdx;
      const float S_old = dev_s_old_r1[p];
      const float S_new = signalFromFixedPoint(dev_pixel_signals_r1[pix]);
      if (S_old == S_new) continue; // no change on this pixel, skip

      for (unsigned k = dev_pix2track_offsets_r1[pix]; k < dev_pix2track_offsets_r1[pix + 1]; k++) {
        const unsigned tp = dev_pix2track_r1[k];
        if (tp == gt) continue;

        const auto cur_pid_tp = pids[tp];
        if (cur_pid_tp == Allen::Rich::ParticleIDType::Unknown) continue;

        const unsigned ph_start_tp = dev_offsets_rich_photons_r1[tp];
        const unsigned ph_end_tp = dev_offsets_rich_photons_r1[tp + 1];
        for (unsigned p2 = ph_start_tp; p2 < ph_end_tp; p2++) {
          if (dev_rich_photons_r1[p2].pixelIdx != pix) continue;
          for (unsigned particle_index = 0; particle_index < Allen::Rich::NParticleTypes; ++particle_index) {
            const auto particle = static_cast<Allen::Rich::ParticleIDType>(particle_index);
            if (particle == cur_pid_tp) continue;
            const float sig_cur = dev_photon_pix_signals_r1[p2][cur_pid_tp];
            const float sig_h = dev_photon_pix_signals_r1[p2][particle];
            const float d_dll =
              (sigFunc(S_new) - sigFunc(S_new - sig_cur + sig_h)) - (sigFunc(S_old) - sigFunc(S_old - sig_cur + sig_h));
            atomicAdd(&dlls[tp][particle], d_dll);
          }
        }
      }
    }

    // RICH 2
    for (unsigned p = ph_start_r2 + lane_id; p < ph_end_r2; p += warp_size) {
      const unsigned pix = dev_rich_photons_r2[p].pixelIdx;
      const float S_old = dev_s_old_r2[p];
      const float S_new = signalFromFixedPoint(dev_pixel_signals_r2[pix]);
      if (S_old == S_new) continue;

      for (unsigned k = dev_pix2track_offsets_r2[pix]; k < dev_pix2track_offsets_r2[pix + 1]; k++) {
        const unsigned tp = dev_pix2track_r2[k];
        if (tp == gt) continue;

        const auto cur_pid_tp = pids[tp];
        if (cur_pid_tp == Allen::Rich::ParticleIDType::Unknown) continue;

        const unsigned ph_start_tp = dev_offsets_rich_photons_r2[tp];
        const unsigned ph_end_tp = dev_offsets_rich_photons_r2[tp + 1];
        for (unsigned p2 = ph_start_tp; p2 < ph_end_tp; p2++) {
          if (dev_rich_photons_r2[p2].pixelIdx != pix) continue;
          for (unsigned particle_index = 0; particle_index < Allen::Rich::NParticleTypes; ++particle_index) {
            const auto particle = static_cast<Allen::Rich::ParticleIDType>(particle_index);
            if (particle == cur_pid_tp) continue;
            const float sig_cur = dev_photon_pix_signals_r2[p2][cur_pid_tp];
            const float sig_h = dev_photon_pix_signals_r2[p2][particle];
            const float d_dll =
              (sigFunc(S_new) - sigFunc(S_new - sig_cur + sig_h)) - (sigFunc(S_old) - sigFunc(S_old - sig_cur + sig_h));
            atomicAdd(&dlls[tp][particle], d_dll);
          }
        }
      }
    }             // End of step 3
    __syncwarp(); // ensure all DLL updates visible before next iteration's Phase A
  }               // iterations while loop
}

void rich_global_pid::rich_global_pid_t::initDLLs(
  const ArgumentReferences<Parameters>& arguments,
  const Allen::Context& context,
  const Allen::Rich::ParticleIDType* pids_in,
  Allen::Rich::ParticleIDType* pids_out) const
{
  const unsigned number_of_tracks = first<host_number_of_tracks_t>(arguments);
  const unsigned number_of_photons_r1 = first<host_number_of_photons_r1_t>(arguments);
  const unsigned number_of_photons_r2 = first<host_number_of_photons_r2_t>(arguments);

  Allen::memset_async<dev_dll_out_t>(arguments, 0, context);

  global_function(rich_acc_pixel_dll_k)(dim3(32), dim3(m_block_dim), context)(
    number_of_tracks,
    number_of_photons_r1,
    pids_in,
    data<dev_offsets_rich_photons_r1_t>(arguments),
    data<dev_rich_photons_r1_t>(arguments),
    data<dev_photon_pix_signals_r1_t>(arguments),
    data<dev_pixel_signals_r1_t>(arguments),
    data<dev_dll_out_t>(arguments));

  global_function(rich_acc_pixel_dll_k)(dim3(32), dim3(m_block_dim), context)(
    number_of_tracks,
    number_of_photons_r2,
    pids_in,
    data<dev_offsets_rich_photons_r2_t>(arguments),
    data<dev_rich_photons_r2_t>(arguments),
    data<dev_photon_pix_signals_r2_t>(arguments),
    data<dev_pixel_signals_r2_t>(arguments),
    data<dev_dll_out_t>(arguments));

  global_function(rich_init_dll_best_hypo_k)(dim3(32), dim3(m_block_dim), context)(
    number_of_tracks,
    pids_in,
    data<dev_track_total_signals_r1_t>(arguments),
    data<dev_track_total_signals_r2_t>(arguments),
    data<dev_dll_out_t>(arguments),
    pids_out);

  // recompute the DLLs based on the best hypo that was selected
  global_function(rich_global_pid_init_update_signals_k)(dim3(32), dim3(m_block_dim), context)(
    number_of_tracks,
    number_of_photons_r1,
    pids_in,
    pids_out,
    data<dev_offsets_rich_photons_r1_t>(arguments),
    data<dev_rich_photons_r1_t>(arguments),
    data<dev_photon_pix_signals_r1_t>(arguments),
    data<dev_pixel_signals_r1_t>(arguments));

  global_function(rich_global_pid_init_update_signals_k)(dim3(32), dim3(m_block_dim), context)(
    number_of_tracks,
    number_of_photons_r2,
    pids_in,
    pids_out,
    data<dev_offsets_rich_photons_r2_t>(arguments),
    data<dev_rich_photons_r2_t>(arguments),
    data<dev_photon_pix_signals_r2_t>(arguments),
    data<dev_pixel_signals_r2_t>(arguments));
}

void rich_global_pid::rich_global_pid_t::doIterations(
  const ArgumentReferences<Parameters>& arguments,
  const Allen::Context& context,
  Allen::Rich::ParticleIDType* pids,
  Allen::Rich::HypoData<float>* dlls,
  float* s_old_r1,
  float* s_old_r2,
  const unsigned* pix2track_offsets_r1,
  const unsigned* pix2track_r1,
  const unsigned* pix2track_offsets_r2,
  const unsigned* pix2track_r2) const
{

  const unsigned number_of_events = first<host_number_of_events_t>(arguments);

  constexpr unsigned warps_per_block = 8;
  const unsigned n_blocks = (number_of_events + warps_per_block - 1) / warps_per_block;

  global_function(rich_global_pid_iterations_k)(dim3(n_blocks), dim3(warps_per_block * warp_size), context)(
    number_of_events,
    data<dev_offsets_tracks_t>(arguments),
    data<dev_offsets_rich_photons_r1_t>(arguments),
    data<dev_rich_photons_r1_t>(arguments),
    data<dev_photon_pix_signals_r1_t>(arguments),
    data<dev_track_total_signals_r1_t>(arguments),
    data<dev_pixel_signals_r1_t>(arguments),
    data<dev_offsets_rich_photons_r2_t>(arguments),
    data<dev_rich_photons_r2_t>(arguments),
    data<dev_photon_pix_signals_r2_t>(arguments),
    data<dev_track_total_signals_r2_t>(arguments),
    data<dev_pixel_signals_r2_t>(arguments),
    pids,
    dlls,
    s_old_r1,
    s_old_r2,
    pix2track_offsets_r1,
    pix2track_r1,
    pix2track_offsets_r2,
    pix2track_r2,
    m_epsilon.value(),
    m_maxEventIterations.value());
}

void rich_global_pid::rich_global_pid_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  const unsigned number_of_events = first<host_number_of_events_t>(arguments);
  const unsigned number_of_tracks = first<host_number_of_tracks_t>(arguments);
  const unsigned number_of_pixels_r1 = first<host_number_of_pixels_r1_t>(arguments);
  const unsigned number_of_pixels_r2 = first<host_number_of_pixels_r2_t>(arguments);

  set_size<dev_pixel_signals_r1_t>(arguments, number_of_pixels_r1);
  set_size<dev_pixel_signals_r2_t>(arguments, number_of_pixels_r2);

  set_size<dev_pix_bkg_r1_t>(
    arguments,
    Allen::Rich::NPDPanelsPerRICH * number_of_events *
      Allen::Rich::Detector::PDPanel<Allen::Rich::Detector::Rich1>::ECsPerPanel);
  set_size<dev_pix_bkg_r2_t>(
    arguments,
    Allen::Rich::NPDPanelsPerRICH * number_of_events *
      Allen::Rich::Detector::PDPanel<Allen::Rich::Detector::Rich2>::ECsPerPanel);

  set_size<dev_pid_out_t>(arguments, number_of_tracks);
  set_size<dev_dll_out_t>(arguments, number_of_tracks);
}

void rich_global_pid::rich_global_pid_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants& constants,
  const Allen::Context& context) const
{
  [[maybe_unused]] const Allen::Rich::RichDetector<Allen::Rich::Detector::Rich1>* rich1 = constants.dev_rich_1_geometry;
  [[maybe_unused]] const Allen::Rich::RichDetector<Allen::Rich::Detector::Rich2>* rich2 = constants.dev_rich_2_geometry;

  // const unsigned number_of_events = first<host_number_of_events_t>(arguments);
  const unsigned number_of_tracks = first<host_number_of_tracks_t>(arguments);
  auto pids_tmp_buffer =
    arguments.template make_buffer<Allen::Store::Scope::Device, Allen::Rich::ParticleIDType>(number_of_tracks);

  // pix2track map
  const unsigned number_of_photons_r1 = first<host_number_of_photons_r1_t>(arguments);
  const unsigned number_of_photons_r2 = first<host_number_of_photons_r2_t>(arguments);
  auto s_old_r1 = arguments.template make_buffer<Allen::Store::Scope::Device, float>(number_of_photons_r1);
  auto s_old_r2 = arguments.template make_buffer<Allen::Store::Scope::Device, float>(number_of_photons_r2);

  Allen::Rich::ParticleIDType* pids_in = pids_tmp_buffer.data();
  Allen::Rich::ParticleIDType* pids_out = data<dev_pid_out_t>(arguments);

  if (m_nLikelihoodIterations.value() % 2 == 0) {
    // Make sure the last iteration writes to dev_pid_out_t
    std::swap(pids_in, pids_out);
  }

  Allen::memcpy_async(
    pids_in,
    data<dev_pid_in_t>(arguments),
    number_of_tracks * sizeof(Allen::Rich::ParticleIDType),
    Allen::memcpyDeviceToDevice,
    context);

  // Init pixel signals:
  pixelSignalsForRich<Allen::Rich::Detector::Rich1>(arguments, context, pids_in);
  pixelSignalsForRich<Allen::Rich::Detector::Rich2>(arguments, context, pids_in);

  for (unsigned it = 0; it < m_nLikelihoodIterations.value(); it++) {
    // Compute backgrounds:
    backgroundsForRichFromReco<Allen::Rich::Detector::Rich1>(arguments, context, it);
    backgroundsForRichFromReco<Allen::Rich::Detector::Rich2>(arguments, context, it);

    // Init DLLs and set to best hypothesis:
    initDLLs(arguments, context, pids_in, pids_out);

    // Do iterations until convergence:
    doIterations(
      arguments,
      context,
      pids_out,
      data<dev_dll_out_t>(arguments),
      s_old_r1.data(),
      s_old_r2.data(),
      data<dev_pix2track_offsets_r1_t>(arguments),
      data<dev_pix2track_r1_t>(arguments),
      data<dev_pix2track_offsets_r2_t>(arguments),
      data<dev_pix2track_r2_t>(arguments));

    std::swap(pids_in, pids_out);
  }

  // Normalise DLLs to pion convention expected by the converter
  global_function(rich_normalise_dlls_k)(dim3(32), dim3(m_block_dim), context)(
    number_of_tracks, data<dev_dll_out_t>(arguments));
}
