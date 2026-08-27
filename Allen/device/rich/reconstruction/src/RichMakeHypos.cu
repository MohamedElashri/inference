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

#include "RichMakeHypos.cuh"
#include <ParabolicExtrapolator.cuh>
#include <RungeKuttaExtrapolator.cuh>
#include <BinarySearch.cuh>
#include <PrefixSum.cuh>

INSTANTIATE_ALGORITHM(rich_make_hypos::rich_make_hypos_t);

template<Allen::Rich::Detector::DetectorType richIdx>
void rich_make_hypos::rich_make_hypos_t::updateRich(const Allen::Rich::RichDetector<richIdx>* rich) const
{
  const double min_photon_E = static_cast<double>(Allen::Rich::MinPhotonEnergy);
  const double max_photon_E = static_cast<double>(Allen::Rich::MaxPhotonEnergy);

  const double RC = static_cast<double>(rich->m_selLorGasFac * rich->m_rho / rich->m_molW);
  const double RF = static_cast<double>(rich->m_selF1 + rich->m_selF2);
  const double RE02 =
    static_cast<double>(rich->m_selF1 * rich->m_selE2 * rich->m_selE2 + rich->m_selF2 * rich->m_selE1 * rich->m_selE1) /
    RF;
  const double RE = static_cast<double>(rich->m_selE2 * rich->m_selE2 + rich->m_selE1 * rich->m_selE1) / RF;
  const double RG = static_cast<double>(rich->m_selE1 * rich->m_selE1 * rich->m_selE2 * rich->m_selE2) / (RF * RE02);
  const double RH = RE02 / RF;
  const double RM = RE + (2.0 * RC);
  const double RS = RG + (2.0 * RC);
  const double RT = std::sqrt(0.25 * RM * RM - RH * RS);
  const double RXSP = std::sqrt((RM / 2.0 + RT) / RH);
  const double RXSM = std::sqrt((RM / 2.0 - RT) / RH);
  const double REP = std::sqrt(RE02) * RXSP;
  const double REM = std::sqrt(RE02) * RXSM;
  const double RXSPscale = (RXSP - 1.0 / RXSP);
  const double RXSMscale = (RXSM - 1.0 / RXSM);
  const double X = (3.0 * RC * std::sqrt(RE02) / (4.0 * RT));

  auto paraW = [&](const double energy) {
    const double A = (RXSPscale * std::log((REP + energy) / (REP - energy)));
    const double B = (RXSMscale * std::log((REM + energy) / (REM - energy)));
    return X * (A - B);
  };

  const unsigned nbins = rich->m_spectraEffs.size();
  const double binw = (max_photon_E - min_photon_E) / nbins;

  // loop over the energy bins
  double avg_refIndexTheta = 0.0;
  double avg_refIndexYield = 0.0;
  double total_efficiency = 0.0;
  for (unsigned iEnBin = 0; iEnBin < nbins; iEnBin++) {
    const double botEn = min_photon_E + iEnBin * binw;
    const double topEn = botEn + binw;
    const double paraWDiff = paraW(topEn) - paraW(botEn);
    m_paraWDiff[iEnBin] = static_cast<float>(paraWDiff);
    const double bin_efficiency = static_cast<double>(rich->m_spectraEffs[iEnBin]);

    // Average ref index for this bin:
    const double refIndex_bin = std::sqrt(1.0 / (1.0 - paraWDiff / binw));

    // Ref index at mid point:
    const double refIndex_mid = static_cast<double>(rich->m_refIndexE[iEnBin]);

    // Weight average by the bin efficiency:
    avg_refIndexTheta += refIndex_mid * bin_efficiency;
    avg_refIndexYield += refIndex_bin * bin_efficiency;
    total_efficiency += bin_efficiency;
  }
  avg_refIndexTheta /= total_efficiency;
  avg_refIndexYield /= total_efficiency;

  m_refIndexTheta = avg_refIndexTheta;
  m_refIndexYield = avg_refIndexYield;
  m_deltaE = (max_photon_E - min_photon_E) * (total_efficiency / nbins);
}

void rich_make_hypos::rich_make_hypos_t::update(const Constants& constants) const
{
  const auto rich = Allen::Rich::Detector::detectorTypeFromNumber(m_current_rich.value());
  if (rich == Allen::Rich::Detector::Rich1) {
    updateRich<Allen::Rich::Detector::Rich1>(constants.host_rich_1_geometry);
  }
  else {
    updateRich<Allen::Rich::Detector::Rich2>(constants.host_rich_2_geometry);
  }
}

inline __device__ void moveState(Extrapolators::State& state, float z, const MagneticField::Magfield& field)
{
  const float dz = z - state.z;
  if (fabsf(dz) < 30.f) { // TODO: take as parameter?
    state.x += state.tx * dz;
    state.y += state.ty * dz;
    state.z = z;
  }
  else {
    Extrapolators::ParabolicExtrapolator<float>::propagate(state, dz, field);
  }
}

inline __device__ auto invalidHypos()
{
  Allen::Rich::ParticleHypos hypos {};
  UNROLL(Allen::Rich::NRealParticleTypes)
  for (unsigned hypo_index = 0; hypo_index < Allen::Rich::NRealParticleTypes; ++hypo_index) {
    const auto hypo = static_cast<Allen::Rich::ParticleIDType>(hypo_index);
    hypos.ckTheta[hypo] = NAN;
    hypos.ckRes[hypo] = 0.f;
    hypos.yield[hypo] = 0.f;
  }
  return hypos;
}

template<Allen::Rich::Detector::DetectorType richIdx, bool useYieldWeightedAngles>
__global__ void rich_hypos_k(
  const Allen::Rich::RichDetector<richIdx>* rich,
  const MagneticField::Magfield magfield,
  const SimpleKalmanState* entry_states,
  const SimpleKalmanState* exit_states,
  const unsigned total_number_of_tracks,
  float3* segs_best_point,
  float3* segs_best_momentum,
  float2* segs_point_at_panel,
  Allen::Rich::ParticleHypos* hypos_tracks,
  const float mirror_shift,
  const float rad_scale,
  const float minRadLength,
  const float refIndexTheta,
  const float refIndexYield,
  const float deltaE,
  const std::array<float, Allen::Rich::NPhotonSpectraBins> paraWDiff,
  Allen::Monitoring::Counter<>::DeviceType failed_ray_traces)
{
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;
  for (unsigned i = threadId; i < total_number_of_tracks; i += stride) {
    // Extrapolate state to the correct side
    Extrapolators::State entry_state = entry_states[i];
    Extrapolators::State exit_state = exit_states[i];

    // Find real exit_state
    const auto detectorSide = Allen::Rich::side<richIdx>(exit_state.pos());
    const auto initialZ = rich->m_radZExit;

    moveState(exit_state, initialZ - mirror_shift, magfield);

    auto gDir = exit_state.dir();
    float3 exit_point = exit_state.pos();
    if (!Allen::Rich::reflectSpherical(
          exit_point, gDir, rich->nominalCentreOfCurvature(detectorSide), rich->sphMirrorRadius())) {
      segs_best_point[i] = {NAN, NAN, NAN};
      segs_best_momentum[i] = {NAN, NAN, NAN};
      segs_point_at_panel[i] = {NAN, NAN};
      hypos_tracks[i] = invalidHypos();
      failed_ray_traces.increment();
      continue;
    }

    // Find real entry state
    float3 entry_point = entry_state.pos();
    entry_point.z = rich->m_radZEntry;

    if constexpr (richIdx == Allen::Rich::Detector::Rich1) {
      rich->beampipeIntersect(entry_point, exit_point);
    }

    moveState(entry_state, entry_point.z, magfield);
    moveState(exit_state, exit_point.z, magfield);
    Extrapolators::State middle_state = entry_state;
    moveState(middle_state, (entry_point.z + exit_point.z) * 0.5f, magfield);

    // Precompute path length
    float pathLength = mag(exit_state.pos() - middle_state.pos()) + mag(middle_state.pos() - entry_state.pos());
    float momentum = middle_state.p();

    segs_best_point[i] = middle_state.pos();
    segs_best_momentum[i] = float3 {middle_state.px(), middle_state.py(), middle_state.pz()};
    segs_point_at_panel[i] = Allen::Rich::radLocalCorrection(
      Allen::Rich::pointAtPanel(rich, middle_state.pos(), middle_state.dir(), detectorSide), rad_scale);

    Allen::Rich::ParticleHypos hypos {};
    if (!std::isfinite(pathLength) || !std::isfinite(momentum) || pathLength < minRadLength) {
      hypos = invalidHypos();
    }
    else if (useYieldWeightedAngles) {
      // Compute emitted and detectable photon yields and spectra, and CK angles
      // then fill ParticleHypos from the computed Cherenkov angles
      hypos =
        Allen::Rich::ParticleHypos(momentum, pathLength, paraWDiff, rich->m_spectraEffs, rich->m_refIndexE, richIdx);
    }
    else { // CK angle computation with average refractive index
      // Create ck angles, res and yields for each hypo
      hypos = Allen::Rich::ParticleHypos(momentum, pathLength, refIndexTheta, refIndexYield, deltaE, richIdx);
    }

    hypos_tracks[i] = hypos;
  }
}

void rich_make_hypos::rich_make_hypos_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  const unsigned number_of_tracks = first<host_number_of_tracks_t>(arguments);

  set_size<dev_segs_best_point_t>(arguments, number_of_tracks);
  set_size<dev_segs_best_momentum_t>(arguments, number_of_tracks);
  set_size<dev_segs_point_at_panel_t>(arguments, number_of_tracks);
  set_size<dev_rich_hypos_t>(arguments, number_of_tracks);
}

template<Allen::Rich::Detector::DetectorType richIdx>
void rich_make_hypos::rich_make_hypos_t::launchForRich(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants& constants,
  const Allen::Context& context,
  const Allen::Rich::RichDetector<richIdx>* rich) const
{
  const unsigned number_of_tracks = first<host_number_of_tracks_t>(arguments);

  const float mirror_shift = m_mirrShift.value()[richIdx];
  const float minRadLength = m_minRadLength.value()[richIdx];
  const float radScale = m_radScale.value()[richIdx];
  const bool useYieldWeightedAngles = m_useYieldWeightedAngles.value();

  // Compute hypos
  const auto& hypos_kernel = useYieldWeightedAngles ? rich_hypos_k<richIdx, true> : rich_hypos_k<richIdx, false>;
  global_function(hypos_kernel)(dim3(32), m_block_dim, context)(
    rich,
    *constants.magnetic_field,
    data<dev_rich_entry_states_t>(arguments),
    data<dev_rich_exit_states_t>(arguments),
    number_of_tracks,
    data<dev_segs_best_point_t>(arguments),
    data<dev_segs_best_momentum_t>(arguments),
    data<dev_segs_point_at_panel_t>(arguments),
    data<dev_rich_hypos_t>(arguments),
    mirror_shift,
    radScale,
    minRadLength,
    m_refIndexTheta,
    m_refIndexYield,
    m_deltaE,
    m_paraWDiff,
    m_failed_ray_traces.data(context));
}

void rich_make_hypos::rich_make_hypos_t::operator()(
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
