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
#pragma once

#include <cstdint>
#include <RichDefinitions.cuh>
#include <RichResolutionParams.cuh>
#include <TabulatedFunction1D.cuh>
#include <RichTrackSegment.cuh>

namespace Allen::Rich {
  template<typename T>
  struct HypoData {
    __device__ const T& operator[](const ParticleIDType particle) const { return m_data[particle]; }
    __device__ T& operator[](const ParticleIDType particle) { return m_data[particle]; }

    __host__ friend inline std::ostream& operator<<(std::ostream& os, const HypoData<T>& data)
    {
      os << "[ ";
      for (const auto particle : particles()) {
        os << data.m_data[particle] << " ";
      }
      return os << "]";
    }

    ParticleArray<T> m_data {};
  };

  struct ParticleHypos {
    __device__ ParticleHypos() = default;
    /**
     * @brief Simple constructor from momentum and average refractive index.
     * Assume constant refractive index through the radiator and energy spectra.
     */
    __device__ ParticleHypos(
      float momentum,
      float pathLength,
      float nTheta,
      float nYield,
      float deltaE,
      const Detector::DetectorType rich)
    {
      UNROLL(NRealParticleTypes)
      for (unsigned particle_index = 0; particle_index < NRealParticleTypes; ++particle_index) {
        const auto particle = static_cast<ParticleIDType>(particle_index);
        const float E2 = momentum * momentum + particleMass2[particle];
        const float beta = momentum / sqrtf(E2);
        ckTheta[particle] = acosf(1.f / (nTheta * beta));

        // Use parametrized resolutions:
        // https://gitlab.cern.ch/lhcb/Rec/-/blob/master/Rich/RichFutureRecTrackAlgorithms/src/RichTrackParameterisedCherenkovResolutions.cpp
        Allen::TabulatedFunction1D interp(
          LongMomentum_x[rich][particle], LongMomentum_y[rich][particle], LongMomentum_size[rich][particle]);
        ckRes[particle] = interp.value(momentum);

        // const float sinTheta = sinf(ckTheta[i]);
        // yield[i] = 37.f * pathLength * sinTheta * sinTheta * deltaE;
        const float invBeta2 = E2 / (momentum * momentum);
        float val = 37.f * pathLength * (1.f - invBeta2 / (nYield * nYield)) * deltaE;
        yield[particle] = val > 0.f ? val : 0.f;
      }
      ckTheta[ParticleIDType::BelowThreshold] = NAN;
      ckRes[ParticleIDType::BelowThreshold] = 0.f;
      yield[ParticleIDType::BelowThreshold] = 0.f;
    }

    /**
     * @brief Constructor computing yield-weighted Cherenkov angles and using detector spectra efficiencies.
     */
    __device__ ParticleHypos(
      const float momentum,
      const float pathLength,
      const std::array<float, Allen::Rich::NPhotonSpectraBins>& paraWDiff,
      const std::array<float, Allen::Rich::NPhotonSpectraBins>& spectraEffs,
      const std::array<float, Allen::Rich::NPhotonSpectraBins>& refIndexPerBin,
      const Detector::DetectorType rich)
    {

      const float momentum2 = momentum * momentum;
      const float binSize =
        (Allen::Rich::MaxPhotonEnergy - Allen::Rich::MinPhotonEnergy) / Allen::Rich::NPhotonSpectraBins;

      // Loop over (real) PID types. Below threshold excluded.
      UNROLL(NRealParticleTypes)
      for (unsigned particle_index = 0; particle_index < NRealParticleTypes; ++particle_index) {
        const auto particle = static_cast<ParticleIDType>(particle_index);

        float detectableSignal = 0.f;
        float angleSum = 0.f;

        // Compute some segment ID dependent parameters
        const auto Esq = momentum2 + particleMass2[particle];
        const auto invBetaSqA = 37.f * (momentum2 > 0.f ? Esq / momentum2 : 0.f);
        const auto invGammaSqE = (Esq > 0.f ? particleMass2[particle] / Esq : 0.f) * binSize;

        // MassHypothesisRings acceptance check would go here.
        // In HLT2, this verifies that at least one point of the CK ring
        // for this hypothesis was ray-traced onto an active PMT

        // compute track beta
        // Adapted from Lbcom/Rich/RichFutureTools/src/RichParticleProperties.cpp
        const auto beta = momentum / sqrtf(Esq);

        // loop over the energy bins
        for (unsigned int iEnBin = 0; iEnBin < Allen::Rich::NPhotonSpectraBins; ++iEnBin) {

          /// Emitted yields and spectra
          // Ported from Rich/RichFutureRecTrackAlgorithms/src/RichEmittedPhotonYields.cpp
          // Compute number of photons
          const auto nPhot = invBetaSqA * (paraWDiff[iEnBin] - invGammaSqE);

          // bin signal
          const auto emittedBinSignal = pathLength * (nPhot > 0.f ? nPhot : 0.f);
          /// End of emitted yields and spectra

          /// Detectable yields and spectra
          // Ported from Rec/Rich/RichFutureRecTrackAlgorithms/src/RichDetectablePhotonYields.cpp
          // Scale emitted signal by precomputed detector efficiency
          const auto detectableBinSignal = emittedBinSignal * spectraEffs[iEnBin];

          if (detectableBinSignal > 0.f) {
            detectableSignal += detectableBinSignal;

            /// End of detectable yields and spectra
            /// CK angle computation
            // Ported from Rec/Rich/RichFutureRecTrackAlgorithms/src/RichTrackCherenkovAngles.cpp
            // if beta > 0?
            const auto temp = beta * refIndexPerBin[iEnBin];
            if (temp > 1.f) {
              angleSum += detectableBinSignal * acosf(1.f / temp);
            }
            /// End of CK angle computation
          }
        } // Energy bins

        yield[particle] = detectableSignal;
        ckTheta[particle] =
          (detectableSignal > 0.f) ? (angleSum / detectableSignal) : NAN; // Normalise the angle by total yield

        // Use parametrized resolutions for now
        // TODO: probably should adapt Rich/RichFutureRecTrackAlgorithms/src/RichTrackFunctionalCherenkovResolutions.cpp
        Allen::TabulatedFunction1D interp(
          LongMomentum_x[rich][particle], LongMomentum_y[rich][particle], LongMomentum_size[rich][particle]);
        ckRes[particle] = interp.value(momentum);
      } // Particle hypos
      ckTheta[ParticleIDType::BelowThreshold] = NAN;
      ckRes[ParticleIDType::BelowThreshold] = 0.f;
      yield[ParticleIDType::BelowThreshold] = 0.f;
    }

    HypoData<float> ckTheta {};
    HypoData<float> ckRes {};
    HypoData<float> yield {};
  };
} // namespace Allen::Rich
